"""Multi-version table storage.

Every row key maps to a *version chain* (a small list).  Each version stores:

* ``row``  – the full column tuple,
* ``xmin`` – id of the transaction that created it,
* ``xmax`` – id of the transaction that deleted/overwrote it (0 = alive).

Visibility is decided by the :class:`~minidb.txn.manager.Snapshot` handed to
each read.  Old versions are garbage collected once every live transaction
is guaranteed to see the newer state.

Concurrency
-----------
A per-table re-entrant *latch* (NOT a transaction lock) guards the B+ tree's
physical structure: a reader that traversed the tree while another thread
was mid-split (or mid key-removal in GC) could otherwise follow stale
pointers.  The latch is held only for the duration of one tree operation –
``scan`` materialises its visible rows under it and releases immediately –
so it never spans I/O or a wait on a row lock and does not break the
readers-never-block-on-writers MVCC property.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator, Optional

from .bptree import BPlusTree
from .schema import TableSchema


class Version:
    __slots__ = ("row", "xmin", "xmax")

    def __init__(self, row: tuple[Any, ...], xmin: int, xmax: int = 0) -> None:
        self.row = row
        self.xmin = xmin
        self.xmax = xmax

    def to_dict(self) -> dict[str, Any]:
        return {"row": list(self.row), "xmin": self.xmin, "xmax": self.xmax}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Version":
        return Version(row=tuple(d["row"]), xmin=d["xmin"], xmax=d.get("xmax", 0))


class DuplicateKeyError(Exception):
    pass


class Table:
    """A B+ tree of version chains."""

    def __init__(self, schema: TableSchema, order: int = 32) -> None:
        self.schema = schema
        self.tree: BPlusTree[Any, list[Version]] = BPlusTree(order=order)
        # monotonic surrogate key for tables without a user primary key
        self.next_rowid = 1
        self._rowid_lock = threading.Lock()
        self.has_user_pk = bool(schema.primary_key_columns)
        # structural latch – see module docstring
        self.latch = threading.RLock()

    def allocate_rowid(self) -> int:
        with self._rowid_lock:
            key = self.next_rowid
            self.next_rowid += 1
            return key

    # ------------------------------------------------------------------ #
    # key helpers
    # ------------------------------------------------------------------ #
    @property
    def pk_indexes(self) -> list[int]:
        return [i for i, c in enumerate(self.schema.columns) if c.primary_key]

    def user_key_of(self, row: tuple[Any, ...]) -> Any:
        idxs = self.pk_indexes
        if len(idxs) == 1:
            return row[idxs[0]]
        return tuple(row[i] for i in idxs)

    def bump_rowid(self, key: Any) -> None:
        """Advance the surrogate-key counter past a replayed key (recovery)."""
        if not self.has_user_pk and isinstance(key, int) and key >= self.next_rowid:
            self.next_rowid = key + 1

    # ------------------------------------------------------------------ #
    # writes (the executor holds the row X lock for ``key``)
    # ------------------------------------------------------------------ #
    def insert(self, txid: int, row: tuple[Any, ...], key: Any = None) -> Any:
        """Insert a new version.

        For user-PK tables the key is derived from the row; for surrogate
        tables the caller must pass an already-allocated ``key`` (so the
        locked key, WAL key and tree key are all the same value).
        """
        if key is None:
            if not self.has_user_pk:
                key = self.allocate_rowid()
            else:
                key = self.user_key_of(row)
        with self.latch:
            chain = self.tree.get(key, None)
            version = Version(row=row, xmin=txid)
            if chain is None:
                self.tree.insert(key, [version])
            else:
                # Caller verified no visible live version exists.
                chain.append(version)
        return key

    def update(self, txid: int, key: Any, new_row: tuple[Any, ...]) -> None:
        """Install a new version; caller holds the row lock and a visible
        version exists."""
        with self.latch:
            chain = self.tree[key]
            for v in reversed(chain):
                if v.xmin == txid and v.xmax == 0:
                    v.row = new_row  # own uncommitted update: mutate in place
                    return
            chain[-1].xmax = txid
            chain.append(Version(row=new_row, xmin=txid))

    def delete(self, txid: int, key: Any) -> bool:
        with self.latch:
            chain = self.tree[key]
            for v in reversed(chain):
                if v.xmin == txid and v.xmax == 0:
                    # own uncommitted version – remove outright
                    chain.remove(v)
                    if not chain:
                        self.tree.delete(key)
                    return True
            current = chain[-1]
            if current.xmax != 0:
                return False
            current.xmax = txid
            return True

    def append_replayed(self, key: Any, version: Version) -> None:
        """Recovery-only: attach a version reconstructed from the WAL."""
        with self.latch:
            chain = self.tree.get(key, None)
            if chain is None:
                self.tree.insert(key, [version])
            else:
                chain.append(version)

    def mark_deleted_replayed(self, txid: int, key: Any) -> None:
        """Recovery-only: apply a DELETE record to the latest version."""
        with self.latch:
            self.tree[key][-1].xmax = txid

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def visible_version(self, key: Any, snapshot) -> Optional[Version]:
        with self.latch:
            chain = self.tree.get(key, None)
            if chain is None:
                return None
            # tuple() snapshots the chain so visibility can't race an append
            return _pick_visible(tuple(chain), snapshot)

    def scan(self, snapshot) -> list[tuple[Any, tuple[Any, ...]]]:
        """Return visible ``(key, row)`` pairs (materialised under latch)."""
        out: list[tuple[Any, tuple[Any, ...]]] = []
        with self.latch:
            for key, chain in self.tree.items():
                v = _pick_visible(tuple(chain), snapshot)
                if v is not None:
                    out.append((key, v.row))
        return out

    def count_visible(self, snapshot) -> int:
        with self.latch:
            return sum(
                1 for chain in self.tree.values()
                if _pick_visible(tuple(chain), snapshot) is not None
            )

    def iter_chains(self):
        """Snapshot every chain for an offline operation (checkpoint/GC)."""
        with self.latch:
            return [(key, list(chain)) for key, chain in self.tree.items()]

    def replace_chains(self, snapshot_pairs) -> None:
        """GC/checkpoint helper: atomically replace all chain contents."""
        with self.latch:
            live_keys: set[Any] = set()
            for key, chain in snapshot_pairs:
                live_keys.add(key)
                if chain:
                    self.tree.insert(key, chain)
            for existing_key, _ in self.tree.items():
                if existing_key not in live_keys:
                    self.tree.delete(existing_key)

    # ------------------------------------------------------------------ #
    # garbage collection
    # ------------------------------------------------------------------ #
    def gc(self, is_committed, oldest_active: int) -> int:
        """Reclaim dead versions.

        A version is reclaimable when every transaction that could still see
        it has finished: aborted creators below the low-water mark, or
        versions deleted by a committed transaction below that mark.
        """
        reclaimed = 0
        new_pairs = []
        for key, chain in self.iter_chains():
            alive: list[Version] = []
            for v in chain:
                if (
                    v.xmin < oldest_active
                    and is_committed(v.xmin) is False
                ):
                    reclaimed += 1
                    continue
                if (
                    v.xmax
                    and v.xmax < oldest_active
                    and is_committed(v.xmax)
                ):
                    reclaimed += 1
                    continue
                alive.append(v)
            new_pairs.append((key, alive))
        self.replace_chains([(k, c) for k, c in new_pairs if c])
        return reclaimed


def _pick_visible(chain, snapshot) -> Optional[Version]:
    """Return the newest version visible under ``snapshot``.

    Newer versions sit at the end of the chain.
    """
    for v in reversed(chain):
        if snapshot.visible(v):
            return v
    return None
