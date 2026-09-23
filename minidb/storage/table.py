"""Schema objects and the MVCC table storage.

Every primary-key value maps in a B+ tree to a *version chain*.  Versions are
never overwritten in place; writers install a new version and readers pick the
version visible to their snapshot, which is how MVCC snapshots work.

Version timestamps
------------------
``xmin_ts`` / ``xmax_ts`` are commit timestamps:

* ``xmin_ts > 0``  - creating transaction committed at that timestamp
* ``xmin_ts == 0`` - creation is still uncommitted (only visible to its txn)
* ``xmax_ts > 0``  - deleting transaction committed at that timestamp
* ``xmax == txn`` and ``xmax_ts == 0`` - uncommitted delete

Rows loaded from a checkpoint are stamped as committed at timestamp 1, and the
commit timestamp counter starts above that.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

from .btree import BPlusTree

# Commit timestamp stamped on all materialised (checkpoint) rows.
BOOTSTRAP_TS = 1


class RWLock:
    """A simple readers/writer lock: many readers, one writer, writer-preferring."""

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    def acquire_read(self):
        with self._cond:
            while self._writer_active or self._writers_waiting:
                self._cond.wait()
            self._readers += 1

    def release_read(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self):
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer_active or self._readers > 0:
                    self._cond.wait()
                self._writer_active = True
            finally:
                self._writers_waiting -= 1

    def release_write(self):
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()


@dataclass
class Column:
    name: str
    type_name: str  # INT | VARCHAR | TEXT | BOOLEAN | FLOAT
    length: Optional[int] = None
    not_null: bool = False
    primary_key: bool = False
    unique: bool = False
    default: object = None  # already a Python value (literals only)
    autoincrement: bool = False

    def to_dict(self):
        return {
            "name": self.name, "type": self.type_name, "length": self.length,
            "not_null": self.not_null, "primary_key": self.primary_key,
            "unique": self.unique, "default": self.default,
            "autoincrement": self.autoincrement,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"], type_name=d["type"], length=d.get("length"),
            not_null=d.get("not_null", False), primary_key=d.get("primary_key", False),
            unique=d.get("unique", False), default=d.get("default"),
            autoincrement=d.get("autoincrement", False),
        )


@dataclass
class Schema:
    name: str
    columns: list  # list[Column]
    pk_columns: list = field(default_factory=list)

    def column(self, name):
        for c in self.columns:
            if c.name == name:
                return c
        return None

    @property
    def column_names(self):
        return [c.name for c in self.columns]

    def to_dict(self):
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "pk_columns": list(self.pk_columns),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            columns=[Column.from_dict(c) for c in d["columns"]],
            pk_columns=list(d.get("pk_columns", [])),
        )


class Version:
    __slots__ = ("xmin", "xmin_ts", "xmax", "xmax_ts", "data")

    def __init__(self, data, xmin, xmin_ts=0):
        self.xmin = xmin          # creating transaction id
        self.xmin_ts = xmin_ts    # commit timestamp, 0 while uncommitted
        self.xmax = 0             # deleting transaction id
        self.xmax_ts = 0          # delete commit timestamp
        self.data = data          # {column: value}

    def visible(self, snapshot_ts, own_txn=0):
        """Visibility for a snapshot taken at ``snapshot_ts``.

        ``own_txn`` is the reader's transaction id (0 for committed-only reads).
        """
        if self.xmin_ts == 0:
            if self.xmin != own_txn or own_txn == 0:
                return False
        elif self.xmin_ts > snapshot_ts:
            return False
        if self.xmax != 0:
            if self.xmax == own_txn and own_txn != 0:
                return False  # deleted by the reader itself
            if self.xmax_ts != 0 and self.xmax_ts <= snapshot_ts:
                return False
        return True

    def to_dict(self):
        return {"xmin": self.xmin, "xmin_ts": self.xmin_ts,
                "xmax": self.xmax, "xmax_ts": self.xmax_ts, "data": self.data}

    @classmethod
    def from_dict(cls, d):
        v = cls(d["data"], d.get("xmin", 0), d.get("xmin_ts", BOOTSTRAP_TS))
        v.xmax = d.get("xmax", 0)
        v.xmax_ts = d.get("xmax_ts", 0)
        return v


class Table:
    """A table: schema plus a B+ tree of version chains keyed by row key."""

    def __init__(self, schema, btree_order=64):
        self.schema = schema
        self.tree = BPlusTree(order=btree_order)
        self.lock = RWLock()
        # Auto-rowid / autoincrement counters.
        self.next_rowid = 1
        self.auto_counters = {}  # column -> next value
        # One secondary B+ tree per UNIQUE column: value -> primary key.
        self.unique_indexes = {}
        for col in schema.columns:
            if col.unique and col.name not in schema.pk_columns:
                self.unique_indexes[col.name] = BPlusTree(order=btree_order)

    def add_unique_index(self, column):
        if column not in self.unique_indexes:
            self.unique_indexes[column] = BPlusTree(order=self.tree.order)
            idx = self.unique_indexes[column]
            for key, v in self.scan_visible(10 ** 18):
                val = v.data.get(column)
                if val is not None:
                    idx.insert(val, key)

    @property
    def name(self):
        return self.schema.name

    @property
    def pk_columns(self):
        return self.schema.pk_columns

    # ------------------------------------------------------------- key utils

    def make_key(self, row):
        pks = self.pk_columns
        if not pks:
            return row.get("_rowid_")
        if len(pks) == 1:
            return row[pks[0]]
        return tuple(row[c] for c in pks)

    def alloc_rowid(self):
        rid = self.next_rowid
        self.next_rowid += 1
        return rid

    def alloc_auto(self, column):
        nxt = self.auto_counters.get(column, 1)
        self.auto_counters[column] = nxt + 1
        return nxt

    # --------------------------------------------------------- version access

    def get_chain(self, key):
        return self.tree.get(key)

    def visible_version(self, key, snapshot_ts, own_txn=0):
        chain = self.tree.get(key)
        if chain is None:
            return None
        for v in chain:  # newest first
            if v.visible(snapshot_ts, own_txn):
                return v
        return None

    def scan_visible(self, snapshot_ts, own_txn=0):
        """Materialise ``(key, Version)`` visible to a snapshot.

        The read latch is held only while walking the B+ tree and copying
        references; afterwards callers iterate a private list with no lock.
        Version *data* dictionaries are never mutated in place, so the
        snapshot stays coherent even when commits land concurrently.
        """
        self.lock.acquire_read()
        try:
            out = []
            for key, chain in self.tree.items():
                for v in chain:
                    if v.visible(snapshot_ts, own_txn):
                        out.append((key, v))
                        break
            return out
        finally:
            self.lock.release_read()

    def create_chain(self, key, version):
        """Install the first version for a brand-new key.

        Structural B+ tree change (may split); exclusive latch. Returns False
        if another thread created the key concurrently.
        """
        self.lock.acquire_write()
        try:
            if self.tree.get(key) is not None:
                return False
            self.tree.insert(key, [version])
            return True
        finally:
            self.lock.release_write()

    def replace_chain(self, key, chain):
        """Atomically swap the version-chain list for an existing key.

        No structural tree change (value slot replacement), and the old list
        object stays valid for any thread still iterating it - chains are
        copy-on-write.
        """
        self.tree.insert(key, chain)

    def remove_key(self, key):
        self.lock.acquire_write()
        try:
            self.tree.delete(key)
        finally:
            self.lock.release_write()

    def install(self, key, version):
        """Put a brand-new version at the head of the chain."""
        chain = self.tree.get(key)
        if chain is None:
            chain = []
            self.tree.insert(key, chain)
        chain.insert(0, version)

    def prune(self, key, oldest_snapshot):
        """GC versions nobody can see any more. Exclusive latch: pruning a
        fully-dead row deletes its key, a structural tree change."""
        self.lock.acquire_write()
        try:
            chain = self.tree.get(key)
            if chain is None:
                return
            keep = []
            for v in chain:
                if v.xmin_ts == 0 or v.xmax_ts == 0 or v.visible(oldest_snapshot):
                    keep.append(v)
            if not keep:
                self.tree.delete(key)
            elif len(keep) != len(chain):
                self.tree.insert(key, keep)
        finally:
            self.lock.release_write()

    # ----------------------------------------------------- secondary indexes

    def unique_get(self, column, value):
        idx = self.unique_indexes.get(column)
        return idx.get(value) if idx is not None else None

    def maintain_unique_indexes(self, chain, txn_id, commit_ts):
        """Update unique indexes when ``txn_id``'s versions on key commit."""
        if not self.unique_indexes:
            return
        # Latest committed version on the chain after stamping.
        latest = None
        for v in chain:
            if v.xmin_ts > 0 and v.xmax_ts == 0:
                latest = v
                break
        latest_deleted = latest is None
        for column, idx in self.unique_indexes.items():
            # Values this transaction created/removed on this key.
            for v in chain:
                if v.xmin == txn_id and v.xmin_ts:
                    val = v.data.get(column)
                    if val is not None and idx.get(val) is None:
                        idx.insert(val, self.make_key(v.data))
            # If the row is now fully deleted, drop its indexed values.
            if latest_deleted:
                for v in chain:
                    val = v.data.get(column)
                    if val is not None and idx.get(val) is not None:
                        idx.delete(val)

    def rebuild_unique_indexes(self):
        for col in list(self.unique_indexes):
            self.unique_indexes.pop(col)
        for col in self.schema.columns:
            if col.unique and col.name not in self.schema.pk_columns:
                self.unique_indexes[col.name] = BPlusTree(order=self.tree.order)
        for key, chain in self.tree.items():
            latest = None
            for v in chain:
                if v.xmin_ts > 0 and v.xmax_ts == 0:
                    latest = v
                    break
            if latest is None:
                continue
            for column, idx in self.unique_indexes.items():
                val = latest.data.get(column)
                if val is not None:
                    idx.insert(val, key)

    def recompute_counters(self):
        max_rowid = 0
        max_auto = {}
        for key, chain in self.tree.items():
            latest = chain[0]
            for v in chain:
                if v.xmin_ts > 0 and v.xmax_ts == 0:
                    latest = v
                    break
            rid = latest.data.get("_rowid_")
            if isinstance(rid, int):
                max_rowid = max(max_rowid, rid)
            for col in self.schema.columns:
                if col.autoincrement:
                    val = latest.data.get(col.name)
                    if isinstance(val, int):
                        max_auto[col.name] = max(max_auto.get(col.name, 0), val)
        self.next_rowid = max_rowid + 1
        for col, val in max_auto.items():
            self.auto_counters[col] = val + 1

    # ----------------------------------------------------------- persistence

    def committed_image(self):
        """Return ``{key: data}`` for latest committed versions (checkpoint)."""
        image = {}
        for key, chain in self.tree.items():
            for v in chain:
                if v.xmin_ts > 0 and (v.xmax_ts == 0):
                    data = dict(v.data)
                    image[key] = data
                    break
        return image

    def load_image(self, image):
        for key, data in image.items():
            v = Version(dict(data), 0, BOOTSTRAP_TS)
            self.tree.insert(key, [v])

    def meta_to_dict(self):
        return {"schema": self.schema.to_dict(),
                "next_rowid": self.next_rowid,
                "auto_counters": self.auto_counters}

    def load_meta(self, d):
        self.schema = Schema.from_dict(d["schema"])
        self.next_rowid = d.get("next_rowid", 1)
        self.auto_counters = dict(d.get("auto_counters", {}))
