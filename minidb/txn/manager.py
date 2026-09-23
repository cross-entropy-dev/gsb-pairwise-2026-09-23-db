"""Transaction manager: ids, snapshots, commit/abort, MVCC visibility.

Visibility model (PostgreSQL-style)
-----------------------------------

Each :class:`~minidb.storage.table.Version` carries ``xmin`` (creating txn)
and ``xmax`` (deleting/overwriting txn, 0 = still alive).  A
:class:`Snapshot` freezes:

* ``taken_at``  – the next-txid counter at snapshot time;
* ``active``    – txids that were in progress then.

A version is visible when its creating transaction committed *before* the
snapshot and its deleting transaction had not committed by then.  A
transaction always sees its own writes.

``READ COMMITTED`` takes a fresh snapshot per statement; ``SERIALIZABLE``
takes one snapshot at first read and, additionally, readers take table-level
``S`` locks (see :mod:`minidb.txn.locks`) which makes schedules
conflict-serializable (S2PL) on top of MVCC.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .locks import LockManager
from .wal import WAL


class Isolation(Enum):
    READ_COMMITTED = "READ COMMITTED"
    SERIALIZABLE = "SERIALIZABLE"


class TxnState(Enum):
    ACTIVE = "active"
    COMMITTED = "committed"
    ABORTED = "aborted"


class TransactionError(Exception):
    pass


class NotInTransaction(TransactionError):
    pass


@dataclass
class Snapshot:
    txid: int
    taken_at: int
    active: frozenset[int]
    # injected by the manager: txid -> True/False/None
    status_fn: Optional[Callable[[int], Optional[bool]]] = None

    def created_visible(self, xmin: int, is_committed=None) -> bool:
        if is_committed is None:
            is_committed = self.status_fn
        if xmin == 0:
            return True  # materialised from a durable checkpoint
        if xmin == self.txid:
            return True
        if xmin >= self.taken_at:
            return False
        if xmin in self.active:
            return False
        return is_committed(xmin) is True

    def deleted_visible(self, xmax: int, is_committed=None) -> bool:
        """True => the deletion is visible (the row is gone)."""
        if is_committed is None:
            is_committed = self.status_fn
        if xmax == 0:
            return False
        if xmax == self.txid:
            return True  # our own delete hides it from us
        if xmax >= self.taken_at:
            return False
        if xmax in self.active:
            return False
        return is_committed(xmax) is True

    def visible(self, version) -> bool:
        return self.version_visible(version)

    def version_visible(self, version, is_committed=None) -> bool:
        if is_committed is None:
            is_committed = self.status_fn
        if not self.created_visible(version.xmin, is_committed):
            return False
        if self.deleted_visible(version.xmax, is_committed):
            return False
        return True


@dataclass
class Transaction:
    txid: int
    isolation: Isolation
    state: TxnState = TxnState.ACTIVE
    snapshot: Optional[Snapshot] = None
    # tables this txn has written (for DDL conflict checks / info)
    written_tables: set[str] = field(default_factory=set)
    did_ddl: bool = False
    # WAL BEGIN is only written when the transaction first mutates, so
    # read-only transactions never touch (or fsync) the log.
    wal_started: bool = False

    @property
    def is_read_only(self) -> bool:
        return not self.wal_started


class TransactionManager:
    def __init__(self, wal: WAL, lock_manager: LockManager) -> None:
        self.wal = wal
        self.locks = lock_manager
        self._counter = itertools.count(1)
        self._next_id = 1
        self._cond = threading.Condition()
        self._active: set[int] = set()
        self._state: dict[int, TxnState] = {}
        self._txns: dict[int, Transaction] = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def begin(self, isolation: Isolation = Isolation.READ_COMMITTED) -> Transaction:
        with self._cond:
            txid = next(self._counter)
            self._next_id = txid + 1
            txn = Transaction(txid=txid, isolation=isolation)
            self._active.add(txid)
            self._state[txid] = TxnState.ACTIVE
            self._txns[txid] = txn
        # NB: the WAL BEGIN record is emitted lazily – see ensure_wal_begin()
        return txn

    def ensure_wal_begin(self, txn: Transaction) -> None:
        """Write the BEGIN record on the transaction's first mutation."""
        if not txn.wal_started:
            self.wal.log_begin(txn.txid)
            txn.wal_started = True

    def get(self, txid: int) -> Transaction:
        try:
            return self._txns[txid]
        except KeyError:
            raise NotInTransaction(f"no transaction {txid}")

    def is_active(self, txid: int) -> bool:
        return self._state.get(txid) == TxnState.ACTIVE

    # ------------------------------------------------------------------ #
    # snapshots
    # ------------------------------------------------------------------ #
    def take_snapshot(self, txid: int) -> Snapshot:
        with self._cond:
            snap = Snapshot(
                txid=txid,
                taken_at=self._next_id,
                active=frozenset(self._active),
                status_fn=self.status,
            )
        return snap

    def snapshot_for(self, txn: Transaction) -> Snapshot:
        """Return the snapshot a statement should read under."""
        if txn.isolation is Isolation.READ_COMMITTED or txn.snapshot is None:
            snap = self.take_snapshot(txn.txid)
            if txn.isolation is Isolation.SERIALIZABLE:
                txn.snapshot = snap  # first read pins it
            return snap
        return txn.snapshot

    def status(self, txid: int) -> Optional[bool]:
        """True committed, False aborted, None in-progress / unknown."""
        st = self._state.get(txid)
        if st == TxnState.COMMITTED:
            return True
        if st == TxnState.ABORTED:
            return False
        return None

    # ------------------------------------------------------------------ #
    # commit / abort
    # ------------------------------------------------------------------ #
    def commit(self, txn: Transaction) -> None:
        if txn.state is not TxnState.ACTIVE:
            raise TransactionError(f"txn {txn.txid} already {txn.state.value}")
        # A read-only transaction wrote nothing: no WAL record needed, so
        # commits of pure reads never fsync and never block on the disk.
        if txn.wal_started:
            # 1. durable commit record BEFORE the commit becomes visible
            self.wal.commit(txn.txid)
        # 2. make visible + wake waiters, then release locks
        with self._cond:
            txn.state = TxnState.COMMITTED
            self._state[txn.txid] = TxnState.COMMITTED
            self._active.discard(txn.txid)
            self._cond.notify_all()
        self.locks.release_all(txn.txid)

    def abort(self, txn: Transaction, reason: Optional[str] = None) -> None:
        if txn.state is not TxnState.ACTIVE:
            return
        if txn.wal_started:
            self.wal.log_abort(txn.txid)
        with self._cond:
            txn.state = TxnState.ABORTED
            self._state[txn.txid] = TxnState.ABORTED
            self._active.discard(txn.txid)
            self._cond.notify_all()
        self.locks.release_all(txn.txid)

    # ------------------------------------------------------------------ #
    # recovery / GC support
    # ------------------------------------------------------------------ #
    def bootstrap_state(self, max_txid: int) -> None:
        """After WAL replay: advance the id counter past replayed txns."""
        with self._cond:
            self._counter = itertools.count(max_txid + 1)
            self._next_id = max_txid + 1
            # mark all replayed ids committed so visibility is stable;
            # non-replayed ids are simply never referenced.
            for i in range(1, max_txid + 1):
                self._state[i] = TxnState.COMMITTED

    def oldest_active(self) -> Optional[int]:
        with self._cond:
            return min(self._active) if self._active else None

    def gc_tick(self, tables) -> int:
        """Run one GC pass over all tables. Returns reclaimed version count."""
        with self._cond:
            oldest = min(self._active) if self._active else self._next_id
        total = 0
        for table in tables.values():
            total += table.gc(self.status, oldest)
        return total

    def active_count(self) -> int:
        with self._cond:
            return len(self._active)

    def shutdown(self) -> None:
        # abort anything left (used by tests / CLI exit)
        with self._cond:
            victims = [self._txns[i] for i in list(self._active)]
        for txn in victims:
            self.abort(txn)
