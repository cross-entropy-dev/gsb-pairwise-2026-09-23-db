"""Lock manager with row locks, table locks, escalation and deadlock detection.

Lock ordering rule (important, avoids AB-BA deadlocks in the manager itself):
every mutation of a resource's holder/waiter sets takes the global ``_meta``
lock first and the resource's ``cv`` second.  Grant decisions read holder
sets while holding ``cv``.  Deadlock detection takes only ``_meta`` and is
invoked while *not* holding any ``cv``.

Lock model
----------

* **Row locks** ``S`` / ``X`` keyed by ``(table, key)``.
* **Table locks** ``IS`` / ``IX`` / ``S`` / ``X`` with the standard
  intention-lock matrix.
* Writers take ``IX`` on the table then ``X`` per touched row.
* ``SERIALIZABLE`` readers take a table ``S`` lock (simple, phantom-free
  S2PL); ``READ COMMITTED`` readers take no long-term locks (MVCC).

Escalation: after ``escalation_threshold`` row locks on one table, row locks
are released and the transaction upgrades to a table ``X`` lock.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Optional

#          held: IS    IX    S     X    (rows = requested mode)
_TABLE_COMPAT = {
    "IS": {"IS": True, "IX": True, "S": True, "X": False},
    "IX": {"IS": True, "IX": True, "S": False, "X": False},
    "S": {"IS": True, "IX": False, "S": True, "X": False},
    "X": {"IS": False, "IX": False, "S": False, "X": False},
}
_MODE_RANK = {"IS": 0, "IX": 1, "S": 2, "X": 3}


class LockError(Exception):
    pass


class DeadlockError(LockError):
    def __init__(self, txid: int, cycle: list[int]) -> None:
        self.txid = txid
        self.cycle = cycle
        super().__init__(
            f"deadlock detected; victim txn {txid} (cycle: "
            f"{' -> '.join(map(str, cycle + [cycle[0]]))})"
        )


class _Resource:
    __slots__ = ("holders", "waiters", "cv")

    def __init__(self) -> None:
        self.holders: dict[int, str] = {}
        self.waiters: list[tuple[int, str, threading.Event]] = []
        self.cv = threading.Condition()


class LockManager:
    def __init__(self, escalation_threshold: int = 100,
                 lock_timeout: float = 30.0) -> None:
        self.escalation_threshold = escalation_threshold
        self.lock_timeout = lock_timeout
        self._resources: dict[tuple, _Resource] = defaultdict(_Resource)
        self._tx_rows: dict[int, dict[str, set]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._tx_tables: dict[int, dict[str, str]] = defaultdict(dict)
        # txid -> {table} whose row locks were escalated away
        self._escalated: dict[int, set[str]] = defaultdict(set)
        self._meta = threading.RLock()

    # ------------------------------------------------------------------ #
    # grant predicates (caller holds the resource cv)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _grantable(res: _Resource, kind: str, mode: str, requester: int) -> bool:
        for txid, held in res.holders.items():
            if txid == requester:
                continue
            if kind == "row":
                if mode == "X" or held == "X":
                    return False
            else:
                if not _TABLE_COMPAT[held][mode]:
                    return False
        return True

    # ------------------------------------------------------------------ #
    # generic wait loop
    # ------------------------------------------------------------------ #
    def _acquire(self, key: tuple, kind: str, txid: int, mode: str,
                 timeout: Optional[float]) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        res = self._resources[key]
        event = threading.Event()

        with res.cv:
            if self._grantable(res, kind, mode, txid):
                return
            res.waiters.append((txid, mode, event))
        try:
            while True:
                # ---- deadlock detection WITHOUT holding any cv ----
                victim = self._detect_deadlock()
                if victim == txid:
                    raise DeadlockError(txid, self._cycle_for(txid))

                with res.cv:
                    if self._grantable(res, kind, mode, txid):
                        return
                    event.clear()
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LockError(
                            f"txn {txid} timed out waiting for {mode} lock "
                            f"on {key[1]!r}"
                        )
                    wait = min(0.25, remaining)
                else:
                    wait = 0.25
                event.wait(timeout=wait)
        finally:
            with res.cv:
                res.waiters = [w for w in res.waiters if w[2] is not event]

    def _wake(self, res: _Resource) -> None:
        """Caller must NOT hold res.cv (may hold _meta)."""
        with res.cv:
            for _, _, ev in res.waiters:
                ev.set()
            res.cv.notify_all()

    # ------------------------------------------------------------------ #
    # table locks
    # ------------------------------------------------------------------ #
    def acquire_table(self, txid: int, table: str, mode: str,
                      timeout: Optional[float] = None) -> None:
        key = ("T", table)
        with self._meta:
            res = self._resources[key]
            with res.cv:
                held = self._tx_tables[txid].get(table)
                if (held is not None
                        and self._mode_covers(held, mode)):
                    return
        self._acquire(key, "table", txid, mode,
                      timeout if timeout is not None else self.lock_timeout)
        with self._meta:
            res = self._resources[key]
            with res.cv:
                self._record_table(txid, table, mode)
                self._wake_locked(res)

    @staticmethod
    def _mode_covers(held: str, requested: str) -> bool:
        if held == "X" or held == requested:
            return True
        if held == "S" and requested in ("IS", "S"):
            return True
        if held == "IX" and requested == "IS":
            return True
        return False

    def _record_table(self, txid: int, table: str, mode: str) -> None:
        """Caller holds _meta + cv."""
        held = self._tx_tables[txid].get(table)
        if held is None:
            new_mode = mode
        elif held == "X":
            new_mode = "X"
        elif {held, mode} == {"S", "IX"}:
            new_mode = "X"
        else:
            new_mode = held if _MODE_RANK[held] >= _MODE_RANK[mode] else mode
        self._tx_tables[txid][table] = new_mode
        self._resources[("T", table)].holders[txid] = new_mode

    # ------------------------------------------------------------------ #
    # row locks + escalation
    # ------------------------------------------------------------------ #
    def acquire_row(self, txid: int, table: str, row_key: object,
                    mode: str = "X", timeout: Optional[float] = None) -> None:
        # escalation already happened: everything goes through the table lock
        with self._meta:
            if table in self._escalated.get(txid, set()):
                self.acquire_table(txid, table, "X", timeout)
                return
        self.acquire_table(
            txid, table, "IX" if mode == "X" else "IS", timeout
        )
        key = ("R", table, row_key)
        self._acquire(key, "row", txid, mode,
                      timeout if timeout is not None else self.lock_timeout)
        with self._meta:
            res = self._resources[key]
            with res.cv:
                res.holders[txid] = mode
                self._tx_rows[txid][table].add(row_key)
                self._wake_locked(res)
                if (len(self._tx_rows[txid][table])
                        >= self.escalation_threshold
                        and table not in self._escalated[txid]):
                    self._escalate(txid, table)

    def _escalate(self, txid: int, table: str) -> None:
        """Caller holds _meta (not the row cvs)."""
        self._escalated[txid].add(table)
        # release all row locks of this txn on the table
        for row_key in list(self._tx_rows[txid][table]):
            rkey = ("R", table, row_key)
            res = self._resources.get(rkey)
            if res is None:
                continue
            with res.cv:
                res.holders.pop(txid, None)
                self._wake_locked(res)
            if not res.holders and not res.waiters:
                self._resources.pop(rkey, None)
        self._tx_rows[txid][table].clear()
        # upgrade table lock if immediately possible; otherwise the next
        # acquire_row() performs the blocking upgrade
        tres = self._resources[("T", table)]
        with tres.cv:
            if self._grantable(tres, "table", "X", txid):
                self._record_table(txid, table, "X")
                self._wake_locked(tres)

    def _wake_locked(self, res: _Resource) -> None:
        for _, _, ev in res.waiters:
            ev.set()
        res.cv.notify_all()

    # ------------------------------------------------------------------ #
    # deadlock detection (wait-for graph cycle)
    # ------------------------------------------------------------------ #
    def _build_graph(self) -> dict[int, set[int]]:
        """Caller holds _meta."""
        graph: dict[int, set[int]] = defaultdict(set)
        for res in self._resources.values():
            holders = set(res.holders)
            for w_txid, _, _ in res.waiters:
                graph[w_txid].update(h for h in holders if h != w_txid)
        return graph

    def _detect_deadlock(self) -> Optional[int]:
        """Youngest txid (highest id) participating in any wait-for cycle."""
        with self._meta:
            graph = self._build_graph()
            victims: set[int] = set()
            for start in list(graph):
                # iterative DFS with path
                stack: list[tuple[int, tuple]] = [(start, (start,))]
                seen_from: dict[int, set] = {}
                while stack:
                    node, path = stack.pop()
                    for nxt in graph.get(node, ()):  # type: ignore[arg-type]
                        if nxt in path:
                            cycle = set(path[path.index(nxt):]) if nxt in path else set(path)
                            victims.update(cycle)
                            break
                        seen = seen_from.setdefault(node, set())
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append((nxt, path + (nxt,)))
            return max(victims) if victims else None

    def _cycle_for(self, txid: int) -> list[int]:
        with self._meta:
            graph = self._build_graph()
            queue = [(txid, [txid])]
            seen = {txid}
            while queue:
                node, path = queue.pop(0)
                for nxt in graph.get(node, ()):  # type: ignore[arg-type]
                    if nxt == txid:
                        return path
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append((nxt, path + [nxt]))
        return [txid]

    # ------------------------------------------------------------------ #
    # release / introspection
    # ------------------------------------------------------------------ #
    def release_all(self, txid: int) -> None:
        touched: list[_Resource] = []
        with self._meta:
            for table in self._tx_tables.pop(txid, {}):
                res = self._resources.get(("T", table))
                if res is not None:
                    with res.cv:
                        res.holders.pop(txid, None)
                    touched.append(res)
                    if not res.holders and not res.waiters:
                        self._resources.pop(("T", table), None)
            for table, keys in self._tx_rows.pop(txid, {}).items():
                for row_key in keys:
                    res = self._resources.get(("R", table, row_key))
                    if res is not None:
                        with res.cv:
                            res.holders.pop(txid, None)
                        touched.append(res)
                        if not res.holders and not res.waiters:
                            self._resources.pop(("R", table, row_key), None)
            self._escalated.pop(txid, None)
        for res in touched:
            self._wake(res)

    def held_row_locks(self, txid: int) -> int:
        with self._meta:
            return sum(len(keys) for keys in self._tx_rows.get(txid, {}).values())

    def held_table_modes(self, txid: int) -> dict[str, str]:
        with self._meta:
            return dict(self._tx_tables.get(txid, {}))
