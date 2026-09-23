"""Lock manager: multi-granularity locking, deadlock detection, escalation.

Lock modes (weakest to strongest):

    IS  intention shared   - transaction may read some rows
    IX  intention exclusive- transaction may write some rows
    S   shared (table)     - reads all rows / predicate read
    X   exclusive          - writes all rows (DDL, escalated writes)

Compatibility matrix (requested vs granted)::

              IS   IX    S     X
        IS    +    +     +     -
        IX    +    +     -     -
        S     +    -     +     -
        X     -    -     -     -

A transaction that holds IX on a table and requests S (or vice versa) is
upgraded to X, since the union is exclusive.

Escalation
----------
More than ``escalation_threshold`` row locks on one table are replaced by a
single table lock (S when all row locks are S, X otherwise).

Deadlock handling
-----------------
Before blocking, a waits-for graph (waiting txn -> conflicting holders) is
searched depth-first for a cycle; the youngest transaction in the cycle
(highest id) is aborted with :class:`DeadlockError`.
"""

import threading
from collections import defaultdict

from ..errors import DeadlockError, LockTimeoutError

IS = "IS"
IX = "IX"
S = "S"
X = "X"

COMPATIBLE = {
    (IS, IS): True,  (IS, IX): True,  (IS, S): True,  (IS, X): False,
    (IX, IS): True,  (IX, IX): True,  (IX, S): False, (IX, X): False,
    (S, IS): True,   (S, IX): False,  (S, S): True,   (S, X): False,
    (X, IS): False,  (X, IX): False,  (X, S): False,  (X, X): False,
}

# Resulting mode when one transaction already holds ``held`` and requests
# ``requested`` on the same resource.
COMBINE = {
    (IS, IS): IS, (IS, IX): IX, (IS, S): S, (IS, X): X,
    (IX, IS): IX, (IX, IX): IX, (IX, S): X, (IX, X): X,
    (S, IS): S,   (S, IX): X,   (S, S): S,  (S, X): X,
    (X, IS): X,   (X, IX): X,   (X, S): X,  (X, X): X,
}

MODE_RANK = {IS: 0, IX: 1, S: 2, X: 3}


class _LockState:
    __slots__ = ("holders", "cond", "waiters")

    def __init__(self):
        self.holders = {}   # txn id -> granted mode
        self.waiters = []   # [(txn id, requested mode)] FIFO


class LockManager:
    def __init__(self, escalation_threshold=200, lock_timeout=10.0):
        self._cond = threading.Condition()
        self._locks = {}                        # resource -> _LockState
        self._txn_locks = defaultdict(dict)     # txn -> {resource: mode}
        self._txn_row_counts = defaultdict(lambda: defaultdict(int))
        self.escalation_threshold = escalation_threshold
        self.lock_timeout = lock_timeout

    # ------------------------------------------------------------- acquire

    def _state(self, resource):
        st = self._locks.get(resource)
        if st is None:
            st = _LockState()
            self._locks[resource] = st
        return st

    def _conflicts_with_others(self, st, txn_id, mode):
        for holder, held_mode in st.holders.items():
            if holder != txn_id and not COMPATIBLE[(mode, held_mode)]:
                return True
        return False

    def _acquire(self, txn_id, resource, mode):
        st = self._state(resource)
        held = st.holders.get(txn_id)
        if held is not None:
            new_mode = COMBINE[(held, mode)]
            if new_mode == held:
                self._txn_locks[txn_id][resource] = held
                return held
            # Upgrade: go to the head of the queue and wait for conflicts to
            # drain (other holders can only shrink as transactions finish).
            st.waiters.insert(0, (txn_id, new_mode))
            try:
                while self._conflicts_with_others(st, txn_id, new_mode):
                    self._wait(txn_id, resource)
            finally:
                if st.waiters and st.waiters[0] == (txn_id, new_mode):
                    st.waiters.pop(0)
            st.holders[txn_id] = new_mode
            self._txn_locks[txn_id][resource] = new_mode
            return new_mode

        st.waiters.append((txn_id, mode))
        try:
            while True:
                # FIFO fairness: only the head waiter may be granted, so a
                # stream of weak-mode requests cannot starve an X request.
                if st.waiters[0] == (txn_id, mode) and \
                        not self._conflicts_with_others(st, txn_id, mode):
                    break
                self._wait(txn_id, resource)
        finally:
            try:
                st.waiters.remove((txn_id, mode))
            except ValueError:
                pass
        st.holders[txn_id] = mode
        self._txn_locks[txn_id][resource] = mode
        return mode

    def _wait(self, txn_id, resource):
        victim = self._find_deadlock_victim(txn_id)
        if victim is not None:
            raise DeadlockError(
                f"deadlock detected; transaction {victim} chosen as victim")
        if not self._cond.wait(timeout=self.lock_timeout):
            raise LockTimeoutError(
                f"transaction {txn_id} timed out waiting on lock {resource}")

    def _find_deadlock_victim(self, waiter):
        """DFS over the waits-for graph starting at ``waiter``."""

        def neighbors(txn):
            edges = set()
            for state in self._locks.values():
                modes = {m for (tid, m) in state.waiters if tid == txn}
                if not modes:
                    continue
                # Use the strongest mode this txn is waiting for here.
                wmode = max(modes, key=lambda m: MODE_RANK[m])
                for holder, held_mode in state.holders.items():
                    if holder != txn and not COMPATIBLE[(wmode, held_mode)]:
                        edges.add(holder)
            return edges

        stack = [(waiter, (waiter,))]
        visited = {waiter}
        while stack:
            node, path = stack.pop()
            for nxt in neighbors(node):
                if nxt == waiter and len(path) > 1:
                    return max(path)   # youngest loses
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append((nxt, path + (nxt,)))
        return None

    # ------------------------------------------------------------- public API

    def lock_row(self, txn_id, table, key, mode=X):
        with self._cond:
            table_res = ("T", table)
            held_table = self._txn_locks.get(txn_id, {}).get(table_res)
            if held_table == X or (held_table == S and mode == S):
                return
            resource = ("R", table, key)
            self._acquire(txn_id, resource, mode)
            self._txn_row_counts[txn_id][table] += 1
            if self._txn_row_counts[txn_id][table] > self.escalation_threshold:
                self._escalate(txn_id, table)

    def lock_table(self, txn_id, table, mode=X):
        with self._cond:
            self._acquire(txn_id, ("T", table), mode)

    def lock_value(self, txn_id, resource, mode=X):
        """Acquire a user-defined lock resource (used for UNIQUE values)."""
        with self._cond:
            self._acquire(txn_id, resource, mode)

    def _escalate(self, txn_id, table):
        """Row locks on ``table`` -> one table lock (row -> table escalation)."""
        want_x = False
        to_release = []
        for res, mode in list(self._txn_locks[txn_id].items()):
            if res[0] == "R" and res[1] == table:
                if mode == X:
                    want_x = True
                to_release.append(res)
        self._acquire(txn_id, ("T", table), X if want_x else S)
        for res in to_release:
            st = self._locks.get(res)
            if st is not None:
                st.holders.pop(txn_id, None)
                if not st.holders and not st.waiters:
                    self._locks.pop(res, None)
            self._txn_locks[txn_id].pop(res, None)
        self._txn_row_counts[txn_id].pop(table, None)
        self._cond.notify_all()

    def release_all(self, txn_id):
        with self._cond:
            for resource in list(self._txn_locks.get(txn_id, {}).keys()):
                st = self._locks.get(resource)
                if st is not None:
                    st.holders.pop(txn_id, None)
                    if not st.holders and not st.waiters:
                        self._locks.pop(resource, None)
            self._txn_locks.pop(txn_id, None)
            self._txn_row_counts.pop(txn_id, None)
            self._cond.notify_all()

    def held_locks(self, txn_id):
        return dict(self._txn_locks.get(txn_id, {}))

    def active_locks(self):
        """Diagnostic snapshot of current grants."""
        with self._cond:
            return {str(r): dict(st.holders) for r, st in self._locks.items()}
