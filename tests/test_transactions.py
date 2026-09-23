"""Transaction tests: atomicity, MVCC snapshots, isolation levels."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minidb import (Database, READ_COMMITTED, SERIALIZABLE,
                    ConstraintViolationError, TransactionError, DeadlockError)


class TransactionTestBase(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.db.executescript("""
            CREATE TABLE acc (id INT PRIMARY KEY, bal INT);
            INSERT INTO acc VALUES (1, 100), (2, 100);
        """)


class AtomicityTests(TransactionTestBase):
    def test_rollback_undoes_insert_update_delete(self):
        t = self.db.begin()
        self.db.execute("INSERT INTO acc VALUES (3, 50)", txn=t)
        self.db.execute("UPDATE acc SET bal = 0 WHERE id = 1", txn=t)
        self.db.execute("DELETE FROM acc WHERE id = 2", txn=t)
        # Own snapshot shows the inserted row and the deleted one gone.
        self.assertEqual(
            self.db.execute("SELECT id, bal FROM acc ORDER BY id",
                            txn=t).tuples(),
            [(1, 0), (3, 50)])
        self.db.rollback(t)
        self.assertEqual(
            self.db.execute("SELECT id, bal FROM acc ORDER BY id").tuples(),
            [(1, 100), (2, 100)])

    def test_failed_statement_autocommit_leaves_no_trace(self):
        with self.assertRaises(ConstraintViolationError):
            self.db.execute("INSERT INTO acc VALUES (1, 1)")
        self.assertEqual(
            self.db.execute("SELECT bal FROM acc WHERE id = 1").tuples(),
            [(100,)])

    def test_rollback_after_constraint_error_keeps_earlier_changes_undone(self):
        t = self.db.begin()
        self.db.execute("INSERT INTO acc VALUES (3, 50)", txn=t)
        with self.assertRaises(ConstraintViolationError):
            self.db.execute("INSERT INTO acc VALUES (3, 50)", txn=t)
        self.db.rollback(t)
        self.assertEqual(
            self.db.execute("SELECT COUNT(*) FROM acc").tuples(), [(2,)])

    def test_ddl_rollback(self):
        t = self.db.begin()
        self.db.execute("CREATE TABLE tmp (a INT PRIMARY KEY)", txn=t)
        self.db.execute("INSERT INTO tmp VALUES (1)", txn=t)
        self.db.rollback(t)
        with self.assertRaises(Exception):
            self.db.execute("SELECT * FROM tmp")

    def test_double_commit_rejected(self):
        t = self.db.begin()
        self.db.commit(t)
        with self.assertRaises(TransactionError):
            self.db.commit(t)


class IsolationTests(TransactionTestBase):
    def test_read_committed_sees_commits_between_statements(self):
        t = self.db.begin(READ_COMMITTED)
        self.assertEqual(
            self.db.execute("SELECT SUM(bal) FROM acc", txn=t).tuples(),
            [(200,)])
        # Another autocommit transaction commits a change.
        self.db.execute("UPDATE acc SET bal = 200 WHERE id = 1")
        # The ongoing RC transaction sees the new commit on its next stmt.
        self.assertEqual(
            self.db.execute("SELECT SUM(bal) FROM acc", txn=t).tuples(),
            [(300,)])
        self.db.rollback(t)

    def test_uncommitted_changes_invisible(self):
        t1 = self.db.begin()
        self.db.execute("UPDATE acc SET bal = 999 WHERE id = 1", txn=t1)
        t2 = self.db.begin(READ_COMMITTED)
        self.assertEqual(
            self.db.execute("SELECT bal FROM acc WHERE id = 1", txn=t2).tuples(),
            [(100,)])
        self.db.rollback(t1)
        self.db.rollback(t2)

    def test_own_writes_visible_inside_txn(self):
        t = self.db.begin()
        self.db.execute("UPDATE acc SET bal = bal + 10 WHERE id = 1", txn=t)
        self.assertEqual(
            self.db.execute("SELECT bal FROM acc WHERE id = 1", txn=t).tuples(),
            [(110,)])
        self.db.commit(t)

    def test_commit_makes_changes_visible(self):
        t = self.db.begin()
        self.db.execute("INSERT INTO acc VALUES (3, 25)", txn=t)
        self.db.commit(t)
        self.assertEqual(
            self.db.execute("SELECT bal FROM acc WHERE id = 3").tuples(),
            [(25,)])

    def test_serializable_blocks_write_after_predicate_read(self):
        import threading
        import time
        t_scan = self.db.begin(SERIALIZABLE)
        self.db.execute("SELECT * FROM acc WHERE bal > 0", txn=t_scan)

        t_ins = self.db.begin(SERIALIZABLE)
        done = threading.Event()

        def insert_later():
            # Must block on the S table lock held by the scanner.
            self.db.execute("INSERT INTO acc VALUES (9, 1)", txn=t_ins)
            self.db.commit(t_ins)
            done.set()

        th = threading.Thread(target=insert_later)
        th.start()
        self.assertFalse(done.wait(0.3))
        self.db.commit(t_scan)
        self.assertTrue(done.wait(2.0))
        th.join()


class WriteSkewLockTests(TransactionTestBase):
    def test_row_locks_serialize_concurrent_updates(self):
        import threading
        errors = []

        def bump():
            try:
                t = self.db.begin(READ_COMMITTED)
                self.db.execute("UPDATE acc SET bal = bal + 1 WHERE id = 1",
                                txn=t)
                self.db.commit(t)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=bump) for _ in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(errors, [])
        self.assertEqual(
            self.db.execute("SELECT bal FROM acc WHERE id = 1").tuples(),
            [(110,)])

    def test_deadlock_is_detected(self):
        import threading
        t1 = self.db.begin(READ_COMMITTED)
        t2 = self.db.begin(READ_COMMITTED)
        barrier = threading.Barrier(2)
        outcomes = {}

        def txn_body(txn, first, second, label):
            try:
                self.db.execute(
                    f"UPDATE acc SET bal = bal + 1 WHERE id = {first}", txn=txn)
                barrier.wait()
                self.db.execute(
                    f"UPDATE acc SET bal = bal + 1 WHERE id = {second}",
                    txn=txn)
                self.db.commit(txn)
                outcomes[label] = "committed"
            except DeadlockError:
                self.db.rollback(txn)
                outcomes[label] = "deadlock"

        th1 = threading.Thread(target=txn_body, args=(t1, 1, 2, "a"))
        th2 = threading.Thread(target=txn_body, args=(t2, 2, 1, "b"))
        th1.start()
        th2.start()
        th1.join()
        th2.join()
        self.assertEqual(sorted(outcomes.values()), ["committed", "deadlock"])
        # Total balance preserved after victim rollback.
        self.assertEqual(
            self.db.execute("SELECT SUM(bal) FROM acc").tuples(), [(202,)])


class LockEscalationTests(TransactionTestBase):
    def test_escalation_row_to_table(self):
        db = Database(":memory:", lock_escalation_threshold=5)
        db.execute("CREATE TABLE wide (id INT PRIMARY KEY, v INT)")
        for i in range(8):
            db.execute(f"INSERT INTO wide VALUES ({i}, 0)")
        t = db.begin()
        # Updating > 5 rows escalates row locks to one table X lock.
        db.execute("UPDATE wide SET v = 1 WHERE id < 7", txn=t)
        held = db.lm.held_locks(t.txn_id)
        table_locks = [r for r in held if r[0] == "T"]
        self.assertTrue(any(r[1] == "wide" for r in table_locks))
        db.rollback(t)


if __name__ == "__main__":
    unittest.main()
