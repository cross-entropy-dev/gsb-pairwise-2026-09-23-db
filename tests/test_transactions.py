"""Transaction / ACID tests: atomicity, MVCC visibility, isolation levels,
rollback, deadlock detection, lock escalation, concurrent writers."""

import os
import shutil
import tempfile
import threading
import unittest

from minidb import Database, Isolation, Session
from minidb.txn.locks import DeadlockError, LockError


class TxnFixture:
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.dir, "db"))
        s = Session(self.db)
        s.sql("CREATE TABLE acct (id INT PRIMARY KEY, bal INT)")
        for i in range(1, 11):
            s.sql(f"INSERT INTO acct VALUES ({i}, {i * 100})")

    def tearDown(self):
        self.db.shutdown()
        shutil.rmtree(self.dir, ignore_errors=True)


class TestAtomicity(TxnFixture, unittest.TestCase):
    def test_rollback_discards_writes(self):
        s = Session(self.db)
        s.sql("BEGIN")
        s.sql("UPDATE acct SET bal = 0 WHERE id = 1")
        s.sql("INSERT INTO acct VALUES (999, 0)")
        s.sql("ROLLBACK")
        self.assertEqual(
            Session(self.db).sql("SELECT bal FROM acct WHERE id = 1").rows[0],
            (100,),
        )
        self.assertEqual(
            Session(self.db).sql("SELECT COUNT(*) FROM acct").rows[0], (10,)
        )

    def test_failed_statement_aborts_autocommit_txn(self):
        s = Session(self.db)
        with self.assertRaises(Exception):
            s.sql("INSERT INTO acct VALUES (1, 0)")  # duplicate PK
        # engine stays usable; locks released
        self.assertEqual(
            Session(self.db).sql("SELECT bal FROM acct WHERE id = 1").rows[0],
            (100,),
        )
        s2 = Session(self.db)
        s2.sql("UPDATE acct SET bal = 101 WHERE id = 1")  # not blocked
        self.assertEqual(
            Session(self.db).sql("SELECT bal FROM acct WHERE id = 1").rows[0],
            (101,),
        )

    def test_partial_multi_insert_is_atomic(self):
        s = Session(self.db)
        with self.assertRaises(Exception):
            s.sql("INSERT INTO acct VALUES (100, 1), (1, 2)")  # 2nd dup
        self.assertEqual(
            Session(self.db).sql("SELECT COUNT(*) FROM acct").rows[0], (10,)
        )


class TestMVCCVisibility(TxnFixture, unittest.TestCase):
    def test_uncommitted_not_visible(self):
        a, b = Session(self.db), Session(self.db)
        a.sql("BEGIN")
        a.sql("UPDATE acct SET bal = -1 WHERE id = 1")
        b.sql("BEGIN")
        self.assertEqual(
            b.sql("SELECT bal FROM acct WHERE id = 1").rows[0], (100,)
        )
        a.sql("COMMIT")
        # READ COMMITTED: a new statement sees the committed value
        self.assertEqual(
            b.sql("SELECT bal FROM acct WHERE id = 1").rows[0], (-1,)
        )
        b.sql("ROLLBACK")

    def test_snapshot_stable_under_serializable(self):
        # SERIALIZABLE = pinned MVCC snapshot + table S lock (S2PL): a
        # concurrent writer blocks, so phantoms are impossible.
        a, b = Session(self.db), Session(self.db)
        b.sql("BEGIN ISOLATION LEVEL SERIALIZABLE")
        b.sql("SELECT COUNT(*) FROM acct")  # pin snapshot + take S lock

        writer_done = threading.Event()

        def writer():
            a.sql("INSERT INTO acct VALUES (100, 100)")
            writer_done.set()

        t = threading.Thread(target=writer)
        t.start()
        self.assertFalse(writer_done.wait(timeout=0.8))
        # repeatable view while the writer is blocked
        self.assertEqual(b.sql("SELECT COUNT(*) FROM acct").rows[0], (10,))
        b.sql("COMMIT")  # release S lock -> writer proceeds
        self.assertTrue(writer_done.wait(timeout=5))
        t.join()
        self.assertEqual(
            Session(self.db).sql("SELECT COUNT(*) FROM acct").rows[0], (11,)
        )

    def test_dirty_row_update_blocks_then_applies(self):
        # two writers to the same row: second blocks, then sees first's commit
        a, b = Session(self.db), Session(self.db)
        a.sql("BEGIN")
        a.sql("UPDATE acct SET bal = 111 WHERE id = 1")

        done = threading.Event()
        result = {}

        def writer():
            b.sql("BEGIN")
            b.sql("UPDATE acct SET bal = 222 WHERE id = 1")
            b.sql("COMMIT")
            result["done"] = True
            done.set()

        t = threading.Thread(target=writer)
        t.start()
        self.assertFalse(done.wait(timeout=0.5))
        a.sql("COMMIT")
        self.assertTrue(done.wait(timeout=5))
        t.join()
        self.assertEqual(
            Session(self.db).sql("SELECT bal FROM acct WHERE id = 1").rows[0],
            (222,),
        )


class TestSerializationConflict(TxnFixture, unittest.TestCase):
    def test_serializable_read_blocks_write(self):
        reader = Session(self.db)
        reader.sql("BEGIN ISOLATION LEVEL SERIALIZABLE")
        reader.sql("SELECT * FROM acct WHERE id = 1")

        writer = Session(self.db)
        blocked = threading.Event()
        proceed = threading.Event()

        def write():
            writer.sql("BEGIN")
            blocked.set()
            writer.sql("UPDATE acct SET bal = 5 WHERE id = 1")
            proceed.set()
            writer.sql("COMMIT")

        t = threading.Thread(target=write)
        t.start()
        blocked.wait(2)
        self.assertFalse(proceed.wait(timeout=0.8))
        reader.sql("COMMIT")
        self.assertTrue(proceed.wait(timeout=5))
        t.join()

    def test_writer_does_not_block_reader(self):
        writer = Session(self.db)
        writer.sql("BEGIN")
        writer.sql("UPDATE acct SET bal = -1 WHERE id = 1")  # uncommitted
        reader = Session(self.db)
        # MVCC: snapshot read is never blocked
        row = reader.sql("SELECT bal FROM acct WHERE id = 1").rows[0]
        self.assertEqual(row, (100,))
        writer.sql("ROLLBACK")


class TestDeadlockAndEscalation(TxnFixture, unittest.TestCase):
    def test_two_transaction_deadlock(self):
        a, b = Session(self.db), Session(self.db)
        a.sql("BEGIN")
        b.sql("BEGIN")
        a.sql("UPDATE acct SET bal = 1 WHERE id = 1")
        b.sql("UPDATE acct SET bal = 2 WHERE id = 2")

        errors: dict[str, BaseException] = {}

        def phase_a():
            try:
                a.sql("UPDATE acct SET bal = 1 WHERE id = 2")
            except BaseException as exc:
                errors["a"] = exc

        def phase_b():
            import time
            time.sleep(0.2)
            try:
                b.sql("UPDATE acct SET bal = 2 WHERE id = 1")
            except BaseException as exc:
                errors["b"] = exc

        t1 = threading.Thread(target=phase_a)
        t2 = threading.Thread(target=phase_b)
        t1.start(); t2.start()
        t1.join(10); t2.join(10)
        self.assertTrue(
            any(isinstance(e, DeadlockError) for e in errors.values()),
            f"expected a deadlock victim, got {errors}",
        )
        # cleanup: rollback whichever survived
        for sess in (a, b):
            try:
                if sess.in_transaction:
                    sess.rollback()
            except Exception:
                pass
        # data stays consistent: victim's updates are gone, other txn rolled
        # back by cleanup
        rows = Session(self.db).sql(
            "SELECT bal FROM acct WHERE id IN (1, 2) ORDER BY id"
        ).rows
        self.assertEqual(rows, [(100,), (200,)])

    def test_lock_escalation_to_table_x(self):
        lm = self.db.lock_manager
        saved = lm.escalation_threshold
        lm.escalation_threshold = 3
        try:
            s = Session(self.db)
            s.sql("BEGIN")
            s.sql("UPDATE acct SET bal = 0 WHERE id <= 5")
            modes = lm.held_table_modes(s._txn.txid)
            self.assertEqual(modes.get("acct"), "X")
            s.sql("ROLLBACK")
        finally:
            lm.escalation_threshold = saved

    def test_lock_timeout(self):
        lm = self.db.lock_manager
        saved = lm.lock_timeout
        lm.lock_timeout = 0.5
        try:
            a, b = Session(self.db), Session(self.db)
            a.sql("BEGIN")
            a.sql("UPDATE acct SET bal = 1 WHERE id = 1")
            b.sql("BEGIN")
            with self.assertRaises(LockError):
                b.sql("UPDATE acct SET bal = 2 WHERE id = 1")
            a.sql("ROLLBACK")
            b.sql("ROLLBACK")
        finally:
            lm.lock_timeout = saved


class TestConcurrentWriters(TxnFixture, unittest.TestCase):
    def test_disjoint_rows_commit_concurrently(self):
        n = 8
        barrier = threading.Barrier(n)
        errors: list[Exception] = []

        def worker(i: int):
            s = Session(self.db)
            try:
                s.sql("BEGIN")
                barrier.wait()
                s.sql(f"UPDATE acct SET bal = bal + 1 WHERE id = {i + 1}")
                s.sql("COMMIT")
            except Exception as exc:
                errors.append(exc)
                if s.in_transaction:
                    s.rollback()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(errors, [])
        for i in range(1, n + 1):
            bal = Session(self.db).sql(
                f"SELECT bal FROM acct WHERE id = {i}"
            ).rows[0][0]
            self.assertEqual(bal, i * 100 + 1)

    def test_concurrent_inserts_unique_pk(self):
        n = 20
        errors: list[Exception] = []

        def worker(i: int):
            s = Session(self.db)
            try:
                s.sql("CREATE TABLE log (id INT PRIMARY KEY)") if False else None
                s.sql(f"INSERT INTO acct VALUES ({1000 + i}, {i})")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(
            Session(self.db).sql("SELECT COUNT(*) FROM acct WHERE id >= 1000").rows[0],
            (n,),
        )

    def test_many_concurrent_transactions(self):
        # requirement: 1000+ concurrent transactions
        n = 1000
        errors: list[Exception] = []

        def worker(i: int):
            s = Session(self.db)
            try:
                s.sql(f"SELECT bal FROM acct WHERE id = {(i % 10) + 1}")
                if i % 3 == 0:
                    s.sql("BEGIN")
                    s.sql(f"UPDATE acct SET bal = bal WHERE id = {(i % 10) + 1}")
                    s.sql("COMMIT")
            except Exception as exc:
                errors.append(exc)
                if s.in_transaction:
                    s.rollback()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
