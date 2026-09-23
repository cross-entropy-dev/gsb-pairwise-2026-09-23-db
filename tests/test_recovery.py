"""Durability and crash-recovery tests (WAL + checkpoints)."""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minidb import Database
from minidb.storage.wal import WAL


class RecoveryTestBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="minidb_recovery_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def reopen(self):
        return Database(self.dir)


class WALRecoveryTests(RecoveryTestBase):
    def test_committed_data_survives_crash(self):
        db = self.reopen()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, v VARCHAR(20))")
        db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        db.execute("UPDATE t SET v = 'aa' WHERE id = 1")
        db.execute("DELETE FROM t WHERE id = 3")
        db.simulate_crash()

        db2 = self.reopen()
        self.assertEqual(
            db2.execute("SELECT id, v FROM t ORDER BY id").tuples(),
            [(1, "aa"), (2, "b")])

    def test_uncommitted_transaction_is_discarded(self):
        db = self.reopen()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
        db.execute("INSERT INTO t VALUES (1)")
        t = db.begin()
        db.execute("INSERT INTO t VALUES (2), (3)", txn=t)
        db.simulate_crash()  # no commit

        db2 = self.reopen()
        self.assertEqual(db2.execute("SELECT id FROM t ORDER BY id").tuples(),
                         [(1,)])

    def test_aborted_transaction_is_discarded(self):
        db = self.reopen()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
        t = db.begin()
        db.execute("INSERT INTO t VALUES (5)", txn=t)
        db.rollback(t)
        db.simulate_crash()

        db2 = self.reopen()
        self.assertEqual(db2.execute("SELECT COUNT(*) FROM t").tuples(), [(0,)])

    def test_multiple_committed_transactions_replay_in_order(self):
        db = self.reopen()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        for i in range(10):
            db.execute(f"INSERT INTO t VALUES ({i}, {i})")
        for i in range(0, 10, 2):
            db.execute(f"UPDATE t SET v = v + 100 WHERE id = {i}")
        db.simulate_crash()

        db2 = self.reopen()
        rows = db2.execute("SELECT id, v FROM t ORDER BY id").tuples()
        expected = [(i, i + (100 if i % 2 == 0 else 0)) for i in range(10)]
        self.assertEqual(rows, expected)


class CheckpointTests(RecoveryTestBase):
    def test_checkpoint_then_wal_replay(self):
        db = self.reopen()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        db.execute("INSERT INTO t VALUES (1, 1), (2, 2)")
        db.checkpoint()
        db.execute("INSERT INTO t VALUES (3, 3)")
        db.simulate_crash()

        db2 = self.reopen()
        self.assertEqual(
            db2.execute("SELECT id, v FROM t ORDER BY id").tuples(),
            [(1, 1), (2, 2), (3, 3)])

    def test_repeated_checkpoints_compact_wal(self):
        db = self.reopen()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        for i in range(20):
            db.execute(f"INSERT INTO t VALUES ({i}, {i})")
            if i % 5 == 4:
                db.checkpoint()
        # After checkpoint the WAL should contain no committed old frames.
        frames = WAL(os.path.join(self.dir, "wal.log")).read_frames()
        self.assertFalse(
            any(f.get("type") == "insert" for f in frames))
        db.simulate_crash()

        db2 = self.reopen()
        self.assertEqual(db2.execute("SELECT COUNT(*) FROM t").tuples(), [(20,)])

    def test_checkpoint_preserves_multiple_tables_and_types(self):
        db = self.reopen()
        db.execute("CREATE TABLE a (id INT PRIMARY KEY, name TEXT, f FLOAT, "
                   "b BOOLEAN)")
        db.execute("INSERT INTO a VALUES (1, 'x', 1.5, TRUE)")
        db.execute("CREATE TABLE c (k INT PRIMARY KEY)")
        db.execute("INSERT INTO c VALUES (9)")
        db.close()  # graceful close checkpoints

        db2 = self.reopen()
        self.assertEqual(
            db2.execute("SELECT id, name, f, b FROM a").tuples(),
            [(1, "x", 1.5, True)])
        self.assertEqual(db2.execute("SELECT k FROM c").tuples(), [(9,)])

    def test_checkpoint_with_active_txn_then_its_commit_survives(self):
        db = self.reopen()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        db.execute("INSERT INTO t VALUES (1, 1)")
        # An active transaction holds uncommitted data across the checkpoint.
        t = db.begin()
        db.execute("INSERT INTO t VALUES (2, 2)", txn=t)
        db.checkpoint()
        db.commit(t)
        db.simulate_crash()

        db2 = self.reopen()
        self.assertEqual(
            db2.execute("SELECT id FROM t ORDER BY id").tuples(), [(1,), (2,)])

    def test_autoincrement_survives_recovery(self):
        db = self.reopen()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY AUTOINCREMENT, x INT)")
        db.execute("INSERT INTO t (x) VALUES (1), (2), (3)")
        db.simulate_crash()

        db2 = self.reopen()
        db2.execute("INSERT INTO t (x) VALUES (4)")
        self.assertEqual(
            db2.execute("SELECT id FROM t WHERE x = 4").tuples(), [(4,)])


class WALIntegrityTests(RecoveryTestBase):
    def test_torn_trailing_frame_is_ignored(self):
        db = self.reopen()
        db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
        db.execute("INSERT INTO t VALUES (1)")
        db.simulate_crash()
        # Append garbage / a torn frame to the WAL.
        path = os.path.join(self.dir, "wal.log")
        with open(path, "ab") as f:
            f.write(b"\x10\x00\x00\x00\xff\xff\xff\xffpartial-garbage")
        db2 = self.reopen()
        self.assertEqual(db2.execute("SELECT id FROM t").tuples(), [(1,)])

    def test_wal_frame_roundtrip(self):
        path = os.path.join(self.dir, "x.log")
        wal = WAL(path)
        for i in range(50):
            wal.append({"type": "insert", "i": i, "s": f"v{i}",
                        "nested": {"a": [1, 2, 3]}})
        records = WAL(path).read_frames()
        self.assertEqual(len(records), 50)
        self.assertEqual(records[7]["i"], 7)
        self.assertEqual(records[7]["nested"], {"a": [1, 2, 3]})


if __name__ == "__main__":
    unittest.main()
