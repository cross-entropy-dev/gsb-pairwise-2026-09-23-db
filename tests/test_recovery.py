"""Crash-recovery tests.

Each scenario starts a *fresh* subprocess that opens the database, performs
work and terminates with ``os._exit`` – no Python cleanup, no checkpoint –
exactly modelling a process crash.  The verification step reopens the
database in-process and asserts the durable state.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

WORKER = os.path.join(os.path.dirname(__file__), "manual_crash_worker.py")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_worker(data_dir: str, mode: str) -> None:
    subprocess.run(
        [sys.executable, WORKER, data_dir, mode],
        check=True, cwd=ROOT,
    )


class TestCrashRecovery(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db_dir = os.path.join(self.dir, "db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_committed_data_survives_crash_without_checkpoint(self):
        run_worker(self.db_dir, "setup")
        # hard crash happened immediately after the inserts' commit
        from minidb import Database, Session
        db = Database(self.db_dir)
        try:
            rows = Session(db).sql(
                "SELECT id, bal FROM acct ORDER BY id"
            ).rows
            self.assertEqual(rows, [(1, 100), (2, 200)])
            self.assertEqual(
                Session(db).sql("SHOW TABLES").rows,
                [("acct",)],
            )
        finally:
            db.shutdown()

    def test_committed_txn_after_crash_is_replayed_from_wal(self):
        run_worker(self.db_dir, "setup")
        run_worker(self.db_dir, "commit_then_crash")
        from minidb import Database, Session
        db = Database(self.db_dir)
        try:
            rows = Session(db).sql(
                "SELECT id, bal FROM acct ORDER BY id"
            ).rows
            # the transfer committed: -50 / +50
            self.assertEqual(rows, [(1, 50), (2, 250)])
        finally:
            db.shutdown()

    def test_uncommitted_txn_rolled_back_after_crash(self):
        run_worker(self.db_dir, "setup")
        run_worker(self.db_dir, "commit_then_crash")
        run_worker(self.db_dir, "mid_txn_crash")
        from minidb import Database, Session
        db = Database(self.db_dir)
        try:
            rows = Session(db).sql(
                "SELECT id, bal FROM acct ORDER BY id"
            ).rows
            # the in-flight transaction never committed -> invisible
            self.assertEqual(rows, [(1, 50), (2, 250)])
        finally:
            db.shutdown()

    def test_checkpoint_then_wal_restart(self):
        from minidb import Database, Session
        run_worker(self.db_dir, "setup")
        # clean open/close writes a checkpoint and truncates the WAL
        db = Database(self.db_dir)
        Session(db).sql("INSERT INTO acct VALUES (5, 500)")
        db.shutdown()
        # now crash right after another committed update (WAL-only)
        db = Database(self.db_dir)
        s = Session(db)
        self.assertEqual(s.sql("SELECT COUNT(*) FROM acct").rows[0], (3,))
        db.shutdown()
        run_worker(self.db_dir, "commit_then_crash")
        db = Database(self.db_dir)
        try:
            rows = Session(db).sql(
                "SELECT id, bal FROM acct ORDER BY id"
            ).rows
            self.assertEqual(rows, [(1, 50), (2, 250), (5, 500)])
        finally:
            db.shutdown()

    def test_torn_wal_tail_ignored(self):
        # append a garbage partial record – recovery must ignore the tail
        run_worker(self.db_dir, "setup")
        wal = os.path.join(self.db_dir, "wal.log")
        with open(wal, "a", encoding="utf-8") as f:
            f.write('{"ty": "insert", "txid": 99, "tab')  # torn line
        from minidb import Database, Session
        db = Database(self.db_dir)
        try:
            rows = Session(db).sql("SELECT id FROM acct ORDER BY id").rows
            self.assertEqual(rows, [(1,), (2,)])
        finally:
            db.shutdown()


if __name__ == "__main__":
    unittest.main()
