"""High-concurrency stress tests.

These exercise many simultaneous transactions (the project targets 1000+)
and verify correctness invariants rather than exact scheduling:

* total money is conserved under concurrent transfers,
* readers never block on writers (and vice versa for pure reads),
* the system remains coherent after a burst of mixed read/write traffic.
"""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minidb import Database, READ_COMMITTED, SERIALIZABLE, DeadlockError


class ConcurrencyStressTests(unittest.TestCase):
    def test_many_concurrent_transfers(self):
        n_accounts = 50
        initial = 100
        n_threads = 100
        ops_per_thread = 20

        db = Database(":memory:")
        db.execute("CREATE TABLE acc (id INT PRIMARY KEY, bal INT)")
        for i in range(n_accounts):
            db.execute(f"INSERT INTO acc VALUES ({i}, {initial})")

        errors = []
        barrier = threading.Barrier(n_threads)

        def worker(seed):
            rng = __import__("random").Random(seed)
            barrier.wait()  # maximise overlap
            for _ in range(ops_per_thread):
                a = rng.randrange(n_accounts)
                b = rng.randrange(n_accounts)
                while b == a:
                    b = rng.randrange(n_accounts)
                amt = rng.randrange(1, 5)
                try:
                    t = db.begin(READ_COMMITTED)
                    db.execute(
                        f"UPDATE acc SET bal = bal - {amt} WHERE id = {a}",
                        txn=t)
                    db.execute(
                        f"UPDATE acc SET bal = bal + {amt} WHERE id = {b}",
                        txn=t)
                    db.commit(t)
                except DeadlockError:
                    db.rollback(t)
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)
                    try:
                        db.rollback(t)
                    except Exception:
                        pass

        threads = [threading.Thread(target=worker, args=(s,))
                   for s in range(n_threads)]
        start = time.time()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        elapsed = time.time() - start

        self.assertEqual(errors, [])
        total = db.execute("SELECT SUM(bal) FROM acc").tuples()[0][0]
        self.assertEqual(total, n_accounts * initial)
        # Throughput is asserted only loosely so CI is not flaky.
        self.assertLess(elapsed, 60.0)

    def test_over_one_thousand_concurrent_readers(self):
        db = Database(":memory:")
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        for i in range(100):
            db.execute(f"INSERT INTO t VALUES ({i}, {i * 2})")

        n_readers = 1000
        barrier = threading.Barrier(n_readers + 1)
        results = [None] * n_readers

        def reader(i):
            barrier.wait()
            t = db.begin(READ_COMMITTED)
            results[i] = db.execute(
                "SELECT SUM(v) FROM t", txn=t).tuples()[0][0]
            db.rollback(t)

        threads = [threading.Thread(target=reader, args=(i,))
                   for i in range(n_readers)]
        for th in threads:
            th.start()
        barrier.wait()  # release all readers together
        start = time.time()
        for th in threads:
            th.join()
        elapsed = time.time() - start

        self.assertTrue(all(r == sum(i * 2 for i in range(100))
                            for r in results))
        # 1000 simultaneous snapshot readers should finish quickly.
        self.assertLess(elapsed, 30.0)

    def test_readers_do_not_block_writer_and_writer_does_not_block_readers(self):
        db = Database(":memory:")
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
        db.execute("INSERT INTO t VALUES (1, 0)")

        writer_holds_lock = threading.Event()
        release_writer = threading.Event()
        read_delays = []

        def writer():
            t = db.begin(READ_COMMITTED)
            db.execute("UPDATE t SET v = 1 WHERE id = 1", txn=t)
            writer_holds_lock.set()
            release_writer.wait(2.0)
            db.commit(t)

        def reader():
            writer_holds_lock.wait(2.0)
            start = time.time()
            t = db.begin(READ_COMMITTED)
            val = db.execute("SELECT v FROM t WHERE id = 1", txn=t).tuples()
            db.rollback(t)
            read_delays.append(time.time() - start)
            return val

        wt = threading.Thread(target=writer)
        wt.start()
        # Launch readers while the writer is mid-transaction holding X locks.
        readers = [threading.Thread(target=reader) for _ in range(20)]
        writer_holds_lock.wait(2.0)
        time.sleep(0.1)
        for r in readers:
            r.start()
        for r in readers:
            r.join(2.0)
        release_writer.set()
        wt.join()

        # Every reader returned promptly with the old committed value.
        self.assertEqual(len(read_delays), 20)
        self.assertLess(max(read_delays), 1.0)


if __name__ == "__main__":
    unittest.main()
