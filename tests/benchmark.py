"""Performance benchmark for minidb.

Usage::

    python -m tests.benchmark [--rows N] [--threads T] [--json]

Measures:

1. B+ tree insert / point lookup / range scan throughput and tree depth;
2. SQL INSERT / SELECT / UPDATE latency on a single connection;
3. mixed concurrent transactions (read/write) across T threads;
4. 1000+ concurrent transactions;
5. WAL commit (fsync) cost and non-blocking reads during heavy writes.

Numbers are printed as a table and, with ``--json``, also written to
``benchmark_results.json`` next to this file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minidb import Database, Session  # noqa: E402
from minidb.storage.bptree import BPlusTree  # noqa: E402


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def bench_bptree(n: int) -> dict:
    tree = BPlusTree(order=32)
    _, t_ins = timed(lambda: [tree.insert(i, i) for i in range(n)])
    _, t_get = timed(
        lambda: [tree.get(i) for i in range(0, n, max(1, n // 10000))]
    )
    lookups = len(range(0, n, max(1, n // 10000)))
    _, t_scan = timed(lambda: list(tree.range(n // 4, 3 * n // 4)))
    _, t_del = timed(lambda: [tree.delete(i) for i in range(0, n, 2)])
    return {
        "rows": n,
        "insert_throughput_ops_s": round(n / t_ins),
        "point_lookup_throughput_ops_s": round(lookups / t_get),
        "range_scan_rows_s": round((n // 2) / t_scan),
        "delete_throughput_ops_s": round((n // 2) / t_del),
        "tree_depth_at_peak": tree.depth() + 0,  # deletes don't shrink depth
    }


def _fresh_db():
    d = tempfile.mkdtemp()
    db = Database(os.path.join(d, "db"))
    s = Session(db)
    s.sql("CREATE TABLE kv (k INT PRIMARY KEY, v INT, label VARCHAR(16))")
    return d, db, s


def bench_sql_single(n: int) -> dict:
    d, db, s = _fresh_db()
    try:
        t0 = time.perf_counter()
        # multi-row inserts in batches of 100
        batch = 100
        for start in range(0, n, batch):
            end = min(start + batch, n)
            values = ",".join(f"({i},{i*2},'k{i}')" for i in range(start, end))
            s.sql(f"INSERT INTO kv VALUES {values}")
        t_insert = time.perf_counter() - t0

        t0 = time.perf_counter()
        reps = min(n, 20000)
        for i in range(reps):
            s.sql(f"SELECT v, label FROM kv WHERE k = {i % n}")
        t_point = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = s.sql("SELECT COUNT(*), SUM(v), AVG(v), MIN(v), MAX(v) FROM kv")
        t_agg = time.perf_counter() - t0
        agg = res.rows[0]

        t0 = time.perf_counter()
        n_upd = min(n, 5000)
        s.sql(f"UPDATE kv SET v = v + 1 WHERE k < {n_upd}")
        t_update = time.perf_counter() - t0

        t0 = time.perf_counter()
        rows = s.sql("SELECT k FROM kv ORDER BY k LIMIT 1000").rows
        t_order = time.perf_counter() - t0

        return {
            "rows": n,
            "insert_rows_s": round(n / t_insert),
            "point_selects_s": round(reps / t_point),
            "aggregate_ms": round(t_agg * 1000, 3),
            "aggregate_result": agg,
            "update_5k_rows_ms": round(t_update * 1000, 1),
            "order_limit_ms": round(t_order * 1000, 2),
        }
    finally:
        db.shutdown()
        shutil.rmtree(d, ignore_errors=True)


def bench_concurrency(n_rows: int, n_threads: int, ops_per_thread: int) -> dict:
    d, db, s = _fresh_db()
    try:
        batch = 500
        for start in range(0, n_rows, batch):
            end = min(start + batch, n_rows)
            values = ",".join(f"({i},{i},'x')" for i in range(start, end))
            s.sql(f"INSERT INTO kv VALUES {values}")

        errors: list[Exception] = []
        latencies: list[float] = []
        lat_lock = threading.Lock()

        def worker(tid: int):
            sess = Session(db)
            rng_state = tid * 2654435761 & 0xFFFFFFFF
            for _ in range(ops_per_thread):
                rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
                k = rng_state % n_rows
                t0 = time.perf_counter()
                try:
                    if rng_state % 4 == 0:
                        sess.sql(f"UPDATE kv SET v = v + 1 WHERE k = {k}")
                    else:
                        sess.sql(f"SELECT v FROM kv WHERE k = {k}")
                except Exception as exc:  # deadlock victim -> count
                    with lat_lock:
                        errors.append(exc)
                with lat_lock:
                    latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(worker, range(n_threads)))
        elapsed = time.perf_counter() - t0
        total_ops = n_threads * ops_per_thread
        latencies.sort()
        return {
            "threads": n_threads,
            "rows": n_rows,
            "total_ops": total_ops,
            "throughput_ops_s": round(total_ops / elapsed),
            "elapsed_s": round(elapsed, 2),
            "latency_ms_p50": round(latencies[len(latencies) // 2], 3),
            "latency_ms_p95": round(latencies[int(len(latencies) * 0.95)], 3),
            "latency_ms_p99": round(latencies[int(len(latencies) * 0.99)], 3),
            "conflict_aborts": len(errors),
        }
    finally:
        db.shutdown()
        shutil.rmtree(d, ignore_errors=True)


def bench_many_concurrent(n_txns: int) -> dict:
    d, db, s = _fresh_db()
    try:
        s.sql("INSERT INTO kv VALUES (1, 1, 'a'), (2, 2, 'b')")
        max_workers = 64
        # barrier parties must equal the number of *workers*, not tasks
        barrier = threading.Barrier(max_workers)

        def worker(i):
            sess = Session(db)
            # only the first wave synchronises; the rest flow through
            if i < max_workers:
                barrier.wait()
            sess.sql("SELECT COUNT(*) FROM kv")

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(worker, range(n_txns)))
        elapsed = time.perf_counter() - t0
        return {"concurrent_transactions": n_txns, "elapsed_s": round(elapsed, 3)}
    finally:
        db.shutdown()
        shutil.rmtree(d, ignore_errors=True)


def bench_wal_and_read_nonblocking(n: int) -> dict:
    d, db, s = _fresh_db()
    try:
        # commit cost (each autocommit insert fsyncs the WAL)
        t0 = time.perf_counter()
        reps = 200
        for i in range(reps):
            s.sql(f"INSERT INTO kv VALUES ({i}, {i}, 'w')")
        t_commit = (time.perf_counter() - t0) / reps

        stop = threading.Event()
        read_lat: list[float] = []
        read_count = 0

        def reader():
            nonlocal read_count
            sess = Session(db)
            sess.sql("INSERT INTO kv VALUES (-1, 0, 'r')")
            while not stop.is_set():
                t0 = time.perf_counter()
                sess.sql("SELECT v FROM kv WHERE k = -1")
                read_lat.append(time.perf_counter() - t0)
                read_count += 1

        th = threading.Thread(target=reader)
        th.start()
        time.sleep(0.2)
        t0 = time.perf_counter()
        for i in range(reps, reps + n):
            s.sql(f"INSERT INTO kv VALUES ({i}, {i}, 'w')")
        write_elapsed = time.perf_counter() - t0
        time.sleep(0.2)
        stop.set()
        th.join()
        return {
            "fsync_commit_ms": round(t_commit * 1000, 3),
            "writer_rows_s": round(n / write_elapsed),
            "reads_during_writes": read_count,
            "read_p95_ms": round(
                statistics.quantiles(read_lat, n=20)[18] * 1000
                if len(read_lat) > 20 else max(read_lat) * 1000, 3
            ),
        }
    finally:
        db.shutdown()
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--ops", type=int, default=500)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = {
        "env": {"python": sys.version.split()[0], "platform": sys.platform},
        "bptree": bench_bptree(args.rows),
        "sql_single": bench_sql_single(args.rows),
        "concurrency": bench_concurrency(
            max(2000, args.rows // 4), args.threads, args.ops
        ),
        "many_txns": bench_many_concurrent(1000),
        "wal": bench_wal_and_read_nonblocking(500),
    }

    def section(title, data: dict):
        print(f"\n== {title} ==")
        for k, v in data.items():
            print(f"  {k:32s} {v}")

    print("minidb benchmark")
    section("B+ tree", report["bptree"])
    section("SQL (single session)", report["sql_single"])
    section(f"Concurrent workload ({args.threads} threads)",
            report["concurrency"])
    section("1000 concurrent transactions", report["many_txns"])
    section("WAL durability / non-blocking reads", report["wal"])

    if args.json:
        out = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
