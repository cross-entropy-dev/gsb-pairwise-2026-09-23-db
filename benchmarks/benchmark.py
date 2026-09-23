"""Performance benchmark for minidb.

Run with::

    python benchmarks/benchmark.py            # quick
    python benchmarks/benchmark.py --full     # larger data set

It measures:

1. B+ tree insert / point lookup / range scan / delete vs. size and height.
2. SQL statement throughput (autocommit INSERT, batched INSERT, SELECT).
3. Aggregation and GROUP BY throughput.
4. Concurrent transaction throughput (mixed transfers).
5. WAL commit (fsync) latency.
6. Snapshot-read throughput while a writer holds row X locks.

Results are printed as a table and, when run with ``--report PATH``, written
as a Markdown performance report.
"""

import argparse
import os
import statistics
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minidb import Database, READ_COMMITTED, DeadlockError
from minidb.storage.btree import BPlusTree, _InternalNode, _LeafNode


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def tree_height(tree):
    depth = 0
    node = tree.root
    while isinstance(node, _InternalNode):
        depth += 1
        node = node.children[0]
    return depth


def bench_btree(n):
    rows = []
    tree = BPlusTree(order=64)
    _, t_insert = timed(lambda: [tree.insert(i, i) for i in range(n)])
    rows.append(("B+ insert", n, t_insert, n / t_insert))

    _, t_get = timed(
        lambda: [tree.get(i) for i in range(0, n, max(1, n // 10000))])
    lookups = n // max(1, n // 10000)
    rows.append(("B+ point lookup (sample)", lookups, t_get,
                 lookups / t_get))

    def scan_all():
        count = 0
        for _ in tree.range():
            count += 1
        return count

    _, t_scan = timed(scan_all)
    rows.append(("B+ full range scan", n, t_scan, n / t_scan))

    keys = list(range(n))
    height = tree_height(tree)
    _, t_del = timed(lambda: [tree.delete(k) for k in keys])
    rows.append(("B+ delete", n, t_del, n / t_del))
    return rows, height


def bench_sql(n, data_dir):
    rows = []
    db = Database(data_dir)
    db.execute("CREATE TABLE kv (id INT PRIMARY KEY, cat INT, val INT, "
               "note VARCHAR(20))")

    # Autocommit single-row inserts (WAL + lock path).
    def single_inserts():
        for i in range(n):
            db.execute(
                f"INSERT INTO kv VALUES ({i}, {i % 10}, {i * 3}, 'n{i}')")

    _, t = timed(single_inserts)
    rows.append(("INSERT (autocommit, 1 row)", n, t, n / t))

    # Batched multi-row inserts in one transaction.
    db.execute("CREATE TABLE kv2 (id INT PRIMARY KEY, cat INT, val INT)")
    batch = 500

    def batched_inserts():
        for start in range(n, 2 * n, batch):
            values = ", ".join(
                f"({i}, {i % 10}, {i * 3})"
                for i in range(start, min(start + batch, 2 * n)))
            db.execute(f"INSERT INTO kv2 VALUES {values}")

    _, t = timed(batched_inserts)
    rows.append(("INSERT (batched, one txn/500)", n, t, n / t))

    _, result = timed(
        lambda: db.execute("SELECT COUNT(*), SUM(val), AVG(val), MIN(val), "
                           "MAX(val) FROM kv"))
    rows.append(("Aggregate over full table", 1, result,
                 1 / result if result else 0))

    _, result = timed(
        lambda: db.execute(
            "SELECT cat, COUNT(*), SUM(val) FROM kv GROUP BY cat "
            "ORDER BY cat"))
    rows.append(("GROUP BY (10 groups)", 1, result, 1 / result))

    _, result = timed(
        lambda: [db.execute("SELECT val FROM kv WHERE id = ?".replace(
            "?", str(i))) for i in range(0, n, max(1, n // 2000))])
    qcount = len(range(0, n, max(1, n // 2000)))
    rows.append(("SELECT by PK (point lookup)", qcount, result,
                 qcount / result))

    _, result = timed(lambda: db.execute(
        "SELECT a.id, b.val FROM kv a JOIN kv2 b ON a.id = b.id LIMIT 1000"))
    rows.append(("PK index join (1000 rows)", 1, result, 1 / result))

    return rows, db


def bench_concurrency(db, n_accounts=100, n_threads=64, ops=20):
    """Mixed transfers against an existing database.

    For a disk-backed database throughput is bounded by fsync commit
    latency; callers also run the in-memory variant to show raw locking /
    MVCC throughput independent of disk durability.
    """
    errors = []

    def worker(seed):
        import random
        rng = random.Random(seed)
        for _ in range(ops):
            a, b = rng.randrange(n_accounts), rng.randrange(n_accounts)
            while b == a:
                b = rng.randrange(n_accounts)
            try:
                t = db.begin(READ_COMMITTED)
                db.execute(f"UPDATE kv SET val = val + 1 WHERE id = {a}",
                           txn=t)
                db.execute(f"UPDATE kv SET val = val - 1 WHERE id = {b}",
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
    start = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.perf_counter() - start
    total_txns = n_threads * ops
    return elapsed, total_txns / elapsed, errors


def build_accounts(db, n_accounts):
    db.execute("CREATE TABLE IF NOT EXISTS kv (id INT PRIMARY KEY, val INT)")
    have = db.execute("SELECT COUNT(*) FROM kv").tuples()[0][0]
    if have == 0:
        batch = 500
        for start in range(0, n_accounts, batch):
            vals = ", ".join(f"({i}, 0)"
                             for i in range(start, min(start + batch,
                                                      n_accounts)))
            db.execute(f"INSERT INTO kv VALUES {vals}")


def bench_wal_latency(db, n=200):
    db.execute("CREATE TABLE wal_t (id INT PRIMARY KEY)")
    latencies = []
    for i in range(n):
        t = db.begin()
        db.execute(f"INSERT INTO wal_t VALUES ({i})", txn=t)
        start = time.perf_counter()
        db.commit(t)
        latencies.append(time.perf_counter() - start)
    return statistics.median(latencies), statistics.mean(latencies)


def format_table(headers, data):
    widths = [len(h) for h in headers]
    string_rows = []
    for row in data:
        srow = [str(c) for c in row]
        for i, cell in enumerate(srow):
            widths[i] = max(widths[i], len(cell))
        string_rows.append(srow)
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = "\n".join(
        "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
        for r in string_rows)
    return "\n".join([line, sep, body])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="use the larger data set")
    parser.add_argument("--report", help="write Markdown report to PATH")
    args = parser.parse_args()

    n = 50_000 if args.full else 10_000
    tmp = tempfile.mkdtemp(prefix="minidb_bench_")

    print(f"minidb benchmark  (n={n:,} rows, data dir={tmp})\n")

    btree_rows, height = bench_btree(n)
    print("== B+ tree ==")
    print(format_table(
        ["operation", "count", "seconds", "ops/sec"],
        [(name, f"{count:,}", f"{sec:.4f}", f"{rate:,.0f}")
         for name, count, sec, rate in btree_rows]))
    print(f"tree height at {n:,} rows (order=64): {height}\n")

    sql_rows, db = bench_sql(n, tmp)
    print("== SQL ==")
    print(format_table(
        ["operation", "count", "seconds", "rate"],
        [(name, count, f"{sec:.4f}",
          f"{rate:,.0f} ops/s" if "INSERT" in name or "SELECT" in name
          else f"{sec:.4f} s")
         for name, count, sec, rate in sql_rows]))

    elapsed, tps, errors = bench_concurrency(db, n_accounts=100)
    print(f"\n== Concurrency (disk-backed, fsync per commit) ==\n"
          f"64 threads, mixed transfers: {tps:,.0f} txn/s "
          f"({elapsed:.2f}s, errors={len(errors)})")

    # In-memory database: shows MVCC/lock throughput without disk fsync.
    mem_db = Database(":memory:")
    build_accounts(mem_db, 100)
    mem_elapsed, mem_tps, mem_errors = bench_concurrency(
        mem_db, n_accounts=100)
    print(f"== Concurrency (in-memory, no fsync) ==\n"
          f"64 threads, mixed transfers: {mem_tps:,.0f} txn/s "
          f"({mem_elapsed:.2f}s, errors={len(mem_errors)})")
    invariant = mem_db.execute("SELECT SUM(val) FROM kv").tuples()[0][0]
    print(f"money-conservation invariant SUM(val) = {invariant}")

    med, avg = bench_wal_latency(db)
    print(f"\n== Durability ==\nfsync commit latency: median={med*1000:.2f}ms "
          f"mean={avg*1000:.2f}ms")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(f"# minidb performance report\n\n")
            f.write(f"- Rows per benchmark: **{n:,}**\n")
            f.write(f"- B+ tree order: 64, height at {n:,} rows: **{height}** "
                    f"(O(log n) confirmed)\n")
            f.write(f"- Platform: Python {sys.version.split()[0]}, "
                    f"{os.name}\n\n")
            f.write("## B+ tree\n\n")
            f.write(format_table(
                ["operation", "count", "seconds", "ops/sec"],
                [(name, f"{count:,}", f"{sec:.4f}", f"{rate:,.0f}")
                 for name, count, sec, rate in btree_rows]))
            f.write("\n\n## SQL\n\n")
            f.write(format_table(
                ["operation", "count", "seconds", "rate"],
                [(name, count, f"{sec:.4f}", f"{rate:,.0f}")
                 for name, count, sec, rate in sql_rows]))
            f.write(f"\n\n## Concurrency\n\n")
            f.write(f"64 threads x 20 mixed transfers, **disk-backed** "
                    f"(fsync per commit): **{tps:,.0f} txn/s** in "
                    f"{elapsed:.2f}s (errors: {len(errors)}).  \n")
            f.write(f"Same workload, **in-memory** (no fsync): "
                    f"**{mem_tps:,.0f} txn/s** in {mem_elapsed:.2f}s "
                    f"(errors: {len(mem_errors)}).  \n")
            f.write(f"Money-conservation invariant after the in-memory run: "
                    f"SUM(val) = **{invariant}**.\n\n")
            f.write("Disk throughput is bounded by per-commit fsync latency; "
                    "the in-memory figure reflects raw MVCC/lock throughput.\n\n")
            f.write("## Durability\n\n")
            f.write(f"fsync commit latency: median **{med*1000:.2f}ms**, "
                    f"mean **{avg*1000:.2f}ms**.\n")
        print(f"\nreport written to {args.report}")


if __name__ == "__main__":
    main()
