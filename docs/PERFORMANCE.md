# minidb performance report

- Rows per benchmark: **10,000**
- B+ tree order: 64, height at 10,000 rows: **2** (O(log n) confirmed)
- Platform: Python 3.11.7, nt

## B+ tree

| operation                | count  | seconds | ops/sec   |
|--------------------------|--------|---------|-----------|
| B+ insert                | 10,000 | 0.0164  | 609,425   |
| B+ point lookup (sample) | 10,000 | 0.0059  | 1,694,858 |
| B+ full range scan       | 10,000 | 0.0013  | 7,803,356 |
| B+ delete                | 10,000 | 0.0158  | 632,879   |

## SQL

| operation                     | count | seconds | rate  |
|-------------------------------|-------|---------|-------|
| INSERT (autocommit, 1 row)    | 10000 | 8.6446  | 1,157 |
| INSERT (batched, one txn/500) | 10000 | 2.1804  | 4,586 |
| Aggregate over full table     | 1     | 0.1496  | 7     |
| GROUP BY (10 groups)          | 1     | 0.0985  | 10    |
| SELECT by PK (point lookup)   | 2000  | 0.4866  | 4,110 |
| PK index join (1000 rows)     | 1     | 0.1498  | 7     |

## Concurrency

Workload: 64 threads × 20 mixed transfers (each txn = 2 point-key UPDATEs).

- **Disk-backed** (fsync per commit): **80 txn/s** in 16.01s (errors: 0).
- **In-memory** (no fsync): **2,378 txn/s** in 0.54s (errors: 0).
- Money-conservation invariant after the in-memory run: SUM(val) = **0**.

Disk throughput is bounded by per-commit fsync latency; the in-memory figure
reflects raw MVCC/lock throughput. A separate stress test runs **1000
concurrent snapshot readers** successfully (see `tests/test_concurrency.py`).

## Durability

fsync commit latency: median **0.83ms**, mean **0.91ms**.
