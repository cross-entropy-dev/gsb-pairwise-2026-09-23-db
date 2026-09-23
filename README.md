# minidb

一个**从零实现**的内存数据库引擎，支持 ACID 事务、MVCC 并发控制与 WAL 持久化。
纯 Python 3.10+ 标准库实现，**不使用任何第三方数据库库**。

## 特性

- **存储引擎**：内存 B+ 树（O(log n) 增删查，叶子链表支持高效范围扫描）+ 磁盘持久化
- **SQL 子集**：`CREATE TABLE / INSERT / SELECT / UPDATE / DELETE`
  - 类型：`INT / FLOAT / VARCHAR(n) / TEXT / BOOLEAN`
  - `WHERE / ORDER BY / LIMIT / OFFSET / DISTINCT`
  - 聚合：`COUNT / SUM / AVG / MIN / MAX`（含 `DISTINCT`）、`GROUP BY / HAVING`
  - `INNER JOIN / LEFT JOIN`
  - WHERE 中的标量子查询、`IN (subquery)`（支持关联子查询）
- **事务**：完整 ACID
  - 隔离级别：`READ COMMITTED`、`SERIALIZABLE`
  - MVCC 多版本快照，读不阻塞写
  - 行锁/表锁（IS/IX/S/X）、**死锁检测**、**行锁→表锁升级**
- **持久化**：后台线程 WAL（仅提交点 fsync，不阻塞读）+ 检查点 + 崩溃恢复
- **查询优化**：谓词下推、主键点查索引访问
- 交互式 CLI、完整单元测试、性能基准

## 快速开始

```python
from minidb import Database, Session, Isolation

db = Database("./data")            # 打开/恢复数据库
s = Session(db)

s.sql("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(50), age INT)")
s.sql("INSERT INTO users VALUES (1, 'Ada', 36), (2, 'Bob', 41)")

print(s.sql("SELECT name, age FROM users WHERE age > 30 ORDER BY age DESC").rows)
# [('Bob', 41), ('Ada', 36)]

# 显式事务
s.sql("BEGIN ISOLATION LEVEL SERIALIZABLE")
s.sql("UPDATE users SET age = age + 1 WHERE id = 1")
s.sql("COMMIT")          # 或 s.sql("ROLLBACK")

db.shutdown()            # 干净关闭：写检查点
```

## 交互式 CLI

```bash
python -m minidb.cli [数据目录]
```

```sql
minidb> CREATE TABLE emp (id INT PRIMARY KEY, name VARCHAR(20), dept_id INT);
minidb> INSERT INTO emp VALUES (1, 'Ada', 10), (2, 'Bob', 10), (3, 'Cyd', 20);
minidb> SELECT dept_id, COUNT(*) FROM emp GROUP BY dept_id HAVING COUNT(*) > 1;
minidb> EXPLAIN SELECT * FROM emp WHERE id = 1;
minidb> .tables
minidb> .describe emp
minidb> .quit
```

## SQL 参考

```
CREATE TABLE t (col TYPE [NOT NULL] [PRIMARY KEY], ... [, PRIMARY KEY (col)])
INSERT INTO t [(cols)] VALUES (...), (...), ...
SELECT [DISTINCT] proj [, ...]
  FROM table [alias] [INNER|LEFT JOIN table2 [alias] ON ...]
  [WHERE expr] [GROUP BY expr ... [HAVING expr]]
  [ORDER BY expr [ASC|DESC] ...] [LIMIT n [OFFSET m]]
UPDATE t SET col = expr [, ...] [WHERE expr]
DELETE FROM t [WHERE expr]
BEGIN [ISOLATION LEVEL READ COMMITTED|SERIALIZABLE]
COMMIT | ROLLBACK
SHOW TABLES | DESCRIBE table | EXPLAIN select
```

表达式：算术 `+ - * /`、比较 `= != < <= > >=`、逻辑 `AND OR NOT`（三值逻辑）、
`IS [NOT] NULL / LIKE / IN / BETWEEN`，以及 `UPPER / LOWER`。
多条 SQL 用 `;` 分隔；自动提交，除非显式 `BEGIN`。

## 运行测试

```bash
python -m unittest discover -s tests -p "test_*.py"     # 全部 92 个测试
```

覆盖：B+ 树随机化压力、SQL 语法、SQL 执行（连接/聚合/子查询/DML）、
事务可见性、隔离级别、死锁检测、锁升级、1000 并发事务、
子进程真实崩溃恢复（`os._exit` 模拟）、CLI。

## 性能基准

```bash
python -m tests.benchmark --rows 20000 --threads 16 --ops 300 --json
```

详见 [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)。

## 文档

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 分层架构、MVCC/锁/WAL 设计、ACID 对照
- [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) — 性能数据与瓶颈分析

## 代码结构

```
minidb/
  storage/  bptree.py  table.py  schema.py      # B+树 / 多版本表 / 类型
  sql/      lexer.py  parser.py  ast.py         # 词法 / 语法 / AST
  txn/      manager.py  locks.py  wal.py        # 事务 / 锁与死锁 / WAL
  engine/   database.py executor.py planner.py  # 内核 / 执行 / 优化
            expressions.py session.py errors.py
  cli.py                                     # 交互式命令行
tests/                                          # 单元测试 + 基准 + 崩溃脚本
```

## 设计取舍

- SERIALIZABLE 用“固定 MVCC 快照 + 表级 S 锁（S2PL）”实现，强一致但并发度
  低于真正的 SSI；READ COMMITTED 下读完全无锁。
- 仅单列主键自动索引（B+ 树）；无用户主键时使用内部代理键。
- 检查点为全量快照，在无活跃事务时原子写出。
