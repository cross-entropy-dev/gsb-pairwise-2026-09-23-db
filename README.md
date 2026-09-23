# minidb

一个从零实现的、支持 **ACID 事务** 的内存数据库引擎，纯 Python 标准库编写，**不使用任何第三方数据库库**（连 pickle 都不用，自带二进制编解码）。

- B+ 树内存存储 + WAL / 检查点磁盘持久化
- 手写 SQL 词法分析器与递归下降语法分析器
- MVCC 多版本并发控制，支持 **READ COMMITTED** 与 **SERIALIZABLE**
- 死锁检测（等待图）、行锁 → 表锁升级、谓词防幻读
- 交互式 CLI、完整单元测试、性能基准

## 快速开始

需要 Python 3.9+（开发使用 3.11），无第三方依赖。

```python
from minidb import Database, SERIALIZABLE

db = Database(":memory:")          # 或 Database("./data") 启用磁盘持久化
db.execute("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(50), age INT)")
db.execute("INSERT INTO users VALUES (1, 'Ada', 36), (2, 'Linus', 50)")

print(db.execute("SELECT name, age FROM users WHERE age > 40 ORDER BY name").dicts())
# [{'name': 'Linus', 'age': 50}]

print(db.execute("SELECT COUNT(*), AVG(age) FROM users").tuples())

# 显式事务
txn = db.begin(SERIALIZABLE)
db.execute("UPDATE users SET age = age + 1 WHERE id = 1", txn=txn)
db.commit(txn)   # 或 db.rollback(txn)
```

### 交互式 CLI

```bash
python -m minidb.cli                 # 内存模式
python -m minidb.cli ./data         # 持久化到 ./data
```

```
minidb> CREATE TABLE t (id INT PRIMARY KEY, v TEXT);
minidb> INSERT INTO t VALUES (1, 'hello'), (2, 'world');
minidb> SELECT * FROM t ORDER BY id;
minidb> .tables
minidb> .schema t
minidb> .help
minidb> .exit
```

CLI 支持跨多行、以 `;` 结尾的语句，以及 `BEGIN; ... COMMIT;` 显式事务。

## SQL 子集

```sql
CREATE TABLE name (
  col INT [PRIMARY KEY] [AUTOINCREMENT] [NOT NULL] [UNIQUE] [DEFAULT literal],
  col VARCHAR(n) | TEXT | BOOLEAN | FLOAT, ...
);
DROP TABLE name [IF EXISTS];

INSERT INTO name [(cols)] VALUES (...), (...);
SELECT [DISTINCT] expr [, ...]
  FROM table [alias]
  [INNER JOIN | LEFT JOIN table ON ...]
  [WHERE ...]
  [GROUP BY ... [HAVING ...]]
  [ORDER BY ... [ASC | DESC]]
  [LIMIT n [OFFSET m]];
UPDATE name SET col = expr [WHERE ...];
DELETE FROM name [WHERE ...];

BEGIN [ISOLATION LEVEL READ COMMITTED | SERIALIZABLE];
COMMIT;
ROLLBACK;
```

- 类型：`INT/INTEGER/BIGINT`、`VARCHAR(n)/CHAR`、`TEXT`、`BOOLEAN/BOOL`、`FLOAT/DOUBLE/REAL`
- 谓词：`= <> != < > <= >=`、`AND/OR/NOT`、`IS [NOT] NULL`、`LIKE`、`IN`、`BETWEEN`、`CASE`
- 聚合：`COUNT / SUM / AVG / MIN / MAX`，支持 `DISTINCT`、`COUNT(*)`
- 连接：`INNER JOIN`、`LEFT JOIN`（含逗号交叉连接，解析支持 RIGHT JOIN）
- 子查询：标量子查询、`IN (SELECT ...)`、`[NOT] EXISTS (SELECT ...)`，支持**相关子查询**
- 约束：`PRIMARY KEY`、`NOT NULL`、`UNIQUE`、`DEFAULT`、`AUTOINCREMENT`

## 项目结构

```
minidb/
├── sql/            lexer.py · parser.py · ast_nodes.py
├── executor/       optimizer.py · executor.py · evaluator.py · types.py
├── storage/        btree.py · table.py · wal.py
├── transaction/    transaction_manager.py · lock_manager.py
├── codec.py        自描述二进制编解码
├── database.py     引擎门面（组装/恢复/自动提交/检查点）
└── cli.py          交互式命令行
tests/              103 个测试：B+树 · SQL · 事务 · 崩溃恢复 · 并发
benchmarks/         benchmark.py
docs/               ARCHITECTURE.md · PERFORMANCE.md
```

## 运行测试与基准

```bash
python -m unittest discover -s tests -v          # 全部测试
python benchmarks/benchmark.py --full            # 大数据集
python benchmarks/benchmark.py --report docs/PERFORMANCE.md
```

## ACID 与并发一览

| 特性 | 实现 |
|------|------|
| 原子性 | 提交时统一打提交时间戳；回滚反向摘除新版本、清除删除标记、回滚 DDL |
| 一致性 | 类型/长度、NOT NULL、PRIMARY KEY、UNIQUE 约束在写入路径强制 |
| 隔离性 | RC：语句级 MVCC 快照，读无锁；SERIALIZABLE：严格两阶段锁 + 谓词锁防幻读 |
| 持久性 | WAL（长度+CRC32 帧）每事务一次 fsync；原子检查点；崩溃重放 |
| 并发 | 读写互不阻塞、等待图死锁检测（最年轻者牺牲）、行锁→表锁升级 |

详细设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，性能数据见 [docs/PERFORMANCE.md](docs/PERFORMANCE.md)。

## 范围与限制

这是一个教学/演示性质的数据库引擎，明确**未实现**：二级索引的查询优化器选择（仅主键索引被用于访问路径）、网络服务端与客户端协议、认证授权、ALTER TABLE、外键/CHECK 约束。它的目标是用清晰的模块展示一个真实数据库引擎的核心机制。
