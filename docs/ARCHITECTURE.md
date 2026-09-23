# minidb 架构设计文档

一个纯 Python 标准库实现的、支持 ACID 事务的内存数据库引擎。不依赖任何第三方数据库库。

- 存储：B+ 树内存索引 + WAL/检查点磁盘持久化
- 查询：手写词法分析器、递归下降语法分析器、规则优化器、执行器
- 事务：MVCC 多版本 + 两阶段锁（READ COMMITTED / SERIALIZABLE）
- 并发：读写不互斥（快照读无锁）、死锁检测、锁升级

---

## 1. 总体架构

```
                ┌──────────────────────────────────────────┐
   SQL 文本  ──▶│ sql/lexer.py   词法分析 (Token)            │
                │ sql/parser.py  递归下降 (AST)              │
                │ sql/ast_nodes.py                          │
                └───────────────────┬──────────────────────┘
                                     ▼
                ┌──────────────────────────────────────────┐
                │ executor/optimizer.py  常量折叠/谓词下推    │
                │ executor/executor.py   查询执行流水线      │
                │ executor/evaluator.py  表达式/子查询求值   │
                │ executor/types.py      类型/三值逻辑/聚合   │
                └───────────────────┬──────────────────────┘
                                     ▼
                ┌──────────────────────────────────────────┐
                │ transaction/transaction_manager.py        │
                │   MVCC 版本管理 · 提交协议 · 崩溃恢复       │
                │ transaction/lock_manager.py               │
                │   多粒度锁 · 死锁检测 · 行→表锁升级         │
                └───────────┬───────────────┬──────────────┘
                            ▼               ▼
        ┌────────────────────────┐  ┌──────────────────────┐
        │ storage/table.py       │  │ storage/wal.py       │
        │  Table / Version 链    │  │  WAL 帧 + checkpoint │
        │ storage/btree.py       │  │  codec.py 二进制编码  │
        │  B+ 树 + 叶子链表      │  │                      │
        └────────────────────────┘  └──────────────────────┘
```

入口为 `database.Database`，它组装以上全部组件并对外暴露 `execute / executescript / begin / commit / rollback / checkpoint / close`。CLI 位于 `minidb/cli.py`。

### 模块清单

| 模块 | 职责 |
|------|------|
| `minidb/sql/lexer.py` | 词法分析：关键字、标识符、数字、字符串、运算符、注释 |
| `minidb/sql/parser.py` | 递归下降分析，生成 AST；运算符优先级、JOIN、子查询、事务语句 |
| `minidb/sql/ast_nodes.py` | AST 数据类 |
| `minidb/executor/optimizer.py` | 常量折叠、永真/永假消解；谓词下推由执行器按 schema 完成 |
| `minidb/executor/executor.py` | DDL/DML/SELECT 执行、嵌套循环连接、聚合、排序 |
| `minidb/executor/evaluator.py` | 标量表达式、三值逻辑、相关子查询 |
| `minidb/executor/types.py` | 类型强制转换、比较、LIKE、聚合状态机 |
| `minidb/storage/btree.py` | B+ 树（点查/范围/插入/删除，叶子链表） |
| `minidb/storage/table.py` | Schema、Version 版本链、二级唯一索引、自增计数 |
| `minidb/storage/wal.py` | WAL 帧格式（长度+CRC32）、原子检查点 |
| `minidb/codec.py` | 自描述二进制编解码（不使用 pickle） |
| `minidb/transaction/lock_manager.py` | IS/IX/S/X 多粒度锁、等待图死锁检测、锁升级 |
| `minidb/transaction/transaction_manager.py` | 事务生命周期、MVCC 可见性、提交/回滚、WAL 恢复 |
| `minidb/database.py` | 引擎门面、自动提交、恢复编排 |
| `minidb/cli.py` | 交互式命令行 |

---

## 2. 存储引擎

### 2.1 B+ 树（`storage/btree.py`）

每张表在主键上维护一棵 B+ 树。

- **阶数 order（默认 64，偶数）**：叶子最多 `order` 个条目；内部节点最多 `order` 个子节点（`order-1` 个分隔键）。
- 非根节点低于 `order/2` 个键时触发再平衡：优先向兄弟借位，否则与兄弟合并。插入到容量上限时对半分裂。
- 点查沿内部节点二分下降，复杂度 **O(log_b n)**。
- 叶子节点通过 `next` 指针串联，范围扫描与全表扫描沿叶子链顺序进行，无需反复从根下降。
- 键支持任意可比较 Python 值（int、str、元组），因此单列与多列主键统一处理。
- `to_dict / from_dict` 只导出有序叶子内容，供检查点使用；恢复时自底向上重建内部层级。

没有显式主键的表使用内部自增 `_rowid_` 作为键。

### 2.2 多版本存储（`storage/table.py`）

每个键映射到一条**版本链**（最新版本在链首）：

```
key ─▶ [Version(xmin,xmin_ts,xmax,xmax_ts,data)] ─▶ [Version ...]
```

Version 字段：

| 字段 | 含义 |
|------|------|
| `xmin` | 创建该版本的事务 id |
| `xmin_ts` | 创建事务的提交时间戳；`0` 表示尚未提交 |
| `xmax` | 删除/覆盖该版本的事务 id；`0` 表示存活 |
| `xmax_ts` | 删除事务的提交时间戳；`0` 表示删除未提交 |
| `data` | 该行的列值字典 |

可见性判定（快照读）：

- 创建版本未提交，且创建者不是本事务 → 不可见；
- 创建提交时间戳晚于快照 → 不可见；
- 版本被本事务删除 → 不可见；
- 版本被某已提交事务在快照之前删除 → 不可见；
- 其余情况可见。

每个 UNIQUE 列维护一棵二级 B+ 树（值 → 主键），在提交时随版本落定更新。

### 2.3 WAL 与检查点（`storage/wal.py`，`codec.py`）

**WAL 帧格式**：

```
uint32 payload_length │ uint32 crc32(payload) │ payload(codec 编码)
```

文件以魔数 `MDBW` 开头。崩溃产生的尾部残缺帧（长度不足或 CRC 不匹配）在恢复时被安全忽略。

记录类型：`begin / insert / update / delete / ddl / commit / abort / checkpoint`。

**提交规则（Write-Ahead）**：

1. DML 记录先追加到 WAL 并 `flush`，之后版本才对本事务可见；
2. `COMMIT` 帧追加并 `fsync`——每个事务仅一次 fsync；
3. 提交时间戳在 fsync 临界区内单调分配，保证磁盘顺序与可见顺序一致。

读操作完全不接触 WAL 锁，因此 **WAL 写入不阻塞读**。

**检查点**：将每张表最新已提交版本写入 `checkpoint.db`（先写临时文件再原子 `os.replace`），随后把 WAL 重写为“仍活跃事务的帧 + checkpoint 标记”。崩溃若发生在检查点与 WAL 重写之间，重放对已存在数据幂等容忍（更新缺失键按插入处理、删除缺失键忽略）。

`codec.py` 是自描述的长度前缀二进制格式（None/bool/int64/float/utf-8/bytes/list/dict），不使用 pickle。

---

## 3. 查询引擎

### 3.1 词法分析（Lexer）

逐字符扫描，输出带类型与位置的 Token：`KEYWORD / IDENT / NUMBER / STRING / OP`。支持：

- 单引号字符串（`''` 转义）、双引号引用标识符；
- 整数、小数、科学计数法；
- 行注释 `--`、块注释 `/* */`；
- 多字符运算符（`<= >= <> !=` 等）。

### 3.2 语法分析（Parser）

手写递归下降，每个语法非终结符对应一个方法。表达式优先级自低到高：

```
OR < AND < NOT < 比较(= <> < > <= >=, IS [NOT] NULL, LIKE, IN, BETWEEN)
   < + - < * / < 一元 -/+ < 基本式(字面量/列/函数/子查询/CASE)
```

支持的语句：`CREATE/DROP TABLE`、单行/多行 `INSERT`、`SELECT`（含 JOIN/GROUP BY/HAVING/ORDER BY/LIMIT/OFFSET/DISTINCT）、`UPDATE`、`DELETE`、`BEGIN/COMMIT/ROLLBACK`。

### 3.3 逻辑查询计划与优化

SELECT 执行流水线：

```
绑定名称 → 扫描计划(谓词下推/主键点查)
  → 嵌套循环连接(主键等值连接走 B+ 树索引)
  → WHERE 残余过滤 → GROUP BY/聚合 → HAVING
  → SELECT 投影 → DISTINCT → ORDER BY → OFFSET/LIMIT
```

**基于规则的优化**：

1. **常量折叠**：纯字面量表达式（`1+2*3`、`'a'='b'`）在计划期一次求值；
2. **永真/永假消解**：`WHERE ... AND TRUE` 去掉 TRUE，常量 FALSE 短路；
3. **谓词下推**：只引用一个基表的 WHERE 合取项，在该表处于“内连接保留侧”时下推到扫描；LEFT JOIN 右表的单表 ON 条件也会下推；
4. **主键访问路径**：下推的主键等值条件变为 B+ 树点查，而非全表扫描；
5. **索引嵌套循环连接**：当 JOIN ON 是右表主键 = 左表表达式时，对每个左行执行一次主键点查。

### 3.4 执行与三值逻辑

- 中间行同时包含限定名（`t.col`）与裸名（`col`），简化列解析与 JOIN 合并。
- WHERE/HAVING 采用 SQL 三值逻辑（TRUE/FALSE/UNKNOWN），仅保留确定为 TRUE 的行。
- 聚合 `COUNT/SUM/AVG/MIN/MAX` 支持 `DISTINCT`；无 GROUP BY 时整张表为一个组（空表 `COUNT(*)=0`）。
- 相关子查询（IN / EXISTS / 标量）执行时携带外层行（`outer_row`），内层无法绑定的限定列在运行时回退到外层行解析。
- 类型系统支持 INT、VARCHAR(n)、TEXT、BOOLEAN、FLOAT，含强制转换与 VARCHAR 长度校验。

---

## 4. 事务与并发控制

### 4.1 ACID 实现

- **原子性 Atomicity**：提交时所有版本一次性打上提交时间戳；失败/回滚时反向撤销——移除本事务安装的版本、清除本事务设置的 `xmax`、回滚 DDL。自动提交语句失败即回滚。
- **一致性 Consistency**：NOT NULL、PRIMARY KEY、UNIQUE、类型与长度约束在写入路径强制；主键不可更新。
- **隔离性 Isolation**：见 4.2。
- **持久性 Durability**：WAL + fsync 提交、检查点、崩溃重放（见 2.3）。

### 4.2 两种隔离级别

**READ COMMITTED（默认）**

- 每条语句获取一个新的已提交快照；读操作**不加任何锁**，绝不被写阻塞，也不阻塞写。
- 写操作取 IX 表意向锁 + X 行锁（首个更新者胜出）。UPDATE 在获得行 X 锁后对**最新已提交版本**重新做当前读并重新求值 WHERE 与 SET，避免基于旧快照丢失更新。
- 能看到语句之间其他事务的提交；未提交数据始终不可见。

**SERIALIZABLE**

- 在 MVCC 之上使用**严格两阶段锁**：点读取 S 行锁，谓词/扫描读取 S 表锁（防止幻读），写取 IX/X；所有锁持有到提交/回滚。
- 因此调度冲突可串行化，包括范围谓词防幻读。

### 4.3 锁管理器（`lock_manager.py`）

- 多粒度锁模式：**IS / IX / S / X**，标准兼容矩阵；同事务重复加锁按模式并集升级（如持 IX 再请求 S 升级为 X）。
- 每个资源一个 FIFO 等待队列，队首优先授权，避免弱锁请求饿死 X 请求。
- **锁升级**：单事务在一张表上的行锁数量超过阈值（默认 200）即升级为单个表锁（存在任一行 X 锁则升级为 X）。
- **死锁检测**：阻塞前构建等待图（等待事务 → 冲突持有者），DFS 搜索环；发现环时选择环中**最年轻（id 最大）**的事务作为牺牲者，抛出 `DeadlockError` 并回滚。另设锁等待超时作为兜底，避免永久挂起。

### 4.4 时间戳与垃圾回收

- 内部维护单调递增的提交时间戳；检查点装载的行标记为引导时间戳。
- 提交后对触碰过的键做机会式版本回收：删除对最老活跃快照已不可见的旧版本。

---

## 5. 崩溃恢复流程

打开数据库时：

1. 读取 `checkpoint.db`（不存在则从空库开始），装载 schema、最新已提交行、自增计数，时间戳水位取检查点 `max_ts`。
2. 读取 `wal.log`，按事务收集帧；忽略 `checkpoint` 标记后的旧段与残缺尾帧。
3. 仅对存在 `commit` 帧且提交时间戳 **高于检查点水位** 的事务，按提交顺序重放其 DDL/DML；无 commit 的事务（崩溃时仍活跃）整体丢弃。
4. 重建唯一索引与自增计数器，提交时间戳推进到最大值。

该模型同时覆盖：仅 WAL 崩溃、检查点后崩溃、检查点与 WAL 重写之间崩溃、尾部撕裂、已回滚事务等情形（见 `tests/test_recovery.py`）。

---

## 6. 关键设计取舍

- **MVCC + 2PL 混合**：RC 下读完全无锁以获得高并发；SERIALIZABLE 下用严格 2PL 在同一套存储上获得强隔离与防幻读，避免实现复杂的 SSI。
- **更新=删旧+插新**：UPDATE 安装新版本并把旧版本标记为本事务删除，提交后统一打时间戳，回滚只需摘除新版本。
- **检查点导出叶子**：相比序列化整棵树结构，导出有序叶子更简单健壮，内部层级按阶数自底向上重建。
- **谓词下推位置**：需要 schema 才能确定无限定列归属，因此下推在执行器绑定阶段完成，而非纯 AST 重写阶段。
- **每事务一次 fsync**：DML 帧只 flush，提交帧 fsync，兼顾持久性与吞吐。

---

## 7. SQL 方言一览

```
CREATE TABLE name ( col TYPE [PRIMARY KEY] [NOT NULL] [UNIQUE]
                    [DEFAULT literal] [AUTOINCREMENT], ... )
DROP TABLE name [IF EXISTS]
INSERT INTO name [(cols)] VALUES (...), (...)
SELECT [DISTINCT] cols|agg|* [FROM ... [JOIN ... ON ...]]
  [WHERE ...] [GROUP BY ... [HAVING ...]]
  [ORDER BY ... [ASC|DESC]] [LIMIT n [OFFSET m]]
UPDATE name SET col = expr [WHERE ...]
DELETE FROM name [WHERE ...]
BEGIN [ISOLATION LEVEL READ COMMITTED|SERIALIZABLE]; COMMIT; ROLLBACK
```

类型：`INT/INTEGER/BIGINT`、`VARCHAR(n)/CHAR`、`TEXT`、`BOOLEAN/BOOL`、`FLOAT/DOUBLE/REAL`。
运算符与谓词：`+ - * /`、`= <> != < > <= >=`、`AND/OR/NOT`、`IS [NOT] NULL`、`LIKE`、`IN`、`BETWEEN`、`CASE`。
聚合：`COUNT/SUM/AVG/MIN/MAX`（支持 `DISTINCT`、`COUNT(*)`）。
连接：`INNER JOIN`、`LEFT [OUTER] JOIN`（以及逗号交叉连接、RIGHT JOIN 解析支持）。
子查询：标量子查询、`IN (SELECT ...)`、`[NOT] EXISTS (SELECT ...)`，支持相关引用。

---

## 8. 测试与性能

- 单元/集成测试位于 `tests/`：B+ 树结构不变量与随机压力、词法/语法、SQL 全功能、事务 ACID 与隔离、崩溃恢复、高并发。
- 基准位于 `benchmarks/benchmark.py`，结果见 [PERFORMANCE.md](PERFORMANCE.md)。
