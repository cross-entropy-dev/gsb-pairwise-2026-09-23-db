"""Database engine facade.

A :class:`Database` owns the catalog (committed tables), the WAL, the lock
manager and the transaction manager.  It parses SQL to AST, runs it through
:class:`QueryExecutor`, and exposes:

* :meth:`begin` / :meth:`commit` / :meth:`rollback` for explicit transactions
* autocommit mode for one-off statements
* :meth:`checkpoint` writing a consistent on-disk snapshot
* automatic recovery on open (checkpoint + WAL replay)
* :meth:`simulate_crash` for durability tests

Typical use::

    db = Database("./data")
    db.execute("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(50))")
    db.execute("INSERT INTO users VALUES (1, 'Ada')")
    print(db.execute("SELECT * FROM users").dicts())
    db.close()
"""

import os
import threading

from .sql import ast_nodes as ast
from .sql.parser import parse
from .errors import (MiniDBError, TransactionError, ExecutionError,
                     DeadlockError)
from .storage.wal import WAL, write_checkpoint, read_checkpoint
from .storage.table import BOOTSTRAP_TS
from .transaction.lock_manager import LockManager
from .transaction.transaction_manager import (
    TransactionManager, READ_COMMITTED, SERIALIZABLE)
from .executor.executor import QueryExecutor, QueryResult

_TRANSACTION_NODES = (ast.Begin, ast.Commit, ast.Rollback)


class _NullWAL:
    """WAL stand-in for fully in-memory databases."""

    def append(self, record, fsync=False):
        pass

    def read_frames(self):
        return []

    def rewrite(self, records, ts):
        pass

    def close(self):
        pass


class Database:
    def __init__(self, directory=":memory:", btree_order=64,
                 lock_escalation_threshold=200):
        self.directory = directory
        self.btree_order = btree_order
        self.catalog = {}
        if directory == ":memory:":
            self._wal_path = None
            self._checkpoint_path = None
            self.wal = _NullWAL()
        else:
            os.makedirs(directory, exist_ok=True)
            self._wal_path = os.path.join(directory, "wal.log")
            self._checkpoint_path = os.path.join(directory, "checkpoint.db")
            self.wal = WAL(self._wal_path)
        self.lm = LockManager(
            escalation_threshold=lock_escalation_threshold)
        self.tm = TransactionManager(self.catalog, self.wal, self.lm)
        self.executor = QueryExecutor(self)
        self._checkpoint_lock = threading.Lock()
        self._closed = False
        self.recover()

    # ------------------------------------------------------------- recovery

    def recover(self):
        checkpoint_doc = (read_checkpoint(self._checkpoint_path)
                          if self._checkpoint_path else None)
        records = self.wal.read_frames() if self._wal_path else []
        self.tm.recover(checkpoint_doc, records)

    def visible_tables(self, txn=None):
        if txn is None:
            return set(self.catalog)
        names = set(self.catalog) - txn.dropped_tables
        names |= set(txn.created_tables)
        return names

    # -------------------------------------------------------- SQL execution

    def execute(self, sql, txn=None, isolation=None):
        """Execute one SQL statement.

        Without an explicit ``txn`` the statement runs as an autocommit
        transaction.  BEGIN/COMMIT/ROLLBACK control an explicit transaction.
        """
        statements = parse(sql)
        if len(statements) != 1:
            raise ExecutionError(
                f"execute() expects one statement, got {len(statements)}; "
                f"use executescript()")
        return self._run(statements[0], txn, isolation)

    def executescript(self, sql, isolation=READ_COMMITTED):
        """Run several statements inside one transaction; returns last result."""
        statements = parse(sql)
        txn = self.begin(isolation)
        result = None
        try:
            for node in statements:
                result = self._run(node, txn, isolation)
            self.commit(txn)
            return result
        except Exception:
            self.rollback(txn)
            raise

    def _run(self, node, txn, isolation=None):
        # Transaction control statements.
        if isinstance(node, ast.Begin):
            if txn is not None:
                raise TransactionError("already inside a transaction")
            return self.begin(node.isolation or READ_COMMITTED)
        if isinstance(node, ast.Commit):
            if txn is None:
                raise TransactionError("no active transaction to commit")
            self.commit(txn)
            return QueryResult(message="COMMIT")
        if isinstance(node, ast.Rollback):
            if txn is None:
                raise TransactionError("no active transaction to rollback")
            self.rollback(txn)
            return QueryResult(message="ROLLBACK")

        own_txn = txn is None
        if own_txn:
            txn = self.begin(isolation or READ_COMMITTED)
        try:
            result = self.executor.execute(node, txn)
            if own_txn:
                self.commit(txn)
            return result
        except (DeadlockError,):
            # A deadlock victim is aborted inside the lock manager; make sure
            # its in-memory state is rolled back too.
            if txn.status == "active":
                self.rollback(txn)
            raise
        except Exception:
            if own_txn and txn.status == "active":
                self.rollback(txn)
            raise

    # ---------------------------------------------------- transaction API

    def begin(self, isolation=READ_COMMITTED):
        return self.tm.begin(isolation)

    def commit(self, txn):
        return self.tm.commit(txn)

    def rollback(self, txn):
        self.tm.abort(txn)

    # ----------------------------------------------------------- checkpoint

    def checkpoint(self):
        """Persist a consistent snapshot of committed data.

        1. Snapshot every table's latest committed rows.
        2. Atomically write checkpoint.db.
        3. Rewrite wal.log keeping only frames of still-active transactions
           and append a checkpoint marker.
        """
        with self._checkpoint_lock:
            with self.tm._mutex:
                max_ts = self.tm._clock
                active_ids = {t.tn_id if False else t.txn_id
                              for t in self.tm.active.values()}
                doc = {"max_ts": max_ts, "tables": {}}
                for name, table in self.catalog.items():
                    rows = {}
                    for key, chain in table.tree.items():
                        for v in chain:
                            if v.xmin_ts > 0 and v.xmax_ts == 0:
                                rows[key] = {
                                    k: val for k, val in v.data.items()
                                    if not k.startswith("_") or k == "_rowid_"}
                                break
                    doc["tables"][name] = {
                        "schema": table.schema.to_dict(),
                        "next_rowid": table.next_rowid,
                        "auto_counters": table.auto_counters,
                        "rows": rows,
                    }
            if self._checkpoint_path is not None:
                write_checkpoint(self._checkpoint_path, doc)

            if self._wal_path is not None:
                keep = []
                for rec in self.wal.read_frames():
                    if rec.get("type") == "checkpoint":
                        keep = []
                        continue
                    if rec.get("txn") in active_ids:
                        keep.append(rec)
                self.wal.rewrite(keep, max_ts)
            return max_ts

    # ------------------------------------------------------------- lifecycle

    def simulate_crash(self):
        """Drop all in-memory state without any graceful shutdown.

        The OS file handle is released (as happens when a process dies) but
        no checkpoint is taken and the WAL is not rewritten.
        """
        self.catalog.clear()
        self.tm.active.clear()
        if self._wal_path is not None:
            try:
                self.wal.close()
            except OSError:
                pass
        self._closed = True

    def close(self):
        if self._closed:
            return
        self.checkpoint()
        if self._wal_path is not None:
            self.wal.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # --------------------------------------------------------------- introspection

    def table_names(self):
        return sorted(self.catalog)
