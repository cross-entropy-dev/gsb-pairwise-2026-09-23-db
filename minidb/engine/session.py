"""User-facing session: SQL parsing, transaction control, EXPLAIN.

A session wraps one :class:`~minidb.engine.database.Database`.  Statements
run in autocommit mode unless an explicit transaction is open
(``BEGIN`` / ``COMMIT`` / ``ROLLBACK``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..sql.parser import parse_sql
from ..sql import ast
from ..txn.manager import Isolation, Transaction
from ..txn.locks import DeadlockError
from .executor import Executor, ResultSet
from .planner import plan_select


@dataclass
class ExplainResult:
    plan: str

    def __str__(self) -> str:
        return self.plan


class Session:
    def __init__(self, db) -> None:
        self.db = db
        self.executor = Executor(db)
        self._txn: Optional[Transaction] = None

    # ------------------------------------------------------------------ #
    @property
    def in_transaction(self) -> bool:
        return self._txn is not None and self._txn.state.value == "active"

    def begin(self, isolation: Isolation = Isolation.READ_COMMITTED) -> int:
        if self.in_transaction:
            raise RuntimeError("already in a transaction")
        self._txn = self.db.txn_manager.begin(isolation)
        return self._txn.txid

    def commit(self) -> None:
        txn = self._require_txn()
        self.db.txn_manager.commit(txn)
        self._txn = None

    def rollback(self) -> None:
        txn = self._require_txn()
        self.db.txn_manager.abort(txn)
        self._txn = None

    def _require_txn(self) -> Transaction:
        if self._txn is None:
            raise RuntimeError("no active transaction")
        return self._txn

    # ------------------------------------------------------------------ #
    def sql(self, text: str):
        """Execute one SQL statement (without trailing ';').

        Returns a :class:`ResultSet`, :class:`ExplainResult`, or a status
        string for transaction-control statements.
        """
        explain = False
        stripped = text.strip()
        upper = stripped.upper()
        if upper == "EXPLAIN" or upper.startswith("EXPLAIN "):
            explain = True
            stripped = stripped[7:].strip()
            if not stripped:
                raise RuntimeError("EXPLAIN requires a SELECT statement")

        node = parse_sql(stripped)

        # transaction control
        if isinstance(node, ast.BeginStmt):
            iso = (
                Isolation.SERIALIZABLE
                if node.isolation == "SERIALIZABLE"
                else Isolation.READ_COMMITTED
            )
            self.begin(iso)
            return f"BEGIN (isolation: {iso.value})"
        if isinstance(node, ast.CommitStmt):
            self.commit()
            return "COMMIT"
        if isinstance(node, ast.RollbackStmt):
            self.rollback()
            return "ROLLBACK"

        if explain:
            if not isinstance(node, ast.SelectStmt) or node.from_table is None:
                raise RuntimeError("EXPLAIN supports SELECT ... FROM")
            tables = [node.from_table] + [j.table for j in node.joins]
            schemas = {
                t: self.db.get_table(t).schema.column_names_lower()
                for t in tables
            }
            return ExplainResult(plan_select(node, schemas).description)

        own_txn = self._txn is None
        if own_txn:
            self.begin(Isolation.READ_COMMITTED)
        txn = self._txn
        assert txn is not None
        try:
            result = self.executor.execute(node, txn)
            if own_txn:
                self.commit()
            return result
        except BaseException:
            # On any failure (including being the deadlock victim) an
            # autocommit transaction must abort so its locks are released.
            if own_txn:
                self.db.txn_manager.abort(txn)
                self._txn = None
            raise

    # convenience ------------------------------------------------------ #
    def close(self) -> None:
        if self.in_transaction:
            self.rollback()
