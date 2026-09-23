"""minidb - a small ACID, MVCC, B+ tree-backed in-memory SQL database.

Pure standard-library Python; no third-party database libraries.

Quick start::

    from minidb import Database
    db = Database(":memory:")
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, n INT)")
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20)")
    for row in db.execute("SELECT SUM(n) FROM t").tuples():
        print(row)
"""

from .database import Database
from .errors import (MiniDBError, LexerError, ParseError, ExecutionError,
                     TableNotFoundError, TableExistsError, ColumnNotFoundError,
                     TypeMismatchError, ConstraintViolationError,
                     TransactionError, DeadlockError, LockTimeoutError,
                     SerializationError)
from .transaction.transaction_manager import READ_COMMITTED, SERIALIZABLE

__all__ = [
    "Database", "MiniDBError", "LexerError", "ParseError", "ExecutionError",
    "TableNotFoundError", "TableExistsError", "ColumnNotFoundError",
    "TypeMismatchError", "ConstraintViolationError", "TransactionError",
    "DeadlockError", "LockTimeoutError", "SerializationError",
    "READ_COMMITTED", "SERIALIZABLE",
]

__version__ = "0.1.0"
