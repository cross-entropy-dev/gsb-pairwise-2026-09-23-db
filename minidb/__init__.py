"""minidb – a small in-memory ACID database engine with MVCC + WAL.

Typical usage::

    from minidb import Database, Session

    db = Database("./data")
    s = Session(db)
    s.sql("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(50))")
    s.sql("INSERT INTO users VALUES (1, 'Ada')")
    print(s.sql("SELECT * FROM users").rows)
"""

from .engine.database import Database
from .engine.session import Session
from .engine.executor import ResultSet
from .txn.manager import Isolation

__all__ = ["Database", "Session", "ResultSet", "Isolation"]
__version__ = "1.0.0"
