"""AST node definitions for the SQL subset.

The classes are plain dataclasses; the parser produces them, the planner /
executor consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# --------------------------------------------------------------------- #
# expressions
# --------------------------------------------------------------------- #
@dataclass
class Literal:
    value: Any  # int, float, str, bool, None


@dataclass
class ColumnRef:
    name: str
    table: Optional[str] = None  # alias / table qualifier


@dataclass
class Star:
    table: Optional[str] = None


@dataclass
class UnaryOp:
    op: str           # "-", "NOT", "IS NULL", "IS NOT NULL"
    operand: Any


@dataclass
class BinaryOp:
    op: str           # = <> < <= > >= + - * / AND OR LIKE IN BETWEEN
    left: Any
    right: Any


@dataclass
class FuncCall:
    name: str         # COUNT/SUM/AVG/MIN/MAX/UPPER/LOWER
    args: list[Any]
    distinct: bool = False
    star: bool = False  # COUNT(*)


@dataclass
class Subquery:
    query: "SelectStmt"


@dataclass
class InSubquery:
    expr: Any
    query: "SelectStmt"
    negated: bool = False


@dataclass
class Exists:
    query: "SelectStmt"
    negated: bool = False


# --------------------------------------------------------------------- #
# statements
# --------------------------------------------------------------------- #
@dataclass
class ColumnDef:
    name: str
    type: str
    length: Optional[int] = None
    nullable: bool = True
    primary_key: bool = False


@dataclass
class CreateTable:
    name: str
    columns: list[ColumnDef]


@dataclass
class Insert:
    table: str
    columns: Optional[list[str]]
    rows: list[list[Any]]  # each is a list of expressions (literals)


@dataclass
class Update:
    table: str
    assignments: list[tuple[str, Any]]
    where: Any = None


@dataclass
class Delete:
    table: str
    where: Any = None


@dataclass
class JoinSpec:
    table: str
    alias: Optional[str]
    kind: str           # "INNER" / "LEFT"
    on: Any


@dataclass
class SelectStmt:
    projections: list[Any]              # expressions / Star / FuncCall
    from_table: Optional[str] = None
    from_alias: Optional[str] = None
    joins: list[JoinSpec] = field(default_factory=list)
    where: Any = None
    group_by: list[Any] = field(default_factory=list)
    having: Any = None
    order_by: list[tuple[Any, str]] = field(default_factory=list)  # (expr, ASC/DESC)
    limit: Optional[int] = None
    offset: Optional[int] = None
    distinct: bool = False


@dataclass
class BeginStmt:
    isolation: Optional[str] = None  # "READ COMMITTED" / "SERIALIZABLE"


@dataclass
class CommitStmt:
    pass


@dataclass
class RollbackStmt:
    pass


@dataclass
class ShowTables:
    pass


@dataclass
class Describe:
    table: str
