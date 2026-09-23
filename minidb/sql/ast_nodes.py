"""AST node definitions produced by the parser and consumed by the executor."""

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------- expressions

@dataclass
class Literal:
    value: Any  # int, float, str, bool, or None


@dataclass
class Star:
    """Unqualified ``*``."""


@dataclass
class QualifiedStar:
    table: str


@dataclass
class Column:
    name: str
    table: Optional[str] = None


@dataclass
class BinaryOp:
    op: str
    left: Any
    right: Any


@dataclass
class UnaryOp:
    op: str
    operand: Any


@dataclass
class FunctionCall:
    name: str  # upper-cased
    args: list
    distinct: bool = False
    star: bool = False  # COUNT(*)


@dataclass
class Between:
    expr: Any
    low: Any
    high: Any
    negated: bool = False


@dataclass
class InList:
    expr: Any
    values: list
    negated: bool = False


@dataclass
class InSubquery:
    expr: Any
    subquery: Any
    negated: bool = False


@dataclass
class Exists:
    subquery: Any
    negated: bool = False


@dataclass
class Subquery:
    select: Any


@dataclass
class CaseExpr:
    operand: Any  # simple CASE operand, or None for searched CASE
    whens: list  # list of (condition, result)
    default: Any = None


# ------------------------------------------------------------------- clauses

@dataclass
class ColumnDef:
    name: str
    type_name: str
    length: Optional[int] = None
    not_null: bool = False
    primary_key: bool = False
    unique: bool = False
    default: Any = None
    autoincrement: bool = False


@dataclass
class TableRef:
    name: str
    alias: Optional[str] = None


@dataclass
class Join:
    join_type: str  # "INNER", "LEFT", "RIGHT", "CROSS"
    left: Any
    right: Any
    on: Any = None


@dataclass
class SelectItem:
    expr: Any
    alias: Optional[str] = None


@dataclass
class OrderByItem:
    expr: Any
    desc: bool = False


# ----------------------------------------------------------------- statements

@dataclass
class CreateTable:
    name: str
    columns: list  # list[ColumnDef]
    if_not_exists: bool = False


@dataclass
class DropTable:
    name: str
    if_exists: bool = False


@dataclass
class Insert:
    table: str
    columns: Optional[list]  # explicit column list, or None
    rows: list = field(default_factory=list)  # list[list[expr]]


@dataclass
class Select:
    items: list  # list[SelectItem]
    source: Any = None  # TableRef | Join
    where: Any = None
    group_by: list = field(default_factory=list)
    having: Any = None
    order_by: list = field(default_factory=list)
    limit: Any = None
    offset: Any = None
    distinct: bool = False


@dataclass
class Update:
    table: str
    assignments: list  # list[(column_name, expr)]
    where: Any = None


@dataclass
class Delete:
    table: str
    where: Any = None


@dataclass
class Begin:
    isolation: Optional[str] = None  # "READ COMMITTED" / "SERIALIZABLE"


@dataclass
class Commit:
    pass


@dataclass
class Rollback:
    pass
