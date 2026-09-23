"""Schema / type system.

Supported column types: ``INT``, ``FLOAT``, ``VARCHAR(n)``, ``TEXT`` and
``BOOLEAN``.  Values are normalised on write so the rest of the engine only
ever sees native Python objects (``int | float | str | bool | None``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# type tags
INT = "INT"
FLOAT = "FLOAT"
VARCHAR = "VARCHAR"
TEXT = "TEXT"
BOOLEAN = "BOOLEAN"

TYPE_ALIASES = {
    "INT": INT,
    "INTEGER": INT,
    "BIGINT": INT,
    "SMALLINT": INT,
    "TINYINT": INT,
    "FLOAT": FLOAT,
    "DOUBLE": FLOAT,
    "REAL": FLOAT,
    "VARCHAR": VARCHAR,
    "CHAR": VARCHAR,
    "TEXT": TEXT,
    "STRING": TEXT,
    "BOOLEAN": BOOLEAN,
    "BOOL": BOOLEAN,
}


class SchemaError(Exception):
    """Raised for invalid DDL."""


class TypeConversionError(Exception):
    """Raised when a value cannot be stored in a column."""


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    length: Optional[int] = None
    nullable: bool = True
    primary_key: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "length": self.length,
            "nullable": self.nullable,
            "primary_key": self.primary_key,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Column":
        return Column(
            name=d["name"],
            type=d["type"],
            length=d.get("length"),
            nullable=d.get("nullable", True),
            primary_key=d.get("primary_key", False),
        )


@dataclass
class TableSchema:
    name: str
    columns: list[Column] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index = {c.name: i for i, c in enumerate(self.columns)}

    # ------------------------------------------------------------------ #
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column_names_lower(self) -> list[str]:
        return [c.name.lower() for c in self.columns]

    def index_of(self, name: str) -> int:
        return self._index[name]

    def has_column(self, name: str) -> bool:
        return name in self._index

    def column(self, name: str) -> Column:
        return self.columns[self._index[name]]

    @property
    def primary_key_columns(self) -> list[Column]:
        return [c for c in self.columns if c.primary_key]

    def validate_and_convert(self, row: tuple[Any, ...]) -> tuple[Any, ...]:
        """Type-check / coerce a full row (one value per column)."""
        if len(row) != len(self.columns):
            raise TypeConversionError(
                f"table {self.name!r} expects {len(self.columns)} values, "
                f"got {len(row)}"
            )
        out: list[Any] = []
        for col, val in zip(self.columns, row):
            out.append(convert_value(col, val))
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "columns": [c.to_dict() for c in self.columns]}

    @staticmethod
    def from_dict(d: dict[str]) -> "TableSchema":
        return TableSchema(
            name=d["name"],
            columns=[Column.from_dict(c) for c in d["columns"]],
        )


def convert_value(col: Column, val: Any) -> Any:
    """Convert a Python/SQL literal value to the column's native type."""
    if val is None:
        if not col.nullable:
            raise TypeConversionError(
                f"column {col.name!r} is NOT NULL"
            )
        return None
    t = col.type
    try:
        if t == INT:
            if isinstance(val, bool):
                return int(val)
            if isinstance(val, int):
                return val
            if isinstance(val, float) and val.is_integer():
                return int(val)
            if isinstance(val, str):
                return int(val.strip())
            raise ValueError(val)
        if t == FLOAT:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
            if isinstance(val, str):
                return float(val.strip())
            raise ValueError(val)
        if t == BOOLEAN:
            if isinstance(val, bool):
                return val
            if isinstance(val, int) and val in (0, 1):
                return bool(val)
            if isinstance(val, str):
                low = val.strip().lower()
                if low in ("true", "t", "1", "yes"):
                    return True
                if low in ("false", "f", "0", "no"):
                    return False
            raise ValueError(val)
        # varchar / text
        if isinstance(val, str):
            s = val
        elif isinstance(val, bool):
            s = "TRUE" if val else "FALSE"
        else:
            s = str(val)
        if t == VARCHAR and col.length is not None and len(s) > col.length:
            raise TypeConversionError(
                f"value too long for {col.name} VARCHAR({col.length}): "
                f"{len(s)} chars"
            )
        return s
    except TypeConversionError:
        raise
    except (ValueError, TypeError):
        raise TypeConversionError(
            f"cannot convert {val!r} to {col.type} for column {col.name!r}"
        )
