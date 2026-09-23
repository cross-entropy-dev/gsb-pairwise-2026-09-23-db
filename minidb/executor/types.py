"""SQL value type coercion and three-valued-logic helpers."""

import math

from ..errors import TypeMismatchError


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def sql_truth(v):
    """Three-valued logic truth: True / False / None (unknown)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if is_number(v):
        return v != 0
    if isinstance(v, str):
        return v != ""
    return bool(v)


def coerce_value(value, col):
    """Coerce a Python value to the declared column type.

    Returns the stored value. Raises TypeMismatchError on bad input.
    """
    if value is None:
        # NOT NULL enforcement is the executor's job (it raises a
        # constraint violation after the whole row has been built).
        return None

    t = col.type_name
    if t == "INT":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise TypeMismatchError(
            f"column {col.name!r} expects INT, got {value!r}")
    if t == "FLOAT":
        if is_number(value):
            return float(value)
        if isinstance(value, bool):
            return float(value)
        raise TypeMismatchError(
            f"column {col.name!r} expects FLOAT, got {value!r}")
    if t in ("VARCHAR", "TEXT"):
        if not isinstance(value, str):
            raise TypeMismatchError(
                f"column {col.name!r} expects {t}, got {value!r}")
        if t == "VARCHAR" and col.length is not None and len(value) > col.length:
            raise TypeMismatchError(
                f"value too long for {col.name!r} "
                f"({len(value)} > VARCHAR({col.length}))")
        return value
    if t == "BOOLEAN":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise TypeMismatchError(
            f"column {col.name!r} expects BOOLEAN, got {value!r}")
    raise TypeMismatchError(f"unknown column type {t!r}")


def compare(a, b):
    """SQL comparison: -1/0/1 or None when not comparable / NULL input."""
    if a is None or b is None:
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        if isinstance(a, bool) and isinstance(b, bool):
            return (a > b) - (a < b)
        return None
    if is_number(a) and is_number(b):
        return (a > b) - (a < b)
    if isinstance(a, str) and isinstance(b, str):
        return (a > b) - (a < b)
    return None


def add(a, b):
    if a is None or b is None:
        return None
    if is_number(a) and is_number(b):
        return a + b
    raise TypeMismatchError(f"cannot add {a!r} and {b!r}")


def sub(a, b):
    if a is None or b is None:
        return None
    if is_number(a) and is_number(b):
        return a - b
    raise TypeMismatchError(f"cannot subtract {b!r} from {a!r}")


def mul(a, b):
    if a is None or b is None:
        return None
    if is_number(a) and is_number(b):
        return a * b
    raise TypeMismatchError(f"cannot multiply {a!r} and {b!r}")


def div(a, b):
    if a is None or b is None:
        return None
    if not (is_number(a) and is_number(b)):
        raise TypeMismatchError(f"cannot divide {a!r} by {b!r}")
    if b == 0:
        return None  # SQL division by zero yields NULL
    return a / b


def like_to_regex(pattern):
    """Translate a SQL LIKE pattern (% and _) into a regex source string."""
    import re
    out = ["^"]
    for ch in pattern:
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    out.append("$")
    return "".join(out)


def like(value, pattern):
    if value is None or pattern is None:
        return None
    import re
    return re.match(like_to_regex(pattern), str(value), re.DOTALL) is not None


def aggregate_init(name):
    if name == "COUNT":
        return 0
    return None


def aggregate_step(name, state, value, distinct_values=None):
    if name == "COUNT":
        if distinct_values is not None:
            distinct_values.add(value)
            return len(distinct_values)
        return state + 1 if value is not None else state
    if value is None:
        return state
    if distinct_values is not None:
        if value in distinct_values:
            return state
        distinct_values.add(value)
    if name == "SUM":
        return value if state is None else state + value
    if name == "AVG":
        total, count = state if state is not None else (0, 0)
        return (total + value, count + 1)
    if name == "MIN":
        return value if state is None or compare(value, state) < 0 else state
    if name == "MAX":
        return value if state is None or compare(value, state) > 0 else state
    raise TypeMismatchError(f"unknown aggregate {name}")


def aggregate_final(name, state):
    if name == "COUNT":
        return state
    if state is None:
        return None
    if name == "AVG":
        total, count = state
        return total / count if count else None
    return state
