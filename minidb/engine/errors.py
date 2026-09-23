"""Exception hierarchy for the engine."""

from __future__ import annotations


class MiniDBError(Exception):
    """Base class for all engine errors."""


class IntegrityError(MiniDBError):
    """Primary-key / NOT NULL / type constraint violation."""


class ExecutionError(MiniDBError):
    pass
