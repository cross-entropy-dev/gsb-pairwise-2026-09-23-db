"""Exception hierarchy for minidb."""


class MiniDBError(Exception):
    """Base class for all minidb errors."""


class LexerError(MiniDBError):
    pass


class ParseError(MiniDBError):
    pass


class ExecutionError(MiniDBError):
    pass


class TableNotFoundError(ExecutionError):
    pass


class TableExistsError(ExecutionError):
    pass


class ColumnNotFoundError(ExecutionError):
    pass


class DuplicateColumnError(ExecutionError):
    pass


class TypeMismatchError(ExecutionError):
    pass


class ConstraintViolationError(ExecutionError):
    pass


class TransactionError(MiniDBError):
    pass


class DeadlockError(TransactionError):
    """Raised when the deadlock detector chooses this transaction as victim."""


class LockTimeoutError(TransactionError):
    pass


class SerializationError(TransactionError):
    """SERIALIZABLE transaction failed validation and was aborted."""
