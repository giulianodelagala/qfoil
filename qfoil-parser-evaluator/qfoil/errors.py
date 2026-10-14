"""Public exceptions raised at QFOIL API boundaries."""


class QFOILError(Exception):
    """Base class for public QFOIL errors."""


class QFOILSyntaxError(QFOILError, ValueError):
    """The supplied QFOIL expression is not syntactically valid."""


class QFOILValidationError(QFOILError, ValueError):
    """The supplied values do not satisfy an API contract."""


class UnsupportedOperationError(QFOILError):
    """The requested operation is outside the supported QFOIL subset."""


__all__ = [
    "QFOILError",
    "QFOILSyntaxError",
    "QFOILValidationError",
    "UnsupportedOperationError",
]
