from __future__ import division


class DslError(Exception):
    """
    Quant DSL exception base class.
    """
    # Todo: Just have one msg parameter (merge 'error' and 'descr')?
    # Todo: Don't keep reference to node, just keep the line number.
    def __init__(self, error, descr=None, node=None):
        self.error = error
        self.descr = descr
        self.node = node
        self.lineno = getattr(node, "lineno", None)

    def __repr__(self):
        msg = self.error
        if self.descr:
            msg += ": %s" % self.descr
        if self.lineno:
            msg += " (line %d)" % (self.lineno)
        return msg

    __str__ = __repr__


class DslSyntaxError(DslError):
    """
    Exception class for user syntax errors.
    """


class DslTestExpressionCannotBeEvaluated(DslSyntaxError):
    """
    Exception class for test expression evaluation errors.
    """


class DslPresentTimeNotInScope(DslSyntaxError):
    """
    Exception class for present time not being in scope.
    """


class DslIfTestExpressionError(DslSyntaxError):
    """
    Exception class for user syntax errors (comparison errors).
    """


class DslCompareArgsError(DslSyntaxError):
    """
    Exception class for user syntax errors (comparison errors).
    """


class DslBinOpArgsError(DslSyntaxError):
    """
    Exception class for user syntax errors (binary operation).
    """


class DslNameError(DslSyntaxError):
    """
    Exception class for undefined names.
    """


class DslSystemError(DslError):
    """
    Exception class for DSL system errors.
    """


class DslCompileError(DslError):
    pass


class CallLimitError(DslCompileError):
    pass


class RecursionDepthError(DslCompileError):
    pass


class TimeoutError(DslError):
    pass


class InterruptSignalReceived(DslError):
    pass