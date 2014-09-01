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


class DslNameError(DslError):
    """
    Exception class for undefined names.
    """


class DslSystemError(DslError):
    """
    Exception class for DSL system errors.
    """