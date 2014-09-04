from __future__ import division
from abc import ABCMeta, abstractmethod
import Queue as queue
import datetime
import itertools
import math
import re
import uuid

import dateutil.parser

from quantdsl.exceptions import DslSystemError, DslSyntaxError, DslNameError, DslError
from quantdsl.priceprocess.base import getDurationYears
from quantdsl import utc


class DslObject(object):
    """
    Base class for DSL language objects.

    Responsible for maintaining reference to original AST (for error reporting),
    and for rendering objects into valid DSL source code. Also has methods for
    validating object arguments, and finding child nodes of a particular type.
    """

    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwds):
        self.node = kwds.pop('node', None)
        self.validate(args)
        self._args = list(args)

    def __str__(self, indent=0):
        """
        Returns DSL source code, that can be parsed to generate a clone of self.
        """
        return "%s(%s)" % (self.__class__.__name__, ", ".join([str(i) for i in self._args]))

    @property
    def hash(self):
        """
        Creates a hash that is unique for this fragment of DSL.
        """
        if not hasattr(self, '_hash'):
            try:
                hashes = ""
                for arg in self._args:
                    if isinstance(arg, list):
                        arg = tuple(arg)
                    hashes += str(hash(arg))
            except TypeError, e:
                raise DslSystemError(e)
            self._hash = hash(hashes)
        return self._hash

    def __hash__(self):
        return self.hash

    def pprint(self, indent=''):
        msg = self.__class__.__name__ + "("
        lenArgs = len(self._args)
        if lenArgs > 1:
            msg += "\n"
        tab = 4
        indent += ' ' * tab
        for i, arg in enumerate(self._args):
            if lenArgs > 1:
                msg += indent
            if isinstance(arg, DslObject):
                msg += arg.pprint(indent)
            else:
                msg += str(arg)
            if i < lenArgs - 1:
                msg += ","
            if lenArgs > 1:
                msg += "\n"
        if lenArgs > 1:
            msg += indent
        msg += ")"
        return msg

    @abstractmethod
    def validate(self, args):
        pass

    # Todo: Rework validation, perhaps by considering a declarative form in which to express the requirements.
    def assertArgsLen(self, args, requiredLen=None, minLen=None):
        if minLen != None and len(args) < minLen:
            error = "%s is broken" % self.__class__.__name__
            descr = "requires at least %s arguments (%s were given)" % (minLen, len(args))
            raise DslSyntaxError(error, descr, self.node)
        if requiredLen != None and len(args) != requiredLen:
            error = "%s is broken" % self.__class__.__name__
            descr = "requires %s arguments (%s were given)" % (requiredLen, len(args))
            raise DslSyntaxError(error, descr, self.node)

    def assertArgsPosn(self, args, posn, requiredType):
        if isinstance(requiredType, list):
            # Ahem, this is a way of saying we require a list of the type (should be a list length 1).
            self.assertArgsPosn(args, posn, list)
            assert len(requiredType) == 1, "List def should only have one item."
            requiredType = requiredType[0]
            listOfArgs = args[posn]
            for i in range(len(listOfArgs)):
                self.assertArgsPosn(listOfArgs, i, requiredType)
        elif not isinstance(args[posn], requiredType):
            error = "%s is broken" % self.__class__.__name__
            if isinstance(requiredType, tuple):
                requiredTypeNames = [i.__name__ for i in requiredType]
                requiredTypeNames = ", ".join(requiredTypeNames)
            else:
                requiredTypeNames = requiredType.__name__
            desc = "argument %s must be %s" % (posn, requiredTypeNames)
            desc += " (but a %s was found): " % (args[posn].__class__.__name__)
            desc += str(args[posn])
            raise DslSyntaxError(error, desc, self.node)

    def findInstances(self, dslType):
        return list(self.findInstancesGenerator(dslType))

    def hasInstances(self, dslType):
        try:
            self.findInstancesGenerator(dslType).next()
        except StopIteration:
            return False
        else:
            return True

    def findInstancesGenerator(self, dslType):
        if isinstance(self, dslType):
            yield self
        for arg in self._args:
            if isinstance(arg, DslObject):
                for dslObj in arg.findInstancesGenerator(dslType):
                    yield dslObj
            elif isinstance(arg, list):
                for arg in arg:
                    if isinstance(arg, DslObject):
                        for dslObj in arg.findInstances(dslType):
                            yield dslObj

    def reduce(self, dslLocals, dslGlobals, effectivePresentTime=None, pendingCallStack=None):
        """
        Reduces by reducing all args, and then using those args
        to create a new instance of self.
        """
        newDslArgs = []
        for dslArg in self._args:
            if isinstance(dslArg, DslObject):
                dslArg = dslArg.reduce(dslLocals, dslGlobals, effectivePresentTime, pendingCallStack=pendingCallStack)
            newDslArgs.append(dslArg)
        return self.__class__(node=self.node, *newDslArgs)


class DslExpression(DslObject):

    @abstractmethod
    def evaluate(self, **kwds):
        pass

    def discount(self, value, date, **kwds):
        r = float(kwds['interestRate']) / 100
        T = getDurationYears(kwds['presentTime'], date)
        return value * math.exp(- r * T)


class DslConstant(DslExpression):
    requiredType = None

    def __str__(self, indent=0):
        return repr(self.value)

    def validate(self, args):
        self.assertArgsLen(args, requiredLen=1)
        if self.requiredType == None:
            raise Exception, "requiredType attribute not set on %s" % self.__class__
        self.assertArgsPosn(args, posn=0, requiredType=self.requiredType)
        self.parse(args[0])

    @property
    def value(self):
        if not hasattr(self, '_value'):
            self._value = self.parse(self._args[0])
        return self._value

    def evaluate(self, **_):
        return self.value

    def parse(self, value):
        return value


class String(DslConstant):
    requiredType = basestring


class Number(DslConstant):

    @property
    def requiredType(self):
        from numpy import ndarray
        return (int, float, ndarray)


class Date(DslConstant):
    requiredType = (basestring, String, datetime.datetime)

    def __str__(self, indent=0):
        return "Date('%04d-%02d-%02d')" % (self.value.year, self.value.month, self.value.day)

    def parse(self, value):
        # Return a datetime.datetime.
        if isinstance(value, (basestring, String)):
            if isinstance(value, String):
                dateStr = value.evaluate()
            else:
                dateStr = value
            try:
                return dateutil.parser.parse(dateStr).replace(tzinfo=utc)
            except ValueError, inst:
                raise DslSyntaxError("invalid date string", dateStr, node=self.node)
        elif isinstance(value, datetime.datetime):
            return value
        else:
            raise DslSystemError("shouldn't get here", value, node=self.node)


class TimeDelta(DslConstant):
    requiredType = (String, datetime.timedelta)

    def __str__(self, indent=0):
        return "%s('%dd')" % (self.__class__.__name__, self.value.days)

    def parse(self, value, regex=re.compile(r'((?P<days>\d+?)d)?')):
        if isinstance(value, String):
            durationStr = value.evaluate()
            parts = regex.match(durationStr)
            if not parts:
                raise DslSyntaxError('invalid time delta string', durationStr, node=self.node)
            parts = parts.groupdict()
            time_params = {}
            for (name, param) in parts.iteritems():
                if param:
                    time_params[name] = int(param)
            return datetime.timedelta(**time_params)
        elif isinstance(value, datetime.timedelta):
            return value
        else:
            raise DslSystemError("shouldn't get here", value, node=self.node)


class UnaryOp(DslExpression):
    opchar = None

    def __str__(self, indent=0):
        return str(self.opchar) + str(self.operand)

    def validate(self, args):
        self.assertArgsLen(args, requiredLen=1)
        self.assertArgsPosn(args, posn=0, requiredType=DslExpression)

    @property
    def operand(self):
        return self._args[0]

    def evaluate(self, **kwds):
        return self.op(self.operand.evaluate(**kwds))

    @abstractmethod
    def op(self, value):
        pass


class UnarySub(UnaryOp):
    opchar = '-'

    def op(self, value):
        return -value


class BoolOp(DslExpression):
    def validate(self, args):
        self.assertArgsLen(args, requiredLen=1)
        self.assertArgsPosn(args, posn=0, requiredType=list)

    @property
    def values(self):
        return self._args[0]

    def evaluate(self, **kwds):
        lenValues = len(self.values)
        assert lenValues >= 2
        for dslExpr in self.values:
            assert isinstance(dslExpr, DslExpression)
            value = dslExpr.evaluate(**kwds)
            # Assert value is a simple value.
            if not isinstance(dslExpr, DslExpression):
                raise DslSyntaxError("not a simple value", str(value), node=self.node)
            if self.op(value):
                return self.op(True)
        return self.op(False)

    @abstractmethod
    def op(self, value):
        pass


class Or(BoolOp):
    def op(self, value):
        return value


class And(BoolOp):
    def op(self, value):
        return not value


class BinOp(DslExpression):
    opchar = ''

    @abstractmethod
    def op(self, left, right):
        pass

    def __str__(self, indent=0):
        if self.opchar:
            def makeStr(dslExpr):
                dslString = str(dslExpr)
                if isinstance(dslExpr, BinOp):
                    dslString = "(" + dslString + ")"
                return dslString
            return makeStr(self.left) + " " + self.opchar + " " + makeStr(self.right)
        else:
            return '%s(%s, %s)' % (self.__class__.__name__, self.left, self.right)

    def validate(self, args):
        self.assertArgsLen(args, requiredLen=2)
        self.assertArgsPosn(args, posn=0, requiredType=(DslExpression, Date, TimeDelta, Underlying))
        self.assertArgsPosn(args, posn=1, requiredType=(DslExpression, Date, TimeDelta, Underlying))

    @property
    def left(self):
        return self._args[0]

    @property
    def right(self):
        return self._args[1]

    def evaluate(self, **kwds):
        left = self.left.evaluate(**kwds)
        right = self.right.evaluate(**kwds)
        if isinstance(left, datetime.timedelta) and isinstance(right, float):
            rightOrig = right
            right = int(right)
            assert rightOrig == right, "Can't %s timedelta and fractional number '%s'" % rightOrig
        elif isinstance(right, datetime.timedelta) and isinstance(left, float):
            leftOrig = left
            left = int(left)
            assert leftOrig == left, "Can't %s timedelta and fractional number '%s'" % leftOrig
        try:
            return self.op(left, right)
        except TypeError, e:
            raise DslSyntaxError("unable to %s" % self.__class__.__name__.lower(), "%s %s: %s" % (left, right, e),
                                 node=self.node)


class Add(BinOp):
    opchar = '+'

    def op(self, left, right):
        return left + right


class Sub(BinOp):
    opchar = '-'

    def op(self, left, right):
        return left - right


class Mult(BinOp):
    opchar = '*'

    def op(self, left, right):
        return left * right


class Div(BinOp):
    opchar = '/'

    def op(self, left, right):
        return left / right


class Max(BinOp):
    def op(self, a, b):
        # Assume a and b have EITHER type ndarray, OR type int or float.
        # Try to 'balance' the sides.
        #  - two scalar numbers are good
        #  - one number with one vector is okay
        #  - two vectors is okay, but they must have the same length.
        import numpy
        aIsaNumber = isinstance(a, (int, float))
        bIsaNumber = isinstance(b, (int, float))
        if aIsaNumber and bIsaNumber:
            # Neither are vectors.
            return max(a, b)
        elif (not aIsaNumber) and (not bIsaNumber):
            # Both are vectors.
            if len(a) != len(b):
                descr = "%s and %s" % (len(a), len(b))
                raise DslSystemError('Vectors have different length: ', descr, self.node)
        elif aIsaNumber and (not bIsaNumber):
            # Todo: Optimise with scipy.zeros() when a equals zero?
            a = numpy.array([a] * len(b))
        elif bIsaNumber and (not aIsaNumber):
            # Todo: Optimise with scipy.zeros() when b equals zero?
            b = numpy.array([b] * len(a))
        c = numpy.array([a, b])
        return c.max(axis=0)


# Todo: Pow, Mod, FloorDiv don't have proofs, so shouldn't really be used for combining random variables? Either prevent usage with ndarray inputs, or do the proofs. :-)

class Pow(BinOp):
    opchar = '**'

    def op(self, left, right):
        return left ** right


class Mod(BinOp):
    opchar = '%'

    def op(self, left, right):
        return left % right


class FloorDiv(BinOp):
    opchar = '//'

    def op(self, left, right):
        return left // right


class Name(DslExpression):

    def __str__(self, indent=0):
        return self.name

    def validate(self, args):
        assert isinstance(args[0], (basestring, String)), type(args[0])

    @property
    def name(self):
        """
        Returns a Python string.
        """
        name = self._args[0]
        if isinstance(name, basestring):
            return name
        elif isinstance(name, String):
            return name.evaluate()

    def reduce(self, dslLocals, dslGlobals, effectivePresentTime=None, pendingCallStack=False):
        """
        Replace name with named value in context (kwds).
        """

        combinedNamespace = DslNamespace(itertools.chain(dslGlobals.items(), dslLocals.items()))

        from numpy import ndarray
        value = self.evaluate(**combinedNamespace)
        if isinstance(value, basestring):
            return String(value, node=self.node)
        elif isinstance(value, (int, float, ndarray)):
            return Number(value, node=self.node)
        elif isinstance(value, datetime.datetime):
            return Date(value, node=self.node)
        elif isinstance(value, datetime.timedelta):
            return TimeDelta(value, node=self.node)
        elif isinstance(value, DslObject):
            return value
        else:
            raise DslSyntaxError("expected number, string or DSL object when reducing name '%s'" % self.name,
                                 repr(value), node=self.node)

    def evaluate(self, **kwds):
        try:
            return kwds[self.name]
        except KeyError:
            raise DslNameError(
                "'%s' is not defined. Current frame defines" % self.name,
                kwds.keys() or "None",
                node=self.node
            )


class Stub(Name):
    """
    Stubs are named values. Stubs are used to associate a value in a stubbed expression
    with the value of another expression in a dependency graph.
    """

    def __str__(self, indent=0):
        # Can't just return a Python string, like with Names, because this
        # is normally a UUID, and UUIDs are not valid Python variable names
        # because they have dashes and sometimes start with numbers.
        return "Stub('%s')" % self.name


class Underlying(DslObject):

    def validate(self, args):
        self.assertArgsLen(args, 1)

    @property
    def expr(self):
        return self._args[0]

    def evaluate(self, **_):
        return self.expr


class FunctionDef(DslObject):
    """
    A DSL function def creates DSL expressions when called. They can be defined as
    simple or conditionally recursive functions. Loops aren't supported, neither
    are assignments.
    """

    def __str__(self, indent=0):
        indentSpaces = 4 * ' '
        msg = ""
        for decoratorName in self.decoratorNames:
            msg += "@" + decoratorName + "\n"
        msg += "def %s(%s):\n" % (self.name, ", ".join(self.callArgNames))
        if isinstance(self.body, DslObject):
            try:
                msg += indentSpaces + self.body.__str__(indent=indent+1)
            except TypeError:
                raise DslSystemError("DSL object can't handle indent: %s" % type(self.body))
        else:
            msg += str(self.body)
        return msg

    def __init__(self, *args, **kwds):
        super(FunctionDef, self).__init__(*args, **kwds)
        # Initialise the function call cache for this function def.
        self.callCache = {}
        self.enclosedNamespace = DslNamespace()

    def validate(self, args):
        self.assertArgsLen(args, requiredLen=4)

    @property
    def name(self):
        return self._args[0]

    @property
    def callArgNames(self):
        if not hasattr(self, '_callArgNames'):
            self._callArgNames = [i.name for i in self._args[1]]
        return self._callArgNames

    @property
    def callArgs(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def decoratorNames(self):
        return self._args[3]

    def validateCallArgs(self, dslLocals):
        for callArgName in self.callArgNames:
            if callArgName not in dslLocals:
                raise DslSyntaxError('expected call arg not found',
                                     "arg '%s' not in call arg namespace %s" % (callArgName, dslLocals.keys()))

    def apply(self, dslGlobals=None, effectivePresentTime=None, pendingCallStack=None, isDestacking=False, **dslLocals):
        # It's a function call, so create a new namespace "context".
        if dslGlobals is None:
            dslGlobals = DslNamespace()
        else:
           assert isinstance(dslGlobals, DslNamespace)
        dslGlobals = DslNamespace(itertools.chain(self.enclosedNamespace.items(), dslGlobals.items()))
        dslLocals = DslNamespace(dslLocals)

        # Validate the call args with the definition.
        self.validateCallArgs(dslLocals)

        # Create the cache key.
        callCacheKeyDict = dslLocals.copy()
        callCacheKeyDict["__effectivePresentTime__"] = effectivePresentTime
        callCacheKey = self.createHash(dslLocals)

        # Check the call cache, to see whether this function has already been evaluated with these args.
        if not isDestacking and callCacheKey in self.callCache:
            return self.callCache[callCacheKey]

        if pendingCallStack and not isDestacking and not 'nostub' in self.decoratorNames:
            # Just stack the call expression and return a stub.
            assert isinstance(pendingCallStack, queue.Queue)

            # Create a new stub - the stub ID is the name of the return value of the function call..
            stubId = str(uuid.uuid4())
            dslStub = Stub(stubId, node=self.node)

            # Put the function call on the call stack, with the stub ID.
            assert isinstance(pendingCallStack, FunctionDefCallStack)
            pendingCallStack.put(
                stubId=stubId,
                stackedCall=self,
                stackedLocals=dslLocals.copy(),
                stackedGlobals=dslGlobals.copy(),
                effectivePresentTime=effectivePresentTime
            )
            # Return the stub so that the containing DSL can be fully evaluated
            # once the stacked function call has been evaluated.
            dslExpr = dslStub
        else:
            # Todo: Make sure the expression can be selected with the dslLocals?
            # - ie the conditional expressions should be functions only of call arg
            # values that can be fully evaluated without evaluating contractual DSL objects.
            selectedExpression = self.selectExpression(self.body, dslLocals)

            # Add this function to the dslNamespace (just in case it's called by itself).
            newDslGlobals = DslNamespace(dslGlobals)
            newDslGlobals[self.name] = self

            # Reduce the selected expression.
            dslExpr = selectedExpression.reduce(dslLocals, newDslGlobals, effectivePresentTime, pendingCallStack=pendingCallStack)

        # Cache the result.
        if not isDestacking:
            self.callCache[callCacheKey] = dslExpr

        return dslExpr

    def selectExpression(self, dslExpr, callArgNamespace):
        # If the DSL expression is an instance of If, then evaluate
        # the test and accordingly select body or orelse expressions. Repeat
        # this method with the selected expression (supports if-elif-elif-else).
        # Otherwise just return the DSL express as the selected expression.

        if isinstance(dslExpr, BaseIf):
            # Todo: Implement a check that this test expression can be evaluated? Or handle case when it can't?
            # Todo: Also allow user defined functions that just do dates or numbers in test expression.
            # it doesn't have or expand into DSL elements that are the functions of time (Wait, Choice, Market, etc).
            if dslExpr.test.evaluate(**callArgNamespace):
                selected = dslExpr.body
            else:
                selected = dslExpr.orelse
            selected = self.selectExpression(selected, callArgNamespace)
        else:
            selected = dslExpr
        return selected

    def createHash(self, obj):
        if isinstance(obj, (int, float, basestring, datetime.datetime, datetime.timedelta)):
            return hash(obj)
        if isinstance(obj, dict):
            return hash(tuple(sorted([(a, self.createHash(b)) for a, b in obj.items()])))
        if isinstance(obj, list):
            return hash(tuple(sorted([self.createHash(a) for a in obj])))
        elif isinstance(obj, DslObject):
            return hash(obj)
        else:
            raise DslSystemError("Can't create hash from obj type '%s'" % type(obj), obj,
                                      node=obj.node if isinstance(obj, DslObject) else None)


class FunctionCall(DslExpression):

    def __str__(self):
        return "%s(%s)" % (self.functionDefName,
            ", ".join([str(arg) for arg in self.callArgExprs]))

    def validate(self, args):
        self.assertArgsLen(args, requiredLen=2)
        self.assertArgsPosn(args, posn=0, requiredType=Name)
        self.assertArgsPosn(args, posn=1, requiredType=list)

    @property
    def functionDefName(self):
        return self._args[0]

    @property
    def callArgExprs(self):
        return self._args[1]

    def reduce(self, dslLocals, dslGlobals, effectivePresentTime=None, pendingCallStack=False):
        """
        Reduces function call to result of evaluating function def with function call args.
        """

        # Replace functionDef names with things in kwds.
        functionDef = self.functionDefName.reduce(dslLocals, dslGlobals, effectivePresentTime, pendingCallStack=pendingCallStack)

        # Function def should have changed from a Name to a FunctionDef.
        assert isinstance(functionDef, FunctionDef)

        # Check lengths of arg names matches length of arg exprs (function signature must
        # satisfy the call). Or the other way around :).
        if len(functionDef.callArgs) != len(self.callArgExprs):
            raise DslSyntaxError(
                "mismatched call args",
                "expected %s but got %s. Expected args: %s. Received exprs: %s" % (
                    len(functionDef.callArgs),
                    len(self.callArgExprs),
                    functionDef.callArgNames,
                    self.callArgExprs,
                ),
                node=self.node
            )

        # Create a new call arg namespace for the new call arg values.
        newDslLocals = DslNamespace()

        # Obtain the call arg values.
        for callArgExpr, callArgDef in zip(self.callArgExprs, functionDef.callArgs):
            # Skip if it's a DSL object that needs to be evaluated later with market data simulation.
            # Todo: Think about and improve the way these levels are separated.
            if not isinstance(callArgExpr, DslObject):
                # It's a simple value - pass through, not much else to do.
                callArgValue = callArgExpr
            else:
                # Substitute names, etc.
                callArgExpr = callArgExpr.reduce(dslLocals, dslGlobals, effectivePresentTime, pendingCallStack=pendingCallStack)
                # Decide whether to evaluate, or just pass the expression into the function call.
                if isinstance(callArgExpr, Underlying):
                    # It's explicitly wrapped as an "underlying", so unwrap it as expected.
                    callArgValue = callArgExpr.evaluate()
                elif callArgExpr.hasInstances((Market, Fixing, Choice, Settlement, FunctionDef, Stub)):
                    # It's an underlying contract, or a stub. In any case, can't evaluate here, so.pass it through.
                    callArgValue = callArgExpr
                else:
                    assert isinstance(callArgExpr, DslExpression)
                    # It's a sum of two constants, or something like that - evaluate the full expression.
                    callArgValue = callArgExpr.evaluate()

            # Add the call arg value to the new call arg namespace.
            newDslLocals[callArgDef.name] = callArgValue

        # Evaluate the function def with the dict of call arg values.
        dslExpr = functionDef.apply(dslGlobals, effectivePresentTime, pendingCallStack=pendingCallStack, isDestacking=False, **newDslLocals)

        # The result of this function call (stubbed or otherwise) should be a DSL expression.
        assert isinstance(dslExpr, DslExpression)

        return dslExpr

    def evaluate(self, **kwds):
        raise DslSyntaxError('call to undefined name', self.functionDefName.name, node=self.node)


class FunctionArg(DslObject):

    def validate(self, args):
        self.assertArgsLen(args, requiredLen=2)

    @property
    def name(self):
        return self._args[0]

    @property
    def dslTypeName(self):
        return self._args[1]


class BaseIf(DslExpression):

    def validate(self, args):
        self.assertArgsLen(args, requiredLen=3)
        self.assertArgsPosn(args, posn=0, requiredType=DslExpression)
        self.assertArgsPosn(args, posn=1, requiredType=DslExpression)
        self.assertArgsPosn(args, posn=2, requiredType=DslExpression)

    @property
    def test(self):
        return self._args[0]

    @property
    def body(self):
        return self._args[1]

    @property
    def orelse(self):
        return self._args[2]

    def evaluate(self, **kwds):
        testResult = self.test.evaluate(**kwds)
        if isinstance(testResult, DslObject):
            raise DslSyntaxError("If test condition result cannot be a DSL object", str(testResult), node=self.node)
        if testResult:
            return self.body.evaluate(**kwds)
        else:
            return self.orelse.evaluate(**kwds)


class If(BaseIf):

    def __str__(self, indent=0):
        INDENT = indent * 4 * ' '
        msg = "\n"
        msg += INDENT+"if %s:\n" % self.test
        msg += INDENT+"    %s\n" % self.body

        def strOrelse(orelse):
            msg = ''
            if isinstance(orelse, If):
                msg += INDENT+"elif %s:\n" % orelse.test
                msg += INDENT+"    %s\n" % orelse.body
                msg += strOrelse(orelse.orelse)
            else:
                msg += INDENT+"else:\n"
                msg += INDENT+"    %s\n"% orelse
            return msg

        msg += strOrelse(self.orelse)
        return msg


class IfExp(If):
    """
    Special case of If, where if-else clause is one expression (no elif support).
    """

    def __str__(self, indent=0):
        return "%s if %s else %s" % (self.body, self.test, self.orelse)


class Compare(DslExpression):
    validOps = {
        'Eq': lambda a, b: a == b,
        'NotEq': lambda a, b: a != b,
        'Lt': lambda a, b: a < b,
        'LtE': lambda a, b: a <= b,
        'Gt': lambda a, b: a > b,
        'GtE': lambda a, b: a >= b,
    }

    opcodes = {
        'Eq': '==',
        'NotEq': '!=',
        'Lt': '<',
        'LtE': '<=',
        'Gt': '>',
        'GtE': '>=',
    }

    def __str__(self, indent=0):
        return str(self.left) + ' ' \
            +  " ".join([str(self.opcodes[op])+' '+str(right) for (op, right) in zip(self.opNames, self.comparators) ])


    def validate(self, args):
        self.assertArgsLen(args, 3)
        self.assertArgsPosn(args, 0, requiredType=(
            DslExpression, Date))  #, Date, Number, String, int, float, basestring, datetime.datetime))
        self.assertArgsPosn(args, 1, requiredType=list)
        self.assertArgsPosn(args, 2, requiredType=list)
        for opName in args[1]:
            if opName not in self.validOps.keys():
                raise DslSyntaxError("Op name '%s' not supported" % opName)

    @property
    def left(self):
        return self._args[0]

    @property
    def opNames(self):
        return self._args[1]

    @property
    def comparators(self):
        return self._args[2]

    def evaluate(self, **kwds):
        left = self.left.evaluate(**kwds)
        for i in range(len(self.opNames)):
            right = self.comparators[i].evaluate(**kwds)
            opName = self.opNames[i]
            op = self.validOps[opName]
            if not op(left, right):
                return False
            left = right
        return True


class Module(DslObject):
    """
    A DSL module has a body, which is a list of DSL objects either
    function defs or expressions.
    """

    def __str__(self):
        return "\n".join([str(statement) for statement in self.body])

    def validate(self, args):
        self.assertArgsLen(args, 1)
        self.assertArgsPosn(args, 0, [(FunctionDef, DslExpression, Date)])

    @property
    def body(self):
        return self._args[0]

    def compile(self, dslLocals=None, dslGlobals=None, dependencyGraphClass=None):
        # It's a module compilation, so create a new namespace "context".
        if dslLocals == None:
            dslLocals = {}
        dslLocals = DslNamespace(dslLocals)
        if dslGlobals == None:
            dslGlobals = {}
        dslGlobals = DslNamespace(dslGlobals)

        # Can't do much with an empty module.
        if len(self.body) == 0:
            raise DslSyntaxError('empty module', node=self.node)

        # Collect function defs and expressions.
        functionDefs = []
        expressions = []
        for dslObj in self.body:
            if isinstance(dslObj, FunctionDef):
                dslGlobals[dslObj.name] = dslObj
                # Share the module level namespace (any function body can call any other function).
                dslObj.enclosedNamespace = dslGlobals
                functionDefs.append(dslObj)
            elif isinstance(dslObj, DslExpression):
                expressions.append(dslObj)
            else:
                raise DslSyntaxError("'%s' not allowed in module" % type(dslObj), dslObj, node=dslObj.node)

        if len(expressions) == 1:
            # Return the expression, but reduce it with function defs if any are defined.
            dslExpr = expressions[0]
            assert isinstance(dslExpr, DslExpression)
            if len(functionDefs):
                # Compile the expression
                if dependencyGraphClass:

                    # Create a stack of discovered calls to function defs.
                    pendingCallStack = FunctionDefCallStack()

                    # Create a stack for the stubbed exprs.
                    stubbedExprs = StubbedExpressionStack()

                    # Start things off. If an expression has a FunctionCall, it will cause a pending
                    # call to be placed on the pending call stack, and the function call will be
                    # replaced with a stub, which acts as a placeholder for the result of the function
                    # call. By looping over the pending call stack until it is empty, evaluating
                    # pending calls to generate stubbed expressions and further pending calls, the
                    # module can be compiled into a stack of stubbed expressions.
                    # Of course if the module's expression doesn't have a function call, there
                    # will just be one expression on the stack of "stubbed" expressions, and it will
                    # not have any stubs.
                    stubbedExpr = dslExpr.reduce(
                        dslLocals,
                        DslNamespace(dslGlobals),
                        pendingCallStack=pendingCallStack
                    )

                    # Create the root stub ID, this will allow the final result to be retrieved.
                    from quantdsl.domain.services import createUuid
                    self.rootStubId = str(createUuid())

                    # Put the module expression (now stubbed) on the stack.
                    stubbedExprs.put(self.rootStubId, stubbedExpr, None)

                    # Continue by looping over any pending calls that have resulted from the module's expression.
                    while not pendingCallStack.empty():
                        # Get the stacked call info.
                        (stubId, stackedCall, stackedLocals, stackedGlobals, effectivePresentTime) = pendingCallStack.get()

                        # Check we've got a function def.
                        assert isinstance(stackedCall, FunctionDef), type(stackedCall)

                        # Apply the stacked call values to the called function def.
                        stubbedExpr = stackedCall.apply(stackedGlobals,
                                                        effectivePresentTime,
                                                        pendingCallStack=pendingCallStack,
                                                        isDestacking=True,
                                                        **stackedLocals)

                        # Put the resulting (potentially stubbed) expression on the stack of stubbed expressions.
                        stubbedExprs.put(stubId, stubbedExpr, effectivePresentTime)

                    # Create an expression stack DSL object from the stack of stubbed expressions.
                    stubbedExprsArray = []
                    while not stubbedExprs.empty():
                        stubbedExprsArray.append(stubbedExprs.get())
                    dslObj = dependencyGraphClass(self.rootStubId, stubbedExprsArray)
                else:
                    # Compile the module expression as and for a single threaded recursive operation (faster but not
                    # distributed, so also limited in space and perhaps time). For smaller computations only.
                    dslObj = dslExpr.reduce(dslLocals, DslNamespace(dslGlobals))
            else:
                # The module just has an expression. Can't break up a monolithic DSL expression in an expression stack
                # yet. So Compile the module expression as and for a single threaded recursive operation.
                dslObj = dslExpr.reduce(dslLocals, DslNamespace(dslGlobals))
            return dslObj
        elif len(expressions) > 1:
            # Can't meaningfully evaluate more than one expression (since assignments are not supported).
            raise DslSyntaxError('more than one expression in module', node=expressions[1].node)
        elif len(functionDefs) == 1:
            # It's just a module with one function, so return the function def.
            return functionDefs[0]
        elif len(functionDefs) > 1:
            # Can't meaningfully evaluate more than one expression (there are no assignments).
            secondDef = functionDefs[1]
            raise DslSyntaxError('more than one function def in module without an expression', '"def %s"' % secondDef.name, node=functionDefs[1].node)
        raise DslSyntaxError("shouldn't get here", node=self.node)


def nostub(*args):
    """
    Dummy 'nostub' Quant DSL decorator - we just want the name in the namespace.
    """
    import mock
    return mock.Mock


class DslNamespace(dict):

    def copy(self):
        copy = self.__class__(self)
        return copy


class StochasticObject(DslObject):
    pass

class DatedDslObject(DslObject):

    @property
    def date(self):
        if not hasattr(self, '_date'):
            date = self._args[0]
            if isinstance(date, Name):
                raise DslSyntaxError("date value name '%s' must be resolved to a datetime before it can be used" % date.name, node=self.node)
            if isinstance(date, datetime.datetime):
                pass
            if isinstance(date, basestring):
                date = String(date)
            if isinstance(date, String):
                date = Date(date, node=date.node)
            if isinstance(date, (Date, BinOp)):
                date = date.evaluate()
            if not isinstance(date, datetime.datetime):
                raise DslSyntaxError("date value should be a datetime.datetime by now, but it's a %s" % date, node=self.node)
            self._date = date
        return self._date


functionalDslClasses = {
    'Add': Add,
    'And': And,
    'Compare': Compare,
    'Date': Date,
    'Div': Div,
    'DslObject': DslObject,
    'FloorDiv': FloorDiv,
    'FunctionArg': FunctionArg,
    'FunctionCall': FunctionCall,
    'FunctionDef': FunctionDef,
    'If': If,
    'IfExp': IfExp,
    'Max': Max,
    'Mod': Mod,
    'Module': Module,
    'Mult': Mult,
    'Name': Name,
    'Number': Number,
    'Or': Or,
    'Pow': Pow,
    'String': String,
    'Stub': Stub,
    'Sub': Sub,
    'TimeDelta': TimeDelta,
    'UnarySub': UnarySub,
    'Underlying': Underlying,
}

class Market(StochasticObject, DslExpression):
    def validate(self, args):
        self.assertArgsLen(args, requiredLen=1)
        self.assertArgsPosn(args, posn=0, requiredType=(basestring, String, Name))

    @property
    def name(self):
        return self._args[0].evaluate() if isinstance(self._args[0], String) else self._args[0]

    def evaluate(self, **kwds):
        try:
            presentTime = kwds['presentTime']
        except KeyError:
            raise DslSyntaxError(
                "Can't evaluate Market '%s' without 'presentTime' in context variables" % self.name,
                ", ".join(kwds.keys()),
                node=self.node
            )
        try:
            allMarketPrices = kwds['allMarketPrices']
        except KeyError:
            raise DslError(
                "Can't evaluate Market '%s' without 'allMarketPrices' in context variables" % self.name,
                ", ".join(kwds.keys()),
                node=self.node
            )

        try:
            marketPrices = allMarketPrices[self.name]
        except KeyError:
            raise DslError(
                "Can't evaluate Market '%s' without market name in 'allMarketPrices'" % self.name,
                ", ".join(allMarketPrices.keys()),
                node=self.node
            )

        try:
            marketPrice = marketPrices[presentTime]
        except KeyError:
            raise DslError(
                "Can't evaluate Market '%s' without present time '%s in market prices" % (self.name, presentTime),
                ", ".join(marketPrices.keys()),
                node=self.node
            )

        return marketPrice


class Settlement(StochasticObject, DatedDslObject, DslExpression):
    """
    Discounts value of expression to 'presentTime'.
    """

    def validate(self, args):
        self.assertArgsLen(args, requiredLen=2)
        self.assertArgsPosn(args, posn=0, requiredType=(String, Date, Name,BinOp))
        self.assertArgsPosn(args, posn=1, requiredType=DslExpression)

    def evaluate(self, **kwds):
        value = self._args[1].evaluate(**kwds)
        return self.discount(value, self.date, **kwds)


class Fixing(StochasticObject, DatedDslObject, DslExpression):
    """
    A fixing defines the 'presentTime' used for evaluating its expression.
    """

    def __str__(self):
        return "%s('%04d-%02d-%02d', %s)" % (
            self.__class__.__name__,
            self.date.year,
            self.date.month,
            self.date.day,
            self.expr)

    def validate(self, args):
        self.assertArgsLen(args, requiredLen=2)
        self.assertArgsPosn(args, posn=0, requiredType=(basestring, String, Date, Name, BinOp))
        self.assertArgsPosn(args, posn=1, requiredType=DslExpression)

    @property
    def expr(self):
        return self._args[1]

    def reduce(self, dslLocals, dslGlobals, effectivePresentTime=None, pendingCallStack=None):
        # Figure out the effectivePresentTime from the fixing date, which might still be a Name.
        # Todo: It might also be a date expression, and so might the
        fixingDate = self._args[0]
        if isinstance(fixingDate, datetime.datetime):
            pass
        if isinstance(fixingDate, basestring):
            fixingDate = String(fixingDate)
        if isinstance(fixingDate, String):
            fixingDate = Date(fixingDate, node=fixingDate.node)
        if isinstance(fixingDate, (Date, BinOp, Name)):
            fixingDate = fixingDate.evaluate(**dslLocals)
        if not isinstance(fixingDate, datetime.datetime):
            raise DslSyntaxError("fixing date value should be a datetime.datetime by now, but it's a %s" % fixingDate, node=self.node)
        effectivePresentTime = fixingDate
        return super(Fixing, self).reduce(dslLocals, dslGlobals, effectivePresentTime, pendingCallStack=pendingCallStack)

    def evaluate(self, **kwds):
        kwds = kwds.copy()
        kwds['presentTime'] = self.date
        return self.expr.evaluate(**kwds)


class On(Fixing):
    """
    A shorter name for Fixing.
    """


class Wait(Fixing):
    """
    A fixing with discounting of the resulting value from date arg to presentTime.
    """
    def evaluate(self, **kwds):
        value = super(Wait, self).evaluate(**kwds)
        return self.discount(value, self.date, **kwds)


class Choice(StochasticObject, DslExpression):
    """
    Encapsulates the Longstaff-Schwartz routine as an element of the language.
    """
    def validate(self, args):
        self.assertArgsLen(args, minLen=2)
        for i in range(len(args)):
            self.assertArgsPosn(args, posn=i, requiredType=DslExpression)

    def evaluate(self, **kwds):
        # Check the results cache, to see whether this function
        # has already been evaluated with these args.
        if not hasattr(self, 'resultsCache'):
            self.resultsCache = {}
        cacheKeyKwdItems = [(k, hash(tuple(sorted(v))) if isinstance(v, dict) else v) for (k, v) in kwds.items()]
        kwdsHash = hash(tuple(sorted(cacheKeyKwdItems)))
        if kwdsHash not in self.resultsCache:
            # Run the least-squares monte-carlo routine.
            presentTime = kwds['presentTime']
            initialState = LongstaffSchwartzState(self, presentTime)
            finalStates = [LongstaffSchwartzState(a, presentTime) for a in self._args]
            longstaffSchwartz = LongstaffSchwartz(initialState, finalStates)
            result = longstaffSchwartz.evaluate(**kwds)
            self.resultsCache[kwdsHash] = result
        return self.resultsCache[kwdsHash]


class LongstaffSchwartz(object):
    """
    Implements a least-squares Monte Carlo simulation, following the Longstaff-Schwartz paper
    on valuing American options (for reference, see Quant DSL paper).
    """
    def __init__(self, initialState, subsequentStates):
        self.initialState = initialState
        for subsequentState in subsequentStates:
            self.initialState.addSubsequentState(subsequentState)
        self.states = None
        self.statesByTime = None

    def evaluate(self, **kwds):
        try:
            allMarketPrices = kwds['allMarketPrices']
        except KeyError:
            raise DslSystemError("'allMarketPrices' not in evaluation kwds", kwds.keys(), node=None)
        if len(allMarketPrices) == 0:
            raise DslSystemError('no rvs', str(kwds))
        firstMarketPrices = allMarketPrices.values()[0]
        allStates = self.getStates()
        allStates.reverse()
        valueOfBeingIn = {}
        import numpy
        import scipy
        for state in allStates:
            lenSubsequentStates = len(state.subsequentStates)
            stateValue = None
            if lenSubsequentStates > 1:
                conditionalExpectedValues = []
                expectedContinuationValues = []
                underlyingValue = firstMarketPrices[state.time]
                for subsequentState in state.subsequentStates:
                    regressionVariables = []
                    dslMarkets = subsequentState.dslObject.findInstances(Market)
                    marketNames = set([m.name for m in dslMarkets])
                    for marketName in marketNames:
                        marketPrices = allMarketPrices[marketName]
                        try:
                            marketPrice = marketPrices[state.time]
                        except KeyError, inst:
                            msg = "Couldn't find time '%s' in random variables. Times are: %s" % (
                                state.time, marketPrices.keys())
                            raise Exception(msg)

                        regressionVariables.append(marketPrice)
                    # Todo: Either use or remove 'getPayoff()', payoffValue not used ATM.
                    #payoffValue = self.getPayoff(state, subsequentState)
                    expectedContinuationValue = valueOfBeingIn[subsequentState]
                    expectedContinuationValues.append(expectedContinuationValue)
                    if len(regressionVariables):
                        conditionalExpectedValue = LeastSquares(regressionVariables, expectedContinuationValue).fit()
                    else:
                        conditionalExpectedValue = expectedContinuationValue
                    conditionalExpectedValues.append(conditionalExpectedValue)
                conditionalExpectedValues = numpy.array(conditionalExpectedValues)
                expectedContinuationValues = numpy.array(expectedContinuationValues)
                argmax = conditionalExpectedValues.argmax(axis=0)
                offsets = numpy.array(range(0, conditionalExpectedValues.shape[1])) * conditionalExpectedValues.shape[0]
                indices = argmax + offsets
                assert indices.shape == underlyingValue.shape
                stateValue = expectedContinuationValues.transpose().take(indices)
                assert stateValue.shape == underlyingValue.shape
            elif lenSubsequentStates == 1:
                subsequentState = state.subsequentStates.pop()
                stateValue = valueOfBeingIn[subsequentState]
            elif lenSubsequentStates == 0:
                stateValue = state.dslObject.evaluate(**kwds)
                if isinstance(stateValue, (int, float)):
                    try:
                        underlyingValue = firstMarketPrices[state.time]
                    except KeyError, inst:
                        msg = "Couldn't find time '%s' in random variables. Times are: %s" % (
                            state.time, sorted(firstMarketPrices.keys()))
                        raise Exception(msg)
                    pathCount = len(underlyingValue)
                    if stateValue == 0:
                        stateValue = scipy.zeros(pathCount)
                    else:
                        ones = scipy.ones(pathCount)
                        stateValue = ones * stateValue
                if not isinstance(stateValue, numpy.ndarray):
                    raise Exception("State value type is '%s' when numpy.ndarray is required: %s" % (
                        type(stateValue), stateValue))
            valueOfBeingIn[state] = stateValue
        return valueOfBeingIn[self.initialState]

    def getTimes(self):
        return self.getStatesByTime().keys()

    def getStatesAt(self, time):
        return self.getStatesByTime()[time]

    def getStatesByTime(self):
        if self.statesByTime is None:
            self.statesByTime = {}
            for state in self.getStates():
                if state.time not in self.statesByTime:
                    self.statesByTime[state.time] = []
                self.statesByTime[state.time].append(state)
        return self.statesByTime

    def getStates(self):
        if self.states is None:
            self.states = self.findStates(self.initialState)
        return self.states

    def findStates(self, state):
        states = [state]
        for subsequentState in state.subsequentStates:
            states += self.findStates(subsequentState)
        return states

    def getPayoff(self, state, nextState):
        return 0


class LongstaffSchwartzState(object):
    """
    Object to represent state in the Longstaff-Schwartz routine.
    """

    def __init__(self, dslObject, time):
        self.subsequentStates = set()
        self.dslObject = dslObject
        self.time = time

    def addSubsequentState(self, state):
        self.subsequentStates.add(state)


class LeastSquares(object):
    """
    Implements the least-squares routine.
    """

    def __init__(self, xs, y):
        self.pathCount = len(y)
        for x in xs:
            if len(x) != self.pathCount:
                raise Exception, "Regression won't work with uneven path counts."
        self.xs = xs
        self.y = y

    def fit(self):
        import scipy
        regressions = list()
        # Regress against unity.
        regressions.append(scipy.ones(self.pathCount))
        # Regress against each variable.
        for x in self.xs:
            regressions.append(x)
        # Regress against squares and cross products.
        indices = range(0, len(self.xs))
        combinations = list()
        for i in indices:
            for j in indices:
                combination = [i, j]
                combination.sort()
                if combination not in combinations:
                    combinations.append(combination)
        for combination in combinations:
            product = self.xs[combination[0]] * self.xs[combination[1]]
            regressions.append(product)
        # Run the regression.
        a = scipy.matrix(regressions).transpose()
        b = scipy.matrix(self.y).transpose()
        if a.shape[0] != b.shape[0]:
            raise Exception, "Regression won't work with uneven path counts."
        c = self.solve(a, b)
        c = scipy.matrix(c)
        #print "a: ", a
        #print "a: ", a.shape, type(a)
        #print "b: ", b
        #print "b: ", b.shape, type(b)
        #print "c: ", c.shape, type(c)
        #print "c: ", c
        if a.shape[1] != c.shape[0]:
            raise Exception, "Matrices are not aligned: %s and %s" % (a.shape, c.shape)
        #else:
        #    raise Exception, "Matrices are aligned: %s and %s" % (a.shape, c.shape)
        d = a * c
        #print "d: ", d
        #print "d: ", d.shape, type(d)
        #print "d A1: ", d.getA1()
        return d.getA1()

    def solve(self, a, b):
        import scipy.linalg
        try:
            c,resid,rank,sigma = scipy.linalg.lstsq(a, b)
        except Exception, inst:
            msg = "Couldn't solve a and b: %s %s: %s" % (a, b, inst)
            raise Exception, msg
        return c

defaultDslClasses = functionalDslClasses.copy()
defaultDslClasses.update({
    'Choice': Choice,
    'Fixing': Fixing,
    'Market': Market,
    'On': On,
    'Settlement': Settlement,
    'Wait': Wait,
})


class FunctionDefCallStack(queue.Queue):

    def put(self, stubId, stackedCall, stackedLocals, stackedGlobals, effectivePresentTime):
        assert isinstance(stubId, basestring), type(stubId)
        assert isinstance(stackedCall, FunctionDef), type(stackedCall)
        assert isinstance(stackedLocals, DslNamespace), type(stackedLocals)
        assert isinstance(stackedGlobals, DslNamespace), type(stackedGlobals)
        assert isinstance(effectivePresentTime, (datetime.datetime, type(None))), type(effectivePresentTime)
        queue.Queue.put(self, (stubId, stackedCall, stackedLocals, stackedGlobals, effectivePresentTime))


class StubbedExpressionStack(queue.LifoQueue):

    def put(self, stubId, stubbedExpr, effectivePresentTime):
        assert isinstance(stubId, basestring), type(stubId)
        assert isinstance(stubbedExpr, DslExpression), type(stubbedExpr)
        assert isinstance(effectivePresentTime, (datetime.datetime, type(None))), type(effectivePresentTime)
        queue.LifoQueue.put(self, (stubId, stubbedExpr, effectivePresentTime))