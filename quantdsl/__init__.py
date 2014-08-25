from __future__ import division
from abc import ABCMeta, abstractmethod
import ast
import datetime
import dateutil.parser
import math
import uuid
from itertools import chain
import multiprocessing
import Queue as queue
import re
import itertools
import scipy
import time
import sys
import threading
try:
    import pytz
except ImportError:
    pytz = None

__version__ = '0.0.3'

# Todo: Build out the persistence support, so it can run with various backends (RDBMS, Redis, Celery, etc.).
# Todo: Move calculation of value based on brownian motion from Market object, so that its evaluation method simply looks up value of random variable for that market at that present time.
# Todo: Develop PriceSimulation object from just a simple "local" in memory, to be an network service client object.
# Todo: Check whether it's better to regress on uncorrelated brownian motions, rather than correlated ones.
# Todo: Develop multi-factor model (e.g. Schwartz-Smith)?
# Todo: Develop closures, function defs within function defs may help to reduce call argument complexity.
# Todo: Make stats available on number of call requirements, number of leaves in dependency graph. And depth of graph? Predicting cost of each node would be possible (but complicated for LongstaffSchwartz cos the LeastSqaures routine is run different numbers of times (spot the formula).
# Todo: Optimization for parallel execution, so if there are four cores, then it might make sense only to stub four large branches?
# Todo: Optimize network traffic by creating a single message containing all data required to evaluate a stubbed expression.
# Todo: Move to an event sourced model for dependency graph changes (CreatedResult, CreatedCallDependency, CreatedCallRequirement, etc.) and use an RDBMS (even sqlite in memory) to manage a single table with everything in.
# Todo: Clean up the str, repr, pprint stuff?
# Todo: Make this works with Python 3.
# Todo: Use function arg annotation to declare types of DSL function args (will only work with Python 3).
# Todo: Support list comprehensions, for things like a strip of options?
# Todo: Support plotting.

# Todo: Raise Quant DSL-specific type mismatch errors at run time (ie e.g. handle situation where datetime and string can't be added).
# Todo: Anyway, identify when type mismatches will occur - can't multiply a date by a number, can't add a date to a date or to a number, can't add a number to a timedelta. Etc?
# Todo: (Long one) Go through all ways of writing broken DSL source code, and make sure there are sensible errors.
# Todo: Figure out how to identify and catch when a loop will be started, perhaps by limiting the number of FunctionDef.apply() calls in one DslParser.parse() to a configurable limit? Need to catch e.g. def f(n): return f(n+1).
# Todo: Figure out behaviour for observationTime > any fixing date, currently leads to a complex numbers (square root of negative time delta).

# Todo: Review the test coverage of the code.
# Todo: Review the separation of concerns between the various test cases.
# Todo: Move these todos to an issue tracker.

def parse(dslSource, filename='<unknown>', dslClasses=None):
    """
    Returns a DSL module, created according to the given DSL source module.
    """
    return DslParser().parse(dslSource, filename=filename, dslClasses=dslClasses)


def compile(dslSource, filename='<unknown>', isParallel=None, dslClasses=None, compileKwds=None, **extraCompileKwds):
    """
    Returns a DSL expression, created according to the given DSL source module.

    That is, if the source module contains a function def and an expression which
    calls that function, then the expression's function call will be evaluated
    and the resulting DSL expression will be substituted for the function call
    in the module's expression, so that calls to user defined functions are eliminated
    and a single DSL expression is obtained.

    If the source module contains a function def, but no expression, the module is compiled
    into a function def object. Calling .apply() on a function def object will return a DSL
    expression object, which can be evaluated by calling its .evaluate() method.
    """
    if compileKwds is None:
        compileKwds = DslNamespace()
    assert isinstance(compileKwds, dict)
    compileKwds.update(extraCompileKwds)

    # Parse the source into a DSL module object.
    dslModule = parse(dslSource, filename=filename, dslClasses=dslClasses)

    # Compile the module into a single (primitive) expression.
    return dslModule.compile(DslNamespace(), compileKwds, isParallel=isParallel)


def eval(dslSource, filename='<unknown>', isParallel=None, dslClasses=None, compileKwds=None, evaluationKwds=None,
         priceSimulatorClass=None, isVerbose=False, **extraEvaluationKwds):
    """
    Returns the result of evaluating a compiled module (an expression, or a user defined function).

    An expression (with optional function defs) will evaluate to a simple value.

    A function def will evaluate to a DSL expression, will may then be evaluated (more than one
    function def without an expression is an error).
    """
    if evaluationKwds is None:
        evaluationKwds = DslNamespace()
    assert isinstance(evaluationKwds, dict)
    evaluationKwds.update(extraEvaluationKwds)

    if isVerbose:
        print "Compiling DSL source:"
        print
        print '"""'
        print dslSource.strip()
        print '"""'
        print

    # Compile the source into a primitive DSL expression.
    startCompile = datetime.datetime.now()

    dslExpr = compile(dslSource, filename=filename, isParallel=isParallel, dslClasses=dslClasses, compileKwds=compileKwds)

    durationCompile = datetime.datetime.now() - startCompile

    if isVerbose:
        print "Duration of compilation: %s" % durationCompile
        print

    assert isinstance(dslExpr, (DslExpression, ExpressionStack)), type(dslExpr)

    if isVerbose:
        if isinstance(dslExpr, ExpressionStack):
            lenDslExpr = len(dslExpr)

            print "Compiled DSL source into %d partial expressions (root ID: %s)." % (lenDslExpr, dslExpr.rootStubId)
            print
            #for stubId, stubbedExpr, _ in dslExpr.stubbedExprs:
            #    print stubId, stubbedExpr

    # Fix up the evaluationKwds with Brownian motions, if necessary.
    if dslExpr.hasInstances(dslType=Market):

        # In this case, evaluationKwds must have 'observationTime' and 'pathCount'.
        observationTime = evaluationKwds['observationTime']
        assert isinstance(observationTime, datetime.datetime)

        # Initialise presentTime as observationTime.
        evaluationKwds['presentTime'] = observationTime

        # Check pathCount.
        pathCount = evaluationKwds['pathCount']
        assert isinstance(pathCount, int)

        if isVerbose:
            print "Path count: %d" % pathCount
            print
        # Check calibration for market dynamics.
        marketCalibration = evaluationKwds['marketCalibration']
        assert isinstance(marketCalibration, dict)

        # Construct the Brownian motions.
        if not 'brownianMotions' in evaluationKwds:

            if isVerbose:
                print "Finding all market names and fixing dates..."
                print
            # Find all unique market names.
            marketNames = set()
            for dslMarket in dslExpr.findInstances(dslType=Market):
                assert isinstance(dslMarket, Market)
                marketNames.add(dslMarket.name)

            # Find all unique fixing dates.
            fixingDates = set()
            for dslFixing in dslExpr.findInstances(dslType=Fixing):
                assert isinstance(dslFixing, Fixing)
                fixingDates.add(dslFixing.date)
            fixingDates = sorted(list(fixingDates))

            if isVerbose:
                print "Computing Brownian motions for market names ('%s') from observation time '%s' through fixing dates: '%s'." % (
                    "', '".join(marketNames),
                    "%04d-%02d-%02d" % (observationTime.year, observationTime.month, observationTime.day),
                    # Todo: Only print first and last few, if there are loads.
                    "', '".join(["%04d-%02d-%02d" % (d.year, d.month, d.day) for d in fixingDates]),
                )
                print

            if not priceSimulatorClass:
                priceSimulatorClass = PriceSimulator
            priceSimulator = priceSimulatorClass()
            brownianMotions = priceSimulator.getBrownianMotions(marketNames, fixingDates, observationTime, pathCount, marketCalibration)
            evaluationKwds['brownianMotions'] = brownianMotions


    if isinstance(dslExpr, ExpressionStack):
        if isVerbose:
            print "Expression stack:"
            for stubbedExprData in dslExpr.stubbedExprs:
                print "  " + str(stubbedExprData[0]) + ": " + str(stubbedExprData[1])
            print

            print "Evaluating %d partial expressions, please wait..." % lenDslExpr
            print
        # Start progress display thread.
        def showProgress():
            progress = 0
            while progress < 100:
                time.sleep(0.2)
                # Avoid race condition.
                if hasattr(dslExpr, 'runner') and hasattr(dslExpr.runner, 'resultsDict'):
                    progress = 100.0 * len(dslExpr.runner.resultsDict) / lenDslExpr
                    sys.stdout.write("\rProgress: %01.2f%%" % progress)
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
            sys.stdout.write("\r")
            sys.stdout.flush()

        thread = threading.Thread(target=showProgress)
        thread.start()

    # Evaluate the primitive DSL expression.
    startEval = datetime.datetime.now()
    value = dslExpr.evaluate(**evaluationKwds)
    durationEval = datetime.datetime.now() - startEval

    if isinstance(dslExpr, ExpressionStack):
        # Join with progress display thread.
        thread.join(timeout=3)

    if isVerbose:
        print "Duration of evaluation: %s" % durationEval
        print

    # Prepare the result.
    if isinstance(value, scipy.ndarray):
        mean = value.mean()
        stderr = value.std() / math.sqrt(pathCount)
    else:
        mean = value
        stderr = 0
    # Todo: Make this a proper object class.
    dslResult = {
        'mean': mean,
        'stderr': stderr
    }

    return dslResult


class DslParser(object):

    def parse(self, dslSource, filename='<unknown>', dslClasses=None):
        """
        Creates a DSL Module object from a DSL source text.
        """
        self.dslClasses = defaultDslClasses.copy()
        if dslClasses:
            assert isinstance(dslClasses, dict)
            self.dslClasses.update(dslClasses)

        if not isinstance(dslSource, basestring):
            raise QuantDslError("Can't parse non-string object", dslSource)

        assert isinstance(dslSource, basestring)
        try:
            # Parse as Python source code, into a Python abstract syntax tree.
            astModule = ast.parse(dslSource, filename=filename, mode='exec')
        except SyntaxError, e:
            raise QuantDslSyntaxError("DSL source code is not valid Python code", e)

        # Generate Quant DSL from Python AST.
        return self.visitAstNode(astModule)

    def visitAstNode(self, node):
        """
        Identifies which "visit" method to call, according to type of node being visited.

        Returns the result of calling the identified "visit" method.
        """
        assert isinstance(node, ast.AST)

        # Construct the "visit" method name.
        dslElementName = node.__class__.__name__
        methodName = 'visit' + dslElementName

        # Try to get the "visit" method object.
        try:
            method = getattr(self, methodName)
        except AttributeError:
            msg = "element '%s' is not supported (visit method '%s' not found on parser): %s" % (
                dslElementName, methodName, node)
            raise QuantDslSyntaxError(msg)

        # Call the "visit" method object, and return the result of visiting the node.
        return method(node=node)

    def visitReturn(self, node):
        """
        Visitor method for ast.Return nodes.

        Returns the result of visiting the expression held by the return statement.
        """
        assert isinstance(node, ast.Return)
        return self.visitAstNode(node.value)

    def visitModule(self, node):
        """
        Visitor method for ast.Module nodes.

        Returns a DSL Module, with a list of DSL expressions as the body.
        """
        assert isinstance(node, ast.Module)
        body = [self.visitAstNode(n) for n in node.body]
        return Module(body, node=node)

    def visitExpr(self, node):
        """
        Visitor method for ast.Expr nodes.

        Returns the result of visiting the contents of the expression node.
        """
        assert isinstance(node, ast.Expr)
        if isinstance(node.value, ast.AST):
            return self.visitAstNode(node.value)
        else:
            raise QuantDslSyntaxError

    def visitNum(self, node):
        """
        Visitor method for ast.Name.

        Returns a DSL Number object, with the number value.
        """
        assert isinstance(node, ast.Num)
        return Number(node.n, node=node)

    def visitStr(self, node):
        """
        Visitor method for ast.Str.

        Returns a DSL String object, with the string value.
        """
        assert isinstance(node, ast.Str)
        return String(node.s, node=node)

    def visitUnaryOp(self, node):
        """
        Visitor method for ast.UnaryOp.

        Returns a specific DSL UnaryOp object (e.g UnarySub), along with the operand.
        """
        assert isinstance(node, ast.UnaryOp)
        args = [self.visitAstNode(node.operand)]
        if isinstance(node.op, ast.USub):
            dslUnaryOpClass = UnarySub
        else:
            raise QuantDslSyntaxError("Unsupported unary operator token: %s" % node.op)
        return dslUnaryOpClass(node=node, *args)

    def visitBinOp(self, node):
        """
        Visitor method for ast.BinOp.

        Returns a specific DSL BinOp object (e.g Add), along with the left and right operands.
        """
        assert isinstance(node, ast.BinOp)
        typeMap = {
            ast.Add: Add,
            ast.Sub: Sub,
            ast.Mult: Mult,
            ast.Div: Div,
            ast.Pow: Pow,
            ast.Mod: Mod,
            ast.FloorDiv: FloorDiv,
        }
        try:
            dslClass = typeMap[type(node.op)]
        except KeyError:
            raise QuantDslSyntaxError("Unsupported binary operator token", node.op, node=node)
        args = [self.visitAstNode(node.left), self.visitAstNode(node.right)]
        return dslClass(node=node, *args)

    def visitBoolOp(self, node):
        """
        Visitor method for ast.BoolOp.

        Returns a specific DSL BoolOp object (e.g And), along with the left and right operands.
        """
        assert isinstance(node, ast.BoolOp)
        typeMap = {
            ast.And: And,
            ast.Or: Or,
        }
        try:
            dslClass = typeMap[type(node.op)]
        except KeyError:
            raise QuantDslSyntaxError("Unsupported boolean operator token: %s" % node.op)
        else:
            values = [self.visitAstNode(v) for v in node.values]
            args = [values]
            return dslClass(node=node, *args)

    def visitName(self, node):
        """
        Visitor method for ast.Name.

        Returns a DSL Name object, along with the name's string.
        """
        return Name(node.id, node=node)

    def visitCall(self, node):
        """
        Visitor method for ast.Call.

        Returns a built-in DSL expression, or a DSL FunctionCall if the name refers to a user
        defined function.
        """
        if node.keywords:
            raise QuantDslSyntaxError("Calling with keywords is not currently supported (positional args only).")
        if node.starargs:
            raise QuantDslSyntaxError("Calling with starargs is not currently supported (positional args only).")
        if node.kwargs:
            raise QuantDslSyntaxError("Calling with kwargs is not currently supported (positional args only).")

        # Collect the call arg expressions (whose values will be passed into the call when it is made).
        callArgExprs = [self.visitAstNode(arg) for arg in node.args]

        # Check the called node is an ast.Name.
        calledNode = node.func
        assert isinstance(calledNode, ast.Name)
        calledNodeName = calledNode.id

        # Construct a DSL object for this call.
        try:
            # Resolve the name with a new instance of a DSL class.
            dslClass = self.dslClasses[calledNodeName]
        except KeyError:
            # Resolve as a FunctionCall, and expect
            # to resolve the name to a function def later.
            dslArgs = [Name(calledNodeName, node=calledNode), callArgExprs]
            return FunctionCall(node=node, *dslArgs)
        else:
            assert issubclass(dslClass, DslObject)
            return dslClass(node=node, *callArgExprs)

    def visitFunctionDef(self, node):
        """
        Visitor method for ast.FunctionDef.

        Returns a named DSL FunctionDef, with a definition of the expected call argument values.
        """
        name = node.name
        callArgDefs = [FunctionArg(arg.id, '') for arg in node.args.args]
        assert len(node.body) == 1, "Function defs with more than one body statement are not supported at the moment."
        decoratorNames = [astName.id for astName in node.decorator_list]
        body = self.visitAstNode(node.body[0])
        dslArgs = [name, callArgDefs, body, decoratorNames]
        functionDef = FunctionDef(node=node, *dslArgs)
        return functionDef

    def visitIfExp(self, node):
        """
        Visitor method for ast.IfExp.

        Returns a named DSL IfExp, with a test DSL expression and expressions whose usage is
        conditional upon the test.
        """
        test = self.visitAstNode(node.test)
        body = self.visitAstNode(node.body)
        orelse = self.visitAstNode(node.orelse)
        args = [test, body, orelse]
        return IfExp(node=node, *args)

    def visitIf(self, node):
        """
        Visitor method for ast.If.

        Returns a named DSL If object, with a test DSL expression and expressions whose usage is
        conditional upon the test.
        """
        test = self.visitAstNode(node.test)
        assert len(node.body) == 1, "If statements with more than one body statement are not supported at the moment."
        body = self.visitAstNode(node.body[0])
        assert len(
            node.orelse) == 1, "If statements with more than one orelse statement are not supported at the moment."
        orelse = self.visitAstNode(node.orelse[0])
        args = [test, body, orelse]
        return If(node=node, *args)

    def visitCompare(self, node):
        """
        Visitor method for ast.Compare.

        Returns a named DSL Compare object, with operators (ops) and operands (comparators).
        """

        left = self.visitAstNode(node.left)
        opNames = [o.__class__.__name__ for o in node.ops]
        comparators = [self.visitAstNode(c) for c in node.comparators]
        args = [left, opNames, comparators]
        return Compare(node=node, *args)


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
                raise QuantDslSystemError(e)
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
            raise QuantDslSyntaxError(error, descr, self.node)
        if requiredLen != None and len(args) != requiredLen:
            error = "%s is broken" % self.__class__.__name__
            descr = "requires %s arguments (%s were given)" % (requiredLen, len(args))
            raise QuantDslSyntaxError(error, descr, self.node)

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
            raise QuantDslSyntaxError(error, desc, self.node)

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


class UTC(datetime.tzinfo):
    """
    UTC implementation taken from Python's docs.

    Used only when pytz isn't available.
    """
    ZERO = datetime.timedelta(0)

    def __repr__(self):
        return "<UTC>"

    def utcoffset(self, dt):
        return self.ZERO

    def tzname(self, dt):
        return "UTC"

    def dst(self, dt):
        return self.ZERO


utc = pytz.utc if pytz else UTC()


class Date(DslConstant):

    requiredType = (String, datetime.datetime)

    def __str__(self, indent=0):
        return "Date('%04d-%02d-%02d')" % (self.value.year, self.value.month, self.value.day)

    def parse(self, value):
        # Return a datetime.datetime.
        if isinstance(value, String):
            dateStr = value.evaluate()
            try:
                return dateutil.parser.parse(dateStr).replace(tzinfo=utc)
            except ValueError, inst:
                raise QuantDslSyntaxError("invalid date string", dateStr, node=self.node)
        elif isinstance(value, datetime.datetime):
            return value
        else:
            raise QuantDslSystemError("shouldn't get here", value, node=self.node)


class TimeDelta(DslConstant):
    requiredType = (String, datetime.timedelta)

    def __str__(self, indent=0):
        return "%s('%dd')" % (self.__class__.__name__, self.value.days)

    def parse(self, value, regex=re.compile(r'((?P<days>\d+?)d)?')):
        if isinstance(value, String):
            durationStr = value.evaluate()
            parts = regex.match(durationStr)
            if not parts:
                raise QuantDslSyntaxError('invalid time delta string', durationStr, node=self.node)
            parts = parts.groupdict()
            time_params = {}
            for (name, param) in parts.iteritems():
                if param:
                    time_params[name] = int(param)
            return datetime.timedelta(**time_params)
        elif isinstance(value, datetime.timedelta):
            return value
        else:
            raise QuantDslSystemError("shouldn't get here", value, node=self.node)


class UnaryOp(DslExpression):

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
    def op(self, value):
        return -value

    opchar = '-'



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
                raise QuantDslSyntaxError("not a simple value", str(value), node=self.node)
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
            raise QuantDslSyntaxError("unable to %s" % self.__class__.__name__.lower(), "%s %s: %s" % (left, right, e),
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
                raise QuantDslSystemError('Vectors have different length: ', descr, self.node)
        elif aIsaNumber and (not bIsaNumber):
            # Todo: Optimise with scipy.zeros() when a equals zero?
            a = numpy.array([a] * len(b))
        elif bIsaNumber and (not aIsaNumber):
            # Todo: Optimise with scipy.zeros() when b equals zero?
            b = numpy.array([b] * len(a))
        c = numpy.array([a, b])
        return c.max(axis=0)


class Market(DslExpression):
    def validate(self, args):
        self.assertArgsLen(args, requiredLen=1)
        self.assertArgsPosn(args, posn=0, requiredType=(String, Name))

    @property
    def name(self):
        return self._args[0].evaluate()

    def evaluate(self, **kwds):
        # Todo: Improve on having various ways of checking variables are defined.
        try:
            observationTime = kwds['observationTime']
        except KeyError:
            raise QuantDslSyntaxError(
                "Can't evaluate Market '%s' without 'observationTime' in context variables." % self.name,
                ", ".join(kwds.keys()),
                node=self.node
            )
        try:
            presentTime = kwds['presentTime']
        except KeyError:
            raise QuantDslSyntaxError(
                "Can't evaluate Market '%s' without 'presentTime' in context variables" % self.name,
                ", ".join(kwds.keys()),
                node=self.node
            )
        try:
            calibration = kwds['marketCalibration']
        except KeyError:
            raise QuantDslError(
                "Can't evaluate Market '%s' without 'calibration' in context variables" % self.name,
                ", ".join(kwds.keys()),
                node=self.node
            )

        # Todo: Move this price construction into the PriceSimulator, and make Market values available by (name, date).

        LAST_PRICE = '%s-LAST-PRICE' % self.name
        try:
            lastPrice = calibration[LAST_PRICE]
        except KeyError:
            raise QuantDslError(
                "Can't evaluate Market '%s' without calibration setting '%s" % (self.name, LAST_PRICE),
                ", ".join(calibration.keys()),
                node=self.node
            )
        ACTUAL_HISTORICAL_VOLATILITY = '%s-ACTUAL-HISTORICAL-VOLATILITY' % self.name
        try:
            actualHistoricalVolatility = calibration[ACTUAL_HISTORICAL_VOLATILITY]
        except KeyError:
            raise QuantDslError(
                "Can't evaluate Market '%s' without calibration setting '%s" % (self.name, actualHistoricalVolatility),
                ", ".join(calibration.keys()),
                node=self.node
            )

        if presentTime == observationTime:
            value = lastPrice
        else:
            try:
                brownianMotions = kwds['brownianMotions']
            except KeyError:
                raise QuantDslSyntaxError(
                    "Can't evaluate Market '%s' without 'brownianMotions' in context variables" % self.name,
                    ", ".join(kwds.keys()),
                    node=self.node
                )

            try:
                marketRvs = brownianMotions[self.name]
            except KeyError:
                raise QuantDslSyntaxError(
                    "Market '%s' not available in rvs" % self.name,
                    ", ".join(brownianMotions.keys()),
                    node=self.node
                )

            try:
                marketRv = marketRvs[presentTime]
            except KeyError:
                raise Exception, "Present time %s not in market rvs: %s" % (presentTime, marketRvs.keys())

            sigma = actualHistoricalVolatility / 100
            import scipy
            T = getDurationYears(observationTime, presentTime)
            value = lastPrice * scipy.exp(sigma * marketRv - 0.5 * sigma * sigma * T)
        return value

    # def getLastPrice(self, image):
    #     return self.getMetricValue(image, 'last-price')
    #
    # def getSigma(self, image):
    #     volatility = self.getMetricValue(image, 'actual-historical-volatility')
    #     return float(volatility) / 100
    #
    # def getMetricValue(self, image, metricName):
    #     return image.getMetricValue(metricName, self.getDomainObject(image))
    #
    # def getDomainObject(self, image):
    #     if not hasattr(self, 'domainObject'):
    #         self.domainObject = None
    #         marketRef = self.name
    #         if marketRef:
    #             if marketRef.startswith('#'):
    #                 marketId = marketRef[1:]
    #                 self.domainObject = image.registry.markets.findSingleDomainObject(id=marketId)
    #         if not self.domainObject:
    #             raise Exception, "Market '%s' could not be found." % marketRef
    #     return self.domainObject


class DatedDslObject(DslObject):

    @property
    def date(self):
        if not hasattr(self, '_date'):
            date = self._args[0]
            if isinstance(date, Name):
                raise QuantDslSyntaxError("date value name '%s' must be resolved to a datetime before it can be used" % date.name, node=self.node)
            if isinstance(date, datetime.datetime):
                pass
            if isinstance(date, basestring):
                date = String(date)
            if isinstance(date, String):
                date = Date(date, node=date.node)
            if isinstance(date, (Date, BinOp)):
                date = date.evaluate()
            if not isinstance(date, datetime.datetime):
                raise QuantDslSyntaxError("date value should be a datetime.datetime by now, but it's a %s" % date, node=self.node)
            self._date = date
        return self._date


class Settlement(DatedDslObject, DslExpression):
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


class Fixing(DatedDslObject, DslExpression):
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
        self.assertArgsPosn(args, posn=0, requiredType=(String, Date, Name, BinOp))
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
            raise QuantDslSyntaxError("fixing date value should be a datetime.datetime by now, but it's a %s" % fixingDate, node=self.node)
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


class Choice(DslExpression):
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
        #     # Erm, this hash is a bit crappy
        #     # Todo: Use a DslNamepace object instead, and make it capable of generating the hash.
        #     id(kwds['brownianMotions']),
        #     str(kwds['presentTime']),
        #     kwds['interestRate'],
        # ))
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
            brownianMotions = kwds['brownianMotions']
        except KeyError:
            raise QuantDslSystemError("'brownianMotions' not in evaluation kwds", kwds.keys(), node=None)
        if len(brownianMotions) == 0:
            raise QuantDslSystemError('no rvs', str(kwds))
        firstMarketRvs = brownianMotions.values()[0]
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
                underlyingValue = firstMarketRvs[state.time]
                for subsequentState in state.subsequentStates:
                    regressionVariables = []
                    dslMarkets = subsequentState.dslObject.findInstances(Market)
                    marketNames = set([m.name for m in dslMarkets])
                    for marketName in marketNames:
                        marketRvs = brownianMotions[marketName]
                        try:
                            marketRv = marketRvs[state.time]
                        except KeyError, inst:
                            msg = "Couldn't find time '%s' in random variables. Times are: %s" % (
                                state.time, marketRvs.keys())
                            raise Exception(msg)

                        regressionVariables.append(marketRv)
                    payoffValue = self.getPayoff(state, subsequentState)
                    # Todo: Either use or remove 'getPayoff()', payoffValue not used ATM.
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
                        underlyingValue = firstMarketRvs[state.time]
                    except KeyError, inst:
                        msg = "Couldn't find time '%s' in random variables. Times are: %s" % (
                            state.time, sorted(firstMarketRvs.keys()))
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
            raise QuantDslSyntaxError(
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
        raise QuantDslSyntaxError('call to undefined name', self.functionDefName.name, node=self.node)


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
                raise QuantDslSystemError("DSL object can't handle indent: %s" % type(self.body))
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
                raise QuantDslSyntaxError('expected call arg not found',
                                     "arg '%s' not in call arg namespace %s" % (callArgName, dslLocals.keys()))

    def apply(self, dslGlobals=None, effectivePresentTime=None, pendingCallStack=None, isDestacking=False, **dslLocals):
        # It's a function call, so create a new namespace "context".
        if dslGlobals is None:
            dslGlobals = DslNamespace()
        else:
           assert isinstance(dslGlobals, DslNamespace)
        dslGlobals = DslNamespace(chain(self.enclosedNamespace.items(), dslGlobals.items()))
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
            # Todo: Implement a check that this test expression can be evaluated (ie
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
            raise QuantDslSystemError("Can't create hash from obj type '%s'" % type(obj), obj,
                                      node=obj.node if isinstance(obj, DslObject) else None)


class FunctionArg(DslObject):

    def validate(self, args):
        self.assertArgsLen(args, requiredLen=2)

    @property
    def name(self):
        return self._args[0]

    @property
    def dslTypeName(self):
        return self._args[1]


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

        combinedNamespace = DslNamespace(chain(dslGlobals.items(), dslLocals.items()))

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
            raise QuantDslSyntaxError("expected number, string or DSL object when reducing name '%s'" % self.name,
                                 repr(value), node=self.node)

    def evaluate(self, **kwds):
        try:
            return kwds[self.name]
        except KeyError:
            raise QuantDslNameError(
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
            raise QuantDslSyntaxError("If test condition result cannot be a DSL object", str(testResult), node=self.node)
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
                raise QuantDslSyntaxError("Op name '%s' not supported" % opName)

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


class Underlying(DslObject):

    def validate(self, args):
        self.assertArgsLen(args, 1)

    @property
    def expr(self):
        return self._args[0]

    def evaluate(self, **_):
        return self.expr


defaultDslClasses = {
    'Add': Add,
    'Choice': Choice,
    'Date': Date,
    'Div': Div,
    'Fixing': Fixing,
    'Number': Number,
    'Market': Market,
    'Max': Max,
    'Mul': Mult,
    'Settlement': Settlement,
    'String': String,
    'Sub': Sub,
    'UnarySub': UnarySub,
    'Wait': Wait,
    'On': On,
    'Name': Name,
    'TimeDelta': TimeDelta,
    'Underlying': Underlying,
    'Stub': Stub,
}



class DslNamespace(dict):

    def copy(self):
        copy = self.__class__(self)
        return copy


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

    def compile(self, dslLocals=None, dslGlobals=None, isParallel=False):
        # It's a module compilation, so create a new namespace "context".
        if dslLocals == None:
            dslLocals = {}
        dslLocals = DslNamespace(dslLocals)
        if dslGlobals == None:
            dslGlobals = {}
        dslGlobals = DslNamespace(dslGlobals)

        # Can't do much with an empty module.
        if len(self.body) == 0:
            raise QuantDslSyntaxError('empty module', node=self.node)

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
                raise QuantDslSyntaxError("'%s' not allowed in module" % type(dslObj), dslObj, node=dslObj.node)

        if len(expressions) == 1:
            # Return the expression, but reduce it with function defs if any are defined.
            dslExpr = expressions[0]
            assert isinstance(dslExpr, DslExpression)
            if len(functionDefs):
                # Compile the expression
                if isParallel:
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
                    dslObj = ExpressionStack(self.rootStubId, stubbedExprsArray)
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
            raise QuantDslSyntaxError('more than one expression in module', node=expressions[1].node)
        elif len(functionDefs) == 1:
            # It's just a module with one function, so return the function def.
            return functionDefs[0]
        elif len(functionDefs) > 1:
            # Can't meaningfully evaluate more than one expression (there are no assignments).
            secondDef = functionDefs[1]
            raise QuantDslSyntaxError('more than one function def in module without an expression', '"def %s"' % secondDef.name, node=functionDefs[1].node)
        raise QuantDslSyntaxError("shouldn't get here", node=self.node)


class ExpressionStack(object):

    def __init__(self, rootStubId, stubbedExprs):
        self.rootStubId = rootStubId
        assert isinstance(stubbedExprs, list)
        assert len(stubbedExprs), "Stubbed expressions is empty!"
        self.stubbedExprs = stubbedExprs

    def __len__(self):
        return len(self.stubbedExprs)

    def hasInstances(self, dslType):
        for _, stubbedExpr, _ in self.stubbedExprs:
            if stubbedExpr.hasInstances(dslType=dslType):
                return True
        return False

    def findInstances(self, dslType):
        instances = []
        for _, stubbedExpr, _ in self.stubbedExprs:
            [instances.append(i) for i in stubbedExpr.findInstances(dslType=dslType)]
        return instances

    def evaluate(self, isMultiprocessing=False, poolSize=None, **kwds):
        assert len(self.stubbedExprs)
        leafIds = []
        callRequirementIds = []
        for stubId, stubbedExpr, effectivePresentTime in self.stubbedExprs:

            assert isinstance(stubbedExpr, DslExpression)

            callRequirementIds.append(stubId)

            # Finding stub instances reveals the dependency graph.
            requiredStubIds = [s.name for s in stubbedExpr.findInstances(Stub)]

            # Stubbed expr has names that need to be replaced with results of other stubbed exprs.
            stubbedExprStr = str(stubbedExpr)
            createCallRequirement(stubId, stubbedExprStr, requiredStubIds, effectivePresentTime)

            if not requiredStubIds:
                # Keep a track of the leaves of the dependency graph (stubbed exprs that don't depend on anything).
                leafIds.append(stubId)

        assert self.rootStubId in callRequirementIds

        # Subscribe to dependencies.
        for callRequirementId in callRequirementIds:
            callRequirement = registry.calls[callRequirementId]
            assert isinstance(callRequirement, CallRequirement)
            for requiredCallId in callRequirement.requiredCallIds:
                requiredCall = registry.calls[requiredCallId]
                assert isinstance(requiredCall, CallRequirement)
                if callRequirementId not in requiredCall.subscribers:
                    requiredCall.subscribers.append(callRequirementId)
                assert requiredCallId not in callRequirement.subscribers, "Circular references."  # Circle of 2, anyway.

        # Run the dependency graph.
        self.runner = DependencyGraphRunner(self.rootStubId, leafIds, isMultiprocessing, poolSize=poolSize)
        self.runner.run(**kwds)

        assert self.rootStubId in self.runner.resultsDict, "Root ID not in runner results."
        if isMultiprocessing:
            # At the moment, the multiprocessing code creates it's own results dict.
            [registry.results.__setitem__(key, value) for key, value in self.runner.resultsDict.items()]

        # Debug and testing info.
        self._runnerCallCount = self.runner.callCount

        try:
            return registry.results[self.rootStubId].value
        except KeyError, e:
            errorData = (self.rootStubId, registry.results.keys())
            raise QuantDslSystemError("root value not found", errorData)


from collections import namedtuple

Registry = namedtuple('Registry', ['results', 'calls', 'functions'])

registry = Registry({}, {}, {})


def createCallRequirement(id, stubbedExprStr, requiredStubIds, effectivePresentTime):
    # Create the domain object.
    callRequirement = CallRequirement(id, stubbedExprStr, requiredStubIds, effectivePresentTime)

    # Register the object with the registry.
    registry.calls[callRequirement.id] = callRequirement
    return callRequirement


def createResult(callRequirementId, returnValue, resultsDict):
    assert isinstance(callRequirementId, basestring), type(callRequirementId)
    result = Result(id=callRequirementId, returnValue=returnValue)
    resultsDict[callRequirementId] = result


def createUuid():
    import uuid
    return uuid.uuid4()


class DomainObject(object): pass

class CallRequirement(DomainObject):
    def __init__(self, id, stubbedExprStr, requiredCallIds, effectivePresentTime):
        self.id = id
        self.stubbedExprStr = stubbedExprStr
        self.requiredCallIds = requiredCallIds
        self.effectivePresentTime = effectivePresentTime
        self.subscribers = []
        # Todo: Validate.

    def isReady(self, resultsRegister):
        for cid in self.requiredCallIds:
            if cid not in resultsRegister:
                return False
        return True

    def registerSubscription(self, callRequirementId):
        if callRequirementId not in self.subscribers:
            self.subscribers.append(callRequirementId)


class Result(DomainObject):
    def __init__(self, id, returnValue):
        self.id = id
        self.value = returnValue


class DependencyGraphRunner(object):

    def __init__(self, rootCallRequirementId, leafIds, isMultiprocessing, poolSize=None):
        self.rootCallRequirementId = rootCallRequirementId
        self.leafIds = leafIds
        self.isMultiprocessing = isMultiprocessing
        self.registry = registry
        self.poolSize = poolSize

    def run(self, **kwds):
        self.runKwds = kwds
        self.callCount = 0
        if self.isMultiprocessing:
            self.executionQueueManager = multiprocessing.Manager()
            self.executionQueue = self.executionQueueManager.Queue()
            self.resultsDict = self.executionQueueManager.dict()
        else:
            self.executionQueue = queue.Queue()
            self.resultsDict = registry.results
        for callRequirementId in self.leafIds:
            self.executionQueue.put(callRequirementId)
        pool = None
        if self.isMultiprocessing:
            pool = multiprocessing.Pool(processes=self.poolSize)
        try:
            while not self.executionQueue.empty():
                if self.isMultiprocessing:
                    batchCallRequirementIds = []
                    while not self.executionQueue.empty():
                        batchCallRequirementIds.append(self.executionQueue.get())
                    pool.map(executeCallRequirement, [(i, kwds, self.resultsDict, self.executionQueue) for i in batchCallRequirementIds])
                    self.callCount += len(batchCallRequirementIds)
                else:
                    callRequirementId = self.executionQueue.get()
                    executeCallRequirement((callRequirementId, kwds, self.resultsDict, self.executionQueue))
                    self.callCount += 1
        finally:
            if self.isMultiprocessing:
                pool.close()
                pool.join()


def executeCallRequirement(args):
    """
    Executes the call requirement, produces a value from the stubbed expr and creates a result..
    """
    try:
        # Get call requirement ID and modelled function objects.
        callRequirementId, evaluationKwds, resultsRegister, executionQueue = args

        # Get the call requirement object (it has the stubbedExpr and effectivePresentTime).
        callRequirement = registry.calls[callRequirementId]

        assert isinstance(callRequirement, CallRequirement), "Call requirement object is not a CallRequirement" \
                                                             ": %s" % callRequirement
        if not callRequirement.isReady(resultsRegister=resultsRegister):
            raise QuantDslSystemError("Call requirement '%s' is not actually ready! It shouldn't have got here" \
                                      " without all required results being available. Is the results register" \
                                      " stale?" % callRequirement.id)

        # If necessary, overwrite the effectivePresentTime as the presentTime in the evaluationKwds.
        if callRequirement.effectivePresentTime:
            evaluationKwds['presentTime'] = callRequirement.effectivePresentTime

        # Evaluate the stubbed expr str.
        try:
            stubbedModule = parse(callRequirement.stubbedExprStr)
        except QuantDslSyntaxError:
            raise

        assert isinstance(stubbedModule, Module), "Parsed stubbed expr string is not an module: %s" % stubbedModule

        dslNamespace = DslNamespace()
        for stubId in callRequirement.requiredCallIds:
            stubResult = resultsRegister[stubId]
            assert isinstance(stubResult, Result), "Not an instance of Result: %s" % stubResult
            dslNamespace[stubId] = Number(stubResult.value)

        simpleExpr = stubbedModule.compile(dslLocals=dslNamespace, dslGlobals={})
        assert isinstance(simpleExpr, DslExpression), "Reduced parsed stubbed expr string is not an " \
                                                      "expression: %s" % type(simpleExpr)
        resultValue = simpleExpr.evaluate(**evaluationKwds)
        handleResult(callRequirementId, resultValue, resultsRegister, executionQueue)
        return "OK"
    except Exception, e:
        import traceback
        msg = traceback.format_exc()
        msg += str(e)
        raise Exception(msg)


def handleResult(callRequirementId, resultValue, resultsDict, executionQueue):
    # Create result object and check if subscribers are ready to be executed.
    createResult(callRequirementId, resultValue, resultsDict)
    callRequirement = registry.calls[callRequirementId]
    for subscriberId in callRequirement.subscribers:
        if subscriberId in resultsDict:
            continue
        subscriber = registry.calls[subscriberId]
        if subscriber.isReady(resultsDict):
            executionQueue.put(subscriberId)


# Todo: Just have one msg parameter (merge 'error' and 'descr').
class QuantDslError(Exception):
    """
    Quant DSL exception base class.
    """
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


class QuantDslSyntaxError(QuantDslError):
    """
    Exception class for user syntax errors.
    """


class QuantDslNameError(QuantDslError):
    """
    Exception class for undefined names.
    """


class QuantDslSystemError(QuantDslError):
    """
    Exception class for DSL system errors.
    """


class PriceSimulator(object):

    def getBrownianMotions(self, marketNames, fixingDates, startDate, pathCount, marketCalibration):
        assert isinstance(startDate, datetime.datetime), type(startDate)
        assert isinstance(pathCount, int), type(pathCount)

        import scipy

        marketNames = list(marketNames)
        allDates = [startDate] + sorted(fixingDates)

        lenMarketNames = len(marketNames)
        lenAllDates = len(allDates)

        # Diffuse random variables through each date for each market (uncorrelated increments).
        import scipy
        brownianMotions = scipy.zeros((lenMarketNames, lenAllDates, pathCount))
        for i in range(lenMarketNames):
            _startDate = allDates[0]
            startRv = brownianMotions[i][0]
            for j in range(lenAllDates - 1):
                fixingDate = allDates[j + 1]
                draws = scipy.random.standard_normal(pathCount)
                T = getDurationYears(_startDate, fixingDate)
                if T < 0:
                    raise QuantDslError("Can't sqrt negative numbers: %s. (Contract starts before observation time?)" % T)
                endRv = startRv + scipy.sqrt(T) * draws
                try:
                    brownianMotions[i][j + 1] = endRv
                except ValueError, e:
                    raise ValueError, "Can't set endRv in brownianMotions: %s" % e
                _startDate = fixingDate
                startRv = endRv

        # Read the market calibration data.
        correlations = {}
        for marketNamePairs in itertools.combinations(marketNames, 2):
            marketNamePairs = tuple(sorted(marketNamePairs))
            calibrationName = "%s-%s-CORRELATION" % marketNamePairs
            try:
                correlation = marketCalibration[calibrationName]
            except KeyError, e:
                msg = "Can't find correlation between '%s' and '%s': '%s' not defined in market calibration: %s" % (
                    marketNamePairs[0],
                    marketNamePairs[1],
                    marketCalibration.keys(),
                    e
                )
                raise QuantDslError(msg)
            else:
                correlations[marketNamePairs] = correlation

        correlationMatrix = scipy.zeros((lenMarketNames, lenMarketNames))
        for i in range(lenMarketNames):
            for j in range(lenMarketNames):
                if marketNames[i] == marketNames[j]:
                    correlation = 1
                else:
                    key = tuple(sorted([marketNames[i], marketNames[j]]))
                    correlation = correlations[key]
                correlationMatrix[i][j] = correlation

        import scipy.linalg
        from numpy.linalg import LinAlgError
        try:
            U = scipy.linalg.cholesky(correlationMatrix)
        except LinAlgError, e:
            raise QuantDslError("Couldn't do Cholesky decomposition with correlation matrix: %s: %s" % (correlationMatrix, e))

        # Correlated increments from uncorrelated increments.
        # Todo: This seems to work, but I would feel better if there was a test making sure this implementation of 'R . U' is correct.
        brownianMotions = brownianMotions.transpose() # Put markets on the last dimension, so the broadcasting works?
        try:
            brownianMotionsCorrelated = brownianMotions.dot(U)
        except Exception, e:
            msg = "Couldn't multiply uncorrelated Brownian increments with decomposed correlation matrix: %s, %s" % (brownianMotions, U)
            raise QuantDslError(msg)
        brownianMotionsCorrelated = brownianMotionsCorrelated.transpose() # Put markets back on the first dimension.

        brownianMotionsDict = {}
        for i, marketName in enumerate(marketNames):
            marketRvs = {}
            for j, fixingDate in enumerate(allDates):
                rv = brownianMotionsCorrelated[i][j]
                marketRvs[fixingDate] = rv
            brownianMotionsDict[marketName] = marketRvs

        return brownianMotionsDict


def getDurationYears(startDate, endDate, daysPerYear=365):
    try:
        timeDelta = endDate - startDate
    except TypeError, inst:
        raise TypeError("%s: start: %s end: %s" % (inst, startDate, endDate))
    return timeDelta.days / daysPerYear



####### Plotting codes #####################################

# FROM top of file

# # Todo: Write plots to file.
# isPlotting = False
# if isPlotting:
# from pylab import *


# FROM Least Squares
#if len(regressionVariables):
#    conditionalExpectedValue = LeastSquares(regressionVariables, expectedContinuationValue).fit()

                #plotCount = 3000

                        # if isPlotting:
                        #     data = scipy.array([underlyingValue[:plotCount], conditionalExpectedValue[:plotCount], expectedContinuationValue[:plotCount]])
                        #     plot(data[0], data[2], 'go')
                        #     plot(data[0], data[1], 'y^')


# FROM LongstaffSchwartz
#assert stateValue.shape == underlyingValue.shape

                # if isPlotting:
                #     data = scipy.array([underlyingValue[:plotCount], stateValue[:plotCount]])
                #     plot(data[0], data[1], 'rs')
                #     draw()
                #     show()
