from __future__ import division
from abc import ABCMeta, abstractmethod
import six.moves.queue as queue
import datetime
import itertools
import math
import re
import uuid

import dateutil.parser
import six

from quantdsl.exceptions import DslSystemError, DslSyntaxError, DslNameError, DslError
from quantdsl.priceprocess.base import get_duration_years
from quantdsl import utc


class DslObject(six.with_metaclass(ABCMeta)):
    """
    Base class for DSL language objects.

    Responsible for maintaining reference to original AST (for error reporting),
    and for rendering objects into valid DSL source code. Also has methods for
    validating object arguments, and finding child nodes of a particular type.
    """

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
            except TypeError as e:
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
    def assert_args_len(self, args, required_len=None, min_len=None):
        if min_len != None and len(args) < min_len:
            error = "%s is broken" % self.__class__.__name__
            descr = "requires at least %s arguments (%s were given)" % (min_len, len(args))
            raise DslSyntaxError(error, descr, self.node)
        if required_len != None and len(args) != required_len:
            error = "%s is broken" % self.__class__.__name__
            descr = "requires %s arguments (%s were given)" % (required_len, len(args))
            raise DslSyntaxError(error, descr, self.node)

    def assert_args_arg(self, args, posn, required_type):
        if isinstance(required_type, list):
            # Ahem, this is a way of saying we require a list of the type (should be a list length 1).
            self.assert_args_arg(args, posn, list)
            assert len(required_type) == 1, "List def should only have one item."
            required_type = required_type[0]
            list_of_args = args[posn]
            for i in range(len(list_of_args)):
                self.assert_args_arg(list_of_args, i, required_type)
        elif not isinstance(args[posn], required_type):
            error = "%s is broken" % self.__class__.__name__
            if isinstance(required_type, tuple):
                required_type_names = [i.__name__ for i in required_type]
                required_type_names = ", ".join(required_type_names)
            else:
                required_type_names = required_type.__name__
            desc = "argument %s must be %s" % (posn, required_type_names)
            desc += " (but a %s was found): " % (args[posn].__class__.__name__)
            desc += str(args[posn])
            raise DslSyntaxError(error, desc, self.node)

    def find_instances(self, dslType):
        return list(self.find_instances_generator(dslType))

    def has_instances(self, dslType):
        for i in self.find_instances_generator(dslType):
            return True
        else:
            return False
        # try:
        #     self.find_instances_generator(dslType).next()
        #     # self.find_instances_generator(dslType)
        # except StopIteration:
        #     return False
        # else:
        #     return True

    def find_instances_generator(self, dslType):
        if isinstance(self, dslType):
            yield self
        for arg in self._args:
            if isinstance(arg, DslObject):
                for dsl_obj in arg.find_instances_generator(dslType):
                    yield dsl_obj
            elif isinstance(arg, list):
                for arg in arg:
                    if isinstance(arg, DslObject):
                        for dsl_obj in arg.find_instances(dslType):
                            yield dsl_obj

    def reduce(self, dsl_locals, dsl_globals, effective_present_time=None, pending_call_stack=None):
        """
        Reduces by reducing all args, and then using those args
        to create a new instance of self.
        """
        new_dsl_args = []
        for dsl_arg in self._args:
            if isinstance(dsl_arg, DslObject):
                dsl_arg = dsl_arg.reduce(dsl_locals, dsl_globals, effective_present_time, pending_call_stack=pending_call_stack)
            new_dsl_args.append(dsl_arg)
        return self.__class__(node=self.node, *new_dsl_args)


class DslExpression(DslObject):

    @abstractmethod
    def evaluate(self, **kwds):
        pass

    def discount(self, value, date, **kwds):
        r = float(kwds['interest_rate']) / 100
        T = get_duration_years(kwds['present_time'], date)
        return value * math.exp(- r * T)


class DslConstant(DslExpression):
    required_type = None

    def __str__(self, indent=0):
        return repr(self.value)

    def validate(self, args):
        self.assert_args_len(args, required_len=1)
        if self.required_type == None:
            raise Exception("required_type attribute not set on %s" % self.__class__)
        self.assert_args_arg(args, posn=0, required_type=self.required_type)
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
    required_type = six.string_types


class Number(DslConstant):

    @property
    def required_type(self):
        from numpy import ndarray
        return (int, float, ndarray)


class Date(DslConstant):
    required_type = (six.string_types, String, datetime.datetime)

    def __str__(self, indent=0):
        return "Date('%04d-%02d-%02d')" % (self.value.year, self.value.month, self.value.day)

    def parse(self, value):
        # Return a datetime.datetime.
        if isinstance(value, (six.string_types, String)):
            if isinstance(value, String):
                dateStr = value.evaluate()
            else:
                dateStr = value
            try:
                return dateutil.parser.parse(dateStr).replace(tzinfo=utc)
            except ValueError:
                raise DslSyntaxError("invalid date string", dateStr, node=self.node)
        elif isinstance(value, datetime.datetime):
            return value
        else:
            raise DslSystemError("shouldn't get here", value, node=self.node)


class TimeDelta(DslConstant):
    required_type = (String, datetime.timedelta)

    def __str__(self, indent=0):
        return "%s('%dd')" % (self.__class__.__name__, self.value.days)

    def parse(self, value, regex=re.compile(r'((?P<days>\d+?)d)?')):
        if isinstance(value, String):
            duration_str = value.evaluate()
            parts = regex.match(duration_str)
            if not parts:
                raise DslSyntaxError('invalid time delta string', duration_str, node=self.node)
            parts = parts.groupdict()
            time_params = {}
            for (name, param) in six.iteritems(parts):
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
        self.assert_args_len(args, required_len=1)
        self.assert_args_arg(args, posn=0, required_type=DslExpression)

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
        self.assert_args_len(args, required_len=1)
        self.assert_args_arg(args, posn=0, required_type=list)

    @property
    def values(self):
        return self._args[0]

    def evaluate(self, **kwds):
        len_values = len(self.values)
        assert len_values >= 2
        for dsl_expr in self.values:
            assert isinstance(dsl_expr, DslExpression)
            value = dsl_expr.evaluate(**kwds)
            # Assert value is a simple value.
            if not isinstance(dsl_expr, DslExpression):
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
            def makeStr(dsl_expr):
                dslString = str(dsl_expr)
                if isinstance(dsl_expr, BinOp):
                    dslString = "(" + dslString + ")"
                return dslString
            return makeStr(self.left) + " " + self.opchar + " " + makeStr(self.right)
        else:
            return '%s(%s, %s)' % (self.__class__.__name__, self.left, self.right)

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(DslExpression, Date, TimeDelta, Underlying))
        self.assert_args_arg(args, posn=1, required_type=(DslExpression, Date, TimeDelta, Underlying))

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
        except TypeError as e:
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
        assert isinstance(args[0], (six.string_types, String)), type(args[0])

    @property
    def name(self):
        """
        Returns a Python string.
        """
        name = self._args[0]
        if isinstance(name, six.string_types):
            return name
        elif isinstance(name, String):
            return name.evaluate()

    def reduce(self, dsl_locals, dsl_globals, effective_present_time=None, pending_call_stack=False):
        """
        Replace name with named value in context (kwds).
        """

        combinedNamespace = DslNamespace(itertools.chain(dsl_globals.items(), dsl_locals.items()))

        from numpy import ndarray
        value = self.evaluate(**combinedNamespace)
        if isinstance(value, six.string_types):
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
        self.assert_args_len(args, 1)

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
        indent_spaces = 4 * ' '
        msg = ""
        for decorator_name in self.decorator_names:
            msg += "@" + decorator_name + "\n"
        msg += "def %s(%s):\n" % (self.name, ", ".join(self.call_arg_names))
        if isinstance(self.body, DslObject):
            try:
                msg += indent_spaces + self.body.__str__(indent=indent+1)
            except TypeError:
                raise DslSystemError("DSL object can't handle indent: %s" % type(self.body))
        else:
            msg += str(self.body)
        return msg

    def __init__(self, *args, **kwds):
        super(FunctionDef, self).__init__(*args, **kwds)
        # Initialise the function call cache for this function def.
        self.call_cache = {}
        self.enclosed_namespace = DslNamespace()

    def validate(self, args):
        self.assert_args_len(args, required_len=4)

    @property
    def name(self):
        return self._args[0]

    @property
    def call_arg_names(self):
        if not hasattr(self, '_call_arg_names'):
            self._call_arg_names = [i.name for i in self._args[1]]
        return self._call_arg_names

    @property
    def callArgs(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def decorator_names(self):
        return self._args[3]

    def validateCallArgs(self, dsl_locals):
        for call_arg_name in self.call_arg_names:
            if call_arg_name not in dsl_locals:
                raise DslSyntaxError('expected call arg not found',
                                     "arg '%s' not in call arg namespace %s" % (call_arg_name, dsl_locals.keys()))

    def apply(self, dsl_globals=None, effective_present_time=None, pending_call_stack=None, is_destacking=False, **dsl_locals):
        # It's a function call, so create a new namespace "context".
        if dsl_globals is None:
            dsl_globals = DslNamespace()
        else:
           assert isinstance(dsl_globals, DslNamespace)
        dsl_globals = DslNamespace(itertools.chain(self.enclosed_namespace.items(), dsl_globals.items()))
        dsl_locals = DslNamespace(dsl_locals)

        # Validate the call args with the definition.
        self.validateCallArgs(dsl_locals)

        # Create the cache key.
        call_cache_key_dict = dsl_locals.copy()
        call_cache_key_dict["__effective_present_time__"] = effective_present_time
        call_cache_key = self.create_hash(dsl_locals)

        # Check the call cache, to see whether this function has already been called with these args.
        if not is_destacking and call_cache_key in self.call_cache:
            return self.call_cache[call_cache_key]

        if pending_call_stack and not is_destacking and not 'nostub' in self.decorator_names:
            # Just stack the call expression and return a stub.
            assert isinstance(pending_call_stack, queue.Queue)

            # Create a new stub - the stub ID is the name of the return value of the function call..
            stub_id = str(uuid.uuid4())
            dslStub = Stub(stub_id, node=self.node)

            # Put the function call on the call stack, with the stub ID.
            assert isinstance(pending_call_stack, FunctionDefCallStack)
            pending_call_stack.put(
                stub_id=stub_id,
                stacked_call=self,
                stacked_locals=dsl_locals.copy(),
                stacked_globals=dsl_globals.copy(),
                effective_present_time=effective_present_time
            )
            # Return the stub so that the containing DSL can be fully evaluated
            # once the stacked function call has been evaluated.
            dsl_expr = dslStub
        else:
            # Todo: Make sure the expression can be selected with the dsl_locals?
            # - ie the conditional expressions should be functions only of call arg
            # values that can be fully evaluated without evaluating contractual DSL objects.
            selectedExpression = self.selectExpression(self.body, dsl_locals)

            # Add this function to the dslNamespace (just in case it's called by itself).
            newDslGlobals = DslNamespace(dsl_globals)
            newDslGlobals[self.name] = self

            # Reduce the selected expression.
            dsl_expr = selectedExpression.reduce(dsl_locals, newDslGlobals, effective_present_time, pending_call_stack=pending_call_stack)

        # Cache the result.
        if not is_destacking:
            self.call_cache[call_cache_key] = dsl_expr

        return dsl_expr

    def selectExpression(self, dsl_expr, call_arg_namespace):
        # If the DSL expression is an instance of If, then evaluate
        # the test and accordingly select body or orelse expressions. Repeat
        # this method with the selected expression (supports if-elif-elif-else).
        # Otherwise just return the DSL express as the selected expression.

        if isinstance(dsl_expr, BaseIf):
            # Todo: Implement a check that this test expression can be evaluated? Or handle case when it can't?
            # Todo: Also allow user defined functions that just do dates or numbers in test expression.
            # it doesn't have or expand into DSL elements that are the functions of time (Wait, Choice, Market, etc).
            if dsl_expr.test.evaluate(**call_arg_namespace):
                selected = dsl_expr.body
            else:
                selected = dsl_expr.orelse
            selected = self.selectExpression(selected, call_arg_namespace)
        else:
            selected = dsl_expr
        return selected

    def create_hash(self, obj):
        if isinstance(obj, (int, float, six.string_types, datetime.datetime, datetime.timedelta)):
            return hash(obj)
        if isinstance(obj, dict):
            return hash(tuple(sorted([(a, self.create_hash(b)) for a, b in obj.items()])))
        if isinstance(obj, list):
            return hash(tuple(sorted([self.create_hash(a) for a in obj])))
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
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=Name)
        self.assert_args_arg(args, posn=1, required_type=list)

    @property
    def functionDefName(self):
        return self._args[0]

    @property
    def callArgExprs(self):
        return self._args[1]

    def reduce(self, dsl_locals, dsl_globals, effective_present_time=None, pending_call_stack=False):
        """
        Reduces function call to result of evaluating function def with function call args.
        """

        # Replace functionDef names with things in kwds.
        functionDef = self.functionDefName.reduce(dsl_locals, dsl_globals, effective_present_time, pending_call_stack=pending_call_stack)

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
                    functionDef.call_arg_names,
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
                callArgExpr = callArgExpr.reduce(dsl_locals, dsl_globals, effective_present_time, pending_call_stack=pending_call_stack)
                # Decide whether to evaluate, or just pass the expression into the function call.
                if isinstance(callArgExpr, Underlying):
                    # It's explicitly wrapped as an "underlying", so unwrap it as expected.
                    callArgValue = callArgExpr.evaluate()
                elif callArgExpr.has_instances((Market, Fixing, Choice, Settlement, FunctionDef, Stub)):
                    # It's an underlying contract, or a stub. In any case, can't evaluate here, so.pass it through.
                    callArgValue = callArgExpr
                else:
                    assert isinstance(callArgExpr, DslExpression)
                    # It's a sum of two constants, or something like that - evaluate the full expression.
                    callArgValue = callArgExpr.evaluate()

            # Add the call arg value to the new call arg namespace.
            newDslLocals[callArgDef.name] = callArgValue

        # Evaluate the function def with the dict of call arg values.
        dsl_expr = functionDef.apply(dsl_globals, effective_present_time, pending_call_stack=pending_call_stack, is_destacking=False, **newDslLocals)

        # The result of this function call (stubbed or otherwise) should be a DSL expression.
        assert isinstance(dsl_expr, DslExpression)

        return dsl_expr

    def evaluate(self, **kwds):
        raise DslSyntaxError('call to undefined name', self.functionDefName.name, node=self.node)


class FunctionArg(DslObject):

    def validate(self, args):
        self.assert_args_len(args, required_len=2)

    @property
    def name(self):
        return self._args[0]

    @property
    def dslTypeName(self):
        return self._args[1]


class BaseIf(DslExpression):

    def validate(self, args):
        self.assert_args_len(args, required_len=3)
        self.assert_args_arg(args, posn=0, required_type=DslExpression)
        self.assert_args_arg(args, posn=1, required_type=DslExpression)
        self.assert_args_arg(args, posn=2, required_type=DslExpression)

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
    valid_ops = {
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
            +  " ".join([str(self.opcodes[op])+' '+str(right) for (op, right) in zip(self.op_names, self.comparators) ])

    def validate(self, args):
        self.assert_args_len(args, 3)
        self.assert_args_arg(args, 0, required_type=(
            DslExpression, Date))  #, Date, Number, String, int, float, six.string_types, datetime.datetime))
        self.assert_args_arg(args, 1, required_type=list)
        self.assert_args_arg(args, 2, required_type=list)
        for opName in args[1]:
            if opName not in self.valid_ops.keys():
                raise DslSyntaxError("Op name '%s' not supported" % opName)

    @property
    def left(self):
        return self._args[0]

    @property
    def op_names(self):
        return self._args[1]

    @property
    def comparators(self):
        return self._args[2]

    def evaluate(self, **kwds):
        left = self.left.evaluate(**kwds)
        for i in range(len(self.op_names)):
            right = self.comparators[i].evaluate(**kwds)
            op_name = self.op_names[i]
            op = self.valid_ops[op_name]
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
        self.assert_args_len(args, 1)
        self.assert_args_arg(args, 0, [(FunctionDef, DslExpression, Date)])

    @property
    def body(self):
        return self._args[0]

    def compile(self, dsl_locals=None, dsl_globals=None, dependency_graph_class=None):
        # It's a module compilation, so create a new namespace "context".
        if dsl_locals == None:
            dsl_locals = {}
        dsl_locals = DslNamespace(dsl_locals)
        if dsl_globals == None:
            dsl_globals = {}
        dsl_globals = DslNamespace(dsl_globals)

        # Can't do much with an empty module.
        if len(self.body) == 0:
            raise DslSyntaxError('empty module', node=self.node)

        # Collect function defs and expressions.
        function_defs = []
        expressions = []
        for dsl_obj in self.body:
            if isinstance(dsl_obj, FunctionDef):
                dsl_globals[dsl_obj.name] = dsl_obj
                # Share the module level namespace (any function body can call any other function).
                dsl_obj.enclosed_namespace = dsl_globals
                function_defs.append(dsl_obj)
            elif isinstance(dsl_obj, DslExpression):
                expressions.append(dsl_obj)
            else:
                raise DslSyntaxError("'%s' not allowed in module" % type(dsl_obj), dsl_obj, node=dsl_obj.node)

        if len(expressions) == 1:
            # Return the expression, but reduce it with function defs if any are defined.
            dsl_expr = expressions[0]
            assert isinstance(dsl_expr, DslExpression)
            if len(function_defs) and dependency_graph_class:
                # Compile the expression

                # Create a stack of discovered calls to function defs.
                pending_call_stack = FunctionDefCallStack()

                # Create a stack for the stubbed exprs.
                stubbed_exprs = StubbedExpressionStack()

                # Start things off. If an expression has a FunctionCall, it will cause a pending
                # call to be placed on the pending call stack, and the function call will be
                # replaced with a stub, which acts as a placeholder for the result of the function
                # call. By looping over the pending call stack until it is empty, evaluating
                # pending calls to generate stubbed expressions and further pending calls, the
                # module can be compiled into a stack of stubbed expressions.
                # Of course if the module's expression doesn't have a function call, there
                # will just be one expression on the stack of "stubbed" expressions, and it will
                # not have any stubs.
                stubbed_expr = dsl_expr.reduce(
                    dsl_locals,
                    DslNamespace(dsl_globals),
                    pending_call_stack=pending_call_stack
                )

                # Create the root stub ID, this will allow the final result to be retrieved.
                from quantdsl.domain.services import create_uuid4
                self.root_stub_id = str(create_uuid4())

                # Put the module expression (now stubbed) on the stack.
                stubbed_exprs.put(self.root_stub_id, stubbed_expr, None)

                # Continue by looping over any pending calls that have resulted from the module's expression.
                while not pending_call_stack.empty():
                    # Get the stacked call info.
                    (stub_id, stacked_call, stacked_locals, stacked_globals, effective_present_time) = pending_call_stack.get()

                    # Check we've got a function def.
                    assert isinstance(stacked_call, FunctionDef), type(stacked_call)

                    # Apply the stacked call values to the called function def.
                    stubbed_expr = stacked_call.apply(stacked_globals,
                                                    effective_present_time,
                                                    pending_call_stack=pending_call_stack,
                                                    is_destacking=True,
                                                    **stacked_locals)

                    # Put the resulting (potentially stubbed) expression on the stack of stubbed expressions.
                    stubbed_exprs.put(stub_id, stubbed_expr, effective_present_time)

                # Create an expression stack DSL object from the stack of stubbed expressions.
                stubbed_exprs_array = []
                while not stubbed_exprs.empty():
                    stubbed_exprs_array.append(stubbed_exprs.get())

                dsl_obj = build_dependency_graph(self.root_stub_id, stubbed_exprs_array, dependency_graph_class)
            else:
                # Compile the module for a single threaded recursive operation (faster but not distributed,
                # so also limited in space and perhaps time). For smaller computations only.
                dsl_obj = dsl_expr.reduce(dsl_locals, DslNamespace(dsl_globals))
            return dsl_obj
        elif len(expressions) > 1:
            # Can't meaningfully evaluate more than one expression (since assignments are not supported).
            raise DslSyntaxError('more than one expression in module', node=expressions[1].node)
        elif len(function_defs) == 1:
            # It's just a module with one function, so return the function def.
            return function_defs[0]
        elif len(function_defs) > 1:
            # Can't meaningfully evaluate more than one expression (there are no assignments).
            secondDef = function_defs[1]
            raise DslSyntaxError('more than one function def in module without an expression', '"def %s"' % secondDef.name, node=function_defs[1].node)
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

    @abstractmethod
    def validate(self, args):
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
            if isinstance(date, six.string_types):
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
        self.assert_args_len(args, required_len=1)
        self.assert_args_arg(args, posn=0, required_type=(six.string_types, String, Name))

    @property
    def name(self):
        return self._args[0].evaluate() if isinstance(self._args[0], String) else self._args[0]

    def evaluate(self, **kwds):
        try:
            present_time = kwds['present_time']
        except KeyError:
            raise DslSyntaxError(
                "Can't evaluate Market '%s' without 'present_time' in context variables" % self.name,
                ", ".join(kwds.keys()),
                node=self.node
            )
        try:
            all_market_prices = kwds['all_market_prices']
        except KeyError:
            raise DslError(
                "Can't evaluate Market '%s' without 'all_market_prices' in context variables" % self.name,
                ", ".join(kwds.keys()),
                node=self.node
            )

        try:
            marketPrices = all_market_prices[self.name]
        except KeyError:
            raise DslError(
                "Can't evaluate Market '%s' without market name in 'all_market_prices'" % self.name,
                ", ".join(all_market_prices.keys()),
                node=self.node
            )

        try:
            marketPrice = marketPrices[present_time]
        except KeyError:
            raise DslError(
                "Can't evaluate Market '%s' without present time '%s in market prices" % (self.name, present_time),
                ", ".join(marketPrices.keys()),
                node=self.node
            )

        return marketPrice


class Settlement(StochasticObject, DatedDslObject, DslExpression):
    """
    Discounts value of expression to 'present_time'.
    """

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(String, Date, Name,BinOp))
        self.assert_args_arg(args, posn=1, required_type=DslExpression)

    def evaluate(self, **kwds):
        value = self._args[1].evaluate(**kwds)
        return self.discount(value, self.date, **kwds)


class Fixing(StochasticObject, DatedDslObject, DslExpression):
    """
    A fixing defines the 'present_time' used for evaluating its expression.
    """

    def __str__(self):
        return "%s('%04d-%02d-%02d', %s)" % (
            self.__class__.__name__,
            self.date.year,
            self.date.month,
            self.date.day,
            self.expr)

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(six.string_types, String, Date, Name, BinOp))
        self.assert_args_arg(args, posn=1, required_type=DslExpression)

    @property
    def expr(self):
        return self._args[1]

    def reduce(self, dsl_locals, dsl_globals, effective_present_time=None, pending_call_stack=None):
        # Figure out the effective_present_time from the fixing date, which might still be a Name.
        # Todo: It might also be a date expression, and so might the
        fixingDate = self._args[0]
        if isinstance(fixingDate, datetime.datetime):
            pass
        if isinstance(fixingDate, six.string_types):
            fixingDate = String(fixingDate)
        if isinstance(fixingDate, String):
            fixingDate = Date(fixingDate, node=fixingDate.node)
        if isinstance(fixingDate, (Date, BinOp, Name)):
            fixingDate = fixingDate.evaluate(**dsl_locals)
        if not isinstance(fixingDate, datetime.datetime):
            raise DslSyntaxError("fixing date value should be a datetime.datetime by now, but it's a %s" % fixingDate, node=self.node)
        effective_present_time = fixingDate
        return super(Fixing, self).reduce(dsl_locals, dsl_globals, effective_present_time, pending_call_stack=pending_call_stack)

    def evaluate(self, **kwds):
        kwds = kwds.copy()
        kwds['present_time'] = self.date
        return self.expr.evaluate(**kwds)


class On(Fixing):
    """
    A shorter name for Fixing.
    """

class Wait(Fixing):
    """
    A fixing with discounting of the resulting value from date arg to present_time.
    """
    def evaluate(self, **kwds):
        value = super(Wait, self).evaluate(**kwds)
        return self.discount(value, self.date, **kwds)


class Choice(StochasticObject, DslExpression):
    """
    Encapsulates the Longstaff-Schwartz routine as an element of the language.
    """
    def validate(self, args):
        self.assert_args_len(args, min_len=2)
        for i in range(len(args)):
            self.assert_args_arg(args, posn=i, required_type=DslExpression)

    def evaluate(self, **kwds):
        # Check the results cache, to see whether this function
        # has already been evaluated with these args.
        if not hasattr(self, 'resultsCache'):
            self.resultsCache = {}
        cacheKeyKwdItems = [(k, hash(tuple(sorted(v))) if isinstance(v, dict) else v) for (k, v) in kwds.items()]
        kwdsHash = hash(tuple(sorted(cacheKeyKwdItems)))
        if kwdsHash not in self.resultsCache:
            # Run the least-squares monte-carlo routine.
            present_time = kwds['present_time']
            initialState = LongstaffSchwartzState(self, present_time)
            finalStates = [LongstaffSchwartzState(a, present_time) for a in self._args]
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
        # sleep(1)
        try:
            all_market_prices = kwds['all_market_prices']
        except KeyError:
            raise DslSystemError("'all_market_prices' not in evaluation kwds", kwds.keys(), node=None)
        if len(all_market_prices) == 0:
            raise DslSystemError('no rvs', str(kwds))
        firstMarketPrices = list(all_market_prices.values())[0]
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
                    dslMarkets = subsequentState.dsl_object.find_instances(Market)
                    marketNames = set([m.name for m in dslMarkets])
                    for marketName in marketNames:
                        marketPrices = all_market_prices[marketName]
                        try:
                            marketPrice = marketPrices[state.time]
                        except KeyError as inst:
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
                stateValue = state.dsl_object.evaluate(**kwds)
                if isinstance(stateValue, (int, float)):
                    try:
                        underlyingValue = firstMarketPrices[state.time]
                    except KeyError:
                        msg = "Couldn't find market price at time %s, available times: %s" % (
                            state.time, sorted(firstMarketPrices.keys()))
                        raise KeyError(msg)
                    path_count = len(underlyingValue)
                    if stateValue == 0:
                        stateValue = scipy.zeros(path_count)
                    else:
                        ones = scipy.ones(path_count)
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

    def __init__(self, dsl_object, time):
        self.subsequentStates = set()
        self.dsl_object = dsl_object
        self.time = time

    def addSubsequentState(self, state):
        self.subsequentStates.add(state)


class LeastSquares(object):
    """
    Implements the least-squares routine.
    """

    def __init__(self, xs, y):
        self.path_count = len(y)
        for x in xs:
            if len(x) != self.path_count:
                raise Exception("Regression won't work with uneven path counts.")
        self.xs = xs
        self.y = y

    def fit(self):
        import scipy
        regressions = list()
        # Regress against unity.
        regressions.append(scipy.ones(self.path_count))
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
            raise Exception("Regression won't work with uneven path counts.")
        c = self.solve(a, b)
        c = scipy.matrix(c)
        #print "a: ", a
        #print "a: ", a.shape, type(a)
        #print "b: ", b
        #print "b: ", b.shape, type(b)
        #print "c: ", c.shape, type(c)
        #print "c: ", c
        if a.shape[1] != c.shape[0]:
            raise Exception("Matrices are not aligned: %s and %s" % (a.shape, c.shape))
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
        except Exception as inst:
            msg = "Couldn't solve a and b: %s %s: %s" % (a, b, inst)
            raise Exception(msg)
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

    def put(self, stub_id, stacked_call, stacked_locals, stacked_globals, effective_present_time):
        assert isinstance(stub_id, six.string_types), type(stub_id)
        assert isinstance(stacked_call, FunctionDef), type(stacked_call)
        assert isinstance(stacked_locals, DslNamespace), type(stacked_locals)
        assert isinstance(stacked_globals, DslNamespace), type(stacked_globals)
        assert isinstance(effective_present_time, (datetime.datetime, type(None))), type(effective_present_time)
        queue.Queue.put(self, (stub_id, stacked_call, stacked_locals, stacked_globals, effective_present_time))


class StubbedExpressionStack(queue.LifoQueue):

    def put(self, stub_id, stubbed_expr, effective_present_time):
        assert isinstance(stub_id, six.string_types), type(stub_id)
        assert isinstance(stubbed_expr, DslExpression), type(stubbed_expr)
        assert isinstance(effective_present_time, (datetime.datetime, type(None))), type(effective_present_time)
        queue.LifoQueue.put(self, (stub_id, stubbed_expr, effective_present_time))


def build_dependency_graph(root_stub_id, stubbed_exprs_array, dependencyGraphClass):
    call_requirement_ids, call_requirements, dependencies_by_stub, dependents_by_stub, leaf_ids = reveal_dependencies(
        root_stub_id, stubbed_exprs_array)
    dsl_obj = dependencyGraphClass(root_stub_id, stubbed_exprs_array, call_requirement_ids, call_requirements,
                                  dependencies_by_stub, dependents_by_stub, leaf_ids)
    return dsl_obj


def reveal_dependencies(root_stub_id, stubbed_exprs_data):
    assert isinstance(stubbed_exprs_data, list)
    assert len(stubbed_exprs_data), "Stubbed expressions is empty!"
    leaf_ids = []
    call_requirement_ids = []
    call_requirements = {}
    dependencies_by_stub = {}
    dependents_by_stub = {root_stub_id: []}
    for stub_id, stubbed_expr, effective_present_time in stubbed_exprs_data:

        assert isinstance(stubbed_expr, DslExpression)

        # Discover the dependency graph by identifying the stubs (if any) each stub depends on.
        dependencies = [s.name for s in stubbed_expr.find_instances(Stub)]

        # Remember which stubs this stub depends on (the "upstream" stubs).
        dependencies_by_stub[stub_id] = dependencies

        # If there are no dependencies, then it's a "leaf" of the dependency graph.
        if len(dependencies) == 0:
            # Remember the leaves - they can be evaluated first.
            leaf_ids.append(stub_id)
        else:
            # Remember which stubs depend on this stub ("downstream").
            for dependency in dependencies:
                if dependency not in dependents_by_stub:
                    dependents_by_stub[dependency] = []
                dependents = dependents_by_stub[dependency]
                if stub_id in dependents:
                    raise DslSystemError("Stub ID already in dependents of required stub. Probably wrong?")
                dependents.append(stub_id)

        # Stubbed expr has names that need to be replaced with results of other stubbed exprs.
        stubbed_expr_str = str(stubbed_expr)

        call_requirements[stub_id] = (stubbed_expr_str, effective_present_time)
        call_requirement_ids.append(stub_id)
    assert root_stub_id in call_requirement_ids
    return call_requirement_ids, call_requirements, dependencies_by_stub, dependents_by_stub, leaf_ids