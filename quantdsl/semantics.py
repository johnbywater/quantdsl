from __future__ import division

import json
from abc import ABCMeta, abstractmethod
from collections import namedtuple
import datetime
import itertools
import math
import re

import scipy
from dateutil.relativedelta import relativedelta
from scipy import ndarray
import scipy.linalg
import six
import six.moves.queue as queue

from quantdsl.domain.model.call_requirement import StubbedCall
from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.domain.services.uuids import create_uuid4
from quantdsl.exceptions import DslSystemError, DslSyntaxError, DslNameError, DslError
from quantdsl.priceprocess.base import get_duration_years


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

    def list_instances(self, dsl_type):
        return list(self.find_instances(dsl_type))

    def has_instances(self, dsl_type):
        for i in self.find_instances(dsl_type):
            return True
        else:
            return False

    def find_instances(self, dsl_type):
        if isinstance(self, dsl_type):
            yield self
        for arg in self._args:
            if isinstance(arg, DslObject):
                for dsl_obj in arg.find_instances(dsl_type):
                    yield dsl_obj
            elif isinstance(arg, list):
                for arg in arg:
                    if isinstance(arg, DslObject):
                        for dsl_obj in arg.list_instances(dsl_type):
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

    def identify_price_simulation_requirements(self, requirements, **kwds):
        for dsl_arg in self._args:
            if isinstance(dsl_arg, DslObject):
                dsl_arg.identify_price_simulation_requirements(requirements, **kwds)

    def identify_perturbation_dependencies(self, dependencies, **kwds):
        for dsl_arg in self._args:
            if isinstance(dsl_arg, DslObject):
                dsl_arg.identify_perturbation_dependencies(dependencies, **kwds)


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
        return six.integer_types + (float, ndarray)


class Date(DslConstant):
    required_type = six.string_types + (String, datetime.date)

    def __str__(self, indent=0):
        return "Date('%04d-%02d-%02d')" % (self.value.year, self.value.month, self.value.day)

    def parse(self, value):
        # Return a datetime.datetime.
        if isinstance(value, (six.string_types, String)):
            if isinstance(value, String):
                date_str = value.evaluate()
            else:
                date_str = value
            try:
                year, month, day = [int(i) for i in date_str.split('-')]
                return datetime.date(year, month, day)
                # return dateutil.parser.parse(date_str).replace()
            except ValueError:
                raise DslSyntaxError("invalid date string", date_str, node=self.node)
        elif isinstance(value, (datetime.datetime, datetime.date)):
            return value
        else:
            raise DslSystemError("shouldn't get here", value, node=self.node)


class TimeDelta(DslConstant):
    required_type = (String, datetime.timedelta, relativedelta)

    def __str__(self, indent=0):
        return "%s('%dd')" % (self.__class__.__name__, self.value.days)

    def parse(self, value, regex=re.compile(r'((?P<days>\d+?)d|(?P<months>\d+?)m|(?P<years>\d+?)y)?')):
        if isinstance(value, String):
            duration_str = value.evaluate()
            parts = regex.match(duration_str)
            if not parts:
                raise DslSyntaxError('invalid time delta string', duration_str, node=self.node)
            parts = parts.groupdict()
            params = dict((name, int(param)) for (name, param) in six.iteritems(parts) if param)
            return relativedelta(**params)
        elif isinstance(value, datetime.timedelta):
            return value
        elif isinstance(value, relativedelta):
            return value
        else:
            raise DslSystemError("shouldn't get here", value, node=self.node)

    def __sub__(self, other):
        raise Exception(str(other))

    def __add__(self, other):
        raise Exception(str(other))


class SnapToMonth(DslExpression):

    def __str__(self, indent=0):
        return "SnapToMonth({})".format(str(self._args[0]))

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(Date, Name))
        self.assert_args_arg(args, posn=1, required_type=(Number, six.integer_types))
        assert args[1].evaluate() < 29, DslSyntaxError("Snap day must be less than or equal to 28", node=self.node)

    def evaluate(self, **kwds):
        value = self._args[0].evaluate(**kwds)
        snap_day = self._args[1].evaluate()
        assert isinstance(value, datetime.date)
        year = value.year
        month = value.month
        day = value.day
        if day < snap_day:
            if month == 1:
                month = 12
                year -= 1
            else:
                month += 1
        return datetime.date(year, month, snap_day)


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
            # assert isinstance(dsl_expr, DslExpression)
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
            a = scipy.array([a] * len(b))
        elif bIsaNumber and (not aIsaNumber):
            # Todo: Optimise with scipy.zeros() when b equals zero?
            b = scipy.array([b] * len(a))
        c = scipy.array([a, b])
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
        Replace commodity_name with named value in context (kwds).
        """

        combined_namespace = DslNamespace(itertools.chain(dsl_globals.items(), dsl_locals.items()))

        value = self.evaluate(**combined_namespace)
        if isinstance(value, datetime.date):
            return Date(value, node=self.node)
        elif isinstance(value, DslObject):
            return value
        elif isinstance(value, six.integer_types + (float, ndarray)):
            return Number(value, node=self.node)
        elif isinstance(value, six.string_types):
            return String(value, node=self.node)
        elif isinstance(value, datetime.timedelta):
            return TimeDelta(value, node=self.node)
        elif isinstance(value, relativedelta):
            return TimeDelta(value, node=self.node)
        # elif isinstance(value, (SynchronizedArray, Synchronized)):
        #     return Number(numpy_from_sharedmem(value), node=self.node)
        else:
            raise DslSyntaxError("expected number, string, date, time delta, or DSL object when reducing name '%s'"
                                 "" % self.name, repr(value), node=self.node)

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
            pass
            # assert isinstance(dsl_globals, DslNamespace)
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

            # Create a new stub - the stub ID is the name of the return value of the function call..
            stub_id = create_uuid4()
            dsl_stub = Stub(stub_id, node=self.node)

            # Put the function call on the call stack, with the stub ID.
            # assert isinstance(pending_call_stack, PendingCallQueue)
            pending_call_stack.put(
                stub_id=stub_id,
                stacked_function_def=self,
                stacked_locals=dsl_locals.copy(),
                stacked_globals=dsl_globals.copy(),
                effective_present_time=effective_present_time
            )
            # Return the stub so that the containing DSL can be fully evaluated
            # once the stacked function call has been evaluated.
            dsl_expr = dsl_stub
        else:
            # Todo: Make sure the expression can be selected with the dsl_locals?
            # - ie the conditional expressions should be functions only of call arg
            # values that can be fully evaluated without evaluating contractual DSL objects.
            selected_expression = self.select_expression(self.body, dsl_locals)

            # Add this function to the dslNamespace (just in case it's called by itself).
            new_dsl_globals = DslNamespace(dsl_globals)
            new_dsl_globals[self.name] = self

            # Reduce the selected expression.
            # assert isinstance(selected_expression, DslExpression)
            dsl_expr = selected_expression.reduce(
                dsl_locals=dsl_locals,
                dsl_globals=new_dsl_globals,
                effective_present_time=effective_present_time,
                pending_call_stack=pending_call_stack
            )

        # Cache the result.
        if not is_destacking:
            self.call_cache[call_cache_key] = dsl_expr

        return dsl_expr

    def select_expression(self, dsl_expr, call_arg_namespace):
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
            selected = self.select_expression(selected, call_arg_namespace)
        else:
            selected = dsl_expr
        return selected

    def create_hash(self, obj):
        if isinstance(obj, relativedelta):
            return hash(repr(obj))
        if isinstance(obj, (int, float, six.string_types, datetime.datetime, datetime.date, datetime.timedelta, relativedelta)):
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

    def __str__(self, indent=0):
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
        functionDef = self.functionDefName.reduce(dsl_locals, dsl_globals, effective_present_time,
                                                  pending_call_stack=pending_call_stack)

        # Function def name (a Name object) should have reduced to a FunctionDef object in the namespace.
        # - it's an error for the name to be defined as anything other than a function, but that's not possible here?
        # assert isinstance(functionDef, FunctionDef)

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
                    # assert isinstance(callArgExpr, DslExpression)
                    # It's a sum of two constants, or something like that - evaluate the full expression.
                    callArgValue = callArgExpr.evaluate()

            # Add the call arg value to the new call arg namespace.
            newDslLocals[callArgDef.name] = callArgValue

        # Evaluate the function def with the dict of call arg values.
        dsl_expr = functionDef.apply(dsl_globals, effective_present_time, pending_call_stack=pending_call_stack, is_destacking=False, **newDslLocals)

        # The result of this function call (stubbed or otherwise) should be a DSL expression.
        # assert isinstance(dsl_expr, DslExpression)

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
    def dsl_typeName(self):
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
        indentation = indent * 4 * ' '

        msg = "\n"
        msg += indentation + "if %s:\n" % self.test
        msg += indentation + "    %s\n" % self.body

        msg += self.orelse_to_str(self.orelse, indentation)
        return msg

    def orelse_to_str(self, orelse, indentation):
        msg = ''
        if isinstance(orelse, If):
            msg += indentation + "elif %s:\n" % orelse.test
            msg += indentation + "    %s\n" % orelse.body
            # Recurse down "linked list" of alternatives...
            msg += self.orelse_to_str(orelse.orelse, indentation)
        else:
            # ...until we reach the final alternative.
            msg += indentation + "else:\n"
            msg += indentation + "    %s\n"% orelse
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

    def __init__(self, *args, **kwds):
        super(Module, self).__init__(*args, **kwds)

    def __str__(self, indent=0):
        return "\n".join([str(statement) for statement in self.body])

    def validate(self, args):
        self.assert_args_len(args, 1)
        self.assert_args_arg(args, 0, [(FunctionDef, DslExpression, Date)])

    @property
    def body(self):
        return self._args[0]


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
            if isinstance(date, datetime.date):
                pass
            if isinstance(date, six.string_types):
                date = String(date)
            if isinstance(date, String):
                date = Date(date, node=date.node)
            if isinstance(date, (SnapToMonth, Date, BinOp)):
                date = date.evaluate()
            if not isinstance(date, datetime.date):
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
    'SnapToMonth': SnapToMonth,
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


# Todo: Add something to Map a contract function to a sequence of values (range, list comprehension).

class AbstractMarket(StochasticObject, DslExpression):
    PERTURBATION_FACTOR = 0.001

    def evaluate(self, **kwds):
        # Get the perturbed market name, if set.
        active_perturbation = kwds.get('active_perturbation', None)

        # Get the effective present time (needed to form the simulated_value_id).
        try:
            present_time = kwds['present_time']
        except KeyError:
            raise DslSyntaxError(
                "'present_time' not found in evaluation kwds" % self.market_name,
                ", ".join(kwds.keys()),
                node=self.node
            )

        # Get the dict of simulated values.
        try:
            simulated_value_dict = kwds['simulated_value_dict']
        except KeyError:
            raise DslError(
                "Not found 'simulated_value_dict' in context variables" % self.market_name,
                ", ".join(kwds.keys()),
                node=self.node
            )

        # Make the simulated price ID.
        simulated_price_id = make_simulated_price_id(kwds['simulation_id'], self.commodity_name, present_time, self.delivery_date or present_time)

        # Get the value from the dict of simulated values.
        try:
            simulated_price_value = simulated_value_dict[simulated_price_id]
        except KeyError:
            raise DslError("Simulated price not found ID: {}".format(simulated_price_id))

        # If this is a perturbed market, perturb the simulated value.
        if self.get_perturbation(present_time) == active_perturbation:
            evaluated_value = simulated_price_value * (1 + Market.PERTURBATION_FACTOR)
        else:
            evaluated_value = simulated_price_value
        return evaluated_value

    @property
    def market_name(self):
        return self.commodity_name

    @property
    def delivery_date(self):
        return None

    @property
    def commodity_name(self):
        return self._args[0].evaluate() if isinstance(self._args[0], String) else self._args[0]

    def identify_price_simulation_requirements(self, requirements, **kwds):
        assert isinstance(requirements, set)
        # Get the effective present time (needed to form the simulation requirement).
        try:
            present_time = kwds['present_time']
        except KeyError:
            raise DslSyntaxError(
                "'present_time' not found in evaluation kwds" % self.market_name,
                ", ".join(kwds.keys()),
                node=self.node
            )
        fixing_date = present_time
        requirement = (self.commodity_name, fixing_date, self.delivery_date or present_time)
        requirements.add(requirement)
        super(AbstractMarket, self).identify_price_simulation_requirements(requirements, **kwds)

    def identify_perturbation_dependencies(self, dependencies, **kwds):
        try:
            present_time = kwds['present_time']
        except KeyError:
            raise DslSyntaxError(
                "'present_time' not found in evaluation kwds" % self.market_name,
                ", ".join(kwds.keys()),
                node=self.node
            )
        perturbation = self.get_perturbation(present_time)
        dependencies.add(perturbation)
        super(AbstractMarket, self).identify_perturbation_dependencies(dependencies, **kwds)

    def get_perturbation(self, present_time):
        # For now, just bucket by commodity name and month of delivery.
        delivery_date = self.delivery_date or present_time
        perturbation = json.dumps((self.commodity_name, delivery_date.year, delivery_date.month))
        return perturbation


class Market(AbstractMarket):

    def validate(self, args):
        self.assert_args_len(args, required_len=1)
        self.assert_args_arg(args, posn=0, required_type=(six.string_types, String, Name))


class ForwardMarket(AbstractMarket):

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(six.string_types, String, Name))
        self.assert_args_arg(args, posn=1, required_type=(String, Date, Name, BinOp, SnapToMonth))

    @property
    def delivery_date(self):
        # Todo: Refactor this w.r.t. the Settlement.date property.
        if not hasattr(self, '_delivery_date'):
            date = self._args[1]
            if isinstance(date, Name):
                raise DslSyntaxError("date value name '%s' must be resolved to a datetime before it can be used" % date.name, node=self.node)
            if isinstance(date, datetime.date):
                pass
            if isinstance(date, six.string_types):
                date = String(date)
            if isinstance(date, String):
                date = Date(date, node=date.node)
            if isinstance(date, (SnapToMonth, Date, BinOp)):
                date = date.evaluate()
            if not isinstance(date, datetime.date):
                raise DslSyntaxError("delivery date value should be a datetime.datetime by now, but it's a %s" % date, node=self.node)
            self._delivery_date = date
        return self._delivery_date


class Settlement(StochasticObject, DatedDslObject, DslExpression):
    """
    Discounts value of expression to 'present_time'.
    """

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(String, Date, SnapToMonth, Name, BinOp))
        self.assert_args_arg(args, posn=1, required_type=DslExpression)

    def evaluate(self, **kwds):
        value = self._args[1].evaluate(**kwds)
        return self.discount(value, self.date, **kwds)


class Fixing(StochasticObject, DatedDslObject, DslExpression):
    """
    A fixing defines the 'present_time' used for evaluating its expression.
    """

    def __str__(self, indent=0):
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
        fixing_date = self._args[0]
        if isinstance(fixing_date, datetime.datetime):
            pass
        if isinstance(fixing_date, six.string_types):
            fixing_date = String(fixing_date)
        if isinstance(fixing_date, String):
            fixing_date = Date(fixing_date, node=fixing_date.node)
        if isinstance(fixing_date, (Date, BinOp, Name)):
            fixing_date = fixing_date.evaluate(**dsl_locals)
        if not isinstance(fixing_date, datetime.date):
            raise DslSyntaxError("fixing date value should be a datetime.date by now, but it's a %s" % fixing_date, node=self.node)
        effective_present_time = fixing_date
        return super(Fixing, self).reduce(dsl_locals, dsl_globals, effective_present_time, pending_call_stack=pending_call_stack)

    def evaluate(self, **kwds):
        kwds = kwds.copy()
        kwds['present_time'] = self.date
        return self.expr.evaluate(**kwds)

    def identify_price_simulation_requirements(self, requirements, **kwds):
        kwds['present_time'] = self.date
        super(Fixing, self).identify_price_simulation_requirements(requirements, **kwds)

    def identify_perturbation_dependencies(self, dependencies, **kwds):
        kwds['present_time'] = self.date
        super(Fixing, self).identify_perturbation_dependencies(dependencies, **kwds)



class On(Fixing):
    """
    A shorter name for Fixing.
    """


class Wait(Fixing):
    """
    A fixing with discounting of the resulting value from date arg to present_time.

    Wait(date, expr) == Settlement(date, Fixing(date, expr))
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
        # Run the least-squares monte-carlo routine.
        present_time = kwds['present_time']
        first_commodity_name = kwds['first_commodity_name']
        simulated_value_dict = kwds['simulated_value_dict']

        simulation_id = kwds['simulation_id']
        initial_state = LongstaffSchwartzState(self, present_time)
        final_states = [LongstaffSchwartzState(a, present_time) for a in self._args]
        longstaff_schwartz = LongstaffSchwartz(initial_state, final_states, first_commodity_name,
                                               simulated_value_dict, simulation_id)
        result = longstaff_schwartz.evaluate(**kwds)
        return result

    def identify_price_simulation_requirements(self, requirements, **kwds):
        present_time = kwds['present_time']
        for dsl_market in self.list_instances(AbstractMarket):
            requirements.add((dsl_market.commodity_name, present_time, present_time))
        super(Choice, self).identify_price_simulation_requirements(requirements, **kwds)


class LongstaffSchwartz(object):
    """
    Implements a least-squares Monte Carlo simulation, following the Longstaff-Schwartz paper
    on valuing American options (for reference, see Quant DSL paper).
    """
    def __init__(self, initial_state, subsequent_states, first_commodity_name, simulated_price_dict, simulation_id):
        self.initial_state = initial_state
        for subsequent_state in subsequent_states:
            self.initial_state.add_subsequent_state(subsequent_state)
        self.states = None
        self.states_by_time = None
        self.first_commodity_name = first_commodity_name
        self.simulated_price_dict = simulated_price_dict
        self.simulation_id = simulation_id

    def evaluate(self, **kwds):
        all_states = self.get_states()
        all_states.reverse()
        value_of_being_in = {}
        for state in all_states:
            # assert isinstance(state, LongstaffSchwartzState)
            len_subsequent_states = len(state.subsequent_states)
            state_value = None
            if len_subsequent_states > 1:
                conditional_expected_values = []
                expected_continuation_values = []

                for subsequent_state in state.subsequent_states:
                    regression_variables = []
                    dsl_markets = subsequent_state.dsl_object.list_instances(AbstractMarket)
                    market_names = set([m.commodity_name for m in dsl_markets])
                    for market_name in market_names:
                        market_price = self.get_simulated_value(market_name, state.time)
                        regression_variables.append(market_price)
                    # Todo: Either use or remove 'get_payoff()', payoffValue not used ATM.
                    #payoffValue = self.get_payoff(state, subsequent_state)
                    expected_continuation_value = value_of_being_in[subsequent_state]
                    expected_continuation_values.append(expected_continuation_value)
                    if len(regression_variables):
                        conditional_expected_value = LeastSquares(regression_variables, expected_continuation_value).fit()
                    else:
                        conditional_expected_value = expected_continuation_value
                    conditional_expected_values.append(conditional_expected_value)

                conditional_expected_values = scipy.array(conditional_expected_values)
                expected_continuation_values = scipy.array(expected_continuation_values)
                argmax = conditional_expected_values.argmax(axis=0)
                offsets = scipy.array(range(0, conditional_expected_values.shape[1])) * conditional_expected_values.shape[0]
                indices = argmax + offsets
                # assert indices.shape == underlying_value.shape
                state_value = expected_continuation_values.transpose().take(indices)
                # assert state_value.shape == underlying_value.shape
            elif len_subsequent_states == 1:
                subsequent_state = state.subsequent_states.pop()
                state_value = value_of_being_in[subsequent_state]
            elif len_subsequent_states == 0:
                state_value = state.dsl_object.evaluate(**kwds)
                if isinstance(state_value, (int, float)):
                    # underlying_value = self.get_simulated_value(self.first_commodity_name, state.time)
                    path_count = kwds['path_count']
                    ones = scipy.ones(path_count)
                    state_value = ones * state_value
                if not isinstance(state_value, scipy.ndarray):
                    raise Exception("State value type is '%s' when scipy.ndarray is required: %s" % (
                        type(state_value), state_value))
            value_of_being_in[state] = state_value
        return value_of_being_in[self.initial_state]

    def get_simulated_value(self, market_name, price_time):
        simulated_price_id = make_simulated_price_id(self.simulation_id, market_name, price_time, price_time)
        try:
            simulated_price = self.simulated_price_dict[simulated_price_id]
        except KeyError:
            msg = "Simulated price ID {} not in simulated price dict keys: {}".format(simulated_price_id, self.simulated_price_dict.keys())
            raise KeyError(msg)
        return simulated_price
        # underlying_value = numpy_from_sharedmem(simulated_price)
        # return underlying_value

    def get_times(self):
        return self.get_states_by_time().keys()

    def get_states_at_time(self, time):
        return self.get_states_by_time()[time]

    def get_states_by_time(self):
        if self.states_by_time is None:
            self.states_by_time = {}
            for state in self.get_states():
                if state.time not in self.states_by_time:
                    self.states_by_time[state.time] = []
                self.states_by_time[state.time].append(state)
        return self.states_by_time

    def get_states(self):
        if self.states is None:
            self.states = self.find_states(self.initial_state)
        return self.states

    def find_states(self, state):
        # assert isinstance(state, LongstaffSchwartzState)
        states = [state]
        for subsequentState in state.subsequent_states:
            states += self.find_states(subsequentState)
        return states

    def get_payoff(self, state, nextState):
        return 0


class LongstaffSchwartzState(object):
    """
    Object to represent state in the Longstaff-Schwartz routine.
    """

    def __init__(self, dsl_object, time):
        self.subsequent_states = set()
        self.dsl_object = dsl_object
        self.time = time

    def add_subsequent_state(self, state):
        self.subsequent_states.add(state)


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
        regressions = list()
        # Regress against unity.
        regressions.append(scipy.ones(self.path_count))
        # Regress against each variable.
        for x in self.xs:
            regressions.append(x)
        # Regress against squares and cross products.
        indices = range(0, len(self.xs))
        for i in indices:
            square = self.xs[i] * self.xs[i]
            regressions.append(square)
        for i, j in itertools.combinations(indices, 2):
            product = self.xs[i] * self.xs[j]
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
        d = a * c
        #print "d: ", d
        #print "d: ", d.shape, type(d)
        #print "d A1: ", d.getA1()
        return d.getA1()

    def solve(self, a, b):
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
    'ForwardMarket': ForwardMarket,
    'On': On,
    'Settlement': Settlement,
    'Wait': Wait,
})


PendingCall = namedtuple('PendingCall', ['stub_id', 'stacked_function_def', 'stacked_locals', 'stacked_globals',
                                         'effective_present_time'])



class PendingCallQueue(object):

    def put(self, stub_id, stacked_function_def, stacked_locals, stacked_globals, effective_present_time):
        pending_call = self.validate_pending_call(effective_present_time, stacked_function_def, stacked_globals,
                                                  stacked_locals, stub_id)
        self.put_pending_call(pending_call)

    def validate_pending_call(self, effective_present_time, stacked_function_def, stacked_globals, stacked_locals,
                              stub_id):
        # assert isinstance(stub_id, six.string_types), type(stub_id)
        # assert isinstance(stacked_function_def, FunctionDef), type(stacked_function_def)
        # assert isinstance(stacked_locals, DslNamespace), type(stacked_locals)
        # assert isinstance(stacked_globals, DslNamespace), type(stacked_globals)
        # assert isinstance(effective_present_time, (datetime.datetime, type(None))), type(effective_present_time)
        pending_call = PendingCall(stub_id, stacked_function_def, stacked_locals, stacked_globals,
                                   effective_present_time)
        return pending_call

    @abstractmethod
    def put_pending_call(self, pending_call):
        pass

    @abstractmethod
    def get(self):
        pass


class PythonPendingCallQueue(PendingCallQueue):

    def __init__(self):
        self.queue = queue.Queue()

    def put_pending_call(self, pending_call):
        self.queue.put(pending_call)

    def empty(self):
        return self.queue.empty()

    def get(self, *args, **kwargs):
        return self.queue.get(*args, **kwargs)


def compile_dsl_module(dsl_module, dsl_locals=None, dsl_globals=None, is_dependency_graph=None):
    """
    Returns something that can be evaluated.
    """

    # It's a module compilation, so create a new namespace "context".
    if dsl_locals is None:
        dsl_locals = {}
    dsl_locals = DslNamespace(dsl_locals)
    if dsl_globals is None:
        dsl_globals = {}
    dsl_globals = DslNamespace(dsl_globals)

    # Can't do much with an empty module.
    if len(dsl_module.body) == 0:
        raise DslSyntaxError('empty module', node=dsl_module.node)

    function_defs, expressions = extract_defs_and_exprs(dsl_module, dsl_globals)

    # Handle different combinations of functions and module level expressions in different ways.
    # Todo: Simplify this, but support library files first?
    # Can't meaningfully evaluate more than one expression (since assignments are not supported).
    if len(expressions) > 1:
        raise DslSyntaxError('more than one expression in module', node=expressions[1].node)

    # Can't meaningfully evaluate more than one function def without a module level expression.
    elif len(expressions) == 0 and len(function_defs) > 1:
        second_def = function_defs[1]
        raise DslSyntaxError('more than one function def in module without an expression', '"def %s"' % second_def.name, node=function_defs[1].node)

    # If it's just a module with one function, then return the function def.
    elif len(expressions) == 0 and len(function_defs) == 1:
        return function_defs[0]

    # If there is one expression, reduce it with the function defs that it calls.
    elif len(expressions) == 1:
        dsl_expr = expressions[0]
        # assert isinstance(dsl_expr, DslExpression), dsl_expr
        # - if a dependency graph is required, then "reduce" into stub expressions
        if is_dependency_graph:
            # Compile the module as a dependency graph.

            from quantdsl.domain.services import uuids
            root_stub_id = uuids.create_uuid4()
            # stubbed_calls = generate_stubbed_calls(root_stub_id, dsl_module, dsl_expr, dsl_globals, dsl_locals)
            raise NotImplementedError("")
            # requirements, dependents, leaf_ids = extract_graph_structure(stubbed_calls)
            # call_requirements = call_requirements_from_stubbed_calls(stubbed_calls)


        else:
            # Compile the module for a single threaded recursive operation (faster but not distributed,
            # so also limited in space and perhaps time). For smaller computations only.
            dsl_obj = dsl_expr.reduce(dsl_locals, DslNamespace(dsl_globals))
        return dsl_obj

    else:
        raise DslSyntaxError("shouldn't get here", node=dsl_module.node)


def extract_defs_and_exprs(dsl_module, dsl_globals):
    # Pick out the expressions and function defs from the module body.
    function_defs = []
    expressions = []
    for dsl_obj in dsl_module.body:

        if isinstance(dsl_obj, FunctionDef):
            # Todo: Move this setting of globals elsewhere, it doesn't belong here.
            dsl_globals[dsl_obj.name] = dsl_obj
            # Todo: Move this setting of the 'enclosed namespace' - is this even a good idea?
            # Share the module level namespace (any function body can call any other function).
            dsl_obj.enclosed_namespace = dsl_globals

            function_defs.append(dsl_obj)
        elif isinstance(dsl_obj, DslExpression):
            expressions.append(dsl_obj)
        else:
            raise DslSyntaxError("'%s' not allowed in module" % type(dsl_obj), dsl_obj, node=dsl_obj.node)

    return function_defs, expressions


def generate_stubbed_calls(root_stub_id, dsl_module, dsl_expr, dsl_globals, dsl_locals):
    # Create a stack of discovered calls to function defs.
    # - since we are basically doing a breadth-first search, the pending call queue
    #   will be the max width of the graph, so it might sometimes be useful to
    #   persist the queue to allow for larger graph. For now, just use a Python queue.
    pending_call_stack = PythonPendingCallQueue()

    # Reduce the module object into a "root" stubbed expression with pending calls on the stack.
    # - If an expression has a FunctionCall, it will cause a pending
    # call to be placed on the pending call stack, and the function call will be
    # replaced with a stub, which acts as a placeholder for the result of the function
    # call. By looping over the pending call stack until it is empty, evaluating
    # pending calls to generate stubbed expressions and further pending calls, the
    # module can be compiled into a stack of stubbed expressions.
    # Of course if the module's expression doesn't have a function call, there
    # will just be one expression on the stack of "stubbed" expressions, and it will
    # not have any stubs, and there will be no pending calls on the pending call stack.
    stubbed_expr = dsl_expr.reduce(
        dsl_locals,
        DslNamespace(dsl_globals),
        pending_call_stack=pending_call_stack
    )

    dependencies = list_stub_dependencies(stubbed_expr)
    yield StubbedCall(root_stub_id, stubbed_expr, None, dependencies)

    # Continue by looping over any pending calls.
    while not pending_call_stack.empty():
        # Get the next pending call.
        pending_call = pending_call_stack.get()
        # assert isinstance(pending_call, PendingCall), pending_call

        # Get the function def.
        function_def = pending_call.stacked_function_def
        # assert isinstance(function_def, FunctionDef), type(function_def)

        # Apply the stacked call values to the called function def.
        stubbed_expr = function_def.apply(pending_call.stacked_globals,
                                          pending_call.effective_present_time,
                                          pending_call_stack=pending_call_stack,
                                          is_destacking=True,
                                          **pending_call.stacked_locals)

        # Put the resulting (potentially stubbed) expression on the stack of stubbed expressions.
        dependencies = list_stub_dependencies(stubbed_expr)

        yield StubbedCall(pending_call.stub_id, stubbed_expr, pending_call.effective_present_time, dependencies)


def list_stub_dependencies(stubbed_expr):
    return [s.name for s in stubbed_expr.list_instances(Stub)]


def list_fixing_dates(dsl_expr):
    # Find all unique fixing dates.
    return sorted(list(find_fixing_dates(dsl_expr)))


def find_fixing_dates(dsl_expr):
    for dsl_fixing in dsl_expr.find_instances(dsl_type=Fixing):
        # assert isinstance(dsl_fixing, Fixing)
        if dsl_fixing.date is not None:
            yield dsl_fixing.date


def find_delivery_points(dsl_expr):
    # Find all unique market names.
    all_delivery_points = set()
    for dsl_market in dsl_expr.find_instances(dsl_type=AbstractMarket):
        # assert isinstance(dsl_market, Market)
        delivery_point = dsl_market.get_delivery_point()
        if delivery_point not in all_delivery_points:  # Deduplicate.
            all_delivery_points.add(delivery_point)
            yield delivery_point
