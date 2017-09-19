from __future__ import division

import datetime
import itertools
import math
import re
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import scipy
import scipy.linalg
import six
from dateutil.relativedelta import relativedelta
from scipy import ndarray

from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.domain.services.uuids import create_uuid4
from quantdsl.exceptions import DslError, DslNameError, DslSyntaxError, DslSystemError
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
        self._hash = None

    def __str__(self):
        """
        Returns DSL source code, that can be parsed to generate a clone of self.
        """
        return self.pprint()

    # Todo: More tests that this round trip actually works.
    def pprint(self, indent=''):
        """Returns Quant DSL source code for the DSL object."""
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
        indent = indent[:-tab]
        if lenArgs > 1:
            msg += indent
        msg += ")"
        return msg

    @property
    def hash(self):
        """
        Creates a hash that is unique for this fragment of DSL.
        """
        if self._hash is None:
            hashes = ""
            for arg in self._args:
                if isinstance(arg, list):
                    arg = tuple(arg)
                hashes += str(hash(arg))
            self._hash = hash(hashes)
        return self._hash

    def __hash__(self):
        return self.hash

    @abstractmethod
    def validate(self, args):
        """
        Raises an exception if the object's args are not valid.
        """

    # Todo: Rework validation, perhaps by considering a declarative form in which to express the requirements.
    def assert_args_len(self, args, required_len=None, min_len=None, max_len=None):
        if min_len != None and len(args) < min_len:
            error = "%s is broken" % self.__class__.__name__
            descr = "requires at least %s arguments (%s were given)" % (min_len, len(args))
            raise DslSyntaxError(error, descr, self.node)
        if max_len != None and len(args) > max_len:
            error = "%s is broken" % self.__class__.__name__
            descr = "requires at most %s arguments (%s were given)" % (max_len, len(args))
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
            if isinstance(required_type, (list, tuple)):
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
                    # elif isinstance(arg, list):
                    #     for arg in arg:
                    #         if isinstance(arg, DslObject):
                    #             for dsl_obj in arg.list_instances(dsl_type):
                    #                 yield dsl_obj

    def reduce(self, dsl_locals, dsl_globals, effective_present_time=None, pending_call_stack=None):
        """
        Reduces by reducing all args, and then using those args
        to create a new instance of self.
        """
        new_dsl_args = []
        for dsl_arg in self._args:
            if isinstance(dsl_arg, DslObject):
                dsl_arg = dsl_arg.reduce(dsl_locals, dsl_globals, effective_present_time,
                                         pending_call_stack=pending_call_stack)
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
        """
        Returns the value of the expression.
        """

    def discount(self, value, date, **kwds):
        r = float(kwds['interest_rate']) / 100
        T = get_duration_years(kwds['present_time'], date)
        return value * math.exp(- r * T)


class DslConstant(DslExpression):
    required_type = None

    def pprint(self, indent=''):
        return repr(self.value)

    def validate(self, args):
        self.assert_args_len(args, required_len=1)
        assert self.required_type is not None, "required_type attribute not set on %s" % self.__class__
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

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)


class String(DslConstant):
    required_type = six.string_types


class Number(DslConstant):
    required_type = six.integer_types + (float, ndarray)


class Date(DslConstant):
    required_type = six.string_types + (String, datetime.date, datetime.datetime)

    def pprint(self, indent=''):
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
                return datetime.datetime(year, month, day)
                # return dateutil.parser.parse(date_str).replace()
            except ValueError:
                raise DslSyntaxError("invalid date string", date_str, node=self.node)
        elif isinstance(value, datetime.datetime):
            return value
        elif isinstance(value, datetime.date):
            return datetime.datetime(value.year, value.month, value.day)


class TimeDelta(DslConstant):
    required_type = (String, datetime.timedelta, relativedelta)

    def pprint(self, indent=''):
        return "{}({})".format(self.__class__.__name__, self._args[0])

    def parse(self, value, regex=re.compile(r'((?P<days>\d+?)d|(?P<months>\d+?)m|(?P<years>\d+?)y)?')):
        if isinstance(value, String):
            duration_str = value.evaluate()
            parts = regex.match(duration_str)
            parts = parts.groupdict()
            params = dict((name, int(param)) for (name, param) in six.iteritems(parts) if param)
            if not params:
                raise DslSyntaxError('invalid "time delta" string', duration_str, node=self.node)
            return relativedelta(**params)
        elif isinstance(value, datetime.timedelta):
            return value
        elif isinstance(value, relativedelta):
            return value
        else:
            raise DslSystemError("shouldn't get here", value, node=self.node)


class UnaryOp(DslExpression):
    opchar = None

    def pprint(self, indent=''):
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
        """
        Returns the result of operating on the given value.
        """


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
            if self.op(value):
                return self.op(True)
        return self.op(False)

    @abstractmethod
    def op(self, value):
        """
        Returns value, or not value, according to implementation.
        """

    def pprint(self, indent=''):
        operator = self.__class__.__name__.lower()
        padded = ' ' + operator + ' '
        text = padded.join([str(i) for i in self._args[0]])
        return indent + '(' + text + ')'


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
        """
        Returns result of operating on two args.
        """

    def pprint(self, indent=''):
        if self.opchar:
            def makeStr(dsl_expr):
                dslString = str(dsl_expr)
                if isinstance(dsl_expr, BinOp):
                    dslString = "(" + dslString + ")"
                return dslString

            text = makeStr(self.left) + " " + self.opchar + " " + makeStr(self.right)
        else:
            text = '%s(%s, %s)' % (self.__class__.__name__, self.left, self.right)
        return indent + text

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


# Todo: Pow, Mod, FloorDiv don't have proofs, so shouldn't really be used for combining random variables? Either
# prevent usage with ndarray inputs, or do the proofs. :-)

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


class NonInfixedBinOp(BinOp):
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
            return self.scalar_op(a, b)
        elif (not aIsaNumber) and (not bIsaNumber):
            # Both are vectors.
            msg = "Vectors have different length: %s and %s" % (len(a), len(b))
            assert len(a) == len(b), msg
        elif aIsaNumber and (not bIsaNumber):
            # Todo: Optimise with scipy.zeros() when a equals zero?
            a = scipy.array([a] * len(b))
        elif bIsaNumber and (not aIsaNumber):
            # Todo: Optimise with scipy.zeros() when b equals zero?
            b = scipy.array([b] * len(a))
        return self.vector_op(a, b)

    @abstractmethod
    def vector_op(self, a, b):
        """Computes result of operation on vector values."""

    @abstractmethod
    def scalar_op(self, a, b):
        """Computes result of operation on scalar values."""


class Min(NonInfixedBinOp):
    def vector_op(self, a, b):
        return scipy.array([a, b]).min(axis=0)

    def scalar_op(self, a, b):
        return min(a, b)


class Max(NonInfixedBinOp):
    def vector_op(self, a, b):
        return scipy.array([a, b]).max(axis=0)

    def scalar_op(self, a, b):
        return max(a, b)


class Name(DslExpression):
    def pprint(self, indent=''):
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

    def pprint(self, indent=''):
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

    def pprint(self, indent=''):
        msg = ""
        for decorator_name in self.decorator_names:
            msg += "@" + decorator_name + "\n"
        msg += "def %s(%s):\n" % (self.name, ", ".join(self.call_arg_names))
        if isinstance(self.body, DslObject):
            try:
                msg += self.body.pprint(indent=indent + '    ')
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

        # Second attempt to implement module namespaces...
        self.module_namespace = None

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

    def apply(self, dsl_globals=None, effective_present_time=None, pending_call_stack=None, is_destacking=False,
              **dsl_locals):
        # It's a function call, so create a new namespace "context".
        if dsl_globals is None:
            dsl_globals = DslNamespace()
        else:
            pass
            # assert isinstance(dsl_globals, DslNamespace)
        dsl_globals = DslNamespace(itertools.chain(self.enclosed_namespace.items(), self.module_namespace.items(),
                                                   dsl_globals.items()))
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

        if pending_call_stack and not is_destacking and not 'inline' in self.decorator_names:
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
        if isinstance(obj, (
                int, float, six.string_types, datetime.datetime, datetime.date, datetime.timedelta, relativedelta)):
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
    def pprint(self, indent=''):
        return indent + "%s(%s)" % (self.functionDefName,
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
                callArgExpr = callArgExpr.reduce(dsl_locals, dsl_globals, effective_present_time,
                                                 pending_call_stack=pending_call_stack)
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
        dsl_expr = functionDef.apply(dsl_globals, effective_present_time, pending_call_stack=pending_call_stack,
                                     is_destacking=False, **newDslLocals)

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
    def pprint(self, indent=''):
        msg = "\n"
        msg += indent + "if %s:\n" % self.test
        msg += indent + "    %s\n" % self.body
        msg += self.orelse_to_str(self.orelse, indent)
        return msg

    def orelse_to_str(self, orelse, indent):
        msg = ''
        if isinstance(orelse, If):
            msg += indent + "elif %s:\n" % orelse.test
            msg += indent + "    %s\n" % orelse.body
            # Recurse down "linked list" of alternatives...
            msg += self.orelse_to_str(orelse.orelse, indent)
        else:
            # ...until we reach the final alternative.
            msg += indent + "else:\n"
            msg += indent + "    %s\n" % orelse
        return msg


class IfExp(If):
    """
    Special case of If, where if-else clause is one expression (no elif support).
    """

    def pprint(self, indent=''):
        return indent + "%s if %s else %s" % (self.body, self.test, self.orelse)


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

    def pprint(self, indent=''):
        return indent + str(self.left) + ' ' + " ".join(
            [str(self.opcodes[op]) + ' ' + str(right) for (op, right) in zip(self.op_names, self.comparators)]
        )

    def validate(self, args):
        self.assert_args_len(args, 3)
        self.assert_args_arg(args, 0, required_type=(
            DslExpression, Date))  # , Date, Number, String, int, float, six.string_types, datetime.datetime))
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
    A DSL module has a body, which is a list of DSL statements either
    function defs or expressions.
    """

    def __init__(self, *args, **kwds):
        super(Module, self).__init__(*args, **kwds)

    def pprint(self, indent=''):
        return indent + "\n".join([str(statement) for statement in self.body])

    def validate(self, args):
        self.assert_args_len(args, 2)
        self.assert_args_arg(args, 0, [(FunctionDef, DslExpression, Date)])
        self.assert_args_arg(args, 1, DslNamespace)

    @property
    def body(self):
        return self._args[0]

    @property
    def namespace(self):
        return self._args[1]


def inline(*args):
    """
    Dummy 'inline' Quant DSL decorator - we just want the name in the namespace.
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
        """
        Returns value of stochastic object.
        """


class DatedDslObject(DslObject):
    @property
    def date(self):
        if not hasattr(self, '_date'):
            date = self._args[0]
            if isinstance(date, Name):
                raise DslSyntaxError(
                    "date value name '%s' must be resolved to a datetime before it can be used" % date.name,
                    node=self.node)
            if isinstance(date, datetime.date):
                pass
            if isinstance(date, six.string_types):
                date = String(date)
            if isinstance(date, String):
                date = Date(date, node=date.node)
            if isinstance(date, (Date, BinOp)):
                date = date.evaluate()
            if not isinstance(date, datetime.date):
                raise DslSyntaxError("date value should be a datetime.datetime by now, but it's a %s" % date,
                                     node=self.node)
            self._date = date
        return self._date


class Lift(DslExpression):
    def validate(self, args):
        self.assert_args_len(args, min_len=2, max_len=3)
        # Name of a commodity to be perturbed.
        self.assert_args_arg(args, posn=0, required_type=(String, Name))
        if len(args) == 2:
            # Expression to be perturbed.
            self.assert_args_arg(args, posn=1, required_type=DslExpression)
        elif len(args) == 3:
            # Periodization of the perturbation.
            self.assert_args_arg(args, posn=1, required_type=(String, Name))
            # Expression to be perturbed.
            self.assert_args_arg(args, posn=2, required_type=DslExpression)

    @property
    def commodity_name(self):
        return self._args[0]

    @property
    def mode(self):
        return self._args[1] if len(self._args) == 3 else String('alltime')

    @property
    def expr(self):
        return self._args[-1]

    def identify_perturbation_dependencies(self, dependencies, **kwds):
        perturbation = self.get_perturbation(**kwds)
        dependencies.add(perturbation)
        super(Lift, self).identify_perturbation_dependencies(dependencies, **kwds)

    def get_perturbation(self, **kwds):
        try:
            present_time = kwds['present_time']
        except KeyError:
            raise DslSyntaxError(
                "'present_time' not found in evaluation kwds" % self.commodity_name,
                ", ".join(kwds.keys()),
                node=self.node
            )
        commodity_name = self.commodity_name.evaluate(**kwds)
        mode = self.mode.evaluate(**kwds)
        if mode.startswith('alltime'):
            perturbation = commodity_name
        elif mode.startswith('year'):
            # perturbation = json.dumps((commodity_name, present_time.year))
            perturbation = "{}-{}".format(commodity_name, present_time.year)
        elif mode.startswith('mon'):
            # perturbation = json.dumps((commodity_name, present_time.year, present_time.month))
            perturbation = "{}-{}-{}".format(commodity_name, present_time.year, present_time.month)
        elif mode.startswith('da'):
            # perturbation = json.dumps((commodity_name, present_time.year, present_time.month, present_time.day))
            perturbation = "{}-{}-{}-{}".format(commodity_name, present_time.year, present_time.month,
                                                present_time.day)
        else:
            raise Exception("Unsupported mode: {}".format(mode))
        return perturbation

    def evaluate(self, **kwds):
        # Get the perturbed market name, if set.
        active_perturbation = kwds.get('active_perturbation', None)
        perturbation_factor = kwds['perturbation_factor']

        # If this is a perturbed market, perturb the simulated value.
        expr_value = self.expr.evaluate(**kwds)
        if active_perturbation and self.get_perturbation(**kwds) == active_perturbation.lstrip('-'):
            sign = -1 if active_perturbation.startswith('-') else 1
            evaluated_value = expr_value * (1 + sign * perturbation_factor)
        else:
            evaluated_value = expr_value
        return evaluated_value


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
    'Lift': Lift,
    'Max': Max,
    'Min': Min,
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


# Todo: Add something to Map a contract function to a sequence of values (range, list comprehension).

class AbstractMarket(StochasticObject, DslExpression):
    def evaluate(self, **kwds):
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
        simulated_price_id = make_simulated_price_id(kwds['simulation_id'], self.commodity_name, present_time,
                                                     self.delivery_date or present_time)

        # Get the value from the dict of simulated values.
        try:
            simulated_price_value = simulated_value_dict[simulated_price_id]
        except KeyError:
            raise DslError("Simulated price not found ID: {}".format(simulated_price_id))

        return simulated_price_value

    @property
    def market_name(self):
        return self.commodity_name

    @property
    def delivery_date(self):
        return None

    @property
    def commodity_name(self):
        name = self._args[0].evaluate() if isinstance(self._args[0], String) else self._args[0]
        # Disallow '-' in market names (it's used to compose / split perturbation names,
        # which probably should work differently, so that this restriction can be removed.
        # Todo: Review perturbation names (now hyphen separated, were JSON strings).
        if '-' in name:
            raise DslSyntaxError("hyphen character '-' not allowed in market names (sorry): {}"
                                 "".format(name), node=self.node)
        return name

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


class Market(AbstractMarket):
    def validate(self, args):
        self.assert_args_len(args, required_len=1)
        self.assert_args_arg(args, posn=0, required_type=(six.string_types, String, Name))


class ForwardMarket(AbstractMarket):
    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(six.string_types, String, Name))
        self.assert_args_arg(args, posn=1, required_type=(String, Date, Name, BinOp))

    @property
    def delivery_date(self):
        # Todo: Refactor this w.r.t. the Settlement.date property.
        if not hasattr(self, '_delivery_date'):
            date = self._args[1]
            if isinstance(date, Name):
                raise DslSyntaxError(
                    "date value name '%s' must be resolved to a datetime before it can be used" % date.name,
                    node=self.node)
            if isinstance(date, datetime.date):
                pass
            if isinstance(date, six.string_types):
                date = String(date)
            if isinstance(date, String):
                date = Date(date, node=date.node)
            if isinstance(date, (Date, BinOp)):
                date = date.evaluate()
            if not isinstance(date, datetime.date):
                raise DslSyntaxError("delivery date value should be a datetime.datetime by now, but it's a %s" % date,
                                     node=self.node)
            self._delivery_date = date
        return self._delivery_date


class Settlement(StochasticObject, DatedDslObject, DslExpression):
    """
    Discounts value of expression to 'present_time'.
    """

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(String, Date, Name, BinOp))
        self.assert_args_arg(args, posn=1, required_type=DslExpression)

    def evaluate(self, **kwds):
        value = self._args[1].evaluate(**kwds)
        return self.discount(value, self.date, **kwds)


class Fixing(StochasticObject, DatedDslObject, DslExpression):
    """
    A fixing defines the 'present_time' used for evaluating its expression.
    """

    def pprint(self, indent=''):
        return indent + "%s('%04d-%02d-%02d', %s)" % (
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
            raise DslSyntaxError("fixing date value should be a datetime.date by now, but it's a %s" % fixing_date,
                                 node=self.node)
        effective_present_time = fixing_date
        return super(Fixing, self).reduce(dsl_locals, dsl_globals, effective_present_time,
                                          pending_call_stack=pending_call_stack)

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
                    # payoffValue = self.get_payoff(state, subsequent_state)
                    expected_continuation_value = value_of_being_in[subsequent_state]
                    expected_continuation_values.append(expected_continuation_value)
                    if len(regression_variables):
                        conditional_expected_value = LeastSquares(regression_variables,
                                                                  expected_continuation_value).fit()
                    else:
                        conditional_expected_value = expected_continuation_value
                    conditional_expected_values.append(conditional_expected_value)

                conditional_expected_values = scipy.array(conditional_expected_values)
                expected_continuation_values = scipy.array(expected_continuation_values)
                argmax = conditional_expected_values.argmax(axis=0)
                offsets = scipy.array(range(0, conditional_expected_values.shape[1])) * \
                          conditional_expected_values.shape[0]
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
            msg = "Simulated price ID {} not in simulated price dict keys: {}".format(simulated_price_id,
                                                                                      self.simulated_price_dict.keys())
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
        # print "a: ", a
        # print "a: ", a.shape, type(a)
        # print "b: ", b
        # print "b: ", b.shape, type(b)
        # print "c: ", c.shape, type(c)
        # print "c: ", c
        if a.shape[1] != c.shape[0]:
            raise Exception("Matrices are not aligned: %s and %s" % (a.shape, c.shape))
        d = a * c
        # print "d: ", d
        # print "d: ", d.shape, type(d)
        # print "d A1: ", d.getA1()
        return d.getA1()

    def solve(self, a, b):
        try:
            c, resid, rank, sigma = scipy.linalg.lstsq(a, b)
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
