from __future__ import division

import datetime
import itertools
import math
import re
from abc import ABCMeta, abstractmethod
from collections import namedtuple

# import numexpr
import scipy
import scipy.linalg
import six
from dateutil.relativedelta import relativedelta
from scipy import ndarray

from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.domain.services.uuids import create_uuid4
from quantdsl.exceptions import DslBinOpArgsError, DslCompareArgsError, DslError, DslIfTestExpressionError, \
    DslNameError, DslPresentTimeNotInScope, DslSyntaxError, DslSystemError, DslTestExpressionCannotBeEvaluated
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
                    for _arg in arg:
                        _hash = self.hash_single_arg(_arg)
                        hashes += _hash

                else:
                    _hash = self.hash_single_arg(arg)
                    hashes += _hash

            self._hash = hash(hashes)
        return self._hash

    def hash_single_arg(self, _arg):
        if isinstance(_arg, DslObject):
            _hash = str(_arg.hash)
        elif isinstance(_arg, relativedelta):
            _hash = str(hash(str(_arg)))
        else:
            _hash = str(hash(_arg))
        return _hash

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
        error_msg = "%s is broken" % self.__class__.__name__
        if isinstance(required_type, list):
            # Ahem, this is a way of saying we require a list of the type (should be a list length 1).
            self.assert_args_arg(args, posn, list)
            assert len(required_type) == 1, "List def should only have one item."
            required_type = required_type[0]
            list_of_args = args[posn]
            for i in range(len(list_of_args)):
                try:
                    self.assert_args_arg(list_of_args, i, required_type)
                except DslSyntaxError as e:
                    desc = "argument %s must be %s" % (posn, required_type)
                    desc += " (but a %s was found): " % (list_of_args)
                    desc += str(args[posn])
                    raise DslSyntaxError(error_msg, desc, self.node)

        elif not isinstance(args[posn], required_type):
            if isinstance(required_type, (list, tuple)):
                try:
                    required_type_names = [i.__name__ for i in required_type]
                    required_type_names = ", ".join(required_type_names)
                except AttributeError as e:
                    raise AttributeError(required_type)
            else:
                required_type_names = required_type.__name__
            desc = "argument %s must be %s" % (posn, required_type_names)
            desc += " (but a %s was found): " % (args[posn].__class__.__name__)
            desc += str(args[posn])
            raise DslSyntaxError(error_msg, desc, self.node)

    def list_instances(self, dsl_type):
        return list(self.find_instances(dsl_type))

    def has_instances(self, dsl_type):
        for _ in self.find_instances(dsl_type):
            return True
        else:
            return False

    def find_instances(self, *dsl_types):
        if isinstance(self, dsl_types):
            yield self
        for arg in self._args:
            if isinstance(arg, DslObject):
                for dsl_obj in arg.find_instances(dsl_types):
                    yield dsl_obj

    def substitute_names(self, namespace):
        return self.process('substitute_names', namespace)

    def call_functions(self, present_time=None, observation_date=None, pending_call_stack=None):
        return self.process('call_functions',
                            present_time=present_time,
                            observation_date=observation_date,
                            pending_call_stack=pending_call_stack)

    def cost_expression(self):
        cost = 0
        for instance in self.find_instances(DslExpression):
            cost += instance.cost_element()
        return cost

    def process(self, method, *args, **kwargs):
        new_dsl_args = []
        for dsl_arg in self._args:
            if isinstance(dsl_arg, DslObject):
                new_dsl_arg = getattr(dsl_arg, method)(*args, **kwargs)
            elif isinstance(dsl_arg, (list, tuple)):
                new_dsl_arg = []
                for _dsl_arg in dsl_arg:
                    if isinstance(_dsl_arg, DslObject):
                        _new_dsl_arg = getattr(_dsl_arg, method)(*args, **kwargs)
                    else:
                        _new_dsl_arg = _dsl_arg
                    new_dsl_arg.append(_new_dsl_arg)
            else:
                new_dsl_arg = dsl_arg
            new_dsl_args.append(new_dsl_arg)
        return self.__class__(node=self.node, *new_dsl_args)

    def identify_price_simulation_requirements(self, requirements, **kwds):
        for dsl_arg in self._args:
            if isinstance(dsl_arg, DslObject):
                dsl_arg.identify_price_simulation_requirements(requirements, **kwds)

    def identify_perturbation_dependencies(self, dependencies, **kwds):
        for dsl_arg in self._args:
            if isinstance(dsl_arg, DslObject):
                dsl_arg.identify_perturbation_dependencies(dependencies, **kwds)

    def get_present_time(self, kwds):
        try:
            present_time = kwds['present_time']
        except KeyError:
            raise DslPresentTimeNotInScope(
                "'present_time' not found in evaluation kwds",
                ", ".join(kwds.keys()),
                node=self.node
            )
        else:
            assert isinstance(present_time, datetime.date), type(present_time)
            return present_time


def discount(value, present_date, value_date, interest_rate):
    r = interest_rate / 100
    T = get_duration_years(present_date, value_date)
    # Assumes continuous compounding rate.
    discount_factor = math.exp(- r * T)
    return value * discount_factor
    # Not annualised equivalent rate.
    # return value / (1 + r) ** T


class DslExpression(DslObject):
    relative_cost = 1

    @abstractmethod
    def evaluate(self, **kwds):
        """
        Returns the value of the expression.
        """

    def cost_element(self):
        return self.relative_cost


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
            kwargs = dict((name, int(param)) for (name, param) in six.iteritems(parts) if param)
            if not kwargs:
                raise DslSyntaxError('invalid "time delta" string', duration_str, node=self.node)
            value = relativedelta(**kwargs)
        return value


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


# NUMEXPR_OPS = ['+', '-', '*', '/', '**', '%']
NUMEXPR_OPS = []


class BinOp(DslExpression):
    relative_cost = 10

    op_code = ''

    def pprint(self, indent=''):
        if self.op_code:
            def makeStr(dsl_expr):
                dslString = str(dsl_expr)
                if isinstance(dsl_expr, BinOp):
                    dslString = "(" + dslString + ")"
                return dslString

            text = makeStr(self.left) + " " + self.op_code + " " + makeStr(self.right)
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
            if self.op_code in NUMEXPR_OPS and (isinstance(left, ndarray) or isinstance(right, ndarray)):
                return self.op_numexpr(left, right)
            else:
                return self.op_python(left, right)
        except TypeError as e:
            raise DslBinOpArgsError("unable to %s" % self.__class__.__name__.lower(),
                                    "%s and %s: %s" % (self.left, self.right, e),
                                    node=self.node)

    @abstractmethod
    def op_python(self, left, right):
        """
        Returns result of operating on two args.
        """

    def op_numexpr(self, left, right):
        # return left ** right
        expr = 'left {} right'.format(self.op_code)
        try:
            return numexpr.evaluate(expr)
        except SyntaxError as e:
            raise SyntaxError("Invalid numexpr syntax in class: {}: '{}'".format(
                self.__class__.__name__, expr
            ))


class Add(BinOp):
    op_code = '+'

    def op_python(self, left, right):
        return left + right


class Sub(BinOp):
    op_code = '-'

    def op_python(self, left, right):
        if isinstance(left, datetime.date) and isinstance(right, datetime.date):
            return relativedelta(left, right)
        else:
            return left - right


class Mult(BinOp):
    op_code = '*'

    def op_python(self, left, right):
        return left * right


class Div(BinOp):
    op_code = '/'

    def op_python(self, left, right):
        return left / right


# Todo: Pow, Mod, FloorDiv don't have proofs, so shouldn't really be used for combining random variables? Either
# prevent usage with ndarray inputs, or do the proofs. :-)

class Pow(BinOp):
    op_code = '**'

    def op_python(self, left, right):
        return left ** right

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(DslExpression, Date, TimeDelta, Underlying))
        self.assert_args_arg(args, posn=1, required_type=(DslExpression, Date, TimeDelta, Underlying))


class Mod(BinOp):
    op_code = '%'

    def op_python(self, left, right):
        return left % right


class FloorDiv(BinOp):
    op_code = '//'

    def op_python(self, left, right):
        return left // right


class NonInfixedBinOp(BinOp):
    def op_python(self, a, b):
        # Assume a and b have EITHER type ndarray, OR type int or float.
        # Try to 'balance' the sides.
        #  - two scalar numbers are good
        #  - one number with one vector is okay
        #  - two vectors is okay, but they must have the same length.
        aIsaNumber = isinstance(a, six.integer_types + (float, relativedelta, datetime.date))
        bIsaNumber = isinstance(b, six.integer_types + (float, relativedelta, datetime.date))
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
    relative_cost = 0

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

    def substitute_names(self, namespace):
        value = self.evaluate(**namespace)
        if isinstance(value, datetime.date):
            value = Date(value, node=self.node)
        elif isinstance(value, six.integer_types + (float, ndarray)):
            value = Number(value, node=self.node)
        elif isinstance(value, six.string_types):
            value = String(value, node=self.node)
        elif isinstance(value, datetime.timedelta):
            value = TimeDelta(value, node=self.node)
        elif isinstance(value, relativedelta):
            value = TimeDelta(value, node=self.node)
        return value

    def evaluate(self, **kwds):
        try:
            return kwds[self.name]
        except KeyError:
            msg = "'{}' is not defined. Current frame defines".format(self.name)
            raise DslNameError(msg, str(kwds.keys()), node=self.node)


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


# Todo: Drop this.
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
        self.assert_args_arg(args, 0, six.string_types)
        self.assert_args_arg(args, 1, [FunctionArg])
        self.assert_args_arg(args, 2, DslExpression)
        self.assert_args_arg(args, 3, [six.string_types])

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

    def apply(self, dsl_globals=None, present_time=None, observation_date=None, pending_call_stack=None,
              is_destacking=False, **raw_dsl_locals):

        # Decide either to stub out the function in the caller, and put
        # the call on the pending stack; or to actually apply the args and
        # generate a DSL expression.
        do_apply = pending_call_stack is None or is_destacking or 'inline' in self.decorator_names

        # Sort out the namespaces.
        if dsl_globals is None:
            dsl_globals = DslNamespace()
        if self.module_namespace is None:
            module_namespace = DslNamespace()
        else:
            module_namespace = self.module_namespace
        # Todo: This can be simpler... module_namespace and dsl_globals are trying to do the same thing.
        new_dsl_globals = DslNamespace(itertools.chain(
            self.enclosed_namespace.items(),
            module_namespace.items(),
            dsl_globals.items())
        )

        # Validate the call args with the definition.
        self.validateCallArgs(raw_dsl_locals)

        # Create the cache key.
        new_dsl_locals = DslNamespace()
        call_cache_key_dict = {}
        for arg_name, arg_value in raw_dsl_locals.items():
            if isinstance(arg_value, FunctionCall):
                if not list(arg_value.functionDef.find_instances(FunctionCall, StochasticObject)):
                    try:
                        arg_value = arg_value.call_functions()
                    except DslError as e:
                        raise Exception("Can't evaluate {}: {}: {}".format(arg_name, arg_value, e))
            elif isinstance(arg_value, DslExpression):
                if not arg_value.find_instances(StochasticObject):
                    try:
                        arg_value = arg_value.evaluate()
                    except DslError as e:
                        raise Exception("Can't evaluate {}, a non-stochastic expression: {}: {}".format(arg_name,
                                                                                                    arg_value, e))

            new_dsl_locals[arg_name] = arg_value
            call_cache_key_dict[arg_name] = arg_value
        call_cache_key_dict["__present_time__"] = present_time
        call_cache_key_dict["__do_apply__"] = do_apply
        call_cache_key = self.create_hash(call_cache_key_dict)

        # Check the call cache, to see whether this function has already been called with these args.
        if call_cache_key in self.call_cache:
            return self.call_cache[call_cache_key]

        if do_apply:
            # Select expression from body.
            dsl_expr = self.body
            ns = new_dsl_globals.combine(new_dsl_locals)
            ns['observation_date'] = observation_date
            ns['present_time'] = present_time
            while isinstance(dsl_expr, BaseIf):
                # Todo: Also allow user defined functions that just do dates or numbers in test expression.
                # it doesn't have or expand into DSL elements that are the functions of time (Wait, Choice, Market,
                # etc).
                dsl_expr = dsl_expr.select_expression(**ns)

            # Add this function to the namespace (it might recurse).
            ns[self.name] = self

            # Reduce the selected expression.
            assert isinstance(dsl_expr, DslExpression)
            dsl_expr = dsl_expr.substitute_names(ns.combine(new_dsl_locals))
            dsl_expr = dsl_expr.call_functions(
                present_time=present_time,
                observation_date=observation_date,
                pending_call_stack=pending_call_stack
            )

        else:
            # Stack the call expression, with the call args,
            # and return a Stub, for the calling expression.

            # Create a new Stub ID.
            stub_id = create_uuid4()

            # Put the stub ID on the call stack, with this
            # FunctionDef, the prepared call args, and the effective
            # present time. This defines a pending call.
            # assert isinstance(pending_call_stack, PendingCallQueue)
            # Todo: Extract object class PendingCall.
            pending_call_stack.put(
                stub_id=stub_id,
                stacked_function_def=self,
                stacked_locals=new_dsl_locals.copy(),
                stacked_globals=new_dsl_globals.copy(),
                present_time=present_time
            )

            # Return the stub so that the containing DSL can be fully evaluated
            # once the stacked function call has been evaluated.
            dsl_expr = Stub(stub_id, node=self.node)

        # Cache the expression.
        self.call_cache[call_cache_key] = dsl_expr

        return dsl_expr

    def create_hash(self, obj):
        if obj is None:
            return hash(obj)

        if isinstance(obj, DslObject):
            return obj.hash

        if isinstance(obj, relativedelta):
            return hash(repr(obj))

        numbers_strings_dates = six.integer_types + six.string_types + (
            float, datetime.date, datetime.timedelta
        )
        if isinstance(obj, numbers_strings_dates):
            return hash(obj)

        if isinstance(obj, dict):
            return hash(tuple(sorted([(a, self.create_hash(b)) for a, b in obj.items()])))

        if isinstance(obj, list):
            return hash(tuple(sorted([self.create_hash(a) for a in obj])))

        raise DslSystemError("Can't create hash from obj type '%s'" % type(obj), obj,
                             node=obj.node if isinstance(obj, DslObject) else None)


class FunctionCall(DslExpression):
    def pprint(self, indent=''):
        return indent + "%s(%s)" % (self.functionDef,
                                    ", ".join([str(arg) for arg in self.callArgExprs]))

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(Name, FunctionDef))
        self.assert_args_arg(args, posn=1, required_type=[(DslExpression, FunctionDef)])

    @property
    def functionDef(self):
        return self._args[0]

    @property
    def callArgExprs(self):
        return self._args[1]

    def evaluate(self, **kwds):
        raise DslSyntaxError('call to undefined name', self.functionDef.name, node=self.node)

    def call_functions(self, present_time=None, observation_date=None, pending_call_stack=None):
        if isinstance(self.functionDef, Name):
            raise DslSystemError('Need to substitute names before calling functions')
        assert isinstance(self.functionDef, FunctionDef)

        # Function def name (a Name object) should have reduced to a FunctionDef object in the namespace.
        # - it's an error for the name to be defined as anything other than a function, but that's not possible
        # here?
        f = self.functionDef

        # Check lengths of arg names matches length of arg exprs (function signature must
        # satisfy the call). Or the other way around :).
        if len(f.callArgs) != len(self.callArgExprs):
            raise DslSyntaxError(
                "mismatched call args",
                "expected %s but got %s. Expected args: %s. Received exprs: %s" % (
                    len(f.callArgs),
                    len(self.callArgExprs),
                    f.call_arg_names,
                    self.callArgExprs,
                ),
                node=self.node
            )

        # Prepare the call args.
        # - evaluate call args that don't involve stochastic elements or function calls
        # - call functions that don't involve stochastic elements or function calls
        call_args = DslNamespace()

        # Obtain the call arg values.
        for call_arg_expr, call_arg_def in zip(self.callArgExprs, f.callArgs):
            # Skip if it's a DSL object that needs to be evaluated later with market data simulation.
            # Todo: Think about and improve the way these levels are separated.
            # if isinstance(call_arg_expr, FunctionCall):
            #     # The call arg is another function call.
            #     call_arg_value = call_arg_expr
            # elif isinstance(call_arg_expr, DslExpression):
            if isinstance(call_arg_expr, DslExpression):
                # Substitute names, etc.
                # Decide whether to evaluate, or just pass the expression into the function call.
                if isinstance(call_arg_expr, Underlying):
                    # It's explicitly wrapped as an "underlying", so unwrap it as expected.
                    call_arg_value = call_arg_expr.evaluate()
                elif isinstance(call_arg_expr, FunctionCall):
                    # It's a function call, so try to call it (attempt to simplify things going forward).
                    # - can't do it with stack, because we don't want to stack calls with wrong effective present time
                    # - can't just do it without stack, because recursive functions risk recursion depth exception
                    # - so check the function body to see if it calls another function
                    if list(call_arg_expr.functionDef.find_instances(FunctionCall, StochasticObject)):
                        call_arg_value = call_arg_expr
                    else:
                        try:
                            call_arg_value = call_arg_expr.call_functions(
                                present_time=present_time,
                                observation_date=observation_date
                            )
                        except (RuntimeError, DslError) as e:
                            call_arg_value = call_arg_expr
                elif call_arg_expr.has_instances((ForwardMarket, Market, Fixing, Choice, Settlement, FunctionDef,
                                                  Stub)):
                    # It's an underlying contract, or a stub. In any case, can't evaluate here, so pass it through.
                    call_arg_value = call_arg_expr
                else:
                    # assert isinstance(call_arg_expr, DslExpression)
                    # It's a sum of two constants, or something like that - evaluate the full expression.
                    call_arg_value = call_arg_expr.evaluate(
                        observation_date=observation_date,
                        present_time=present_time,
                    )
            else:
                # It's a simple value - pass through, not much else to do.
                call_arg_value = call_arg_expr

            # Add the call arg value to the new call arg namespace.
            call_args[call_arg_def.name] = call_arg_value

        # Evaluate the function def with the dict of call arg values.
        # The result of this function call (stubbed or otherwise) should be a DSL expression.
        dsl_expr = f.apply(present_time=present_time,
                           observation_date=observation_date,
                           pending_call_stack=pending_call_stack,
                           **call_args)

        # assert isinstance(dsl_expr, DslExpression)

        return dsl_expr


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
        expression = self.select_expression(**kwds)
        return expression.evaluate(**kwds)

    def select_expression(self, **kwds):
        test_expr = self.test.substitute_names(kwds)
        test_expr = test_expr.call_functions(
            present_time=kwds.get('present_time'),
            observation_date=kwds.get('observation_date'),
        )

        # Check for stochastic elements.
        stochastic_elements = test_expr.find_instances(StochasticObject)
        stochastic_elements = list(stochastic_elements)
        if stochastic_elements:
            msg = "if statement test expression can't evaluate stochastic elements: {}".format(
                ", ".join([str(i) for i in stochastic_elements]))
            raise DslTestExpressionCannotBeEvaluated(msg, '', self.node)
        # Check for stub elements.
        stub_elements = test_expr.find_instances(Stub)
        stub_elements = list(stub_elements)
        if stub_elements:
            msg = "if statement test expression can't evaluate stubbed expressions: {}".format(
                ", ".join([str(i) for i in stub_elements]))
            raise DslTestExpressionCannotBeEvaluated(msg, '', self.node)
        try:
            test_value = test_expr.evaluate(**kwds)
        except DslSyntaxError as e:
            msg = "Cannot evaluate if statement's test expression: {}: {}".format(self.test, e)
            raise DslSyntaxError(msg, '', self.node)
        if isinstance(test_value, DslObject):
            raise DslIfTestExpressionError("If statement test expression evaluated to a DSL object",
                                           str(test_value), node=self.node)
        if test_value:
            expression = self.body
        else:
            expression = self.orelse
        return expression


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
    relative_cost = 10


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
        left_value = self.left.evaluate(**kwds)
        for i in range(len(self.op_names)):
            right_value = self.comparators[i].evaluate(**kwds)
            self.check_types(left_value, right_value, i)
            op_name = self.op_names[i]
            op = self.valid_ops[op_name]

            if not op(left_value, right_value):
                return False
            left_value = right_value
        else:
            return True

    comparable_types = {}
    for integer_type in six.integer_types:
        comparable_types[integer_type] = (float, six.integer_types)
    comparable_types[float] = (float, six.integer_types)
    comparable_types[datetime.timedelta] = (datetime.timedelta)
    comparable_types[datetime.date] = (datetime.date, datetime.datetime)
    comparable_types[datetime.datetime] = (datetime.date, datetime.datetime)
    for string_type in six.string_types:
        comparable_types[string_type] = (float, six.string_types)

    def check_types(self, left, right, i):
        # if isinstance(value, Stub):
        #     raise DslSyntaxError("Test expression can't use function calls", node=self.node)
        try:
            assert isinstance(right, self.comparable_types[type(left)])
        except (KeyError, AssertionError) as e:
            msg = ("Test expression needs comparable values, "
                   "not: {} and {}").format(type(left), type(right))
            raise DslCompareArgsError(msg, '', node=self.node)


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

    def combine(self, other):
        copy = self.copy()
        copy.update(other)
        return copy


class StochasticObject(DslObject):
    @abstractmethod
    def validate(self, args):
        """
        Returns value of stochastic object.
        """


class DatedDslObject(DslExpression):
    def get_date_expr(self, **kwargs):
        return self._args[0]

    def get_date(self, **kwargs):
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
                date = date.evaluate(**kwargs)
            if not isinstance(date, datetime.date):
                raise DslSyntaxError("date value should be a datetime.datetime by now, but it's a %s" % date,
                                     node=self.node)
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
    relative_cost = 0
    _commodity_name_arg_index = 0

    def evaluate(self, **kwds):
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
        fixing_date, delivery_date = self.get_fixing_and_delivery_dates(kwds)
        simulated_price_id = make_simulated_price_id(kwds['simulation_id'], self.commodity_name, fixing_date,
                                                     delivery_date)

        # Get the value from the dict of simulated values.
        try:
            simulated_price_value = simulated_value_dict[simulated_price_id]
        except KeyError:
            raise DslError("Simulated price not found ID: {}".format(simulated_price_id))

        # # Discount the value from delivery date to fixing date.
        # if present_time != delivery_date:
        #     simulated_price_value = self.discount(
        #         value=simulated_price_value,
        #         start_date=present_time,
        #         end_date=delivery_date,
        #         **kwds
        #     )

        # Get the active perturbation, if set.
        active_perturbation = kwds.get('active_perturbation', None)
        perturbation_factor = kwds['perturbation_factor']

        expr_value = simulated_price_value
        if active_perturbation and self.get_perturbation(**kwds) == active_perturbation.lstrip('-'):
            # If this perturbation is active, perturb the simulated value.
            sign = -1 if active_perturbation.startswith('-') else 1
            evaluated_value = expr_value * (1 + sign * perturbation_factor)
        else:
            evaluated_value = expr_value
        return evaluated_value

    @property
    def market_name(self):
        return self.commodity_name

    @property
    def commodity_name(self):
        i = self._commodity_name_arg_index
        name = self._args[i].evaluate() if isinstance(self._args[i], String) else self._args[i]
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
        fixing_date, delivery_date = self.get_fixing_and_delivery_dates(kwds)
        requirements.add((self.commodity_name, fixing_date, delivery_date))

        # Support calculating deltas using.
        if fixing_date != delivery_date:
            requirements.add((self.commodity_name, delivery_date, delivery_date))
        if kwds['periodisation'] == 'alltime':
            observation_date = kwds['observation_date']
            requirements.add((self.commodity_name, observation_date, observation_date))

        super(AbstractMarket, self).identify_price_simulation_requirements(requirements, **kwds)

    def identify_perturbation_dependencies(self, dependencies, **kwds):
        perturbation = self.get_perturbation(**kwds)
        if perturbation is not None:
            dependencies.add(perturbation)
        super(AbstractMarket, self).identify_perturbation_dependencies(dependencies, **kwds)

    def get_perturbation(self, **kwds):
        _, delivery_date = self.get_fixing_and_delivery_dates(kwds)
        periodisation = kwds.get('periodisation')
        perturbation = None
        if periodisation is not None:
            commodity_name = self.commodity_name
            if periodisation.startswith('alltime'):
                perturbation = commodity_name
            elif periodisation.startswith('year'):
                perturbation = "{}-{}".format(commodity_name, delivery_date.year)
            elif periodisation.startswith('mon'):
                perturbation = "{}-{}-{}".format(commodity_name, delivery_date.year, delivery_date.month)
            elif periodisation.startswith('da'):
                perturbation = "{}-{}-{}-{}".format(commodity_name, delivery_date.year, delivery_date.month,
                                                    delivery_date.day)
        return perturbation

    def get_fixing_and_delivery_dates(self, kwds):
        present_time = self.get_present_time(kwds)
        return present_time, present_time



class Market(AbstractMarket):
    def validate(self, args):
        self.assert_args_len(args, required_len=1)
        self.assert_args_arg(args, posn=0, required_type=(six.string_types + (String, Name)))


class ForwardMarket(AbstractMarket):
    _commodity_name_arg_index = 1

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(BinOp, Date, Name, String))
        self.assert_args_arg(args, posn=1, required_type=six.string_types + (String, Name))

    def get_fixing_and_delivery_dates(self, kwds):
        fixing_date = self.get_present_time(kwds)

        # Todo: Refactor this w.r.t. the Settlement.date property.
        if not hasattr(self, '_delivery_date'):
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
                date = date.evaluate(**kwds)
            if not isinstance(date, datetime.date):
                raise DslSyntaxError("delivery date value should be a datetime.datetime by now, but it's a %s" % date,
                                     node=self.node)
            self._delivery_date = date
        return fixing_date, self._delivery_date


class Settlement(StochasticObject, DatedDslObject):
    """
    Discounts value of expression to 'present_time'.
    """
    relative_cost = 10

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=(String, Date, Name, BinOp))
        self.assert_args_arg(args, posn=1, required_type=DslExpression)

    def evaluate(self, **kwds):
        included_value = self._args[1].evaluate(**kwds)

        present_time = self.get_present_time(kwds)
        interest_rate = kwds['interest_rate']
        discounted_value = discount(included_value, present_time, self.get_date(**kwds), interest_rate)
        return discounted_value


class Fixing(DatedDslObject):
    """
    A fixing defines the 'present_time' used for evaluating its expression.
    """
    relative_cost = 0


    def pprint(self, indent=''):
        date_expr = self.get_date_expr()
        return indent + "%s(%s, %s)" % (
            self.__class__.__name__,
            date_expr,
            self.expr)

    def validate(self, args):
        self.assert_args_len(args, required_len=2)
        self.assert_args_arg(args, posn=0, required_type=six.string_types + (String, Date, Name, BinOp))
        self.assert_args_arg(args, posn=1, required_type=DslExpression)

    @property
    def expr(self):
        return self._args[1]

    def call_functions(self, present_time=None, observation_date=None, pending_call_stack=None):
        # Figure out the present_time from the fixing date, which might still be a Name.
        fixing_date = self._args[0]
        if isinstance(fixing_date, datetime.datetime):
            pass
        if isinstance(fixing_date, six.string_types):
            fixing_date = String(fixing_date)
        if isinstance(fixing_date, String):
            fixing_date = Date(fixing_date, node=fixing_date.node)
        if isinstance(fixing_date, (Date, BinOp, Name)):
            fixing_date = fixing_date.evaluate(
                present_time=present_time,
                observation_date=observation_date,
            )
        if not isinstance(fixing_date, datetime.date):
            raise DslSyntaxError("fixing date value should be a datetime.date by now, but it's a %s" % fixing_date,
                                 node=self.node)
        present_time = fixing_date
        return super(Fixing, self).call_functions(
            present_time=present_time,
            observation_date=observation_date,
            pending_call_stack=pending_call_stack)

    def evaluate(self, **kwds):
        kwds = kwds.copy()
        kwds['present_time'] = self.get_date(**kwds)
        return self.expr.evaluate(**kwds)

    def identify_price_simulation_requirements(self, requirements, **kwds):
        kwds['present_time'] = self.get_date(**kwds)
        super(Fixing, self).identify_price_simulation_requirements(requirements, **kwds)

    def identify_perturbation_dependencies(self, dependencies, **kwds):
        kwds['present_time'] = self.get_date(**kwds)
        super(Fixing, self).identify_perturbation_dependencies(dependencies, **kwds)


class On(Fixing):
    """
    A shorter name for Fixing.
    """


class Wait(Fixing):
# class Wait(Settlement, Fixing):
    """
    A fixing with discounting of the resulting value from date arg to present_time.

    Wait(date, expr) == Settlement(date, Fixing(date, expr))
    """
    relative_cost = 10

    def evaluate(self, **kwds):
        # return super(Wait, self).evaluate(**kwds)
        value = super(Wait, self).evaluate(**kwds)
        present_time = self.get_present_time(kwds)
        interest_rate = kwds['interest_rate']
        return discount(value, present_time, self.get_date(**kwds), interest_rate)


class Choice(StochasticObject, DslExpression):
    """
    Encapsulates the Longstaff-Schwartz routine as an element of the language.
    """
    relative_cost = 300

    def validate(self, args):
        self.assert_args_len(args, min_len=2)
        for i in range(len(args)):
            self.assert_args_arg(args, posn=i, required_type=DslExpression)

    def evaluate(self, **kwds):
        # Run the least-squares monte-carlo routine.
        present_time = self.get_present_time(kwds)
        # Todo: Check this is ok. Is it symmetric, e.g. with two commdities does a choice with the second work as if
        # the second were the first?
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


class ObservationDate(Date):
    def pprint(self, indent=''):
        return "ObservationDate()"

    def validate(self, args):
        self.assert_args_len(args, required_len=0)

    def evaluate(self, **kwds):
        return kwds['observation_date']


class PresentTime(Date):
    def pprint(self, indent=''):
        return "PresentTime()"

    def validate(self, args):
        self.assert_args_len(args, required_len=0)

    def evaluate(self, **kwds):
        present_time = kwds['present_time']
        assert present_time is not None
        return present_time


class IsDayOfMonth(DatedDslObject):
    def validate(self, args):
        self.assert_args_len(args, required_len=1)
        self.assert_args_arg(args, 0, required_type=Number)

    def evaluate(self, **kwds):
        date = self.get_present_time(kwds)
        day = self._args[0].evaluate(**kwds)
        return date.day == day


defaultDslClasses = functionalDslClasses.copy()
defaultDslClasses.update({
    'Choice': Choice,
    'Fixing': Fixing,
    'Market': Market,
    'ForwardMarket': ForwardMarket,
    'On': On,
    'Settlement': Settlement,
    'Wait': Wait,
    'ObservationDate': ObservationDate,
    'PresentTime': PresentTime,
    'IsDayOfMonth': IsDayOfMonth,
})

PendingCall = namedtuple('PendingCall', ['stub_id', 'stacked_function_def', 'stacked_locals', 'stacked_globals',
                                         'present_time'])
