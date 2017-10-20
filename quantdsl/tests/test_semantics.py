import datetime
from unittest.case import TestCase

from dateutil.relativedelta import relativedelta
from mock import Mock
from scipy import array

from quantdsl.exceptions import DslNameError, DslSyntaxError, DslSystemError, DslPresentTimeNotInScope
from quantdsl.semantics import Add, And, Date, Div, DslNamespace, DslObject, Max, Min, Mult, Name, Number, Or, \
    String, Sub, TimeDelta, Pow, FunctionDef, FunctionCall, Stub, FunctionArg, PresentTime


class Subclass(DslObject):
    def validate(self, args):
        pass


class TestDslObject(TestCase):
    def setUp(self):
        super(TestDslObject, self).setUp()
        self.obj = Subclass()

    def test_assert_args_len(self):
        self.obj.assert_args_len([1], min_len=1)
        with self.assertRaises(DslSyntaxError):
            self.obj.assert_args_len([1], min_len=2)
        self.obj.assert_args_len([1], max_len=1)
        with self.assertRaises(DslSyntaxError):
            self.obj.assert_args_len([1, 2], max_len=1)
        self.obj.assert_args_len([1], required_len=1)
        with self.assertRaises(DslSyntaxError):
            self.obj.assert_args_len([1, 2], required_len=1)

    def test_assert_args_arg(self):
        self.obj.assert_args_arg([1], 0, int)
        with self.assertRaises(DslSyntaxError):
            self.obj.assert_args_arg([1], 0, str)

        self.obj.assert_args_arg([[1]], 0, [int])
        with self.assertRaises(DslSyntaxError):
            self.obj.assert_args_arg([[1, 'a']], 0, [int])

        self.obj.assert_args_arg([1], 0, (int, float))
        with self.assertRaises(DslSyntaxError):
            self.obj.assert_args_arg(['1'], 0, (int, float))

    def test_str(self):
        self.assertEqual(str(self.obj), "Subclass()")
        self.assertEqual(str(Subclass(Subclass())), "Subclass(Subclass())")

    def test_get_present_time(self):
        # Check method returns given value.
        present_time = self.obj.get_present_time({'present_time': datetime.datetime(2011, 1, 1)})
        self.assertEqual(present_time, datetime.datetime(2011, 1, 1))

        # Check method raises exception when value not in scope.
        with self.assertRaises(DslPresentTimeNotInScope):
            self.obj.get_present_time({})



class TestString(TestCase):
    def test_value(self):
        obj_a = String('a')
        obj_b = String('b')

        # Check the value attribute.
        self.assertEqual(obj_a.value, 'a')
        self.assertEqual(obj_b.value, 'b')

        # Check object equality.
        self.assertEqual(obj_b, String('b'))

        # Check object inequality.
        self.assertNotEqual(obj_a, obj_b)

        # Check errors.
        with self.assertRaises(DslSyntaxError):
            String()
        with self.assertRaises(DslSyntaxError):
            String('a', 'b')
        with self.assertRaises(DslSyntaxError):
            String(1)

    def test_str(self):
        obj = String('a')
        self.assertEqual(str(obj), "'a'")
        self.assertEqual(str(Subclass(obj)), "Subclass('a')")

    def test_hash(self):
        obj = String('a')
        self.assertIsInstance(obj.hash, int)


class TestNumber(TestCase):
    def test_value(self):
        # Integers are ok.
        obj = Number(1)
        self.assertEqual(obj.value, 1)

        # Floats are ok.
        obj = Number(1.1)
        self.assertEqual(obj.value, 1.1)

        # Numpy arrays are ok.
        obj = Number(array([1, 2]))
        self.assertEqual(list(obj.value), list(array([1, 2])))

        # No args is not ok.
        with self.assertRaises(DslSyntaxError):
            Number()

        # Two args is not ok.
        with self.assertRaises(DslSyntaxError):
            Number(1, 1.1)

        # A list is not ok.
        with self.assertRaises(DslSyntaxError):
            Number([1, 1.1])

        # A string is not ok.
        with self.assertRaises(DslSyntaxError):
            Number('1')

    def test_str(self):
        obj = Number(1)
        self.assertEqual(str(obj), '1')
        self.assertEqual(str(Subclass(obj)), 'Subclass(1)')


class TestDate(TestCase):
    def test_value(self):
        # A Python string is ok.
        obj = Date('2011-1-1')
        self.assertEqual(obj.value, datetime.datetime(2011, 1, 1))

        # A Quant DSL String is ok.
        obj = Date(String('2011-1-1'))
        self.assertEqual(obj.value, datetime.datetime(2011, 1, 1))

        # A date is ok.
        obj = Date(datetime.date(2011, 1, 1))
        self.assertEqual(obj.value, datetime.datetime(2011, 1, 1))

        # A datetime is ok.
        obj = Date(datetime.datetime(2011, 1, 1))
        self.assertEqual(obj.value, datetime.datetime(2011, 1, 1))

        # No args is not ok.
        with self.assertRaises(DslSyntaxError):
            Date()

        # Two args is not ok.
        with self.assertRaises(DslSyntaxError):
            Date(1, 1.1)

        # A string that doesn't look like a date is not ok.
        with self.assertRaises(DslSyntaxError):
            Date('1')

    def test_str(self):
        obj = Date(String('2011-1-1'))
        self.assertEqual(str(obj), "Date('2011-01-01')")
        self.assertEqual(str(Subclass(obj)), "Subclass(Date('2011-01-01'))")


class TestTimeDelta(TestCase):
    def test_value(self):
        # Days, months, or years is ok.
        obj = TimeDelta(String('1d'))
        self.assertEqual(obj.value, relativedelta(days=1))
        obj = TimeDelta(String('2d'))
        self.assertEqual(obj.value, relativedelta(days=2))
        obj = TimeDelta(String('1m'))
        self.assertEqual(obj.value, relativedelta(months=1))
        obj = TimeDelta(String('1y'))
        self.assertEqual(obj.value, relativedelta(years=1))

        # An invalid time delta string is not ok.
        with self.assertRaises(DslSyntaxError):
            TimeDelta(String('1j'))

    def test_str(self):
        obj = TimeDelta(String('1d'))
        self.assertEqual(str(obj), "TimeDelta('1d')")
        self.assertEqual(str(Subclass(obj)), "Subclass(TimeDelta('1d'))")


class TestAnd(TestCase):
    def test_evaluate(self):
        obj = And([Number(1), Number(1)])
        self.assertTrue(obj.evaluate())

        obj = And([Number(1), Number(0)])
        self.assertFalse(obj.evaluate())

        obj = And([Number(0), Number(1)])
        self.assertFalse(obj.evaluate())

        obj = And([Number(0), Number(0)])
        self.assertFalse(obj.evaluate())

    def test_str(self):
        obj = And([Number(1), Number(2), Number(3)])
        self.assertEqual(str(obj), '(1 and 2 and 3)')


class TestOr(TestCase):
    def test_evaluate(self):
        obj = Or([Number(1), Number(1)])
        self.assertTrue(obj.evaluate())

        obj = Or([Number(1), Number(0)])
        self.assertTrue(obj.evaluate())

        obj = Or([Number(0), Number(1)])
        self.assertTrue(obj.evaluate())

        obj = Or([Number(0), Number(0)])
        self.assertFalse(obj.evaluate())

    def test_str(self):
        obj = Or([Number(1), Number(2), Number(3)])
        self.assertEqual(str(obj), '(1 or 2 or 3)')


class TestAndOr(TestCase):
    def test_str(self):
        obj = And([Number(1), Or([Number(2), Number(3)])])
        self.assertEqual(str(obj), '(1 and (2 or 3))')
        # Check the indentation isn't propagated.
        self.assertEqual(str(obj), '(1 and (2 or 3))')


class TestAdd(TestCase):
    def test_evaluate(self):
        obj = Add(Number(1), Number(1))
        self.assertEqual(obj.evaluate(), 2)

        obj = Add(String('a'), String('b'))
        self.assertEqual(obj.evaluate(), 'ab')

        obj = Add(Number(1), String('a'))
        with self.assertRaises(DslSyntaxError):
            obj.evaluate()


class TestSub(TestCase):
    def test_evaluate(self):
        obj = Sub(Number(1), Number(1))
        self.assertEqual(obj.evaluate(), 0)

        obj = Sub(Number(1), String('a'))
        with self.assertRaises(DslSyntaxError):
            obj.evaluate()

        obj = Sub(Date('2011-1-2'), Date('2011-1-1'))
        self.assertEqual(obj.evaluate(), relativedelta(days=1))


class TestMul(TestCase):
    def test_evaluate(self):
        obj = Mult(Number(2), Number(2))
        self.assertEqual(obj.evaluate(), 4)

        obj = Mult(Number(2), String('a'))
        self.assertEqual(obj.evaluate(), 'aa')

        obj = Mult(Number(2.0), String('a'))
        with self.assertRaises(DslSyntaxError):
            obj.evaluate()

        obj = Mult(String('a'), String('a'))
        with self.assertRaises(DslSyntaxError):
            obj.evaluate()

        obj = Mult(Number(2.1), String('a'))
        with self.assertRaises(DslSyntaxError):
            obj.evaluate()


class TestPow(TestCase):
    def test_evaluate(self):
        obj = Pow(Number(2), Number(2))
        self.assertEqual(obj.evaluate(), 4)

        obj = Pow(Number(2.0), String('a'))
        with self.assertRaises(DslSyntaxError):
            obj.evaluate()



class TestDiv(TestCase):
    def test_evaluate(self):
        obj = Div(Number(5), Number(2))
        self.assertEqual(obj.evaluate(), 2.5)

        obj = Div(TimeDelta(String('2d')), Number(2))
        self.assertEqual(obj.evaluate(), relativedelta(days=1))

        obj = Div(Number(5), Number(0))
        with self.assertRaises(ZeroDivisionError):
            obj.evaluate()

        obj = Div(Number(2.1), String('a'))
        with self.assertRaises(DslSyntaxError):
            obj.evaluate()


class TestMin(TestCase):
    def test_evaluate(self):
        obj = Min(Number(1), Number(2))
        self.assertEqual(obj.evaluate(), 1)

        obj = Min(Number(1), Number(array([1, 2, 3])))
        self.assertEqual(list(obj.evaluate()), list(array([1, 1, 1])))

        obj = Min(Number(2), Number(array([1, 2, 3])))
        self.assertEqual(list(obj.evaluate()), list(array([1, 2, 2])))

        obj = Min(Number(array([3, 2, 1])), Number(array([1, 2, 3])))
        self.assertEqual(list(obj.evaluate()), list(array([1, 2, 1])))


class TestMax(TestCase):
    def test_evaluate(self):
        obj = Max(Number(1), Number(2))
        self.assertEqual(obj.evaluate(), 2)

        obj = Max(Number(1), Number(array([1, 2, 3])))
        self.assertEqual(list(obj.evaluate()), list(array([1, 2, 3])))

        obj = Max(Number(2), Number(array([1, 2, 3])))
        self.assertEqual(list(obj.evaluate()), list(array([2, 2, 3])))

        obj = Max(Number(array([3, 2, 1])), Number(array([1, 2, 3])))
        self.assertEqual(list(obj.evaluate()), list(array([3, 2, 3])))

        # Swap args.
        obj = Max(Number(array([1, 2, 3])), Number(1))
        self.assertEqual(list(obj.evaluate()), list(array([1, 2, 3])))

        obj = Max(Number(array([1, 2, 3])), Number(2))
        self.assertEqual(list(obj.evaluate()), list(array([2, 2, 3])))

        obj = Max(Number(array([1, 2, 3])), Number(array([3, 2, 1])))
        self.assertEqual(list(obj.evaluate()), list(array([3, 2, 3])))


class TestName(TestCase):
    def test_substitute(self):
        # Maybe if Name can take a string, perhaps also other things can?
        # Maybe the parser should return Python string, numbers etc?
        # Maybe String and Number etc don't add anything?
        obj = Name('a')
        self.assertEqual(obj.name, 'a')

        obj = Name(String('a'))
        self.assertEqual(obj.name, 'a')

        ns = DslNamespace()
        with self.assertRaises(DslNameError):
            obj.substitute_names(ns)

        ns = DslNamespace({'a': 1})
        self.assertEqual(obj.substitute_names(ns), Number(1))

        ns = DslNamespace({'a': datetime.timedelta(1)})
        self.assertEqual(obj.substitute_names(ns), TimeDelta(datetime.timedelta(1)))

        function_def = FunctionDef('f', [], Number(1), [])
        ns = DslNamespace({'a': function_def})
        self.assertEqual(obj.substitute_names(ns), function_def)


class TestFunctionDef(TestCase):
    def test_pprint(self):
        fd = FunctionDef('f', [], Name('a'), [])
        self.assertEqual(str(fd), "def f():\n    a")


class TestFunctionCall(TestCase):
    def test_substitute_names(self):
        fc = FunctionCall(Name('f'), [Name('x')])
        fd = FunctionDef('f', [], Name('a'), [])
        ns = DslNamespace({
            'f': fd,
            'x': Number(1),
        })
        fc1 = fc.substitute_names(ns)
        self.assertEqual(fc1.functionDef, fd)
        self.assertEqual(fc1.callArgExprs[0], Number(1))

    def test_call_functions_with_pending_call_stack(self):
        fc = FunctionCall(Name('f'), [Name('x')])
        fd = FunctionDef('f', [FunctionArg('a', '')], Name('a'), [])
        namespace = DslNamespace({fd.name: fd})
        fd.module_namespace = namespace
        number = Number(1234)
        ns = DslNamespace({
            'f': fd,
            'x': number,
        })

        # Substitute names.
        fc1 = fc.substitute_names(ns)
        self.assertEqual(fc1.functionDef, fd)
        self.assertEqual(fc1.callArgExprs[0], number)

        # Call functions.
        expr = fc1.call_functions()
        self.assertEqual(expr, number)

    def test_call_functions_without_pending_call_stack(self):
        fc = FunctionCall(Name('f'), [Name('x')])
        fd = FunctionDef('f', [FunctionArg('a', '')], Name('a'), [])
        number1 = Number(1234)
        number2 = Number(2345)
        ns1 = DslNamespace({
            'f': fd,
            'x': number1,
        })
        ns2 = DslNamespace({
            'f': fd,
            'x': number2,
        })

        # Call functions with pending call stack.
        queue = Mock()
        fc1 = fc.substitute_names(ns1)
        fc2 = fc.substitute_names(ns2)
        self.assertNotEqual(fd.create_hash(fc1), fd.create_hash(fc2))

        t1 = datetime.datetime(2011, 1, 1)
        expr = fc1.call_functions(pending_call_stack=queue, present_time=t1)

        # Check we got a stub.
        self.assertIsInstance(expr, Stub)
        self.assertEqual(queue.put.call_count, 1)

        # Check the call to the stub was queued.
        first_call = queue.put.mock_calls[0]
        self.assertEqual(first_call[2]['stub_id'], expr.name)
        self.assertEqual(first_call[2]['stacked_function_def'], fd)
        self.assertEqual(first_call[2]['stacked_locals'], {'a': 1234})  # Maybe this should be Number(1234)?
        self.assertEqual(first_call[2]['present_time'], t1)

    def test_must_substitute_names_before_call_functions(self):
        fc = FunctionCall(Name('f'), [Name('x')])

        # Call functions with pending call stack.
        queue = Mock()
        t1 = datetime.datetime(2011, 1, 1)
        with self.assertRaises(DslSystemError):
            fc.call_functions(pending_call_stack=queue, present_time=t1)


class TestPresentTime(TestCase):
    def test_pprint(self):
        pt = PresentTime()
        self.assertEqual(str(pt), "PresentTime()")

    def test_validate(self):
        with self.assertRaises(DslSyntaxError):
            PresentTime(String('a'))

    def test_evaluate(self):
        pt = PresentTime()
        with self.assertRaises(KeyError):
            pt.evaluate()

        value = pt.evaluate(present_time=datetime.date(2011, 1, 1))
        self.assertEqual(value, datetime.date(2011, 1, 1))
