import datetime
from unittest.case import TestCase

from dateutil.relativedelta import relativedelta
from scipy import array

from quantdsl.exceptions import DslNameError, DslSyntaxError
from quantdsl.semantics import Add, And, Date, Div, DslNamespace, DslObject, Max, Min, Mult, Name, Number, Or, \
    String, Sub, TimeDelta


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


class TestString(TestCase):
    def test_value(self):
        obj = String('a')
        self.assertEqual(obj.value, 'a')

        obj = String('b')
        self.assertEqual(obj.value, 'b')

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
        self.assertEqual(obj.pprint('    '), '    (1 and (2 or 3))')


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


class TestName(TestCase):
    def test_evaluate(self):
        # Maybe if Name can take a string, perhaps also other things can?
        # Maybe the parser should return Python string, numbers etc?
        # Maybe String and Number etc don't add anything?
        obj = Name('a')
        self.assertEqual(obj.name, 'a')

        obj = Name(String('a'))
        self.assertEqual(obj.name, 'a')

        with self.assertRaises(DslNameError):
            obj.reduce(DslNamespace(), DslNamespace())

        self.assertEqual(obj.reduce(DslNamespace({'a': 1}), DslNamespace()), Number(1))
        self.assertEqual(obj.reduce(DslNamespace({'a': datetime.timedelta(1)}), DslNamespace()),
                         TimeDelta(datetime.timedelta(1)))
