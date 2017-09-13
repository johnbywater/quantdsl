import datetime
from unittest.case import TestCase

from dateutil.relativedelta import relativedelta
from scipy import array

from quantdsl.exceptions import DslSyntaxError
from quantdsl.semantics import Date, DslObject, Number, String, TimeDelta


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

    def test_pprint(self):
        text = self.obj.pprint()
        self.assertEqual(text, "Subclass()")
        self.assertEqual(Subclass(Subclass()).pprint(), "Subclass(Subclass())")
        self.assertEqual(Subclass(Subclass(), 1).pprint(), """Subclass(
    Subclass(),
    1
)""")

    class TestString(TestCase):
        def test(self):
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

    class TestNumber(TestCase):
        def test(self):
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

    class TestDate(TestCase):
        def test(self):
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

    class TestTimeDelta(TestCase):
        def test(self):
            obj = TimeDelta(String('1d'))
            self.assertEqual(obj.value, relativedelta(days=1))
            obj = TimeDelta(String('2d'))
            self.assertEqual(obj.value, relativedelta(days=2))
            obj = TimeDelta(String('1m'))
            self.assertEqual(obj.value, relativedelta(months=1))
            obj = TimeDelta(String('1y'))
            self.assertEqual(obj.value, relativedelta(years=1))
