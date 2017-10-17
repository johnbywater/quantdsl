from unittest import TestCase

import datetime

from quantdsl.priceprocess.forwardcurve import ForwardCurve


class TestForwardCurve(TestCase):
    def test_get_price(self):
        # Check without any data.
        curve = ForwardCurve('name', [])

        # Check raises KeyError for any dates.
        with self.assertRaises(KeyError):
            curve.get_price(date=datetime.datetime(2011, 12, 31))

        # Check with data.
        curve = ForwardCurve('name', [('2011-1-1', 1), ('2011-1-3', 3)])
        # Get first value using exact date.
        self.assertEqual(curve.get_price(date=datetime.datetime(2011, 1, 1)), 1)
        # Get first value using later date.
        self.assertEqual(curve.get_price(date=datetime.datetime(2011, 1, 2)), 1)
        # Get second value using exact date.
        self.assertEqual(curve.get_price(date=datetime.datetime(2011, 1, 3)), 3)
        # Get second value using later date.
        self.assertEqual(curve.get_price(date=datetime.datetime(2011, 1, 4)), 3)

        # Check raises KeyError for values before first date.
        with self.assertRaises(KeyError):
            curve.get_price(date=datetime.datetime(2010, 12, 31))
