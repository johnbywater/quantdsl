import unittest

import scipy

from quantdsl.semantics import LeastSquares


class TestLeastSquares(unittest.TestCase):

    DECIMALS = 12

    def assertFit(self, fixture_x, fixture_y, expected_values):
        assert expected_values is not None
        ls = LeastSquares(scipy.array(fixture_x), scipy.array(fixture_y))
        fit_data = ls.fit()
        for i, expected_value in enumerate(expected_values):
            fit_value = round(fit_data[i], self.DECIMALS)
            msg = "expected_values value: %s, fit value: %s, expected_values data: %s, fit data: %s" % (
                expected_value, fit_value, expected_values, fit_data)
            self.assertEqual(expected_value, fit_value, msg)

    def test_fit1(self):
        self.assertFit(
            fixture_x=[
                [0, 1, 2],
                [3, 4, 5]],
            fixture_y=[1, 1, 1],
            expected_values=[1, 1, 1],
        )

    def test_fit2(self):
        self.assertFit(
            fixture_x=[[0, 1, 2], [3, 4, 5]],
            fixture_y=[0, 1, 2],
            expected_values=[0, 1, 2],
        )
