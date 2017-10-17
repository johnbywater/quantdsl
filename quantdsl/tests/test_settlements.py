# Todo: Get the evaluation to return settlements by date, by picking paths.

# import datetime
# from unittest.case import TestCase
#
# from quantdsl.semantics import Settlement, Date, Number
#
# class TestSettlements(TestCase):
#     def _test(self):
#         e = Settlement(Date('2012-1-1'), Number(10))
#         result = e.evaluate(
#             interest_rate=2.5,
#             present_time=datetime.datetime(2011, 1, 1)
#         )
#         self.fail(result)