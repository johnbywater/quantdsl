from quantdsl.tests.test_application import ContractValuationTestCase, TestCase


class ExpressionTests(ContractValuationTestCase, TestCase):
    def test_generate_valuation_addition(self):
        specification_tmpl = """
from quantdsl.lib.storage1 import GasStorage, Date

GasStorage(Date('%(start_date)s'), Date('%(end_date)s'), '%(commodity)s', %(quantity)s, %(limit)s, TimeDelta('1m'))
"""
        # No capacity.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 0
        }
        self.assert_contract_value(specification, 0.00, {}, expected_call_count=2)

        # Capacity, zero inventory.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 10
        }
        self.assert_contract_value(specification, 0.00, {}, expected_call_count=6)

        # Capacity, zero inventory, option in the future.
        specification = specification_tmpl % {
            'start_date': '2013-1-1',
            'end_date': '2013-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 10
        }
        self.assert_contract_value(specification, 0.0270, {}, expected_call_count=6)

        # Capacity, and inventory to discharge.
        specification = specification_tmpl % {
            'start_date': '2011-1-1',
            'end_date': '2011-3-1',
            'commodity': 'NBP',
            'quantity': 2,
            'limit': 2
        }
        self.assert_contract_value(specification, 19.9857, {}, expected_call_count=10)

        # Capacity, and inventory to discharge in future.
        specification = specification_tmpl % {
            'start_date': '2021-1-1',
            'end_date': '2021-3-1',
            'commodity': 'NBP',
            'quantity': 2,
            'limit': 2
        }
        self.assert_contract_value(specification, 15.3496, {}, expected_call_count=10)

        # Capacity, zero inventory, in future.
        specification = specification_tmpl % {
            'start_date': '2021-1-1',
            'end_date': '2021-3-1',
            'commodity': 'NBP',
            'quantity': 0,
            'limit': 2
        }
        self.assert_contract_value(specification, 0.0123, {}, expected_call_count=6)
