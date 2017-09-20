import datetime

import scipy
import six

from quantdsl.domain.model.call_dependencies import CallDependencies
from quantdsl.domain.model.call_dependents import CallDependents
from quantdsl.domain.model.call_requirement import CallRequirement, register_call_requirement
from quantdsl.domain.model.call_result import CallResult, make_call_result_id, register_call_result
from quantdsl.domain.model.contract_specification import ContractSpecification
from quantdsl.domain.model.market_calibration import MarketCalibration
from quantdsl.domain.model.simulated_price import SimulatedPrice, make_simulated_price_id, register_simulated_price
from quantdsl.domain.services.uuids import create_uuid4
from quantdsl.services import DEFAULT_PRICE_PROCESS_NAME
from quantdsl.tests.test_application import TestCase


class TestEventSourcedRepos(TestCase):
    def test_register_market_calibration(self):
        price_process_name = DEFAULT_PRICE_PROCESS_NAME
        calibration_params = {'param1': 10, 'param2': 20}

        market_calibration = self.app.register_market_calibration(price_process_name, calibration_params)

        assert isinstance(market_calibration, MarketCalibration)
        assert market_calibration.id
        market_calibration = self.app.market_calibration_repo[market_calibration.id]
        assert isinstance(market_calibration, MarketCalibration)
        self.assertEqual(market_calibration.price_process_name, DEFAULT_PRICE_PROCESS_NAME)
        self.assertEqual(market_calibration.calibration_params['param1'], 10)
        self.assertEqual(market_calibration.calibration_params['param2'], 20)

    def test_register_contract_specification(self):
        contract_spec = self.app.register_contract_specification('1 + 1')
        self.assertIsInstance(contract_spec, ContractSpecification)
        self.assertIsInstance(contract_spec.id, six.string_types)
        contract_spec = self.app.contract_specification_repo[contract_spec.id]
        assert isinstance(contract_spec, ContractSpecification)
        self.assertEqual(contract_spec.source_code, '1 + 1')

    def test_register_call_requirements(self):
        call_id = create_uuid4()

        self.assertRaises(KeyError, self.app.call_requirement_repo.__getitem__, call_id)

        dsl_source = '1 + 1'
        effective_present_time = datetime.datetime(2015, 9, 7, 0, 0, 0)

        register_call_requirement(call_id=call_id, dsl_source=dsl_source,
                                  effective_present_time=effective_present_time)

        call_requirement = self.app.call_requirement_repo[call_id]
        assert isinstance(call_requirement, CallRequirement)
        self.assertEqual(call_requirement.dsl_source, dsl_source)
        self.assertEqual(call_requirement.effective_present_time, effective_present_time)

    def test_register_call_dependencies(self):
        call_id = create_uuid4()

        self.assertRaises(KeyError, self.app.call_dependencies_repo.__getitem__, call_id)

        dependencies = ['123', '456']

        self.app.register_call_dependencies(call_id=call_id, dependencies=dependencies)

        call_dependencies = self.app.call_dependencies_repo[call_id]
        assert isinstance(call_dependencies, CallDependencies)
        self.assertEqual(call_dependencies.dependencies, dependencies)

    def test_register_call_dependents(self):
        call_id = create_uuid4()

        self.assertRaises(KeyError, self.app.call_dependents_repo.__getitem__, call_id)

        dependents = ['123', '456']

        self.app.register_call_dependents(call_id=call_id, dependents=dependents)

        call_dependents = self.app.call_dependents_repo[call_id]
        assert isinstance(call_dependents, CallDependents)
        self.assertEqual(call_dependents.dependents, dependents)

    def test_register_call_result(self):
        contract_specification_id = create_uuid4()
        contract_valuation_id = create_uuid4()
        call_id = create_uuid4()

        call_result_id = make_call_result_id(contract_valuation_id, call_id)
        self.assertRaises(KeyError, self.app.call_result_repo.__getitem__, call_result_id)

        register_call_result(call_id=call_id, result_value=123, perturbed_values={},
                             contract_valuation_id=contract_valuation_id,
                             contract_specification_id=contract_specification_id)

        call_result = self.app.call_result_repo[call_result_id]
        assert isinstance(call_result, CallResult)
        self.assertEqual(call_result.result_value, 123)

    def test_register_simulated_price(self):
        price_time = datetime.datetime(2011, 1, 1)
        price_value = scipy.array([1.1, 1.2, 1.367345987359734598734598723459872345987235698237459862345])
        simulation_id = create_uuid4()
        self.assertRaises(KeyError, self.app.simulated_price_repo.__getitem__, simulation_id)

        price = register_simulated_price(simulation_id, '#1', price_time, price_time, price_value)

        assert isinstance(price, SimulatedPrice), price
        assert price.id
        simulated_price_id = make_simulated_price_id(simulation_id, '#1', price_time, price_time)
        self.assertEqual(price.id, simulated_price_id)
        self.app.simulated_price_repo[price.id] = price
        price = self.app.simulated_price_repo[simulated_price_id]
        assert isinstance(price, SimulatedPrice)
        import numpy
        numpy.testing.assert_equal(price.value, price_value)
