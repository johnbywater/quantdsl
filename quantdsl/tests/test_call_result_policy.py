from unittest.case import TestCase

from eventsourcing.domain.model.events import assert_event_handlers_empty
from eventsourcing.infrastructure.event_store import EventStore
from eventsourcing.infrastructure.persistence_subscriber import PersistenceSubscriber
from eventsourcing.infrastructure.stored_events.python_objects_stored_events import PythonObjectsStoredEventRepository

from quantdsl.application.call_result_policy import CallResultPolicy
from quantdsl.domain.model.call_dependencies import register_call_dependencies
from quantdsl.domain.model.call_dependents import register_call_dependents
from quantdsl.domain.model.call_result import register_call_result
from quantdsl.infrastructure.event_sourced_repos.call_dependencies_repo import CallDependenciesRepo
from quantdsl.infrastructure.event_sourced_repos.call_dependents_repo import CallDependentsRepo
from quantdsl.infrastructure.event_sourced_repos.call_requirement_repo import CallRequirementRepo


class TestCallResultPolicy(TestCase):
    def setUp(self):
        assert_event_handlers_empty()
        self.es = EventStore(PythonObjectsStoredEventRepository())
        self.ps = PersistenceSubscriber(self.es)
        # self.call_result_repo = CallResultRepo(self.es)
        self.call_result_repo = {}
        self.call_dependencies_repo = CallDependenciesRepo(self.es)
        self.call_dependents_repo = CallDependentsRepo(self.es)
        self.call_requirement_repo = CallRequirementRepo(self.es)
        self.policy = CallResultPolicy(call_result_repo=self.call_result_repo)

    def tearDown(self):
        self.ps.close()
        self.policy.close()
        assert_event_handlers_empty()

    def test_delete_result(self):
        # In this test, there are two "calls": call1 and call2.
        # It is supposed that call1 happens first, and call2 uses the result of call1.
        # Therefore call2 depends upon call1, call1 is a dependency of call2, and call2 is a dependent of call1.
        call1_id = 'call1'
        call2_id = 'call2'
        contract_valuation_id = 'val1'
        contract_specification_id = 'spec1'
        # call1_id = uuid4().hex
        # call2_id = uuid4().hex
        # contract_valuation_id = uuid4().hex
        # contract_specification_id = uuid4().hex

        register_call_dependencies(call2_id, [call1_id])

        # Check the policy has the dependencies for call2.
        self.assertEqual(self.policy.dependencies[call2_id], [call1_id])

        # Register dependents of call1, as call2.
        register_call_dependents(call1_id, [call2_id])

        # Check the policy has the dependencies for call2.
        self.assertEqual(self.policy.dependents[call1_id], [call2_id])

        # Register call result for call1.
        # - this should trigger deletion of call2 result
        call1_result = register_call_result(call1_id, 1.0, {}, contract_valuation_id, contract_specification_id)

        # Check the policy has the result for call1.
        self.assertTrue(call1_result.id in self.policy.result)

        # Check the result for call1 exists.
        self.assertTrue(call1_result.id in self.call_result_repo)

        # Register call result for call2.
        call2_result = register_call_result(call2_id, 1.0, {}, contract_valuation_id, contract_specification_id)

        # Check the policy has the result for call2.
        self.assertTrue(call2_result.id in self.policy.result)

        # Check the result for call2 exists.
        self.assertTrue(call2_result.id in self.call_result_repo)

        # Check the policy does not have the result for call1.
        self.assertFalse(call1_result.id in self.policy.result)

        # Check the result for call1 doesn't exist (because it's dependents have results).
        self.assertFalse(call1_result.id in self.call_result_repo)
