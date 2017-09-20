from eventsourcing.domain.model.events import subscribe, unsubscribe

from quantdsl.domain.model.call_dependencies import CallDependencies
from quantdsl.domain.model.call_dependents import CallDependents
from quantdsl.domain.model.call_result import CallResult, make_call_result_id
from quantdsl.infrastructure.event_sourced_repos.call_result_repo import CallResultRepo


class CallResultPolicy(object):
    def __init__(self, call_result_repo, call_evaluation_queue=None):
        self.call_result_repo = call_result_repo
        self.call_evaluation_queue = call_evaluation_queue
        self.result = {}
        self.dependents = {}
        self.dependencies = {}
        self.outstanding_dependents = {}
        self.outstanding_dependencies = {}

        subscribe(self.is_call_dependencies_created, self.cache_dependencies)
        subscribe(self.is_call_dependents_created, self.cache_dependents)
        subscribe(self.is_call_result_created, self.cache_result)
        subscribe(self.is_call_result_discarded, self.purge_result)

    def close(self):
        unsubscribe(self.is_call_dependencies_created, self.cache_dependencies)
        unsubscribe(self.is_call_dependents_created, self.cache_dependents)
        unsubscribe(self.is_call_result_created, self.cache_result)
        unsubscribe(self.is_call_result_discarded, self.purge_result)

    def is_call_dependencies_created(self, event):
        return isinstance(event, CallDependencies.Created)

    def is_call_dependents_created(self, event):
        return isinstance(event, CallDependents.Created)

    def is_call_result_created(self, event):
        return isinstance(event, CallResult.Created)

    def is_call_result_discarded(self, event):
        return isinstance(event, CallResult.Discarded)

    def cache_dependencies(self, event):
        assert isinstance(event, CallDependencies.Created)
        call_result_id = event.entity_id
        self.dependencies[call_result_id] = event.dependencies[:]

        # Count one outstanding dependency for each dependency of the call.
        # - minus 1, so that pop() raises at the right time, see below
        if self.call_evaluation_queue:
            self.outstanding_dependencies[call_result_id] = [None] * (len(event.dependencies) - 1)

    def cache_dependents(self, event):
        assert isinstance(event, CallDependents.Created)
        call_result_id = event.entity_id
        self.dependents[call_result_id] = event.dependents[:]

        # Count one outstanding dependent for each dependent of the call.
        # - minus 1, so that pop() raises at the right time, see below
        self.outstanding_dependents[call_result_id] = [None] * (len(event.dependents) - 1)

    def cache_result(self, event):
        assert isinstance(event, CallResult.Created)

        # Remember the call result entity.
        this_result_id = event.entity_id
        call_result = CallResult.mutator(event=event)
        self.result[this_result_id] = call_result
        if isinstance(self.call_result_repo, dict):
            self.call_result_repo[this_result_id] = call_result

        # Decrement outstanding dependents for each dependency of this result.
        for dependency_id in self.dependencies.get(event.call_id, ()):
            try:
                self.outstanding_dependents[dependency_id].pop()
            except IndexError:
                # Discard the result when there are no longer any outstanding dependents.
                dependent_result_id = make_call_result_id(event.contract_valuation_id, dependency_id)
                self.result[dependent_result_id].discard()

        # Remove one outstanding dependency for each dependent of this result.
        if self.call_evaluation_queue:
            for dependent_id in self.dependents.get(event.call_id, ()):
                try:
                    self.outstanding_dependencies[dependent_id].pop()
                except IndexError:
                    # Queue the call if there are no more outstanding results.
                    job = (event.contract_specification_id, event.contract_valuation_id, dependent_id)
                    self.call_evaluation_queue.put(job)

    def purge_result(self, event):
        assert isinstance(event, CallResult.Discarded)
        # Remove from the local results dict.
        del (self.result[event.entity_id])

        # Remove from the call result repo.
        if isinstance(self.call_result_repo, dict):
            del (self.call_result_repo[event.entity_id])
        elif isinstance(self.call_result_repo, CallResultRepo):
            del (self.call_result_repo._cache[event.entity_id])
