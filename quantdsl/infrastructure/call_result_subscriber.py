from eventsourcing.domain.model.events import subscribe, unsubscribe

from quantdsl.domain.model.call_result import CallResult
from quantdsl.domain.services.call_links import get_next_call_id


# def get_next_call_id(call_link_repo, call_result_id):
#     pass


class CallResultSubscriber(object):
    """
    Listens for CallResult.Created events.
    """

    def __init__(self, call_result_queue, call_evaluation_queue, call_link_repo, call_dependencies_repo,
                 call_dependents_repo, call_result_repo):
        self.call_result_queue = call_result_queue
        self.call_evaluation_queue = call_evaluation_queue
        self.call_link_repo = call_link_repo
        self.call_dependencies_repo = call_dependencies_repo
        self.call_dependents_repo = call_dependents_repo
        self.call_result_repo = call_result_repo
        subscribe(self.call_result_created, self.continue_computation)

    def close(self):
        unsubscribe(self.call_result_created, self.continue_computation)

    def call_result_created(self, event):
        return isinstance(event, CallResult.Created)

    def continue_computation(self, event):
        """
        If it has a result queue, puts the result on the result queue. In this case it
        is expected that there is a result queue worker operating on the result queue, but
        that is out of scope of this object.

        If there is no result queue, then it is the responsibility of this class to identify
        the call ids of the next calls to be evaluated. The call links can be used to
        find the next in series. That depends on starting by evaluating the first item in
        the evaluation call order. Alternatively, results can be used to see if any dependents
        have all their dependencies satisfied. That depends on starting by evaluating all
        the leaf nodes.

        If it has an evaluation queue, it puts the next call id(s) on the evaluation queue.

        If there is no evaluation queue, then it is not the responsibility of this class to
        evaluate the calls, because that would soon hit recursion limits.
        """
        assert isinstance(event, CallResult.Created)
        contract_valuation_id = event.contract_valuation_id
        dependency_graph_id = event.dependency_graph_id
        call_result_id = event.entity_id

        # If we have a result queue, just put the call ID on the result queue.
        if self.call_result_queue:
            self.call_result_queue.put((contract_valuation_id, call_result_id))

        # If we have an evaluation queue, identify the next calls to be evaluated and put them on the queue.
        elif self.call_evaluation_queue:
            # Todo: What decides between following a series order and parallel order?
            # Todo: - without parallel order, there is no point having more than one evaluation worker. So
            # Todo:   if there are more than one evaluation workers, you need to do it in parallel? Not
            # Todo:   necessarily because the other workers could be working on other computations at the same time.
            # Todo: So you may benefit from parallel mode if the computation is widely branched.
            # Todo: But there might be race conditions so that doing it in parallel is unstable.
            # Todo: Therefore do it in parallel by default? At the moment it is done in series...
            # Follow the series call order, the last call_result_id is the dependency_graph_id.
            if call_result_id != dependency_graph_id:
                next_call_id = get_next_call_id(self.call_link_repo, call_result_id)
                if next_call_id != None:
                    self.call_evaluation_queue.put((dependency_graph_id, contract_valuation_id, next_call_id))

        # Otherwise, assume the calls are being made on a loop. Don't start recursing in Python. :-)

        # """
        # To continue the calculation, either follow the serial execution order (call link repo)
        # or look at which call dependents have no call dependencies without a call result.
        #
        # To avoid deep recursion, the next call cannot be made directly from here. So we need a queue.
        #
        # The queue could processed by a loop, waiting to get, with the put done here before returning
        # in the loop.
        #
        # The loop could be a celery worker, in which case this would need to call a celery task. The
        # task would need to access the application to do it's evaluation work. The call here would be
        # asynchronous, and the worker would register the result of the evaluation and this subscriber
        # running in the worker's application would call the celery task successively.
        #
        # The loop could be in the caller, but this subscriber would need access to the queue. The celery task
        # is a way of putting something on a queue. So in general we can say that:
        #
        # When this subscriber gets a call id that is ready to be executed, it puts the call on a call execution queue.
        #
        # That could be implemented either by calling a celery task async, by putting something on an in-process
        # queue and returning so the caller loop can do its work, or putting something on a shared or multiprocessing queue
        # and having the caller loop use a pool to make async call evaluation requests.
        #
        # It would also be possible for this subscriber to post call result ids to a queue, perhaps by making an async
        # celery task call. In which case a results worker could respond by identifying the next calls (one way or
        # another, see above) and adding call ids to the call execution queue.
        #
        # Alternatively, a results worker could directly evaluate all the dependents that are ready and publish results
        # to the call result queue it is working on. The leaf calls would need to be evaluated by the caller, to seed the
        # process. In this case, there would not need to be an evaluation queue or any evaluation workers.
        #
        # Having two workers would seem to allow highest throughput, since evaluation workers can get on with evaluating
        # ready calls, and the calls needed to work out how to continue the computation can be done in parallel. This gives
        # an opportunity to serialise the results, reducing the chance of a race condition terminating the sequence early.
        #
        # So basically if this subscriber has results queue, it will put call result ids on the results queue. If there
        # is no results queue, it will identify the calls that are ready to run. If it has an execution queue, it will
        # put the call ids that are ready on the call execution queue. If there is no results queue, it will directly
        # evaluate the call which will publish a result (which will be recursive and therefore limiting).
        #
        # If there is a results queue, there will need to be a results queue loop, which handles the result by identifying
        # calls that are ready to be executed and doing something with those call ids.
        #
        # If the results queue worker has an execution queue, it will put call ids that are ready to run on the execution
        # queue. If is has no execution queue, then it will directly evaluation the call, which will publish a result that
        # will be handled by this subscriber.
        #
        # If celery is used, then making a task call is effectively posting to the queue. Celery workers
        # need to be started independently, and the stored events backend needs to be a persistent database.
        #
        # If multiprocessing is used, then mp.Queue objects can be passed to the mp.Process objects, and the
        # process objects can put items onto the queue. The multiprocessing worker processes do not need
        # to be started independently. The stored events backend needs to be a persistent database. It might
        # be possible to have a shared memory python objects stored event repo, but we don't have one yet.
        #
        # If a single process is used with many threads, then native Queue objects can be used with
        # native Thread objects.
        #
        # Alternatively if single thread is used, the execution order can be regenerated and used
        # to conduct the computation. In that case, this subscriber should do nothing.
        #
        #
        # (Aside: If the end of the calculation is completed, it seems to make sense to publish a domain event.
        # Perhaps a "ContractValuation.Completed" event would be useful to notify when the calculation has
        # completed, and to allow the status of the contract valuation to be seen. Although, the existence
        # of the root call result indicates when the calculation is completed.)
        #
        # """
