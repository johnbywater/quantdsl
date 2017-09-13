from abc import abstractmethod
from collections import defaultdict

import six.moves.queue as queue

from quantdsl.domain.model.call_dependencies import CallDependenciesRepository, register_call_dependencies
from quantdsl.domain.model.call_dependents import CallDependentsRepository, register_call_dependents
from quantdsl.domain.model.call_leafs import register_call_leafs
from quantdsl.domain.model.call_link import register_call_link
from quantdsl.domain.model.call_requirement import StubbedCall, register_call_requirement
from quantdsl.domain.model.contract_specification import ContractSpecification
from quantdsl.domain.services.parser import dsl_parse
from quantdsl.exceptions import DslSyntaxError
from quantdsl.semantics import DslExpression, DslNamespace, FunctionDef, Module, PendingCall, Stub


def generate_dependency_graph(contract_specification, call_dependencies_repo, call_dependents_repo, call_leafs_repo,
                              call_requirement_repo):
    assert isinstance(contract_specification, ContractSpecification)
    dsl_module = dsl_parse(dsl_source=contract_specification.source_code)
    assert isinstance(dsl_module, Module)
    dsl_globals = dsl_module.namespace.copy()
    function_defs, expressions = extract_defs_and_exprs(dsl_module, dsl_globals)
    dsl_expr = expressions[0]
    assert isinstance(dsl_expr, DslExpression)
    dsl_locals = DslNamespace()

    leaf_ids = []
    all_dependents = defaultdict(list)

    # Generate stubbed call from the parsed DSL module object.
    for stub in generate_stubbed_calls(contract_specification.id, dsl_module, dsl_expr, dsl_globals, dsl_locals):
        # assert isinstance(stub, StubbedCall)

        # Register the call requirements.
        call_id = stub.call_id
        dsl_source = str(stub.dsl_expr)
        effective_present_time = stub.effective_present_time
        call_requirement = register_call_requirement(call_id, dsl_source, effective_present_time)

        # Hold onto the dsl_expr, helps in "single process" modes....
        call_requirement._dsl_expr = stub.dsl_expr
        # - put the entity directly in the cache, otherwise the entity will be regenerated when it is next accessed
        #   and the _dsl_expr will be lost.
        call_requirement_repo.add_cache(call_id, call_requirement)

        # Register the call requirements.
        dependencies = stub.requirements
        register_call_dependencies(call_id, dependencies)

        # Keep track of the leaves and the dependents.
        if len(dependencies) == 0:
            leaf_ids.append(call_id)
        else:
            for dependency_call_id in dependencies:
                all_dependents[dependency_call_id].append(call_id)

    # Register the call dependents.
    for call_id, dependents in all_dependents.items():
        register_call_dependents(call_id, dependents)
    register_call_dependents(contract_specification.id, [])

    # Generate and register the call order.
    link_id = contract_specification.id
    for call_id in generate_execution_order(leaf_ids, call_dependents_repo, call_dependencies_repo):
        register_call_link(link_id, call_id)
        link_id = call_id

    # Register the leaf ids.
    register_call_leafs(contract_specification.id, leaf_ids)


def generate_execution_order(leaf_call_ids, call_dependents_repo, call_dependencies_repo):
    """
    Topological sort, using Kahn's algorithm.
    """
    assert isinstance(call_dependents_repo, CallDependentsRepository)
    assert isinstance(call_dependencies_repo, CallDependenciesRepository)

    # Initialise set of nodes that have no outstanding requirements with the leaf nodes.
    S = set(leaf_call_ids)
    removed_edges = defaultdict(set)
    while S:

        # Pick a node, n, that has zero outstanding requirements.
        n = S.pop()

        # Yield node n.
        yield n

        # Get dependents, if any were registered.
        try:
            dependents = call_dependents_repo[n]
        except KeyError:
            continue

        # Visit the nodes that are dependent on n.
        for m in dependents:

            # Remove the edge n to m from the graph.
            removed_edges[m].add(n)

            # If there are zero edges to m that have not been removed, then we
            # can add m to the set of nodes with zero outstanding requirements.
            for d in call_dependencies_repo[m]:
                if d not in removed_edges[m]:
                    break
            else:
                # Forget about removed edges to m.
                removed_edges.pop(m)

                # Add m to the set of nodes that have zero outstanding requirements.
                S.add(m)

                # Todo: Restore the check for remaining (unremoved) edges. Hard to do from the domain model,
                # so perhaps get all nodes in memory and actually remove them from a
                # collection so that we can see if anything remains unremoved (indicates cyclical dependencies).


def generate_stubbed_calls(root_stub_id, dsl_module, dsl_expr, dsl_globals, dsl_locals):
    # Create a stack of discovered calls to function defs.
    # - since we are basically doing a breadth-first search, the pending call queue
    #   will be the max width of the graph, so it might sometimes be useful to
    #   persist the queue to allow for larger graph. For now, just use a Python queue.
    pending_call_stack = PythonPendingCallQueue()

    # Reduce the module object into a "root" stubbed expression with pending calls on the stack.
    # - If an expression has a FunctionCall, it will cause a pending
    # call to be placed on the pending call stack, and the function call will be
    # replaced with a stub, which acts as a placeholder for the result of the function
    # call. By looping over the pending call stack until it is empty, evaluating
    # pending calls to generate stubbed expressions and further pending calls, the
    # module can be compiled into a stack of stubbed expressions.
    # Of course if the module's expression doesn't have a function call, there
    # will just be one expression on the stack of "stubbed" expressions, and it will
    # not have any stubs, and there will be no pending calls on the pending call stack.
    stubbed_expr = dsl_expr.reduce(
        dsl_locals,
        DslNamespace(dsl_globals),
        pending_call_stack=pending_call_stack
    )

    dependencies = list_stub_dependencies(stubbed_expr)
    yield StubbedCall(root_stub_id, stubbed_expr, None, dependencies)

    # Continue by looping over any pending calls.
    while not pending_call_stack.empty():
        # Get the next pending call.
        pending_call = pending_call_stack.get()
        # assert isinstance(pending_call, PendingCall), pending_call

        # Get the function def.
        function_def = pending_call.stacked_function_def
        # assert isinstance(function_def, FunctionDef), type(function_def)

        # Apply the stacked call values to the called function def.
        stubbed_expr = function_def.apply(pending_call.stacked_globals,
                                          pending_call.effective_present_time,
                                          pending_call_stack=pending_call_stack,
                                          is_destacking=True,
                                          **pending_call.stacked_locals)

        # Put the resulting (potentially stubbed) expression on the stack of stubbed expressions.
        dependencies = list_stub_dependencies(stubbed_expr)

        yield StubbedCall(pending_call.stub_id, stubbed_expr, pending_call.effective_present_time, dependencies)


def list_stub_dependencies(stubbed_expr):
    return [s.name for s in stubbed_expr.list_instances(Stub)]


def extract_defs_and_exprs(dsl_module, dsl_globals):
    # Pick out the expressions and function defs from the module body.
    function_defs = []
    expressions = []
    for dsl_obj in dsl_module.body:

        if isinstance(dsl_obj, FunctionDef):
            # Todo: Move this setting of globals elsewhere, it doesn't belong here.
            dsl_globals[dsl_obj.name] = dsl_obj
            # Todo: Move this setting of the 'enclosed namespace' - is this even a good idea?
            # Share the module level namespace (any function body can call any other function).
            dsl_obj.enclosed_namespace = dsl_globals

            function_defs.append(dsl_obj)
        elif isinstance(dsl_obj, DslExpression):
            expressions.append(dsl_obj)
        else:
            raise DslSyntaxError("'%s' not allowed in module" % type(dsl_obj), dsl_obj, node=dsl_obj.node)

    return function_defs, expressions


class PendingCallQueue(object):
    def put(self, stub_id, stacked_function_def, stacked_locals, stacked_globals, effective_present_time):
        pending_call = self.validate_pending_call(effective_present_time, stacked_function_def, stacked_globals,
                                                  stacked_locals, stub_id)
        self.put_pending_call(pending_call)

    def validate_pending_call(self, effective_present_time, stacked_function_def, stacked_globals, stacked_locals,
                              stub_id):
        # assert isinstance(stub_id, six.string_types), type(stub_id)
        # assert isinstance(stacked_function_def, FunctionDef), type(stacked_function_def)
        # assert isinstance(stacked_locals, DslNamespace), type(stacked_locals)
        # assert isinstance(stacked_globals, DslNamespace), type(stacked_globals)
        # assert isinstance(effective_present_time, (datetime.datetime, type(None))), type(effective_present_time)
        pending_call = PendingCall(stub_id, stacked_function_def, stacked_locals, stacked_globals,
                                   effective_present_time)
        return pending_call

    @abstractmethod
    def put_pending_call(self, pending_call):
        """
        Puts pending call on the queue.
        """

    @abstractmethod
    def get(self):
        """
        Gets pending call from the queue.
        """


class PythonPendingCallQueue(PendingCallQueue):
    def __init__(self):
        self.queue = queue.Queue()

    def put_pending_call(self, pending_call):
        self.queue.put(pending_call)

    def empty(self):
        return self.queue.empty()

    def get(self, *args, **kwargs):
        return self.queue.get(*args, **kwargs)
