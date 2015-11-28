from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish
from quantdsl.domain.services.create_uuid4 import create_uuid4


class DependencyGraph(EventSourcedEntity):

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, **kwargs):
        super(DependencyGraph, self).__init__(**kwargs)


def register_dependency_graph():
    created_event = DependencyGraph.Created(entity_id=create_uuid4())
    contract_specification = DependencyGraph.mutator(event=created_event)
    publish(created_event)
    return contract_specification


class Repository(EntityRepository):
    pass

# class DependencyGraph(object):
#     """
#     A dependency graph of stubbed expressions.
#     """
#     def __init__(self, root_stub_id, call_requirements, dependencies, dependents, leaf_ids):
#         self.root_stub_id = root_stub_id
#         self.leaf_ids = leaf_ids
#         self.call_requirements = call_requirements
#         self.dependencies = dependencies
#         self.dependents = dependents
#
#     def __len__(self):
#         return len(self.call_requirements)

