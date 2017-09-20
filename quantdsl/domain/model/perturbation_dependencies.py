from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class PerturbationDependencies(EventSourcedEntity):
    """
    Perturbation requirements are the names of the perturbed values required by call requirement with this entity ID.
    """

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, dependencies, **kwargs):
        super(PerturbationDependencies, self).__init__(**kwargs)
        self._dependencies = dependencies

    @property
    def dependencies(self):
        return self._dependencies


def register_perturbation_dependencies(call_requirement_id, dependencies):
    created_event = PerturbationDependencies.Created(entity_id=call_requirement_id, dependencies=dependencies)
    perturbation_dependencies = PerturbationDependencies.mutator(event=created_event)
    publish(created_event)
    return perturbation_dependencies


class PerturbationDependenciesRepository(EntityRepository):
    pass
