from eventsourcing.domain.model.entity import EventSourcedEntity, EntityRepository
from eventsourcing.domain.model.events import publish


class SimulatedPriceRequirements(EventSourcedEntity):
    """
    Simulated price requirements are the IDs of the simulated prices required by the call requirement with the same ID.
    """

    class Created(EventSourcedEntity.Created):
        pass

    class Discarded(EventSourcedEntity.Discarded):
        pass

    def __init__(self, requirements, **kwargs):
        super(SimulatedPriceRequirements, self).__init__(**kwargs)
        self._requirements = requirements

    @property
    def requirements(self):
        return self._requirements


def register_simulated_price_requirements(call_requirement_id, requirements):
    assert isinstance(requirements, list), type(requirements)
    event = SimulatedPriceRequirements.Created(entity_id=call_requirement_id, requirements=requirements)
    entity = SimulatedPriceRequirements.mutator(event=event)
    publish(event)
    return entity


class SimulatedPriceRequirementsRepository(EntityRepository):
    pass
