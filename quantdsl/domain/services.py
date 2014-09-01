from quantdsl.domain.model import CallRequirement
# Todo: Change things so domain services don't depend on application services and infrastructure.
from quantdsl.infrastructure.registry import registry

## Domain services.


# def createCallRequirement(id, stubbedExprStr, requiredStubIds, effectivePresentTime):
#     # Create the domain object.
#     callRequirement = CallRequirement(id, stubbedExprStr, requiredStubIds, effectivePresentTime)
#
#     # Register the object with the registry.
#     registry.calls[callRequirement.id] = callRequirement
#     return callRequirement


def createUuid():
    import uuid
    return uuid.uuid4()
