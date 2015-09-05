from quantdsl.domain.model import CallRequirement
# Todo: Change things so domain services don't depend on application services and infrastructure.
from quantdsl.infrastructure.registry import registry

## Domain services.


# def createCallRequirement(id, stubbed_expr_str, requiredStubIds, effective_present_time):
#     # Create the domain object.
#     callRequirement = CallRequirement(id, stubbed_expr_str, requiredStubIds, effective_present_time)
#
#     # Register the object with the registry.
#     registry.calls[callRequirement.id] = callRequirement
#     return callRequirement


def createUuid():
    import uuid
    return uuid.uuid4()
