## Domain model objects.

class DomainObject(object): pass


class CallRequirement(DomainObject):
    def __init__(self, id, stubbedExprStr, effectivePresentTime, requiredCallIds, notifyIds):
        self.id = id
        self.stubbedExprStr = stubbedExprStr
        self.effectivePresentTime = effectivePresentTime
        self.requiredCallIds = requiredCallIds
        self.notifyIds = notifyIds
        # Todo: Validate.


class Result(DomainObject):
    def __init__(self, id, returnValue):
        self.id = id
        self.value = returnValue
