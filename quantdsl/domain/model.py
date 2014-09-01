## Domain model objects.

class DomainObject(object): pass


class CallRequirement(DomainObject):
    def __init__(self, id, stubbedExprStr, requiredCallIds, effectivePresentTime):
        self.id = id
        self.stubbedExprStr = stubbedExprStr
        self.requiredCallIds = requiredCallIds
        self.effectivePresentTime = effectivePresentTime
        self.subscribers = []
        # Todo: Validate.

    def isReady(self, resultsRegister):
        for cid in self.requiredCallIds:
            if cid not in resultsRegister:
                return False
        return True

    def registerSubscription(self, callRequirementId):
        if callRequirementId not in self.subscribers:
            self.subscribers.append(callRequirementId)


class Result(DomainObject):
    def __init__(self, id, returnValue):
        self.id = id
        self.value = returnValue


