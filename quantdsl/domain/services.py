from quantdsl.exceptions import QuantDslSystemError, QuantDslSyntaxError
from quantdsl.semantics import Module, DslNamespace, Number, DslExpression
from quantdsl.domain.model import CallRequirement, Result
# Todo: Change things so domain services don't depend on application services and infrastructure.
from quantdsl.infrastructure.registry import registry

## Domain services.


def createCallRequirement(id, stubbedExprStr, requiredStubIds, effectivePresentTime):
    # Create the domain object.
    callRequirement = CallRequirement(id, stubbedExprStr, requiredStubIds, effectivePresentTime)

    # Register the object with the registry.
    registry.calls[callRequirement.id] = callRequirement
    return callRequirement


def createResult(callRequirementId, returnValue, resultsDict):
    assert isinstance(callRequirementId, basestring), type(callRequirementId)
    result = Result(id=callRequirementId, returnValue=returnValue)
    resultsDict[callRequirementId] = result


def createUuid():
    import uuid
    return uuid.uuid4()


def executeCallRequirement(args):
    """
    Executes the call requirement, produces a value from the stubbed expr and creates a result..
    """
    try:
        # Get call requirement ID and modelled function objects.
        callRequirementId, evaluationKwds, resultsRegister, executionQueue, callsDict = args

        # Get the call requirement object (it has the stubbedExpr and effectivePresentTime).
        callRequirement = callsDict[callRequirementId]

        assert isinstance(callRequirement, CallRequirement), "Call requirement object is not a CallRequirement" \
                                                             ": %s" % callRequirement
        if not callRequirement.isReady(resultsRegister=resultsRegister):
            raise QuantDslSystemError("Call requirement '%s' is not actually ready! It shouldn't have got here" \
                                      " without all required results being available. Is the results register" \
                                      " stale?" % callRequirement.id)

        # If necessary, overwrite the effectivePresentTime as the presentTime in the evaluationKwds.
        if callRequirement.effectivePresentTime:
            evaluationKwds['presentTime'] = callRequirement.effectivePresentTime

        # Evaluate the stubbed expr str.
        try:
            # Todo: Rework this dependency.
            from quantdsl.services import parse
            stubbedModule = parse(callRequirement.stubbedExprStr)
        except QuantDslSyntaxError:
            raise

        assert isinstance(stubbedModule, Module), "Parsed stubbed expr string is not an module: %s" % stubbedModule

        dslNamespace = DslNamespace()
        for stubId in callRequirement.requiredCallIds:
            stubResult = resultsRegister[stubId]
            assert isinstance(stubResult, Result), "Not an instance of Result: %s" % stubResult
            dslNamespace[stubId] = Number(stubResult.value)

        simpleExpr = stubbedModule.compile(dslLocals=dslNamespace, dslGlobals={})
        assert isinstance(simpleExpr, DslExpression), "Reduced parsed stubbed expr string is not an " \
                                                      "expression: %s" % type(simpleExpr)
        resultValue = simpleExpr.evaluate(**evaluationKwds)
        handleResult(callRequirementId, resultValue, resultsRegister, executionQueue, callsDict)
        return "OK"
    except Exception, e:
        import traceback
        msg = "Error whilst calling 'executeCallRequirement': %s" % traceback.format_exc()
        msg += str(e)
        raise Exception(msg)


def handleResult(callRequirementId, resultValue, resultsDict, executionQueue, callsDict):
    # Create result object and check if subscribers are ready to be executed.
    createResult(callRequirementId, resultValue, resultsDict)
    callRequirement = callsDict[callRequirementId]
    for subscriberId in callRequirement.subscribers:
        if subscriberId in resultsDict:
            continue
        subscriber = callsDict[subscriberId]
        if subscriber.isReady(resultsDict):
            executionQueue.put(subscriberId)





