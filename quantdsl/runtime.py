import datetime
import multiprocessing
import Queue as queue

from quantdsl.semantics import FunctionDef, DslNamespace, DslExpression
from quantdsl.domain.services import executeCallRequirement
from quantdsl.infrastructure.registry import registry


## "Run time environment."


class FunctionDefCallStack(queue.Queue):

    def put(self, stubId, stackedCall, stackedLocals, stackedGlobals, effectivePresentTime):
        assert isinstance(stubId, basestring), type(stubId)
        assert isinstance(stackedCall, FunctionDef), type(stackedCall)
        assert isinstance(stackedLocals, DslNamespace), type(stackedLocals)
        assert isinstance(stackedGlobals, DslNamespace), type(stackedGlobals)
        assert isinstance(effectivePresentTime, (datetime.datetime, type(None))), type(effectivePresentTime)
        queue.Queue.put(self, (stubId, stackedCall, stackedLocals, stackedGlobals, effectivePresentTime))


class StubbedExpressionStack(queue.LifoQueue):

    def put(self, stubId, stubbedExpr, effectivePresentTime):
        assert isinstance(stubId, basestring), type(stubId)
        assert isinstance(stubbedExpr, DslExpression), type(stubbedExpr)
        assert isinstance(effectivePresentTime, (datetime.datetime, type(None))), type(effectivePresentTime)
        queue.LifoQueue.put(self, (stubId, stubbedExpr, effectivePresentTime))


class DependencyGraphRunner(object):

    def __init__(self, rootCallRequirementId, leafIds, isMultiprocessing, poolSize=None):
        self.rootCallRequirementId = rootCallRequirementId
        self.leafIds = leafIds
        self.isMultiprocessing = isMultiprocessing
        self.registry = registry
        self.poolSize = poolSize

    def run(self, **kwds):
        self.runKwds = kwds
        self.callCount = 0
        if self.isMultiprocessing:
            self.executionQueueManager = multiprocessing.Manager()
            self.executionQueue = self.executionQueueManager.Queue()
            self.resultsDict = self.executionQueueManager.dict()
            self.callsDict = self.executionQueueManager.dict()
            self.callsDict.update(registry.calls)
        else:
            self.executionQueue = queue.Queue()
            self.resultsDict = registry.results
            self.callsDict = registry.calls
        for callRequirementId in self.leafIds:
            self.executionQueue.put(callRequirementId)
        pool = None
        if self.isMultiprocessing:
            pool = multiprocessing.Pool(processes=self.poolSize)

        while not self.executionQueue.empty():
            if self.isMultiprocessing:
                batchCallRequirementIds = []
                while not self.executionQueue.empty():
                    batchCallRequirementIds.append(self.executionQueue.get())
                pool.map_async(executeCallRequirement,
                    [(i, kwds, self.resultsDict, self.executionQueue, self.callsDict) for i in batchCallRequirementIds]
                ).get(99999999)  # Do this rather than just call map(), because otherwise Ctrl-C doesn't work.
                self.callCount += len(batchCallRequirementIds)
            else:
                callRequirementId = self.executionQueue.get()
                executeCallRequirement((callRequirementId, kwds, self.resultsDict, self.executionQueue, self.callsDict))
                self.callCount += 1


