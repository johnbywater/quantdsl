from abc import ABCMeta, abstractmethod
from quantdsl.exceptions import DslSyntaxError, DslSystemError
from quantdsl.semantics import Module, DslNamespace, DslExpression, Stub


class DependencyGraph(object):
    """
    Constructs dependency graph from stack of stubbed expressions.

    The calls are stored in a dict, keyed by ID. They call IDs are
    kept in a list to maintain order. The calls which don't depend
    on any other calls are identified as the 'leafIds'. The calls
    which are dependent on a call are known as 'dependencyIds'. And
    the calls which depend on a call are called 'notifyIds'.
    """
    def __init__(self, rootStubId, stubbedExprsData):
        self.rootStubId = rootStubId
        assert isinstance(stubbedExprsData, list)
        assert len(stubbedExprsData), "Stubbed expressions is empty!"
        self.stubbedExprsData = stubbedExprsData
        self.leafIds = []
        self.callRequirementIds = []
        self.callRequirements = {}
        self.dependencyIds = {}
        self.notifyIds = {self.rootStubId: []}
        for stubId, stubbedExpr, effectivePresentTime in self.stubbedExprsData:

            assert isinstance(stubbedExpr, DslExpression)

            # Finding stub instances reveals the dependency graph.
            requiredStubIds = [s.name for s in stubbedExpr.findInstances(Stub)]

            self.dependencyIds[stubId] = requiredStubIds

            if len(requiredStubIds):
                # Each required stub needs to notify its dependents.
                for requiredStubId in requiredStubIds:
                    if requiredStubId not in self.notifyIds:
                        notifyIds = []
                        self.notifyIds[requiredStubId] = notifyIds
                    else:
                        notifyIds = self.notifyIds[requiredStubId]
                    if stubId in notifyIds:
                        raise DslSystemError("Stub ID already in dependents of required stub. Probably wrong?")
                    notifyIds.append(stubId)
            else:
                # Keep a track of the leaves of the dependency graph (stubbed exprs that don't depend on anything).
                self.leafIds.append(stubId)

            # Stubbed expr has names that need to be replaced with results of other stubbed exprs.
            stubbedExprStr = str(stubbedExpr)


            self.callRequirements[stubId] = (stubbedExprStr, effectivePresentTime)
            self.callRequirementIds.append(stubId)

        # Sanity check.
        assert self.rootStubId in self.callRequirementIds

    def __len__(self):
        return len(self.stubbedExprsData)

    def hasInstances(self, dslType):
        for _, stubbedExpr, _ in self.stubbedExprsData:
            if stubbedExpr.hasInstances(dslType=dslType):
                return True
        return False

    def findInstances(self, dslType):
        instances = []
        for _, stubbedExpr, _ in self.stubbedExprsData:
            [instances.append(i) for i in stubbedExpr.findInstances(dslType=dslType)]
        return instances


    def evaluate(self, dependencyGraphRunnerClass=None, poolSize=None, **kwds):
        # Make sure we've got a dependency graph runner.
        if dependencyGraphRunnerClass is None:
            dependencyGraphRunnerClass = SingleThreadedDependencyGraphRunner
        assert issubclass(dependencyGraphRunnerClass, DependencyGraphRunner)

        # Init and run the dependency graph runner.
        if poolSize:
            self.runner = dependencyGraphRunnerClass(self, poolSize=poolSize)
        else:
            self.runner = dependencyGraphRunnerClass(self)
        self.runner.run(**kwds)

        try:
            return self.runner.resultsDict[self.rootStubId]
        except KeyError, e:
            errorData = (self.rootStubId, self.runner.resultsDict.keys())
            raise DslSystemError("root value not found", str(errorData))


class DependencyGraphRunner(object):

    __metaclass__ = ABCMeta

    def __init__(self, dependencyGraph):
        self.dependencyGraph = dependencyGraph
        self.executionQueue = None

    def run(self, **kwds):
        self.runKwds = kwds
        self.callCount = 0
        self.initQueuesAndDicts()
        # Put the leaves on the execution queue.
        for callRequirementId in self.dependencyGraph.leafIds:
            self.executionQueue.put(callRequirementId)
        # Process items on the queue until queue is empty.
        while not self.executionQueue.empty():
            self.executeWaitingCalls()

    @abstractmethod
    def initQueuesAndDicts(self):
        pass

    @abstractmethod
    def executeWaitingCalls(self):
        pass


class SingleThreadedDependencyGraphRunner(DependencyGraphRunner):

    def initQueuesAndDicts(self):
        import Queue as queue
        self.executionQueue = queue.Queue()
        self.resultsDict = {}
        self.resultIds = {}
        self.callsDict = self.dependencyGraph.callRequirements.copy()
        self.dependencyDict = self.dependencyGraph.dependencyIds.copy()
        self.notifyDict = self.dependencyGraph.notifyIds.copy()

    def executeWaitingCalls(self):
        callRequirementId = self.executionQueue.get()
        executeCallRequirement((callRequirementId, self.runKwds, self.resultsDict, self.resultIds, self.executionQueue,
                                               self.callsDict, self.dependencyDict, self.notifyDict))
        self.callCount += 1


class MultiProcessingDependencyGraphRunner(DependencyGraphRunner):

    def __init__(self, dependencyGraph, poolSize=None):
        super(MultiProcessingDependencyGraphRunner, self).__init__(dependencyGraph)
        self.poolSize = poolSize

    @property
    def multiProcessingPool(self):
        import multiprocessing
        if not hasattr(self, '_multiProcessingPool'):
            pool = multiprocessing.Pool(processes=self.poolSize)
            self._multiProcessingPool = pool
        return self._multiProcessingPool

    def initQueuesAndDicts(self):
        # Make shared memory objects, from the native Python data objects.
        import multiprocessing
        self.executionQueueManager = multiprocessing.Manager()
        self.executionQueue = self.executionQueueManager.Queue()
        self.resultsDict = self.executionQueueManager.dict()
        self.resultIds = self.executionQueueManager.dict()
        self.callsDict = self.executionQueueManager.dict()
        self.callsDict.update(self.dependencyGraph.callRequirements)
        self.dependencyDict = self.executionQueueManager.dict()
        self.dependencyDict.update(self.dependencyGraph.dependencyIds)
        self.notifyDict = self.executionQueueManager.dict()
        self.notifyDict.update(self.dependencyGraph.notifyIds)
        # if 'allMarketPrices' in self.runKwds:
        #     allMarketPrices = self.runKwds.pop('allMarketPrices')
        #     allMarketPricesDict = self.executionQueueManager.dict()
        #     for marketName, marketPrices in allMarketPrices.items():
        #         marketPricesDict = self.executionQueueManager.dict()
        #         allMarketPricesDict[marketName] = marketPricesDict
        #         for fixingDate, marketPrice in marketPrices.items():
        #             marketPricesDict[fixingDate] = marketPrice
        #     self.runKwds['allMarketPrices'] = allMarketPricesDict

    def executeWaitingCalls(self):
        batchCallRequirementIds = []
        while not self.executionQueue.empty():
            batchCallRequirementIds.append(self.executionQueue.get())
        self.multiProcessingPool.map_async(executeCallRequirement,
                       [(i, self.runKwds, self.resultsDict, self.resultIds, self.executionQueue, self.callsDict, self.dependencyDict, self.notifyDict) for i in
                        batchCallRequirementIds]
        ).get(99999999)  # Do this rather than just call map(), because otherwise Ctrl-C doesn't work.
        self.callCount += len(batchCallRequirementIds)


def executeCallRequirement(args):
    """
    Evaluates the stubbed expr, and checks if any dependents are ready to execute.

    Needs to be a module-level function so that multiprocessing can find it.
    """
    try:
        # Get call requirement ID and modelled function objects.
        callRequirementId, evaluationKwds, resultsDict, resultIds, executionQueue, callsDict, dependencyDict, notifyDict = args

        # Get the call requirement object (it has the stubbedExpr and effectivePresentTime).
        stubbedExprStr, effectivePresentTime = callsDict[callRequirementId]

        # If necessary, overwrite the effectivePresentTime as the presentTime in the evaluationKwds.
        if effectivePresentTime:
            evaluationKwds['presentTime'] = effectivePresentTime

        # Evaluate the stubbed expr str.
        try:
            # Todo: Rework this dependency. Especially figure out how to use alternative set of DSL classes when multiprocessing.
            from quantdsl.services import parse
            stubbedModule = parse(stubbedExprStr)
        except DslSyntaxError:
            raise

        assert isinstance(stubbedModule, Module), "Parsed stubbed expr string is not an module: %s" % stubbedModule

        # Get all the required stub expr result values in a namespace object.
        dslNamespace = DslNamespace()
        for stubId in dependencyDict[callRequirementId]:
            stubResult = resultsDict[stubId]
            dslNamespace[stubId] = stubResult

        simpleExpr = stubbedModule.compile(dslLocals=dslNamespace, dslGlobals={})
        assert isinstance(simpleExpr, DslExpression), "Reduced parsed stubbed expr string is not an " \
                                                      "expression: %s" % type(simpleExpr)
        resultValue = simpleExpr.evaluate(**evaluationKwds)

        # Create result object and check if subscribers are ready to be executed.
        resultsDict[callRequirementId] = resultValue
        resultIds[callRequirementId] = None
        for subscriberId in notifyDict[callRequirementId]:
            if subscriberId in resultsDict:
                continue
            subscriberRequiredIds = dependencyDict[subscriberId]
            # It's ready unless it requires a call that doesn't have a result yet.
            isSubscriberReady = True
            for requiredId in subscriberRequiredIds:
                if requiredId == callRequirementId:
                    continue # We know we're done.
                if requiredId not in resultsDict:
                    isSubscriberReady = False
                    break
            if isSubscriberReady:
                executionQueue.put(subscriberId)

        # Check if we can delete dependency results.
        for notifierId in dependencyDict[callRequirementId]:
            # - are there results for all dependents of this result?
            isNotifierDone = True
            for subscriberId in notifyDict[notifierId]:
                if subscriberId == callRequirementId:
                    continue
                if subscriberId not in resultsDict:
                    isNotifierDone = False
                    break
            if isNotifierDone:
                resultsDict.pop(notifierId)

        return "OK"
    except Exception, e:
        import traceback
        msg = "Error whilst calling 'executeCallRequirement': %s" % traceback.format_exc()
        msg += str(e)
        raise Exception(msg)
