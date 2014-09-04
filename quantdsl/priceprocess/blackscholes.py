from __future__ import division
from quantdsl.exceptions import DslError
from quantdsl.priceprocess.base import PriceProcess
from quantdsl.priceprocess.base import getDurationYears
import datetime
import itertools


class BlackScholesPriceProcess(PriceProcess):

    def simulateFuturePrices(self, marketNames, fixingDates, observationTime, pathCount, marketCalibration):
        allBrownianMotions = self.getBrownianMotions(marketNames, fixingDates, observationTime, pathCount, marketCalibration)
        # Compute market prices, so the Market object doesn't do this.
        import scipy
        allMarketPrices = {}
        for (marketName, brownianMotions) in allBrownianMotions.items():
            lastPrice = marketCalibration['%s-LAST-PRICE' % marketName.upper()]
            actualHistoricalVolatility = marketCalibration['%s-ACTUAL-HISTORICAL-VOLATILITY' % marketName.upper()]
            marketPrices = {}
            for (fixingDate, brownianRv) in brownianMotions.items():
                sigma = actualHistoricalVolatility / 100.0
                T = getDurationYears(observationTime, fixingDate)
                marketRv = lastPrice * scipy.exp(sigma * brownianRv - 0.5 * sigma * sigma * T)
                marketPrices[fixingDate] = marketRv

            allMarketPrices[marketName] = marketPrices
        return allMarketPrices

    def getBrownianMotions(self, marketNames, fixingDates, observationTime, pathCount, marketCalibration):
        assert isinstance(observationTime, datetime.datetime), type(observationTime)
        assert isinstance(pathCount, int), type(pathCount)

        marketNames = list(marketNames)
        allDates = [observationTime] + sorted(fixingDates)

        lenMarketNames = len(marketNames)
        lenAllDates = len(allDates)

        # Diffuse random variables through each date for each market (uncorrelated increments).
        import numpy
        import scipy.linalg
        from numpy.linalg import LinAlgError
        brownianMotions = scipy.zeros((lenMarketNames, lenAllDates, pathCount))
        for i in range(lenMarketNames):
            _startDate = allDates[0]
            startRv = brownianMotions[i][0]
            for j in range(lenAllDates - 1):
                fixingDate = allDates[j + 1]
                draws = numpy.random.standard_normal(pathCount)
                T = getDurationYears(_startDate, fixingDate)
                if T < 0:
                    raise DslError("Can't really square root negative time durations: %s. Contract starts before observation time?" % T)
                endRv = startRv + scipy.sqrt(T) * draws
                try:
                    brownianMotions[i][j + 1] = endRv
                except ValueError, e:
                    raise ValueError, "Can't set endRv in brownianMotions: %s" % e
                _startDate = fixingDate
                startRv = endRv

        # Read the market calibration data.
        correlations = {}
        for marketNamePairs in itertools.combinations(marketNames, 2):
            marketNamePairs = tuple(sorted(marketNamePairs))
            calibrationName = "%s-%s-CORRELATION" % marketNamePairs
            try:
                correlation = marketCalibration[calibrationName]
            except KeyError, e:
                msg = "Can't find correlation between '%s' and '%s': '%s' not defined in market calibration: %s" % (
                    marketNamePairs[0],
                    marketNamePairs[1],
                    marketCalibration.keys(),
                    e
                )
                raise DslError(msg)
            else:
                correlations[marketNamePairs] = correlation

        correlationMatrix = numpy.zeros((lenMarketNames, lenMarketNames))
        for i in range(lenMarketNames):
            for j in range(lenMarketNames):
                if marketNames[i] == marketNames[j]:
                    correlation = 1
                else:
                    key = tuple(sorted([marketNames[i], marketNames[j]]))
                    correlation = correlations[key]
                correlationMatrix[i][j] = correlation

        try:
            U = scipy.linalg.cholesky(correlationMatrix)
        except LinAlgError, e:
            raise DslError("Couldn't do Cholesky decomposition with correlation matrix: %s: %s" % (correlationMatrix, e))

        # Correlated increments from uncorrelated increments.
        brownianMotions = brownianMotions.transpose() # Put markets on the last dimension, so the broadcasting works.
        try:
            brownianMotionsCorrelated = brownianMotions.dot(U)
        except Exception, e:
            msg = "Couldn't multiply uncorrelated Brownian increments with decomposed correlation matrix: %s, %s" % (brownianMotions, U)
            raise DslError(msg)
        brownianMotionsCorrelated = brownianMotionsCorrelated.transpose() # Put markets back on the first dimension.

        brownianMotionsDict = {}
        for i, marketName in enumerate(marketNames):
            marketRvs = {}
            for j, fixingDate in enumerate(allDates):
                rv = brownianMotionsCorrelated[i][j]
                marketRvs[fixingDate] = rv
            brownianMotionsDict[marketName] = marketRvs

        return brownianMotionsDict
