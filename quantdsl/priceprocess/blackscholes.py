from __future__ import division
import math
import scipy
from quantdsl.exceptions import DslError
from quantdsl.priceprocess.base import PriceProcess
from quantdsl.priceprocess.base import get_duration_years
import datetime
import itertools


class BlackScholesPriceProcess(PriceProcess):

    def simulateFuturePrices(self, market_names, fixing_dates, observation_time, path_count, market_calibration):
        allBrownianMotions = self.getBrownianMotions(market_names, fixing_dates, observation_time, path_count, market_calibration)
        # Compute market prices, so the Market object doesn't do this.
        import scipy
        all_market_prices = {}
        for (marketName, brownianMotions) in allBrownianMotions.items():
            lastPrice = market_calibration['%s-LAST-PRICE' % marketName.upper()]
            actualHistoricalVolatility = market_calibration['%s-ACTUAL-HISTORICAL-VOLATILITY' % marketName.upper()]
            marketPrices = {}
            for (fixingDate, brownianRv) in brownianMotions.items():
                sigma = actualHistoricalVolatility / 100.0
                T = get_duration_years(observation_time, fixingDate)
                marketRv = lastPrice * scipy.exp(sigma * brownianRv - 0.5 * sigma * sigma * T)
                marketPrices[fixingDate] = marketRv

            all_market_prices[marketName] = marketPrices
        return all_market_prices

    def getBrownianMotions(self, market_names, fixing_dates, observation_time, path_count, market_calibration):
        assert isinstance(observation_time, datetime.datetime), type(observation_time)
        assert isinstance(path_count, int), type(path_count)

        market_names = list(market_names)
        allDates = [observation_time] + sorted(fixing_dates)

        lenMarketNames = len(market_names)
        lenAllDates = len(allDates)

        # Diffuse random variables through each date for each market (uncorrelated increments).
        import numpy
        import scipy.linalg
        from numpy.linalg import LinAlgError
        brownianMotions = scipy.zeros((lenMarketNames, lenAllDates, path_count))
        for i in range(lenMarketNames):
            _start_date = allDates[0]
            startRv = brownianMotions[i][0]
            for j in range(lenAllDates - 1):
                fixingDate = allDates[j + 1]
                draws = numpy.random.standard_normal(path_count)
                T = get_duration_years(_start_date, fixingDate)
                if T < 0:
                    raise DslError("Can't really square root negative time durations: %s. Contract starts before observation time?" % T)
                endRv = startRv + scipy.sqrt(T) * draws
                try:
                    brownianMotions[i][j + 1] = endRv
                except ValueError as e:
                    raise ValueError("Can't set endRv in brownianMotions: %s" % e)
                _start_date = fixingDate
                startRv = endRv

        # Read the market calibration data.
        correlations = {}
        for marketNamePairs in itertools.combinations(market_names, 2):
            marketNamePairs = tuple(sorted(marketNamePairs))
            calibrationName = "%s-%s-CORRELATION" % marketNamePairs
            try:
                correlation = market_calibration[calibrationName]
            except KeyError as e:
                msg = "Can't find correlation between '%s' and '%s': '%s' not defined in market calibration: %s" % (
                    marketNamePairs[0],
                    marketNamePairs[1],
                    market_calibration.keys(),
                    e
                )
                raise DslError(msg)
            else:
                correlations[marketNamePairs] = correlation

        correlationMatrix = numpy.zeros((lenMarketNames, lenMarketNames))
        for i in range(lenMarketNames):
            for j in range(lenMarketNames):
                if market_names[i] == market_names[j]:
                    correlation = 1
                else:
                    key = tuple(sorted([market_names[i], market_names[j]]))
                    correlation = correlations[key]
                correlationMatrix[i][j] = correlation

        try:
            U = scipy.linalg.cholesky(correlationMatrix)
        except LinAlgError as e:
            raise DslError("Couldn't do Cholesky decomposition with correlation matrix: %s: %s" % (correlationMatrix, e))

        # Correlated increments from uncorrelated increments.
        #brownianMotions = brownianMotions.transpose() # Put markets on the last dimension, so the broadcasting works.
        try:
            brownianMotionsCorrelated = brownianMotions.T.dot(U)
        except Exception as e:
            msg = "Couldn't multiply uncorrelated Brownian increments with decomposed correlation matrix: %s, %s: %s" % (brownianMotions, U, e)
            raise DslError(msg)
        brownianMotionsCorrelated = brownianMotionsCorrelated.transpose() # Put markets back on the first dimension.
        brownianMotionsDict = {}
        for i, marketName in enumerate(market_names):
            marketRvs = {}
            for j, fixingDate in enumerate(allDates):
                rv = brownianMotionsCorrelated[i][j]
                marketRvs[fixingDate] = rv
            brownianMotionsDict[marketName] = marketRvs

        return brownianMotionsDict


class BlackScholesVolatility(object):

    def calcActualHistoricalVolatility(self, market, observation_time):
        priceHistory = market.getPriceHistory(observation_time=observation_time)
        prices = scipy.array([i.value for i in priceHistory])
        dates = [i.observation_time for i in priceHistory]
        volatility = 100 * prices.std() / prices.mean()
        duration = max(dates) - min(dates)
        years = (duration.days) / 365.0 # Assumes zero seconds.
        if years == 0:
            raise Exception("Can't calculate volatility for price series with zero duration: %s" % (priceHistory))
        return float(volatility) / math.sqrt(years)
