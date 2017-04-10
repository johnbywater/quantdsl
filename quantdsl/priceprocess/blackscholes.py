from __future__ import division
import math
import scipy
import scipy.linalg
from scipy.linalg import LinAlgError
from quantdsl.exceptions import DslError
from quantdsl.priceprocess.base import PriceProcess
from quantdsl.priceprocess.base import get_duration_years
import datetime


class BlackScholesPriceProcess(PriceProcess):

    def simulate_future_prices(self, market_names, fixing_dates, observation_date, path_count, calibration_params):
        # Compute correlated Brownian motions for each market and fixing date.
        if market_names:
            all_brownian_motions = self.get_brownian_motions(market_names, fixing_dates, observation_date, path_count,
                                                             calibration_params)

            # Compute simulated market prices using the correlated Brownian
            # motions, the actual historical volatility, and the last price.
            for market_name, brownian_motions in all_brownian_motions:
                last_price = calibration_params['%s-LAST-PRICE' % market_name]
                actual_historical_volatility = calibration_params['%s-ACTUAL-HISTORICAL-VOLATILITY' % market_name]
                sigma = actual_historical_volatility / 100.0
                for fixing_date, brownian_rv in brownian_motions:
                    T = get_duration_years(observation_date, fixing_date)
                    simulated_value = last_price * scipy.exp(sigma * brownian_rv - 0.5 * sigma * sigma * T)
                    yield market_name, fixing_date, simulated_value

    def get_brownian_motions(self, market_names, fixing_dates, observation_date, path_count, calibration_params):
        assert isinstance(market_names, list), market_names
        assert isinstance(fixing_dates, list), fixing_dates
        assert isinstance(observation_date, datetime.date), observation_date
        assert isinstance(path_count, int), path_count

        # Get an ordered list of all the dates.
        fixing_dates = set(fixing_dates)
        fixing_dates.add(observation_date)
        all_dates = sorted(fixing_dates)

        len_market_names = len(market_names)
        len_all_dates = len(all_dates)

        if len_market_names == 0:
            return []

        # Diffuse random variables through each date for each market (uncorrelated increments).
        brownian_motions = scipy.zeros((len_market_names, len_all_dates, path_count))
        for i in range(len_market_names):
            _start_date = all_dates[0]
            start_rv = brownian_motions[i][0]
            for j in range(len_all_dates - 1):
                fixing_date = all_dates[j + 1]
                draws = scipy.random.standard_normal(path_count)
                T = get_duration_years(_start_date, fixing_date)
                if T < 0:
                    raise DslError("Can't really square root negative time durations: %s. Contract starts before observation time?" % T)
                end_rv = start_rv + scipy.sqrt(T) * draws
                try:
                    brownian_motions[i][j + 1] = end_rv
                except ValueError as e:
                    raise ValueError("Can't set end_rv in brownian_motions: %s" % e)
                _start_date = fixing_date
                start_rv = end_rv

        if len_market_names > 1:
            correlation_matrix = scipy.zeros((len_market_names, len_market_names))
            for i in range(len_market_names):
                for j in range(len_market_names):

                    # Get the correlation between market i and market j...
                    name_i = market_names[i]
                    name_j = market_names[j]
                    if name_i == name_j:
                        # - they are identical
                        correlation = 1
                    else:
                        # - correlation is expected to be in the "calibration" data
                        correlation = self.get_correlation_from_calibration(calibration_params, name_i, name_j)

                    # ...and put the correlation in the correlation matrix.
                    correlation_matrix[i][j] = correlation

            # Compute lower triangular matrix, using Cholesky decomposition.
            try:
                U = scipy.linalg.cholesky(correlation_matrix)
            except LinAlgError as e:
                raise DslError("Cholesky decomposition failed with correlation matrix: %s: %s" % (correlation_matrix, e))

            # Construct correlated increments from uncorrelated increments
            # and lower triangular matrix for the correlation matrix.
            try:
                # Put markets on the last axis, so the broadcasting works, before computing
                # the dot product with the lower triangular matrix of the correlation matrix.
                brownian_motions_correlated = brownian_motions.T.dot(U)
            except Exception as e:
                msg = ("Couldn't multiply uncorrelated Brownian increments with decomposed correlation matrix: "
                       "%s, %s: %s" % (brownian_motions, U, e))
                raise DslError(msg)

            # Put markets back on the first dimension.
            brownian_motions_correlated = brownian_motions_correlated.transpose()
            brownian_motions = brownian_motions_correlated

        # Put random variables into a nested Python dict, keyed by market commodity_name and fixing date.
        all_brownian_motions = []
        for i, market_name in enumerate(market_names):
            market_rvs = []
            for j, fixing_date in enumerate(all_dates):
                rv = brownian_motions[i][j]
                market_rvs.append((fixing_date, rv))
            all_brownian_motions.append((market_name, market_rvs))

        return all_brownian_motions

    def get_correlation_from_calibration(self, market_calibration, name_i, name_j):
        market_name_pair = tuple(sorted([name_i, name_j]))
        calibration_name = "%s-%s-CORRELATION" % market_name_pair
        try:
            correlation = market_calibration[calibration_name]
        except KeyError as e:
            msg = "Can't find correlation between '%s' and '%s' in market calibration params: %s: %s" % (
                market_name_pair[0],
                market_name_pair[1],
                market_calibration.keys(),
                e
            )
            raise DslError(msg)
        else:
            assert isinstance(correlation, (float, int)), correlation
        return correlation


class BlackScholesVolatility(object):

    def calc_actual_historical_volatility(self, market, observation_date):
        price_history = market.getPriceHistory(observation_date=observation_date)
        prices = scipy.array([i.value for i in price_history])
        dates = [i.observation_date for i in price_history]
        volatility = 100 * prices.std() / prices.mean()
        duration = max(dates) - min(dates)
        years = (duration.days) / 365.0 # Assumes zero seconds.
        if years == 0:
            raise Exception("Can't calculate volatility for price series with zero duration: %s" % (price_history))
        return float(volatility) / math.sqrt(years)
