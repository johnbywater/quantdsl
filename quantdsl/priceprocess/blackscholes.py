from __future__ import division

import datetime
from collections import defaultdict

import numpy
import numpy as np
import scipy
import scipy.linalg
from dateutil.relativedelta import relativedelta
from scipy.linalg import LinAlgError

from quantdsl.exceptions import DslError
from quantdsl.priceprocess.base import PriceProcess, get_duration_years
from quantdsl.priceprocess.common import get_historical_data
from quantdsl.priceprocess.forwardcurve import ForwardCurve


class BlackScholesPriceProcess(PriceProcess):
    def simulate_future_prices(self, observation_date, requirements, path_count, calibration_params):
        # Compute correlated Brownian motions for each market.

        if not requirements:
            return

        all_brownian_motions = self.get_brownian_motions(observation_date, requirements, path_count,
                                                         calibration_params)

        delivery_dates = defaultdict(set)
        for requirement in requirements:
            fixing_date = requirement[1]
            delivery_date = requirement[2]
            delivery_dates[fixing_date].add(delivery_date)

        # delivery_dates[observation_date].add(observation_date)

        # Compute simulated market prices using the correlated Brownian
        # motions, the actual historical volatility, and the last price.
        for commodity_name, brownian_motions in all_brownian_motions:
            # Get the 'last price' for this commodity.

            index = calibration_params['market'].index(commodity_name)
            sigma = calibration_params['sigma'][index]
            curve = ForwardCurve(commodity_name, calibration_params['curve'][commodity_name])
            for fixing_date, brownian_rv in brownian_motions:
                for delivery_date in sorted(delivery_dates[fixing_date]):
                    forward_price = curve.get_price(delivery_date)
                    T = get_duration_years(observation_date, fixing_date)
                    simulated_value = forward_price * scipy.exp(sigma * brownian_rv - 0.5 * sigma * sigma * T)
                    yield commodity_name, fixing_date, delivery_date, simulated_value

    def get_brownian_motions(self, observation_date, requirements, path_count, calibration_params):
        assert isinstance(observation_date, datetime.datetime), observation_date
        assert isinstance(requirements, list), requirements
        assert isinstance(path_count, int), path_count

        commodity_names, fixing_dates = self.get_commodity_names_and_fixing_dates(observation_date, requirements)

        len_commodity_names = len(commodity_names)

        len_fixing_dates = len(fixing_dates)

        # Check the observation date equals the first fixing date.
        assert observation_date == fixing_dates[0], "Observation date {} not equal to first fixing date: {}" \
                                                    "".format(observation_date, fixing_dates[0])

        # Diffuse random variables through each date for each market (uncorrelated increments).
        brownian_motions = scipy.zeros((len_commodity_names, len_fixing_dates, path_count))
        all_brownian_motions = []

        if len_fixing_dates and len_commodity_names:
            for i in range(len_commodity_names):
                _start_date = fixing_dates[0]
                start_rv = brownian_motions[i][0]
                for j in range(len_fixing_dates - 1):
                    fixing_date = fixing_dates[j + 1]
                    draws = scipy.random.standard_normal(path_count)
                    T = get_duration_years(_start_date, fixing_date)
                    if T < 0:
                        raise DslError(
                            "Can't really square root negative time durations: %s. Contract starts before "
                            "observation time?" % T)
                    end_rv = start_rv + scipy.sqrt(T) * draws
                    try:
                        brownian_motions[i][j + 1] = end_rv
                    except ValueError as e:
                        raise ValueError("Can't set end_rv in brownian_motions: %s" % e)
                    _start_date = fixing_date
                    start_rv = end_rv

        if len_commodity_names > 1:
            correlation_matrix = scipy.zeros((len_commodity_names, len_commodity_names))
            for i in range(len_commodity_names):
                for j in range(len_commodity_names):

                    # Get the correlation between market i and market j...
                    name_i = commodity_names[i]
                    name_j = commodity_names[j]
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
                msg = "Cholesky decomposition failed with correlation matrix: %s: %s" % (correlation_matrix, e)
                raise DslError(msg)

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
        for i, commodity_name in enumerate(commodity_names):
            market_rvs = []
            for j, fixing_date in enumerate(fixing_dates):
                rv = brownian_motions[i][j]
                market_rvs.append((fixing_date, rv))
            all_brownian_motions.append((commodity_name, market_rvs))

        return all_brownian_motions

    def get_correlation_from_calibration(self, market_calibration, name_i, name_j):
        index_i = market_calibration['market'].index(name_i)
        index_j = market_calibration['market'].index(name_j)
        try:
            correlation = market_calibration['rho'][index_i][index_j]
        except KeyError as e:
            msg = "Can't find correlation between '%s' and '%s' in market calibration params: %s: %s" % (
                name_i,
                name_j,
                market_calibration,
                e
            )
            raise DslError(msg)
        else:
            assert isinstance(correlation, (float, int)), correlation
        return correlation


def generate_calibration_params(start, end, markets, get_historical_data=get_historical_data):
    name = 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess'
    all_quotes = []
    sigmas = []
    date = start
    all_market_names = []
    # Iterate over all "markets" (e.g. 'GAS').
    all_curves = {}
    for market_name, market_spec in markets.items():
        all_market_names.append(market_name)
        forward_vols = []
        forward_curve = []
        while date <= end:
            forward_year = date.year
            forward_month = date.month
            kwargs = market_spec.copy()
            kwargs['sym'] += quandl_month_codes[forward_month] + str(forward_year)

            # Get the data.
            quotes = get_historical_data(**kwargs)

            all_quotes.append(quotes)
            vol = calc_historical_volatility(quotes)
            forward_vols.append(vol)
            last = pick_last_price(quotes)
            forward_curve.append((date, last))

            # Next month.
            date = date + relativedelta(months=1)
        sigma = np.median(forward_vols)
        sigmas.append(sigma)
        all_curves[market_name] = forward_curve
    # Todo: Align dates and drop rows that aren't full.
    rho = calc_correlation(all_quotes)
    return {
        'name': name,
        'market': all_market_names,
        'sigma': sigmas,
        'rho': rho,
        'curve': all_curves,
    }


def calc_historical_volatility(quotes):
    logreturns = np.log(quotes / quotes.shift(1))
    return np.sqrt(252 * logreturns.var())


def pick_last_price(quotes):
    if len(quotes):
        return quotes[-1]
    else:
        return None


def calc_correlation(*args):
    if len(args) == 1:
        return numpy.array([[1]])
    else:
        raise NotImplementedError('Need to align dates. Also, which time series?')
        return numpy.corrcoef(numpy.array(args))


quandl_month_codes = {
    1: 'F',
    2: 'G',
    3: 'H',
    4: 'J',
    5: 'K',
    6: 'M',
    7: 'N',
    8: 'Q',
    9: 'U',
    10: 'V',
    11: 'X',
    12: 'Z',
}


QuandlProducts = {
    'Endex Dutch TTF Gas Base Load Futures': 'ICE/TFM',
    'Endex Belgian Power Base Load Futures': 'ICE/BPB',
    'UK Natural Gas Futures': 'ICE/M'
}
