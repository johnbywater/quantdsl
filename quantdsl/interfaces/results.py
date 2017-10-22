import datetime
from collections import defaultdict

import pandas
import pandas.plotting
import scipy
from matplotlib import pylab as plt
from numpy import nanpercentile
from numpy.core.multiarray import array
from pandas import DataFrame, Series

from quantdsl.defaults import DEFAULT_CONFIDENCE_INTERVAL
from quantdsl.domain.model.call_result import CallResult
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.market_simulation import MarketSimulation


class Results(object):
    def __init__(self, valuation_result, periods, contract_valuation, market_simulation):
        assert isinstance(contract_valuation, ContractValuation), type(contract_valuation)
        assert isinstance(market_simulation, MarketSimulation), type(market_simulation)
        assert isinstance(valuation_result, CallResult), type(valuation_result)
        assert isinstance(periods, list)
        self.valuation_result = valuation_result
        self.fair_value = valuation_result.result_value
        self.periods = periods
        self.contract_valuation = contract_valuation
        self.market_simulation = market_simulation
        self.observation_date = market_simulation.observation_date
        self.path_count = market_simulation.path_count
        self.perturbation_factor = market_simulation.perturbation_factor
        self.interest_rate = market_simulation.interest_rate
        self.init_dataframes()
        self.confidence_interval = None

    def init_dataframes(self):
        if not self.periods:
            return
        names = set()
        dates = set()
        self.periods_by_market_and_date = defaultdict(dict)
        self.periods_by_date_and_market = defaultdict(dict)
        for p in self.periods:
            name = p['market_name']
            names.add(name)
            date = p['delivery_date']
            dates.add(date)
            self.periods_by_market_and_date[name][date] = p
            self.periods_by_date_and_market[date][name] = p

        self.names = sorted(names)
        self.dates = sorted(dates)

        # Price.
        # self.prices_raw = get_dataframe('price_simulated', measure='direct')
        self.prices_mean = self.get_dataframe('price_simulated')

        # Hedge units.
        self.hedges_mean = self.get_dataframe('hedge_units')

        # Cash.
        self.cash_mean = self.get_dataframe('cash', sum=True, cum=True)

        self.deltas = {p['perturbation_name']: p['delta'] for p in self.periods}

        #
        # raise Exception(data)
        # # Market names.
        # names = sorted(set([p['market_name'] for p in periods]))
        #
        # # Dates.
        # dates = sorted(set([p['delivery_date'] for p in periods]))
        #
        # # Periods by date.
        # periods_by_date = defaultdict(list)
        # [periods_by_date[p['delivery_date']].append(p) for p in self.periods]
        #
        # # Cash.
        # self._cash = [sum([d['cash'] for d in periods_by_date[date]]) for date in dates]
        #
        # # Hedges.
        # self._hedges = {}
        # self.hedges_mean = {}
        #
        # hedges_mean = []
        # for date in dates:
        #     _periods = periods_by_date[date]
        #     hedges = {name: 0 for name in names}
        #     for p in _periods:
        #         hedges.update({p['market_name']: p['hedge_units']})
        #     for name, hedge_units in sorted(hedges.items()):
        #         hedges_mean.append(hedge_units.mean())
        #
        # self.hedges_mean = DataFrame(hedges_mean, index=self.index)
        # for name in names:
        #     _hedges = [d['hedge_units'] for d in periods if d['market_name'] == name]
        #     self._hedges[name] = _hedges
        #     self.hedges_mean[name] = DataFrame(
        #         data=[1, 2, 3],
        #         # data=[hedge.mean() for hedge in _hedges],
        #         index=self.index
        #     )
        #
        # self.by_delivery_date = defaultdict(list)
        # self.by_market_name = defaultdict(list)
        # [self.by_delivery_date[p['delivery_date']].append(p) for p in self.periods]
        # [self.by_market_name[p['market_name']].append(p) for p in self.periods]

    def init_dataframe_errors(self, confidence_interval):
        self.confidence_interval = confidence_interval
        self.prices_errors = self.get_dataframe('price_simulated', measure='quantile')
        self.hedges_errors = self.get_dataframe('hedge_units', measure='quantile')
        self.cash_errors = self.get_dataframe('cash', sum=True, cum=True, measure='quantile')

    def get_dataframe(self, attr, measure='mean', sum=False, cum=False):
        """
        :rtype NDFrame
        """
        values = []
        for name in self.names:
            market_periods_by_date = self.periods_by_market_and_date[name]
            values.append([market_periods_by_date[date][attr] for date in self.dates])

        values = array(values)  # shape is names-dates-samples

        if cum:
            # Accumulate over the dates, the second axis.
            # shape is the same: names-dates-samples
            values = values.cumsum(axis=1)

        if sum:
            # Sum over the names, the first axis.
            # shape is dates-samples
            values = values.sum(axis=0)
            pass

        if measure == 'mean':
            values = values.mean(axis=-1)
        elif measure == 'std':
            values = values.std(axis=-1)
        elif measure == 'quantile':
            assert self.confidence_interval is not None
            low_percentile = (100 - self.confidence_interval) / 2.0
            high_percentile = 100 - low_percentile
            mean = values.mean(axis=-1)
            low = mean - nanpercentile(values, q=low_percentile, axis=-1)
            high = nanpercentile(values, q=high_percentile, axis=-1) - mean
            errors = []
            if sum:
                # Need to return 2-len(dates) sized array, for a Series.
                errors.append([low, high])
            else:
                # Need to return len(names)-2-len(dates) sized array, for a DateFrame.
                for i in range(len(self.names)):
                    errors.append([low[i], high[i]])
            values = array(errors)
            return values
        # elif measure == 'direct':
        #     raise NotImplementedError()
        #     if len(values) == 1:
        #         values = values[0]
        #     else:
        #         raise NotImplementedError()
        #     return DataFrame(values, index=dates, columns=names)
        else:
            raise Exception("Measure '{}' not supported".format(measure))

        if sum:
            return Series(values, index=self.dates)
        else:
            return DataFrame(values.T, index=self.dates, columns=self.names)

    @property
    def fair_value_mean(self):
        if isinstance(self.fair_value, scipy.ndarray):
            fair_value_mean = self.fair_value.mean()
        else:
            fair_value_mean = self.fair_value
        return fair_value_mean

    def plot(self, title='', confidence_interval=DEFAULT_CONFIDENCE_INTERVAL, block=False, pause=0, figsize=(14, 14)):

        self.init_dataframe_errors(confidence_interval)

        assert isinstance(self, Results)

        if not self.periods:
            raise ValueError("Results have no periods to plot")

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize)

        if title:
            fig.canvas.set_window_title(title)

        if isinstance(self.observation_date, datetime.datetime):
            observation_date = self.observation_date.date()
        else:
            observation_date = self.observation_date

        fig.suptitle('On {}, rate {}%, paths {}, pert {}%, conf {}%'.format(
            observation_date, self.interest_rate, self.path_count,
            self.perturbation_factor * 100,
            confidence_interval))

        with pandas.plotting.plot_params.use('x_compat', False):

            # Todo: Try to get the box plots working:
            # https://stackoverflow.com/questions/38120688/pandas-box-plot-for-multiple-column

            # if len(results.prices_raw) == 1:
            #     prices = results.prices_raw[0]
            #     seaborn.boxplot(prices, prices.to_series().apply(lambda x: x.strftime('%Y%m%d')), ax=axes[0])

            self.prices_mean.plot(ax=axes[0], kind='bar', yerr=self.prices_errors)
            axes[0].set_title('Prices')
            axes[0].get_xaxis().set_visible(False)

            self.hedges_mean.plot(ax=axes[1], kind='bar', yerr=self.hedges_errors).axhline(0, color='0.5')
            axes[1].set_title('Hedges')
            axes[1].get_xaxis().set_visible(False)

            self.cash_mean.plot(ax=axes[2], kind='bar', yerr=self.cash_errors, color='g').axhline(0, color='0.5')
            axes[2].set_title('Cash')
            axes[2].get_xaxis().set_visible(True)

        fig.autofmt_xdate(rotation=30)

        if pause and not block:
            plt.pause(pause)
        else:
            plt.show(block=block)
