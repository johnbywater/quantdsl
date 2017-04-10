from __future__ import division
from abc import ABCMeta, abstractmethod
import six


class PriceProcess(six.with_metaclass(ABCMeta)):

    @abstractmethod
    def simulate_future_prices(self, market_names, fixing_dates, observation_date, path_count, calibration_params):
        """
        Returns dict (keyed by market commodity_name) of dicts (keyed by fixing date) with correlated random future prices.
        """


def get_duration_years(start_date, end_date, days_per_year=365):
    try:
        time_delta = end_date - start_date
    except TypeError as inst:
        raise TypeError("%s: start: %s end: %s" % (inst, start_date, end_date))
    return time_delta.days / float(days_per_year)