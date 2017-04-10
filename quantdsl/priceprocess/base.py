from __future__ import division
from abc import ABCMeta, abstractmethod
import six


class PriceProcess(six.with_metaclass(ABCMeta)):

    @abstractmethod
    def simulate_future_prices(self, observation_date, requirements, path_count, calibration_params):
        """
        Returns a generator that yields a sequence of simulated prices.
        """


def get_duration_years(start_date, end_date, days_per_year=365):
    try:
        time_delta = end_date - start_date
    except TypeError as inst:
        raise TypeError("%s: start: %s end: %s" % (inst, start_date, end_date))
    return time_delta.days / float(days_per_year)