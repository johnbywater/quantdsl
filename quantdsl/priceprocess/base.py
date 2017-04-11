from __future__ import division

import datetime
from abc import ABCMeta, abstractmethod
import six

DAYS_PER_YEAR = 365
SECONDS_PER_DAY = 86400


class PriceProcess(six.with_metaclass(ABCMeta)):

    @abstractmethod
    def simulate_future_prices(self, observation_date, requirements, path_count, calibration_params):
        """
        Returns a generator that yields a sequence of simulated prices.
        """

    def get_commodity_names_and_fixing_dates(self, observation_date, requirements):
        # Get an ordered list of all the commodity names and fixing dates.
        commodity_names = sorted(set([r[0] for r in requirements]))
        observation_date = datetime_from_date(observation_date)

        requirement_datetimes = [datetime_from_date(r[1]) for r in requirements]

        fixing_dates = sorted(set([observation_date] + requirement_datetimes))
        return commodity_names, fixing_dates


def get_duration_years(start_date, end_date, days_per_year=DAYS_PER_YEAR):
    try:
        time_delta = datetime_from_date(end_date) - datetime_from_date(start_date)
    except TypeError as inst:
        raise TypeError("%s: start: %s end: %s" % (inst, start_date, end_date))
    return time_delta.total_seconds() / float(days_per_year * SECONDS_PER_DAY)


def datetime_from_date(observation_date):
    if isinstance(observation_date, datetime.date):
        return datetime.datetime(observation_date.year, observation_date.month, observation_date.day)
    else:
        return observation_date
