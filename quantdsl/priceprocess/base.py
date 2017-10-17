from __future__ import division

import datetime
from abc import ABCMeta, abstractmethod
import six
from dateutil.relativedelta import relativedelta

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
    assert isinstance(start_date, datetime.date), type(start_date)
    assert isinstance(end_date, datetime.date), type(end_date)
    r = relativedelta(end_date, start_date)
    return r.years + r.months / 12.0 + (r.days + r.hours / 24) / float(days_per_year)


def datetime_from_date(date):
    return datetime.datetime(date.year, date.month, date.day)
