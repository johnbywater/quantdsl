import datetime

import six
from dateutil.relativedelta import relativedelta
from pandas import Series
from pandas_datareader import DataReader


def get_historical_data(service, sym, days=30, start=None, end=None, col=None, limit=None):
    if end is None:
        end = datetime.datetime.now()
    if start is None:
        start = end - relativedelta(days=days)
    df = DataReader(sym, service, start=start, end=end)
    if col is not None:
        df = df[col]
    if limit is not None:
        df = df[-limit:]
    return df


def from_csvtext(csvtext, cls=Series):
    return cls.from_csv(six.StringIO(csvtext))


def to_csvtext(ndframe):
    f = six.StringIO()
    ndframe.to_csv(f)
    f.seek(0)
    return f.read()