from __future__ import division
from abc import ABCMeta, abstractmethod


class PriceProcess(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def simulateFuturePrices(self, marketNames, fixingDates, observationTime, pathCount, marketCalibration):
        """
        Returns dict (keyed by market name) of dicts (keyed by fixing date) with correlated random future prices.
        """


def getDurationYears(startDate, endDate, daysPerYear=365):
    try:
        timeDelta = endDate - startDate
    except TypeError, inst:
        raise TypeError("%s: start: %s end: %s" % (inst, startDate, endDate))
    return timeDelta.days / float(daysPerYear)