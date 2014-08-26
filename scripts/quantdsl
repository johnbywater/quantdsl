#!/usr/bin/env python
import os

import argh
import multiprocessing
from quantdsl import QuantDslError


@argh.arg('SOURCE', help='DSL source URL or file path ("-" to read from STDIN)')
@argh.arg('-p', '--price-process', help='price process model of market dynamics')
@argh.arg('-c', '--calibration', help='market calibration URL or file path')
@argh.arg('-n', '--num-paths', help='paths in Monte Carlo simulation', type=int)
@argh.arg('-w', '--workers', help='number workers in multiprocessing pool', default=multiprocessing.cpu_count(), type=int)
@argh.arg('-q', '--quiet', help='don\'t show any progress info')
def main(SOURCE, price_process='quantdsl:BlackScholesPriceProcess', calibration=None, num_paths=50000, workers=None, quiet=False):
    """Evaluates DSL module from SOURCE, given market calibration params from MARKET_CALIB."""
    import quantdsl
    import datetime
    import sys

    source_url = SOURCE
    calibration_url = calibration
    isVerbose = not quiet

    def getResource(url):
        if url == '-':
            return sys.stdin.read()
        elif url.startswith('file://'):
            return open(url[7:]).read()
        elif url.startswith('http://'):
            import requests
            return requests.get(url)
        elif os.path.exists(url) and os.path.isfile(url):
            return open(url).read()
        else:
            raise QuantDslError("Can't open resource: %s" % url)

    dslSource = getResource(source_url)

    if calibration_url:
        marketCalibrationJson = getResource(calibration_url)
        import json
        try:
            marketCalibration = json.loads(marketCalibrationJson)
        except Exception, e:
            msg = "Unable to load JSON from %s: %s: %s" % (calibration_url, e, marketCalibrationJson)
            raise ValueError(msg)
    else:
        marketCalibration = {}

    observationTime = datetime.datetime.now().replace(tzinfo=quantdsl.utc)

    try:
        result = quantdsl.eval(dslSource,
            filename=source_url,
            isParallel=True,
            marketCalibration=marketCalibration,
            interestRate=2.5,
            pathCount=num_paths,
            observationTime=observationTime,
            isMultiprocessing=True,
            poolSize=workers,
            isVerbose=isVerbose,
            priceProcessName=price_process,
        )
    except quantdsl.QuantDslError, e:
        print "Failed to eval DSL source:"
        print dslSource
        print
        print "Error:", e
        print
    else:
        if isVerbose:
            print "Result:"
            print "    mean: %.4f" % result['mean']
            print "    stderr: %.4f" % result['stderr']
            print
        else:
            import pprint
            print pprint.pformat(result)

if __name__ == '__main__':
    argh.dispatch_command(main)