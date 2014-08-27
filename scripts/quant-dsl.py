#!/usr/bin/env python
import os
import datetime
import sys
import argh
import multiprocessing as mp
import json

@argh.arg('SOURCE', help='DSL source URL or file path ("-" to read from STDIN)')
@argh.arg('-q', '--quiet', help='don\'t show progress info' )
@argh.arg('-c', '--calibration', help='market calibration URL or file path')
@argh.arg('-n', '--num-paths', help='number of paths in price simulations', type=int)
@argh.arg('-p', '--price-process', help='price process model of market dynamics')
@argh.arg('-m', '--multiprocessing-pool', help='evaluate with multiprocessing pool (option value is pool size, which defaults to cpu count)', nargs='?', type=int)
def main(SOURCE, quiet=False, calibration=None, num_paths=50000, price_process='quantdsl:BlackScholesPriceProcess', multiprocessing_pool=0):
    """Evaluates DSL module from SOURCE, given market calibration params from MARKET_CALIB."""
    import quantdsl

    if multiprocessing_pool is None:
        multiprocessing_pool = mp.cpu_count()

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
            raise quantdsl.QuantDslError("Can't open resource: %s" % url)

    print "DSL source from: %s" % (source_url if source_url != '-' else 'STDIN')
    print
    dslSource = getResource(source_url)

    if calibration_url:
        print "Calibration from: %s" % (calibration_url if calibration_url != '-' else 'STDIN')
        print
        marketCalibrationJson = getResource(calibration_url)
        try:
            marketCalibration = json.loads(marketCalibrationJson)
        except Exception, e:
            msg = "Unable to load JSON from %s: %s: %s" % (
                calibration_url if calibration_url != '-' else 'STDIN',
                e,
                marketCalibrationJson
            )
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
            isMultiprocessing=bool(multiprocessing_pool),
            poolSize=multiprocessing_pool,
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
            sys.stdout.write("Result: ")
            sys.stdout.flush()
        print json.dumps(result, indent=4, sort_keys=True)

if __name__ == '__main__':
    argh.dispatch_command(main)
