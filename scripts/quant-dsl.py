#!/usr/bin/env python
import os
import datetime
import sys
import argh
import multiprocessing as mp
import json
from quantdsl.exceptions import DslError
from quantdsl.services import eval, DEFAULT_PRICE_PROCESS_NAME, DEFAULT_PATH_COUNT


@argh.arg('SOURCE', help='DSL source URL or file path ("-" to read from STDIN)')
@argh.arg('-c', '--calibration', help='market calibration URL or file path')
@argh.arg('-n', '--num-paths', help='number of paths in price simulations', type=int)
@argh.arg('-p', '--price-process', help='price process model of market dynamics')
@argh.arg('-i', '--interest-rate', help='annual percent interest rate', type=float)
@argh.arg('-m', '--multiprocessing-pool', help='evaluate with multiprocessing pool (option value is pool size, which defaults to cpu count)', nargs='?', type=int)
@argh.arg('-q', '--quiet', help='don\'t show progress info')
@argh.arg('-s', '--show-source', help='show source code and compiled expression stack')

def main(SOURCE, calibration=None, num_paths=DEFAULT_PATH_COUNT, price_process=DEFAULT_PRICE_PROCESS_NAME,
         interest_rate=2.5, multiprocessing_pool=0, quiet=False, show_source=False):
    """Evaluates 'Quant DSL' code in SOURCE, given price process parameters in CALIBRATION."""
    import quantdsl
    import quantdsl.semantics

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
            raise DslError("Can't open resource: %s" % url)

    print "DSL source from: %s" % source_url
    print
    dslSource = getResource(source_url)

    if calibration_url:
        print "Calibration from: %s" % calibration_url
        print
        marketCalibrationJson = getResource(calibration_url)
        try:
            marketCalibration = json.loads(marketCalibrationJson)
        except Exception, e:
            msg = "Unable to load JSON from %s: %s: %s" % (calibration_url, e, marketCalibrationJson)
            raise ValueError(msg)
    else:
        marketCalibration = {}

    observationTime = datetime.datetime.now().replace(tzinfo=quantdsl.semantics.utc)

    try:
        result = eval(dslSource,
            filename=source_url if source_url != '-' else 'STDIN',
            isParallel=True,
            marketCalibration=marketCalibration,
            interestRate=interest_rate,
            pathCount=num_paths,
            observationTime=observationTime,
            isMultiprocessing=bool(multiprocessing_pool),
            poolSize=multiprocessing_pool,
            isVerbose=isVerbose,
            isShowSource=show_source,
            priceProcessName=price_process,
        )
    except DslError, e:
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
