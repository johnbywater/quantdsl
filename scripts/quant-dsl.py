#!/usr/bin/env python
import os
import datetime
import sys
import argh
import multiprocessing as mp
import json

from eventsourcing.utils.time import UTC

from quantdsl.exceptions import DslError
from quantdsl.services import dsl_eval, DEFAULT_PRICE_PROCESS_NAME, DEFAULT_PATH_COUNT

now = datetime.datetime.now(tz=UTC)
defaultObservationTime = int("%04d%02d%02d" % (now.year, now.month, now.day))


@argh.arg('source', help='DSL source URL or file path ("-" to read from STDIN)')
@argh.arg('-o', '--observation-time', help='observation time, format YYYYMMDD', type=int)
@argh.arg('-c', '--calibration', help='market calibration URL or file path')
@argh.arg('-n', '--num-paths', help='number of paths in price simulations', type=int)
@argh.arg('-p', '--price-process', help='price process model of market dynamics')
@argh.arg('-i', '--interest-rate', help='annual percent interest rate', type=float)
@argh.arg('-m', '--multiprocessing-pool', help='evaluate with multiprocessing pool (option value is pool size, which '
                                               'defaults to cpu count)', nargs='?', type=int)
@argh.arg('-q', '--quiet', help='don\'t show progress info')
@argh.arg('-s', '--show-source', help='show source code and compiled expression stack')
def main(source, observation_date=defaultObservationTime, calibration=None, num_paths=DEFAULT_PATH_COUNT,
         price_process=DEFAULT_PRICE_PROCESS_NAME, interest_rate=2.5, multiprocessing_pool=0, quiet=False,
         show_source=False):
    """
    Evaluates 'Quant DSL' code in SOURCE, given price process parameters in CALIBRATION.
    """
    import quantdsl
    import quantdsl.semantics

    if multiprocessing_pool is None:
        multiprocessing_pool = mp.cpu_count()

    source_url = source
    calibration_url = calibration
    is_verbose = not quiet

    def get_resource(url):
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

    # Todo: Make this work with Python 3.

    print("DSL source from: %s" % source_url)
    print()
    dsl_source = get_resource(source_url)

    if calibration_url:
        print("Calibration from: %s" % calibration_url)
        print()
        market_calibration_json = get_resource(calibration_url)
        try:
            market_calibration = json.loads(market_calibration_json)
        except Exception:
            msg = "Unable to load JSON from %s: %s" % (calibration_url, market_calibration_json)
            raise ValueError(msg)
    else:
        market_calibration = {}

    observation_date = datetime.datetime(
        int(''.join(str(observation_date)[0:4])),
        int(''.join(str(observation_date)[4:6])),
        int(''.join(str(observation_date)[6:8]))
    ).replace(tzinfo=UTC)

    try:
        result = dsl_eval(
            dsl_source,
            filename=source_url if source_url != '-' else 'STDIN',
            is_parallel=True,
            market_calibration=market_calibration,
            interest_rate=interest_rate,
            path_count=num_paths,
            observation_date=observation_date,
            is_multiprocessing=bool(multiprocessing_pool),
            pool_size=multiprocessing_pool,
            is_verbose=is_verbose,
            is_show_source=show_source,
            price_process_name=price_process,
        )
    except DslError as e:
        print("Failed to dsl_eval DSL source:")
        print(dsl_source)
        print()
        print("Error:", e)
        print()
    else:
        if is_verbose:
            sys.stdout.write("Result: ")
            sys.stdout.flush()
        print(json.dumps(result, indent=4, sort_keys=True))

if __name__ == '__main__':
    argh.dispatch_command(main)
