# coding=utf-8
from __future__ import print_function

import collections
import datetime
import math
import signal
import sys
from collections import defaultdict
from threading import Event, Thread
from time import sleep

import dateutil.parser
import numpy
import six
from eventsourcing.domain.model.events import subscribe, unsubscribe
from numpy.lib.nanfunctions import nanpercentile

from quantdsl.application.base import DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE, Results
from quantdsl.application.with_multithreading_and_python_objects import \
    QuantDslApplicationWithMultithreadingAndPythonObjects
from quantdsl.domain.model.call_dependents import CallDependents
from quantdsl.domain.model.call_link import CallLink
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_result import CallResult, ResultValueComputed, make_call_result_id
from quantdsl.domain.model.simulated_price_requirements import SimulatedPriceRequirements
from quantdsl.exceptions import InterruptSignalReceived, TimeoutError
from quantdsl.priceprocess.base import datetime_from_date


def calc_print_plot(source_code, title='', observation_date=None, periodisation=None, interest_rate=0,
                    path_count=20000, perturbation_factor=0.01, price_process=None, dsl_classes=None,
                    max_dependency_graph_size=DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE,
                    timeout=None, verbose=False, is_double_sided_deltas=True):
    # Calculate and print the results.
    results = calc_print(source_code,
                         max_dependency_graph_size=max_dependency_graph_size,
                         observation_date=observation_date,
                         interest_rate=interest_rate,
                         path_count=path_count,
                         perturbation_factor=perturbation_factor,
                         price_process=price_process,
                         periodisation=periodisation,
                         dsl_classes=dsl_classes,
                         timeout=timeout,
                         verbose=verbose,
                         is_double_sided_deltas=is_double_sided_deltas
                         )

    # Plot the results.
    if results.periods:
        plot_periods(
            periods=results.periods,
            title=title,
            periodisation=periodisation,
            interest_rate=interest_rate,
            path_count=path_count,
            perturbation_factor=perturbation_factor,
        )
    return results


def calc_print(source_code, observation_date=None, interest_rate=0, path_count=20000, perturbation_factor=0.01,
               price_process=None, periodisation=None, dsl_classes=None,
               max_dependency_graph_size=DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE,
               timeout=None, verbose=False, is_double_sided_deltas=True):
    # Calculate the results.
    results = calc(
        source_code=source_code,
        interest_rate=interest_rate,
        path_count=path_count,
        observation_date=observation_date,
        perturbation_factor=perturbation_factor,
        price_process=price_process,
        periodisation=periodisation,
        dsl_classes=dsl_classes,
        max_dependency_graph_size=max_dependency_graph_size,
        timeout=timeout,
        verbose=verbose,
        is_double_sided_deltas=is_double_sided_deltas
    )

    # Print the results.
    print_results(results, path_count)
    return results


def calc(source_code, observation_date=None, interest_rate=0, path_count=20000,
         perturbation_factor=0.01, price_process=None, periodisation=None, dsl_classes=None,
         max_dependency_graph_size=DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE,
         timeout=None, verbose=False, is_double_sided_deltas=True):
    cmd = Calculate(
        source_code=source_code,
        observation_date=observation_date,
        interest_rate=interest_rate,
        path_count=path_count,
        perturbation_factor=perturbation_factor,
        price_process=price_process,
        periodisation=periodisation,
        dsl_classes=dsl_classes,
        max_dependency_graph_size=max_dependency_graph_size,
        timeout=timeout,
        verbose=verbose,
        is_double_sided_deltas=is_double_sided_deltas
    )
    with cmd:
        try:
            return cmd.calculate()
        except (TimeoutError, InterruptSignalReceived) as e:
            print()
            print()
            print(e)
            sys.exit(1)


class Calculate(object):
    def __init__(self, source_code, observation_date=None, interest_rate=0, path_count=20000, perturbation_factor=0.01,
                 price_process=None, periodisation=None, dsl_classes=None,
                 max_dependency_graph_size=DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE,
                 timeout=None, verbose=False, is_double_sided_deltas=True):
        self.timeout = timeout
        self.source_code = source_code
        if observation_date is not None:
            observation_date = datetime_from_date(dateutil.parser.parse(observation_date))
        self.observation_date = observation_date
        self.interest_rate = interest_rate
        self.path_count = path_count
        self.perturbation_factor = perturbation_factor
        # Todo: Optional double or single sided deltas.
        # self.double_sided_deltas = double_sided_deltas
        self.price_process = price_process
        self.periodisation = periodisation
        self.max_dependency_graph_size = max_dependency_graph_size
        self.verbose = verbose
        # Todo: Repetitions - number of times the computation will be repeated (multiprocessing).
        # self.repetitions = repetitions
        self.dsl_classes = dsl_classes
        self.is_double_sided_deltas = is_double_sided_deltas

    def __enter__(self):
        self.orig_sigterm_handler = signal.signal(signal.SIGTERM, self.shutdown)
        self.orig_sigint_handler = signal.signal(signal.SIGINT, self.shutdown)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGTERM, self.orig_sigterm_handler)
        signal.signal(signal.SIGINT, self.orig_sigint_handler)

    def calculate(self):
        self.result_cost = 0
        self.result_count = 0
        self.root_result_id = None
        self.is_timed_out = Event()
        self.is_interrupted = Event()
        self.timeout_msg = ''
        self.is_finished = Event()
        self.started = datetime.datetime.now()
        self.started_evaluating = None
        self.times = collections.deque()
        self.call_result_count = 0
        self.call_requirement_count = 1

        if self.timeout:
            timeout_thread = Thread(target=self.wait_then_set_is_timed_out)
            timeout_thread.setDaemon(True)
            timeout_thread.start()

        with QuantDslApplicationWithMultithreadingAndPythonObjects(
                max_dependency_graph_size=self.max_dependency_graph_size,
                dsl_classes=self.dsl_classes,
        ) as app:

            # Subscribe after the application, so events are received after the application.
            # - this means the final result is persisted before this interface is notified
            #   the result is available and tries to get it, avoiding a race condition
            self.subscribe()
            try:

                # Compile.
                start_compile = datetime.datetime.now()
                contract_specification = app.compile(self.source_code, self.observation_date)
                end_compile = datetime.datetime.now()
                if self.verbose:
                    # Todo: Separate this, not all users want print statements.
                    print("")  # Get a new line after the compilation progress.
                    print("Compilation in {:.3f}s".format((end_compile - start_compile).total_seconds()))

                # Get simulation args.
                if self.price_process is not None:
                    price_process_name = self.price_process['name']
                    calibration_params = {k: v for k, v in self.price_process.items() if k != 'name'}
                else:
                    price_process_name = 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess'
                    calibration_params = {}

                # Simulate the market prices.
                start_simulate = datetime.datetime.now()
                market_simulation = app.simulate(
                    contract_specification,
                    price_process_name=price_process_name,
                    calibration_params=calibration_params,
                    path_count=self.path_count,
                    observation_date=self.observation_date,
                    interest_rate=self.interest_rate,
                    perturbation_factor=self.perturbation_factor,
                    periodisation=self.periodisation,
                )
                end_simulate = datetime.datetime.now()
                if self.verbose:
                    # Todo: Separate this, not all users want print statements.
                    print("Simulation in {:.3f}s".format((end_simulate - start_simulate).total_seconds()))

                # Estimate the cost of the evaluation (to show progress).
                # Todo: Improve the call cost estimation, perhaps by running over the depenendency graph and coding
                # each DSL class to know how long it will take relative to others.
                call_counts, call_costs = app.calc_counts_and_costs(contract_specification.id, self.is_double_sided_deltas)
                self.result_count_expected = sum(call_counts.values())
                self.result_cost_expected = sum(call_costs.values())
                if self.verbose:
                    print("Starting {} node evaluations, please wait...".format(self.result_count_expected))
                # self.expected_num_call_requirements = len(call_costs)

                # Evaluate the contract specification.
                start_calc = datetime.datetime.now()
                self.started_evaluating = datetime.datetime.now()
                evaluation = app.evaluate(
                    contract_specification_id=contract_specification.id,
                    market_simulation_id=market_simulation.id,
                    periodisation=self.periodisation,
                    is_double_sided_deltas=self.is_double_sided_deltas
                )

                # Wait for the result.
                self.root_result_id = make_call_result_id(evaluation.id, evaluation.contract_specification_id)
                if not self.root_result_id in app.call_result_repo:
                    while not self.is_finished.wait(timeout=1):
                        self.check_has_app_thread_errored(app)
                    self.check_is_timed_out()
                    self.check_is_interrupted()

                # Todo: Separate this, not all users want print statements.
                end_calc = datetime.datetime.now()
                if self.verbose:
                    print("")
                    print("Evaluation in {:.3f}s".format((end_calc - start_calc).total_seconds()))

                # Read the results.
                results = app.read_results(evaluation)
            finally:

                self.unsubscribe()

        return results

    def check_has_app_thread_errored(self, app):
        try:
            app.check_has_thread_errored()
        except:
            self.set_is_finished()
            raise

    def subscribe(self):
        subscribe(self.is_call_requirement_created, self.print_compilation_progress)
        subscribe(self.is_calculating, self.check_is_timed_out)
        subscribe(self.is_calculating, self.check_is_interrupted)
        subscribe(self.is_evaluation_complete, self.set_is_finished)
        subscribe(self.is_result_value_computed, self.inc_result_value_computed_count)
        subscribe(self.is_result_value_computed, self.print_evaluation_progress)
        subscribe(self.is_call_result_created, self.inc_call_result_count)
        subscribe(self.is_call_result_created, self.print_evaluation_progress)

    def unsubscribe(self):
        unsubscribe(self.is_call_requirement_created, self.print_compilation_progress)
        unsubscribe(self.is_calculating, self.check_is_timed_out)
        unsubscribe(self.is_calculating, self.check_is_interrupted)
        unsubscribe(self.is_evaluation_complete, self.set_is_finished)
        unsubscribe(self.is_result_value_computed, self.inc_result_value_computed_count)
        unsubscribe(self.is_result_value_computed, self.print_evaluation_progress)
        unsubscribe(self.is_call_result_created, self.inc_call_result_count)
        unsubscribe(self.is_call_result_created, self.print_evaluation_progress)

    @staticmethod
    def is_call_requirement_created(event):
        return isinstance(event, CallRequirement.Created)

    def print_compilation_progress(self, event):
        if self.verbose:
            msg = "\rCompiled {} nodes ".format(self.call_requirement_count)
            self.call_requirement_count += 1
            sys.stdout.write(msg)
            sys.stdout.flush()

    @staticmethod
    def is_calculating(event):
        return isinstance(event, (
            ResultValueComputed,
            CallRequirement.Created,
            CallLink.Created,
            CallDependents.Created,
            SimulatedPriceRequirements.Created,
        ))

    def is_evaluation_complete(self, event):
        return isinstance(event, CallResult.Created) and event.entity_id == self.root_result_id

    def set_is_finished(self, *_):
        self.is_finished.set()

    @staticmethod
    def is_call_result_created(event):
        return isinstance(event, CallResult.Created)

    def inc_call_result_count(self, event):
        self.call_result_count += 1

    @staticmethod
    def is_result_value_computed(event):
        return isinstance(event, ResultValueComputed)

    def inc_result_value_computed_count(self, event):
        self.result_count += 1
        self.result_cost += event.cost

    def print_evaluation_progress(self, event):
        self.check_is_timed_out(event)

        # Todo: Settle this down, needs closer accounting of cost of element evaluation to stop estimate being so
        # jumpy. Perhaps estimating the complexity is something to do when building the dependency graph?
        # i = len(self.times) // 2
        i = 0
        j = len(self.times) - 1
        self.times.append(datetime.datetime.now())

        if self.verbose and j > i:
            duration = self.times[j] - self.times[i]
            rate_cost = self.result_cost / duration.total_seconds()
            rate_count = self.result_count / duration.total_seconds()
            eta = (self.result_cost_expected - self.result_cost) / rate_cost
            seconds_running = (datetime.datetime.now() - self.started).total_seconds()
            seconds_evaluating = (datetime.datetime.now() - self.started_evaluating).total_seconds()

            percent_complete = (100.0 * self.result_cost) / self.result_cost_expected
            msg = (
                "\r"
                "{}/{} "
                "{:.2f}% complete "
                "{:.2f} eval/s "
                "running {:.0f}s "
                "eta {:.0f}s").format(
                self.result_count,
                self.result_count_expected,
                percent_complete,
                rate_count,
                seconds_running,
                eta,
            )

            if self.timeout:
                msg += ' timeout in {:.0f}s'.format(self.timeout - seconds_running)
            sys.stdout.write(msg)
            sys.stdout.flush()

            # Abort if there isn't enough time left.
            if self.timeout:
                out_of_time = self.timeout < seconds_running + eta
                if out_of_time and seconds_evaluating > 15 and eta > 2:
                    msg = ('eta still {:.0f}s after {:.0f}s, so '
                           'aborting in anticipation of {:.0f}s timeout'
                           ).format(eta, seconds_running, self.timeout)
                    self.set_is_timed_out(msg)
                    raise Exception(msg)

    def wait_then_set_is_timed_out(self):
        sleep(self.timeout)
        if not self.is_finished.is_set():
            msg = 'Timed out after {}s'.format(self.timeout)
            self.set_is_timed_out(msg)

    def set_is_timed_out(self, msg):
        self.timeout_msg = msg
        self.is_timed_out.set()
        self.set_is_finished()

    def shutdown(self, signal, frame):
        self.set_is_interrupted('Interrupted by signal {}'.format(signal))

    def set_is_interrupted(self, msg):
        self.interruption_msg = msg
        self.is_interrupted.set()
        self.set_is_finished()

    def check_is_timed_out(self, *_):
        if self.is_timed_out.is_set():
            raise TimeoutError(self.timeout_msg)

    def check_is_interrupted(self, *_):
        if self.is_interrupted.is_set():
            self.set_is_finished()
            raise InterruptSignalReceived(self.interruption_msg)


def print_results(results, path_count):
    print("")
    print("")

    dates = []
    for period in results.periods:
        date = period['delivery_date']
        if date not in dates:
            dates.append(date)

    sqrt_path_count = math.sqrt(path_count)
    if isinstance(results.fair_value, six.integer_types + (float,)):
        fair_value_mean = results.fair_value
        fair_value_stderr = 0
    else:
        fair_value_mean = results.fair_value.mean()
        fair_value_stderr = results.fair_value.std() / sqrt_path_count

    if results.periods:
        net_hedge_cost = 0.0
        net_hedge_units = defaultdict(int)

        market_name_width = max([len(k) for k in results.by_market_name.keys()])
        for delivery_date, markets_results in sorted(results.by_delivery_date.items()):

            for market_result in markets_results:
                market_name = market_result['market_name']
                if delivery_date:
                    print("{} {}".format(delivery_date, market_name))
                else:
                    print(market_name)
                delta = market_result['delta'].mean()
                print("Delta: {: >8.2f}".format(delta))
                price_simulated = market_result['price_simulated'].mean()
                print("Price: {: >8.2f}".format(price_simulated))
                hedge_units = market_result['hedge_units']
                hedge_units_mean = hedge_units.mean()
                hedge_units_stderr = hedge_units.std() / sqrt_path_count
                print("Hedge: {: >8.2f} ± {:.2f}".format(hedge_units_mean, hedge_units_stderr))
                hedge_cost = market_result['hedge_cost']
                hedge_cost_mean = hedge_cost.mean()
                hedge_cost_stderr = hedge_cost.std() / sqrt_path_count
                net_hedge_cost += hedge_cost
                print("Cash:  {: >8.2f} ± {:.2f}".format(-hedge_cost_mean, 3 * hedge_cost_stderr))
                if len(dates) > 1:
                    market_name = market_result['market_name']
                    net_hedge_units[market_name] += hedge_units

                print()

        for commodity in sorted(net_hedge_units.keys()):
            units = net_hedge_units[commodity]
            print("Net hedge {:6} {: >8.2f} ± {: >.2f}".format(
                commodity+':', units.mean(), 3 * units.std() / sqrt_path_count)
            )

        net_hedge_cost_mean = net_hedge_cost.mean()
        net_hedge_cost_stderr = net_hedge_cost.std() / sqrt_path_count

        print("Net hedge cash:  {: >8.2f} ± {: >.2f}".format(-net_hedge_cost_mean, 3 * net_hedge_cost_stderr))
        print()
    # if isinstance(results.fair_value, ndarray):
    #     print("nans: {}".format(isnan(results.fair_value).sum()))
    print("Fair value: {:.2f} ± {:.2f}".format(fair_value_mean, 3 * fair_value_stderr))


def plot_periods(periods, title, periodisation, interest_rate, path_count, perturbation_factor):
    from matplotlib import dates as mdates, pylab as plt

    names = set([p['market_name'] for p in periods])

    f, subplots = plt.subplots(1 + 2 * len(names), sharex=True)
    f.canvas.set_window_title(title)
    f.suptitle('paths:{} perturbation:{} interest:{}% '.format(
        path_count, perturbation_factor, interest_rate))

    if periodisation == 'monthly':
        subplots[0].xaxis.set_major_locator(mdates.MonthLocator())
        subplots[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif periodisation == 'daily':
        subplots[0].xaxis.set_major_locator(mdates.DayLocator())
        subplots[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        return

    NUM_STD_DEVS = 2
    OUTER_COLOUR = 'y'
    MID_COLOUR = 'g'
    INNER_COLOUR = 'b'
    MEAN_COLOUR = '0.1'
    for i, name in enumerate(names):

        _periods = [p for p in periods if p['market_name'] == name]

        dates = [p['delivery_date'] for p in _periods]
        price_plot = subplots[i]

        prices_mean = [p['price_simulated'].mean() for p in _periods]
        # prices_1 = [p['price'][0] for p in _periods]
        # prices_2 = [p['price'][1] for p in _periods]
        # prices_3 = [p['price'][2] for p in _periods]
        # prices_4 = [p['price'][3] for p in _periods]
        prices_std = [p['price_simulated'].std() for p in _periods]
        prices_plus = list(numpy.array(prices_mean) + NUM_STD_DEVS * numpy.array(prices_std))
        prices_minus = list(numpy.array(prices_mean) - NUM_STD_DEVS * numpy.array(prices_std))

        price_plot.set_title('Prices - {}'.format(name))
        price_plot.plot(
            # dates, prices_1, 'g',
            # dates, prices_2, 'b',
            # dates, prices_3, 'm',
            # dates, prices_4, 'c',
            dates, prices_plus, '0.75',
            dates, prices_minus, '0.75',
            dates, prices_mean, '0.25',
        )

        ymin = min(0, min(prices_minus)) - 1
        ymax = max(0, max(prices_plus)) + 1
        price_plot.set_ylim([ymin, ymax])

        cum_pos = []
        for p in _periods:
            pos = p['hedge_units']
            if cum_pos:
                pos += cum_pos[-1]
            cum_pos.append(pos)
        cum_pos_mean = [p.mean() for p in cum_pos]
        cum_pos_p5 = [nanpercentile(p, 5) for p in cum_pos]
        cum_pos_p10 = [nanpercentile(p, 10) for p in cum_pos]
        cum_pos_p25 = [nanpercentile(p, 25) for p in cum_pos]
        cum_pos_p50 = [nanpercentile(p, 50) for p in cum_pos]
        cum_pos_p75 = [nanpercentile(p, 75) for p in cum_pos]
        cum_pos_p90 = [nanpercentile(p, 90) for p in cum_pos]
        cum_pos_p95 = [nanpercentile(p, 95) for p in cum_pos]

        cum_pos_std = [p.std() for p in cum_pos]
        cum_pos_stderr = [p / math.sqrt(path_count) for p in cum_pos_std]
        cum_pos_std_offset = NUM_STD_DEVS * numpy.array(cum_pos_std)
        cum_pos_std_plus = list(numpy.array(cum_pos_mean) + cum_pos_std_offset)
        cum_pos_std_minus = list(numpy.array(cum_pos_mean) - cum_pos_std_offset)
        com_pos_stderr_offset = NUM_STD_DEVS * numpy.array(cum_pos_stderr)
        cum_pos_stderr_plus = list(numpy.array(cum_pos_mean) + com_pos_stderr_offset)
        cum_pos_stderr_minus = list(numpy.array(cum_pos_mean) - com_pos_stderr_offset)

        pos_plot = subplots[len(names) + i]
        pos_plot.set_title('Position - {}'.format(name))

        pos_plot.plot(
            # dates, cum_pos_std_plus, '0.85',
            # dates, cum_pos_std_minus, '0.85',
            dates, cum_pos_p25, INNER_COLOUR,
            dates, cum_pos_p75, INNER_COLOUR,
            dates, cum_pos_p10, MID_COLOUR,
            dates, cum_pos_p90, MID_COLOUR,
            # dates, cum_pos_p5, OUTER_COLOUR,
            # dates, cum_pos_p95, OUTER_COLOUR,
            # dates, cum_pos_stderr_plus, '0.5',
            # dates, cum_pos_stderr_minus, '0.5',
            dates, cum_pos_p50, MEAN_COLOUR,
        )

        ymin = min(0, min(cum_pos_std_minus)) - 1
        ymax = max(0, max(cum_pos_std_plus)) + 1
        pos_plot.set_ylim([ymin, ymax])

    profit_plot = subplots[-1]
    profit_plot.set_title('Profit')

    hedge_cost_by_date = defaultdict(list)

    dates = []
    for period in periods:
        date = period['delivery_date']
        if date not in dates:
            dates.append(date)
        hedge_cost_by_date[date].append(-period['hedge_cost'])

    cum_hedge_cost = []
    for date in dates:
        hedge_cost = sum(hedge_cost_by_date[date])
        if cum_hedge_cost:
            hedge_cost += cum_hedge_cost[-1]
        cum_hedge_cost.append(hedge_cost)

    cum_cash_p5 = [nanpercentile(p, 5) for p in cum_hedge_cost]
    cum_cash_p10 = [nanpercentile(p, 10) for p in cum_hedge_cost]
    cum_cash_p25 = [nanpercentile(p, 25) for p in cum_hedge_cost]
    cum_cash_p50 = [nanpercentile(p, 50) for p in cum_hedge_cost]
    cum_cash_p75 = [nanpercentile(p, 75) for p in cum_hedge_cost]
    cum_cash_p90 = [nanpercentile(p, 90) for p in cum_hedge_cost]
    cum_cash_p95 = [nanpercentile(p, 95) for p in cum_hedge_cost]
    cum_cash_mean = [p.mean() for p in cum_hedge_cost]
    cum_cash_std = [p.std() for p in cum_hedge_cost]
    cum_cash_stderr = [p / math.sqrt(path_count) for p in cum_cash_std]

    cum_cash_std_offset = NUM_STD_DEVS * numpy.array(cum_cash_std)
    cum_cash_std_plus = list(numpy.array(cum_cash_mean) + cum_cash_std_offset)
    cum_cash_std_minus = list(numpy.array(cum_cash_mean) - cum_cash_std_offset)

    cum_cash_stderr_offset = NUM_STD_DEVS * numpy.array(cum_cash_stderr)
    cum_cash_stderr_plus = list(numpy.array(cum_cash_mean) + cum_cash_stderr_offset)
    cum_cash_stderr_minus = list(numpy.array(cum_cash_mean) - cum_cash_stderr_offset)

    profit_plot.plot(
        # dates, cum_cash_std_plus, '0.8',
        # dates, cum_cash_std_minus, '0.8',
        dates, cum_cash_p25, INNER_COLOUR,
        dates, cum_cash_p75, INNER_COLOUR,
        dates, cum_cash_p10, MID_COLOUR,
        dates, cum_cash_p90, MID_COLOUR,
        # dates, cum_cash_p5, OUTER_COLOUR,
        # dates, cum_cash_p95, OUTER_COLOUR,
        # dates, cum_cash_stderr_plus, '0.5',
        # dates, cum_cash_stderr_minus, '0.5',
        dates, cum_cash_p50, MEAN_COLOUR
    )

    f.autofmt_xdate(rotation=60)

    [p.grid() for p in subplots]

    plt.show()
