# coding=utf-8
from __future__ import print_function

import collections
import datetime
import math
import os
import sys
from collections import defaultdict
from threading import Event, Thread
from time import sleep

import dateutil.parser
import numpy
from eventsourcing.domain.model.events import subscribe, unsubscribe
from matplotlib import dates as mdates, pylab as plt

from quantdsl.application.base import DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE
from quantdsl.application.with_multithreading_and_python_objects import \
    QuantDslApplicationWithMultithreadingAndPythonObjects
from quantdsl.domain.model.call_dependents import CallDependents
from quantdsl.domain.model.call_link import CallLink
from quantdsl.domain.model.call_requirement import CallRequirement
from quantdsl.domain.model.call_result import CallResult, ResultValueComputed, make_call_result_id
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.domain.model.simulated_price_requirements import SimulatedPriceRequirements
from quantdsl.exceptions import TimeoutError
from quantdsl.priceprocess.base import datetime_from_date


class Results(object):
    def __init__(self, fair_value, periods):
        self.fair_value = fair_value
        self.periods = periods


def calc_print_plot(source_code, title='', observation_date=None, periodisation=None, interest_rate=0,
                    path_count=20000, perturbation_factor=0.01, price_process=None, supress_plot=False,
                    max_dependency_graph_size=DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE, timeout=None, verbose=False,
                    approximate_discounting=False):
    # Calculate and print the results.
    results = calc_print(source_code,
                         max_dependency_graph_size=max_dependency_graph_size,
                         observation_date=observation_date,
                         interest_rate=interest_rate,
                         path_count=path_count,
                         perturbation_factor=perturbation_factor,
                         price_process=price_process,
                         periodisation=periodisation,
                         timeout=timeout,
                         verbose=verbose,
                         approximate_discounting=approximate_discounting,
                         )

    # Plot the results.
    if results.periods and not supress_plot and not os.getenv('SUPRESS_PLOT'):
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
               price_process=None, periodisation=None, max_dependency_graph_size=DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE,
               timeout=None, verbose=False, approximate_discounting=False):
    # Calculate the results.
    results = calc(
        source_code=source_code,
        interest_rate=interest_rate,
        path_count=path_count,
        observation_date=observation_date,
        perturbation_factor=perturbation_factor,
        price_process=price_process,
        periodisation=periodisation,
        max_dependency_graph_size=max_dependency_graph_size,
        timeout=timeout,
        verbose=verbose,
        approximate_discounting=approximate_discounting,
    )

    # Print the results.
    print_results(results, path_count)
    return results


def calc(source_code, observation_date=None, interest_rate=0, path_count=20000, perturbation_factor=0.01,
         price_process=None, periodisation=None, max_dependency_graph_size=DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE,
         timeout=None, verbose=False, approximate_discounting=False):
    cmd = Calculate(
        source_code=source_code,
        observation_date=observation_date,
        interest_rate=interest_rate,
        path_count=path_count,
        perturbation_factor=perturbation_factor,
        price_process=price_process,
        periodisation=periodisation,
        max_dependency_graph_size=max_dependency_graph_size,
        timeout=timeout,
        verbose=verbose,
        approximate_discounting=approximate_discounting,
    )
    return cmd.calculate()


class Calculate(object):
    def __init__(self, source_code, observation_date=None, interest_rate=0, path_count=20000, perturbation_factor=0.01,
                 price_process=None, periodisation=None, max_dependency_graph_size=DEFAULT_MAX_DEPENDENCY_GRAPH_SIZE,
                 timeout=None, verbose=False, approximate_discounting=False):
        self.timeout = timeout
        self.source_code = source_code
        self.observation_date = observation_date
        self.interest_rate = interest_rate
        self.path_count = path_count
        self.perturbation_factor = perturbation_factor
        # Todo: Optional double or single sided deltas.
        # self.double_sided_deltas = double_sided_deltas
        self.price_process = price_process
        self.periodisation = periodisation
        self.approximate_discounting = approximate_discounting
        self.max_dependency_graph_size = max_dependency_graph_size
        self.verbose = verbose
        # Todo: Repetitions - number of times the computation will be repeated (multiprocessing).
        # self.repetitions = repetitions

    def calculate(self):
        self.node_evaluations_count = 0
        self.root_result_id = None
        self.is_timed_out = Event()
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

        # Compile.
        with QuantDslApplicationWithMultithreadingAndPythonObjects(
                max_dependency_graph_size=self.max_dependency_graph_size) as app:

            # Subscribe after the application, so events are received after the application.
            # - this means the final result is persisted before this interface is notified
            #   the result is available and tries to get it, avoiding a race condition
            self.subscribe()
            try:

                start_compile = datetime.datetime.now()
                contract_specification = app.compile(self.source_code)
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

                if self.observation_date is not None:
                    observation_date = datetime_from_date(dateutil.parser.parse(self.observation_date))
                else:
                    observation_date = None

                # Simulate the market prices.
                start_simulate = datetime.datetime.now()
                market_simulation = app.simulate(
                    contract_specification,
                    price_process_name=price_process_name,
                    calibration_params=calibration_params,
                    path_count=self.path_count,
                    observation_date=observation_date,
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
                call_costs = app.calc_call_costs(contract_specification.id)
                self.node_evaluations_num_expected = sum(call_costs.values())
                if self.verbose:
                    print("Starting {} node evaluations, please wait...".format(self.node_evaluations_num_expected))
                self.expected_num_call_requirements = len(call_costs)

                # Evaluate the contract specification.
                start_calc = datetime.datetime.now()
                self.started_evaluating = datetime.datetime.now()
                evaluation = app.evaluate(
                    contract_specification_id=contract_specification.id,
                    market_simulation_id=market_simulation.id,
                    periodisation=self.periodisation,
                    approximate_discounting=self.approximate_discounting,
                )

                # Wait for the result.
                self.root_result_id = make_call_result_id(evaluation.id, evaluation.contract_specification_id)
                if not self.root_result_id in app.call_result_repo:
                    while not self.is_finished.wait():
                        self.check_is_timed_out()
                    self.check_is_timed_out()

                # Todo: Separate this, not all users want print statements.
                end_calc = datetime.datetime.now()
                if self.verbose:
                    print("")
                    print("Evaluation in {:.3f}s".format((end_calc - start_calc).total_seconds()))

                # Read the results.
                results = self.read_results(app, evaluation, market_simulation)
            finally:

                self.unsubscribe()

        return results

    def read_results(self, app, evaluation, market_simulation):
        assert isinstance(evaluation, ContractValuation)

        call_result_id = make_call_result_id(evaluation.id, evaluation.contract_specification_id)
        call_result = app.call_result_repo[call_result_id]

        fair_value = call_result.result_value

        perturbed_names = call_result.perturbed_values.keys()
        perturbed_names = [i for i in perturbed_names if not i.startswith('-')]
        perturbed_names = sorted(perturbed_names, key=lambda x: [int(i) for i in x.split('-')[1:]])

        periods = []
        for perturbed_name in perturbed_names:

            perturbed_value = call_result.perturbed_values[perturbed_name]
            perturbed_value_negative = call_result.perturbed_values['-' + perturbed_name]
            # Assumes format: NAME-YEAR-MONTH
            perturbed_name_split = perturbed_name.split('-')
            commodity_name = perturbed_name_split[0]

            if commodity_name == perturbed_name:
                simulated_price_id = make_simulated_price_id(market_simulation.id, commodity_name,
                                                             market_simulation.observation_date,
                                                             market_simulation.observation_date)

                simulated_price = app.simulated_price_repo[simulated_price_id]
                price = simulated_price.value
                dy = perturbed_value - perturbed_value_negative
                dx = 2 * market_simulation.perturbation_factor * price
                contract_delta = dy / dx
                hedge_units = - contract_delta
                cash_in = - hedge_units * price
                periods.append({
                    'commodity': perturbed_name,
                    'date': None,
                    'hedge_units': hedge_units,
                    'price': price,
                    'cash_in': cash_in,
                })

            elif len(perturbed_name_split) > 2:
                year = int(perturbed_name_split[1])
                month = int(perturbed_name_split[2])
                if len(perturbed_name_split) > 3:
                    day = int(perturbed_name_split[3])
                    price_date = datetime.date(year, month, day)
                else:
                    price_date = datetime.date(year, month, 1)
                simulated_price_id = make_simulated_price_id(market_simulation.id, commodity_name, price_date,
                                                             price_date)
                try:
                    simulated_price = app.simulated_price_repo[simulated_price_id]
                except KeyError as e:
                    raise Exception("Simulated price for date {} is unavailable".format(price_date, e))

                dy = perturbed_value - perturbed_value_negative
                price = simulated_price.value
                dx = 2 * market_simulation.perturbation_factor * price
                contract_delta = dy / dx
                hedge_units = -contract_delta
                cash_in = - hedge_units * price
                periods.append({
                    'commodity': perturbed_name,
                    'date': price_date,
                    'hedge_units': hedge_units,
                    'price': price,
                    'cash_in': cash_in,
                })

        return Results(fair_value, periods)

    def subscribe(self):
        subscribe(self.is_call_requirement_created, self.print_compilation_progress)
        subscribe(self.is_calculating, self.check_is_timed_out)
        subscribe(self.is_evaluation_complete, self.set_is_finished)
        subscribe(self.is_result_value_computed, self.inc_result_value_computed_count)
        subscribe(self.is_result_value_computed, self.print_evaluation_progress)
        subscribe(self.is_call_result_created, self.inc_call_result_count)
        subscribe(self.is_call_result_created, self.print_evaluation_progress)

    def unsubscribe(self):
        unsubscribe(self.is_call_requirement_created, self.print_compilation_progress)
        unsubscribe(self.is_calculating, self.check_is_timed_out)
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

    def check_is_timed_out(self, *_):
        if self.is_timed_out.is_set():
            raise TimeoutError(self.timeout_msg)

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

    def inc_result_value_computed_count(self, _):
        self.node_evaluations_count += 1

    def print_evaluation_progress(self, event):
        self.check_is_timed_out(event)

        if self.verbose:
            self.times.append(datetime.datetime.now())
            if len(self.times) > max(0.5 * self.node_evaluations_num_expected, 100):
                self.times.popleft()
            if len(self.times) > 1:
                duration = self.times[-1] - self.times[0]
                rate = len(self.times) / duration.total_seconds()
            else:
                rate = 0.001
            eta = (self.node_evaluations_num_expected - self.node_evaluations_count) / rate
            seconds_running = (datetime.datetime.now() - self.started).total_seconds()
            seconds_evaluating = (datetime.datetime.now() - self.started_evaluating).total_seconds()

            msg = (
                "\r"
                "{}/{} "
                "{:.2f}% complete "
                "{:.2f} eval/s "
                "running {:.0f}s "
                "eta {:.0f}s").format(
                    self.node_evaluations_count,
                    self.node_evaluations_num_expected,
                (100.0 * self.node_evaluations_count) / self.node_evaluations_num_expected,
                rate,
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
            msg = 'Timeout after {} seconds'.format(self.timeout)
            self.set_is_timed_out(msg)

    def set_is_timed_out(self, msg):
        self.timeout_msg = msg
        self.is_timed_out.set()
        self.set_is_finished()


def print_results(results, path_count):
    print("")
    print("")

    dates = []
    for period in results.periods:
        date = period['date']
        if date not in dates:
            dates.append(date)

    sqrt_path_count = math.sqrt(path_count)
    if isinstance(results.fair_value, (int, float, long)):
        fair_value_mean = results.fair_value
        fair_value_stderr = 0
    else:
        fair_value_mean = results.fair_value.mean()
        fair_value_stderr = results.fair_value.std() / sqrt_path_count

    if results.periods:
        net_cash_in = 0
        net_hedge_units = defaultdict(int)
        for period in results.periods:
            period_commodity = period['commodity']
            print(period_commodity)
            print("Price: {:.2f}".format(period['price'].mean()))
            hedge_units = period['hedge_units']
            hedge_units_mean = hedge_units.mean()
            hedge_units_stderr = hedge_units.std() / sqrt_path_count
            if len(dates) > 1:
                net_hedge_units[period_commodity.split('-')[0]] += hedge_units
            cash_in = period['cash_in']
            cash_in_mean = cash_in.mean()
            cash_in_stderr = cash_in.std() / sqrt_path_count
            net_cash_in += cash_in
            print("Hedge: {:.2f} ± {:.2f} units".format(hedge_units_mean, 3 * hedge_units_stderr))
            print("Cash: {:.2f} ± {:.2f}".format(cash_in_mean, 3 * cash_in_stderr))
            print()

        for commodity in sorted(net_hedge_units.keys()):
            units = net_hedge_units[commodity]
            print("Net {}: {:.2f} ± {:.2f}".format(
                commodity, units.mean(), 3 * units.std() / sqrt_path_count)
            )

        net_cash_in_mean = net_cash_in.mean()
        net_cash_in_stderr = net_cash_in.std() / sqrt_path_count

        print()
        print("Net cash: {:.2f} ± {:.2f}".format(net_cash_in_mean, 3 * net_cash_in_stderr))
        print()
    print("Fair value: {:.2f} ± {:.2f}".format(fair_value_mean, 3 * fair_value_stderr))


def plot_periods(periods, title, periodisation, interest_rate, path_count, perturbation_factor):
    names = set([p['commodity'].split('-')[0] for p in periods])

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

    for i, name in enumerate(names):

        _periods = [p for p in periods if p['commodity'].startswith(name)]

        dates = [p['date'] for p in _periods]
        price_plot = subplots[i]

        prices_mean = [p['price'].mean() for p in _periods]
        prices_1 = [p['price'][0] for p in _periods]
        prices_2 = [p['price'][1] for p in _periods]
        prices_3 = [p['price'][2] for p in _periods]
        prices_4 = [p['price'][3] for p in _periods]
        prices_std = [p['price'].std() for p in _periods]
        prices_plus = list(numpy.array(prices_mean) + 2 * numpy.array(prices_std))
        prices_minus = list(numpy.array(prices_mean) - 2 * numpy.array(prices_std))

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
        cum_pos_std = [p.std() for p in cum_pos]
        cum_pos_stderr = [p / math.sqrt(path_count) for p in cum_pos_std]
        cum_pos_std_plus = list(numpy.array(cum_pos_mean) + 1 * numpy.array(cum_pos_std))
        cum_pos_std_minus = list(numpy.array(cum_pos_mean) - 1 * numpy.array(cum_pos_std))
        cum_pos_stderr_plus = list(numpy.array(cum_pos_mean) + 3 * numpy.array(cum_pos_stderr))
        cum_pos_stderr_minus = list(numpy.array(cum_pos_mean) - 3 * numpy.array(cum_pos_stderr))

        pos_plot = subplots[len(names) + i]
        pos_plot.set_title('Position - {}'.format(name))

        pos_plot.plot(
            dates, cum_pos_std_plus, '0.85',
            dates, cum_pos_std_minus, '0.85',
            dates, cum_pos_stderr_plus, '0.5',
            dates, cum_pos_stderr_minus, '0.5',
            dates, cum_pos_mean, '0.1',
        )

        ymin = min(0, min(cum_pos_std_minus)) - 1
        ymax = max(0, max(cum_pos_std_plus)) + 1
        pos_plot.set_ylim([ymin, ymax])

    profit_plot = subplots[-1]
    profit_plot.set_title('Profit')

    cash_in_by_date = defaultdict(list)

    dates = []
    for period in periods:
        date = period['date']
        if date not in dates:
            dates.append(date)
        cash_in_by_date[date].append(period['cash_in'])

    cum_cash_in = []
    for date in dates:
        cash_in = sum(cash_in_by_date[date])
        if cum_cash_in:
            cash_in += cum_cash_in[-1]
        cum_cash_in.append(cash_in)

    cum_cash_mean = [p.mean() for p in cum_cash_in]
    cum_cash_std = [p.std() for p in cum_cash_in]
    cum_cash_stderr = [p / math.sqrt(path_count) for p in cum_cash_std]
    cum_cash_std_plus = list(numpy.array(cum_cash_mean) + 1 * numpy.array(cum_cash_std))
    cum_cash_std_minus = list(numpy.array(cum_cash_mean) - 1 * numpy.array(cum_cash_std))
    cum_cash_stderr_plus = list(numpy.array(cum_cash_mean) + 3 * numpy.array(cum_cash_stderr))
    cum_cash_stderr_minus = list(numpy.array(cum_cash_mean) - 3 * numpy.array(cum_cash_stderr))

    profit_plot.plot(
        dates, cum_cash_std_plus, '0.8',
        dates, cum_cash_std_minus, '0.8',
        dates, cum_cash_stderr_plus, '0.5',
        dates, cum_cash_stderr_minus, '0.5',
        dates, cum_cash_mean, '0.1'
    )

    f.autofmt_xdate(rotation=60)

    [p.grid() for p in subplots]

    plt.show()
