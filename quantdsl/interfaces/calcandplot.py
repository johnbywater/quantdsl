# coding=utf-8
from __future__ import print_function

import collections
import datetime
import math
import os
import sys
from collections import defaultdict
from threading import Event

import dateutil.parser
import numpy
from eventsourcing.domain.model.events import subscribe, unsubscribe
from matplotlib import dates as mdates, pylab as plt
from numpy import cumsum, zeros

from quantdsl.application.with_multithreading_and_python_objects import \
    QuantDslApplicationWithMultithreadingAndPythonObjects
from quantdsl.domain.model.call_result import CallResult, ResultValueComputed, make_call_result_id
from quantdsl.domain.model.contract_valuation import ContractValuation
from quantdsl.domain.model.simulated_price import make_simulated_price_id
from quantdsl.priceprocess.base import datetime_from_date


class Results(object):
    def __init__(self, fair_value, periods):
        self.fair_value = fair_value
        self.periods = periods


def calc_print_plot(title, source_code, observation_date, periodisation, interest_rate, path_count,
                    perturbation_factor, price_process, supress_plot=False):

    results = calc_print(source_code, observation_date, interest_rate, path_count, perturbation_factor,
                           price_process)

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


def calc_print(source_code, observation_date, interest_rate, path_count, perturbation_factor, price_process):
    results = calc(
        source_code=source_code,
        interest_rate=interest_rate,
        path_count=path_count,
        observation_date=observation_date,
        perturbation_factor=perturbation_factor,
        price_process=price_process,
    )
    print_results(results, path_count)
    return results


def calc(source_code, observation_date=None, interest_rate=0, path_count=20000, perturbation_factor=0.01,
         price_process=None):
    with Calculate(source_code, observation_date, interest_rate, path_count, perturbation_factor, price_process) as \
            cmd:
        return cmd.run()


class Calculate(object):
    def __init__(self, source_code, observation_date, interest_rate, path_count, perturbation_factor, price_process):
        self.result_values_computed_count = 0
        self.call_result_id = None
        self.is_completed = Event()
        subscribe(self.is_result_value_computed, self.count_result_values_computed)
        subscribe(self.is_evaluation_complete, self.on_evaluation_complete)

        self.source_code = source_code
        self.observation_date = observation_date
        self.interest_rate = interest_rate
        self.path_count = path_count
        self.perturbation_factor = perturbation_factor
        self.price_process = price_process
        self._run_once = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        unsubscribe(self.is_result_value_computed, self.count_result_values_computed)
        unsubscribe(self.is_evaluation_complete, self.on_evaluation_complete)

    def run(self):
        assert not self._run_once, "Already run once"
        self._run_once = True
        with QuantDslApplicationWithMultithreadingAndPythonObjects() as app:

            start_compile = datetime.datetime.now()
            contract_specification = app.compile(self.source_code)
            end_compile = datetime.datetime.now()
            print("Compilation in {}s".format((end_compile - start_compile).total_seconds()))

            start_calc = datetime.datetime.now()

            evaluation, market_simulation = self.calc_results(app, contract_specification)

            self.call_result_id = make_call_result_id(evaluation.id, evaluation.contract_specification_id)

            while self.call_result_id not in app.call_result_repo:
                if self.is_completed.wait(timeout=2):
                    break

            results = self.read_results(app, evaluation, market_simulation)

            end_calc = datetime.datetime.now()
            print("")
            print("Results in {}s".format((end_calc - start_calc).total_seconds()))

        return results

    def calc_results(self, app, contract_specification):
        if self.price_process is not None:
            price_process_name = self.price_process['name']
            calibration_params = {k: v for k, v in self.price_process.items() if k != 'name'}
        else:
            price_process_name = 'quantdsl.priceprocess.blackscholes.BlackScholesPriceProcess'
            calibration_params = {}
        market_calibration = app.register_market_calibration(
            price_process_name=price_process_name,
            calibration_params=calibration_params
        )

        if self.observation_date is not None:
            observation_date = datetime_from_date(dateutil.parser.parse(self.observation_date))
        else:
            observation_date = None
        market_simulation = app.simulate(
            contract_specification,
            market_calibration,
            path_count=self.path_count,
            observation_date=observation_date,
            interest_rate=self.interest_rate,
            perturbation_factor=self.perturbation_factor,
        )

        call_costs = app.calc_call_costs(contract_specification.id)
        self.total_cost = sum(call_costs.values())

        self.times = collections.deque()

        evaluation = app.evaluate(contract_specification.id, market_simulation.id)
        return evaluation, market_simulation

    def read_results(self, app, evaluation, market_simulation):
        assert isinstance(evaluation, ContractValuation)

        call_result_id = make_call_result_id(evaluation.id, evaluation.contract_specification_id)
        call_result = app.call_result_repo[call_result_id]

        sqrt_path_count = math.sqrt(self.path_count)

        fair_value = call_result.result_value

        perturbed_names = call_result.perturbed_values.keys()
        perturbed_names = [i for i in perturbed_names if not i.startswith('-')]
        perturbed_names = sorted(perturbed_names, key=lambda x: [int(i) for i in x.split('-')[1:]])

        total_cash_in = zeros(self.path_count)
        total_units = zeros(self.path_count)
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
                price_mean = simulated_price.value.mean()
                price_std = simulated_price.value.std()
                dy = perturbed_value - perturbed_value_negative
                dx = 2 * market_simulation.perturbation_factor * simulated_price.value
                contract_delta = dy / dx
                hedge_units = - contract_delta
                hedge_units_mean = hedge_units.mean()
                hedge_units_stderr = hedge_units.std() / sqrt_path_count
                cash_in = - hedge_units * simulated_price.value
                cash_in_mean = cash_in.mean()
                cash_in_stderr = cash_in.std() / sqrt_path_count
                periods.append({
                    'commodity': perturbed_name,
                    'date': None,
                    'hedge_units_mean': hedge_units_mean,
                    'hedge_units_stderr': hedge_units_stderr,
                    'price_mean': price_mean,
                    'price_std': price_std,
                    'cash_in_mean': cash_in_mean,
                    'cash_in_stderr': cash_in_stderr,
                    'cum_cash_mean': cash_in_mean,
                    'cum_cash_stderr': cash_in_stderr,
                    'cum_pos_mean': hedge_units_mean,
                    'cum_pos_stderr': hedge_units_stderr,
                    # 'total_unit_stderr': total_units_stderr,
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
                price_mean = price.mean()
                dx = 2 * market_simulation.perturbation_factor * price
                contract_delta = dy / dx
                hedge_units = -contract_delta
                cash_in = - hedge_units * price
                total_units += hedge_units
                total_cash_in += cash_in
                hedge_units_mean = hedge_units.mean()
                hedge_units_stderr = hedge_units.std() / sqrt_path_count
                cash_in_stderr = cash_in.std() / sqrt_path_count
                total_units_stderr = total_units.std() / sqrt_path_count
                cash_in_mean = cash_in.mean()
                total_units_mean = total_units.mean()
                total_cash_in_mean = total_cash_in.mean()
                periods.append({
                    'commodity': perturbed_name,
                    'date': price_date,
                    'hedge_units_mean': hedge_units_mean,
                    'hedge_units_stderr': hedge_units_stderr,
                    'price_mean': price_mean,
                    'price_std': price.std(),  # / math.sqrt(path_count),
                    'cash_in_mean': cash_in_mean,
                    'cash_in_stderr': cash_in_stderr,
                    'cum_cash_mean': total_cash_in_mean,
                    'cum_cash_stderr': total_cash_in.std() / sqrt_path_count,
                    'cum_pos_mean': total_units_mean,
                    'cum_pos_stderr': total_units.std() / sqrt_path_count,
                    'total_unit_stderr': total_units_stderr,
                })

        return Results(fair_value, periods)

    def is_result_value_computed(self, event):
        return isinstance(event, ResultValueComputed)

    def count_result_values_computed(self, event):
        self.times.append(datetime.datetime.now())
        if len(self.times) > 0.5 * self.total_cost:
            self.times.popleft()
        if len(self.times) > 1:
            duration = self.times[-1] - self.times[0]
            rate = len(self.times) / duration.total_seconds()
        else:
            rate = 0.001
        eta = (self.total_cost - self.result_values_computed_count) / rate
        assert isinstance(event, ResultValueComputed)
        self.result_values_computed_count += 1
        sys.stdout.write(
            "\r{:.2f}% complete ({}/{}) {:.2f}/s eta {:.0f}s".format(
                (100.0 * self.result_values_computed_count) / self.total_cost,
                self.result_values_computed_count,
                self.total_cost,
                rate,
                eta
            )
        )
        sys.stdout.flush()

    def is_evaluation_complete(self, event):
        return isinstance(event, CallResult.Created) and event.entity_id == self.call_result_id

    def on_evaluation_complete(self, _):
        self.is_completed.set()


def print_results(results, path_count):
    print("")
    print("")

    if isinstance(results.fair_value, (int, float, long)):
        fair_value_mean = results.fair_value
        fair_value_stderr = 0
    else:
        fair_value_mean = results.fair_value.mean()
        fair_value_stderr = results.fair_value.std() / math.sqrt(path_count)

    if results.periods:
        for period in results.periods:
            print(period['commodity'])
            print("Price: {:.2f}".format(period['price_mean']))
            print("Hedge: {:.2f} ± {:.2f} units of {}".format(period['hedge_units_mean'],
                                                              3 * period['hedge_units_stderr'],
                                                              period['commodity']))
            print("Cash in: {:.2f} ± {:.2f}".format(period['cash_in_mean'], 3 * period['cash_in_stderr']))
            print("Cum posn: {:.2f} ± {:.2f}".format(period['cum_pos_mean'], 3 * period['cum_pos_stderr']))
            print()
        last_data = results.periods[-1]
        print("Net cash in: {:.2f} ± {:.2f}".format(last_data['cum_cash_mean'], 3 * last_data['cum_cash_stderr']))
        print("Net position: {:.2f} ± {:.2f}".format(last_data['cum_pos_mean'], 3 * last_data['cum_pos_stderr']))
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
        raise NotImplementedError(periodisation)

    for i, name in enumerate(names):

        _periods = [p for p in periods if p['commodity'].startswith(name)]

        dates = [p['date'] for p in _periods]
        price_plot = subplots[i]

        prices_mean = [p['price_mean'] for p in _periods]
        prices_std = [p['price_std'] for p in _periods]
        prices_plus = list(numpy.array(prices_mean) + 2 * numpy.array(prices_std))
        prices_minus = list(numpy.array(prices_mean) - 2 * numpy.array(prices_std))

        price_plot.set_title('Prices - {}'.format(name))
        price_plot.plot(dates, prices_plus, '0.75', dates, prices_minus, '0.75', dates, prices_mean, '0.25')

        ymin = min(0, min(prices_minus)) - 1
        ymax = max(0, max(prices_plus)) + 1
        price_plot.set_ylim([ymin, ymax])

        cum_pos_mean = cumsum([p['hedge_units_mean'] for p in _periods])
        cum_pos_stderr = cumsum([p['hedge_units_stderr'] for p in _periods])
        cum_pos_plus = list(numpy.array(cum_pos_mean) + 3 * numpy.array(cum_pos_stderr))
        cum_pos_minus = list(numpy.array(cum_pos_mean) - 3 * numpy.array(cum_pos_stderr))

        pos_plot = subplots[len(names) + i]
        pos_plot.set_title('Position - {}'.format(name))
        pos_plot.plot(dates, cum_pos_plus, '0.75', dates, cum_pos_minus, '0.75', dates, cum_pos_mean, '0.25')
        ymin = min(0, min(cum_pos_minus)) - 1
        ymax = max(0, max(cum_pos_plus)) + 1
        pos_plot.set_ylim([ymin, ymax])

    profit_plot = subplots[-1]
    profit_plot.set_title('Profit')

    dates = []

    cash_in_mean = defaultdict(int)
    cash_in_stderr = defaultdict(int)

    for period in periods:
        date = period['date']
        if date not in dates:
            dates.append(date)
        cash_in_mean[date] += period['cash_in_mean']
        cash_in_stderr[date] += period['cash_in_stderr']

    # [cash_in[p['date']].append(p['cash_in_mean']) for p in _periods]

    cum_cash_mean = cumsum([cash_in_mean[date] for date in dates])
    cum_cash_stderr = cumsum([cash_in_stderr[date] for date in dates])
    # cum_cash_mean = [p['cum_cash_mean'] for p in periods]
    # cum_cash_stderr = [p['cum_cash_stderr'] for p in periods]
    cum_cash_plus = list(numpy.array(cum_cash_mean) + 3 * numpy.array(cum_cash_stderr))
    cum_cash_minus = list(numpy.array(cum_cash_mean) - 3 * numpy.array(cum_cash_stderr))

    profit_plot.plot(dates, cum_cash_plus, '0.75', dates, cum_cash_minus, '0.75', dates, cum_cash_mean, '0.25')
    # profit_plot.plot(dates, cum_cash_mean, '0.25')

    f.autofmt_xdate(rotation=60)

    [p.grid() for p in subplots]

    plt.show()
