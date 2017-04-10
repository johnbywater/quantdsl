from __future__ import division
# Schwartz Smith two factor model.
# http://indices.riskamerica.cl/papers/Short%20Long%20Variations%20Commodity.pdf

# This implementation follows "Calibration Techniques in Energy Markets" by Moeti Ncube (September 13, 2011)
# http://www.mathworks.co.uk/matlabcentral/fileexchange/31381-calibration-of-forward-price--volatility--and-correlations-across-multiple-assets

# Quote from work by Moeti Ncube 2011:
#% This code uses the Schwartz-Smith model to calibrate and simulate multiple assets such that:
#%
#% 1. The Calibration and simulation is consistent with the observable forward curve, adjusted for seasonality, at any given date
#% 2. The Calibration and simulation is consistent with the observable ATM volatility at a given date
#% 3. The Calibration and simulation is consistent with the observable correlation structure between the forward curve vectors at
#% a given date
#%
#%
#% In this example, I use four assets: 5x16,2x16,7x8 PJM forward prices and ATM volatilities along with natural gas forward prices
#% and ATM volatilities
#%
#% Calibration of the parameters is done in Excel. By inputing the vectors of Forward and ATM volatilites for each commodity, I
#% can compute the theoretical Schwartz Smith Forward Prices and Standard Deviations for a given maturity. I then use Excel solver
#% to minimize the difference between the observed market values and their theoretical values to obtain the Schwartz-Smith
#% parameters for each commodity. Note this methodology is drastically different from the one described in "Short Term Variation
#% and Long-Term Dynamics in Commodity Prices" in which Schwartz and Smith calibrate their model to historical futures prices.
#% Here I calibrate the model to the current forward and volatility curve as well as adjust for seasonality. This procedure is
#% much more practical pricing methodology.
#%
#% The more difficult step was insuring that the correlation between and  asset(i) and asset(j) at maturity (t) was consistent
#% with the implied correlation between the forward price vectors. This was done by adjusting the correlation between the
#% short-term factors of commodities at each maturity. The matlab code factors in this adjustment and simulates the 4 commodities
#% in this example to show that the theoretical prices, volatilities, and correlations match up with the observed market data.
#%
#% There is not, to my knowledge, a commodities methodology that incorporates so many market factors across multiple commodities
#% into one simulation. The advantages of such a model allows for more accurate modeling of spark spreads and pricing of deals
#% that are dependent on multiple commodities prices. I have included all files, including excel, associated with this calibration
#% and simulation.
from quantdsl.priceprocess.base import PriceProcess

try:
    from matplotlib import pylab as plt, pylab
except RuntimeError:
    pass
import scipy as np
from scipy.optimize import basinhopping

np.seterr(over='raise')


class SchwartzSmithFromFuturesAndImpliedVols(PriceProcess):

    def simulate_future_prices(self, observation_date, requirements, path_count, calibration_params):

        fixing_dates = set()
        for requirement in requirements:
            fixing_dates.add(requirement[1])

        fixing_dates = sorted(list(fixing_dates))

        allMarketNames = []
        allOptimizedParams = []
        allSeasonalParams = []
        allRhos = []
        # Restructure calibration params.
        for params in calibration_params:
            allMarketNames.append(params['name'])
            allOptimizedParams.append(params['schwartzsmith'])
            allSeasonalParams.append(np.array(params['seasonal']))
            allRhos.append(params['rhos'])

        allRhos = np.array(allRhos)
        all_simulated_prices = simulate_prices(observation_date, fixing_dates, allOptimizedParams,
                                         allSeasonalParams, allRhos, path_count=path_count)

        for i in range(len(all_simulated_prices)):
            spath, mpath, estdfwd, stdpath = all_simulated_prices[i]

            market_name = allMarketNames[i]
            simulated_prices = spath.T
            for j in range(len(fixing_dates)):
                simulated_price = simulated_prices[j]

                fixing_date = fixing_dates[j]
                yield market_name, fixing_date, fixing_date, simulated_price

# Replacement spread sheet.

schwartzparamnames = [
    'kappa',
    'mue',
    'sigmax',
    'sigmae',
    'lambdae',
    'lambdax',
    'pxe',
    'x0',
    'e0',
]

NUM_MONTHS_IN_YEAR = 12

enable_optimize_seasonal_params = True


def calibrate(allData, niter=100, path_count=1000):
    results = []

    num_commodities = len(allData)
    initial_seasonal_params = [1.0] * 12

    observation_date = allData[0]['observation_date']
    months = allData[0]['months']
    maturities = calc_maturities(months, observation_date)

    for commodity_data in allData:
        commodity_name = commodity_data['name']
        futures = commodity_data['futures']
        impvols = commodity_data['impliedAtmVols']

        def objective(x):
            # print "Calling objective function with x:", x
            schwartz_params = convert_to_schwartz_params_dict(x)
            if enable_optimize_seasonal_params:
                seasonalParams = x[-NUM_MONTHS_IN_YEAR:]
            else:
                seasonalParams = initial_seasonal_params
            try:
                score = scoreSchwartz(months, maturities, futures, impvols, seasonalParams, schwartz_params)
            except FloatingPointError:
                score = np.inf
            if np.isnan(score):
                score = np.inf
            # Constrain seasonal_weightings.
            #penalty = ((1 + abs(1 - sum(seasonal_factors) / NUM_MONTHS_IN_YEAR))) ** 5
            try:
                penalty = (1 + abs(1 - sum(seasonalParams) / NUM_MONTHS_IN_YEAR)) ** 5
                score = score * penalty
            except FloatingPointError:
                score = np.inf
            return score

        bounds = [
            (0.0000001, 100),  # kappa
            (-1, 1),  # mue
            (0, 5),  # sigmax
            (0, 5),  # sigmae
            (-5, 5),  # lambdae
            (-5, 5),  # lambdax
            (-1, 1),  # pxe
            (0, 100),  # x0
            (0, 100),  # e0
        ]

        if enable_optimize_seasonal_params:
            bounds += [(0.2, 5)] * NUM_MONTHS_IN_YEAR

        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)

        x0 = [0.1] * len(schwartzparamnames)
        if enable_optimize_seasonal_params:
            x0 += [1] * len(initial_seasonal_params)

        # Run the "basin hopping" routine.
        result = basinhopping(objective, x0, T=0.5, stepsize=0.1, niter=niter, minimizer_kwargs=minimizer_kwargs)

        results.append(result)
        print(commodity_name, result.fun, convert_to_schwartz_params_dict(result.x), result.x[9:])
    all_optimized_schwartz_params = []
    all_optimized_seasonal_params = []
    for c in range(num_commodities):
        result = results[c]
        commodity_data = allData[c]
        observation_date = commodity_data['observation_date']
        months = commodity_data['months']
        futures = commodity_data['futures']
        impvols = commodity_data['impliedAtmVols']
        if len(result.x) - len(schwartzparamnames) == NUM_MONTHS_IN_YEAR:
            seasonal_params = result.x[-NUM_MONTHS_IN_YEAR:]
        else:
            seasonal_params = initial_seasonal_params
        all_optimized_seasonal_params.append(seasonal_params)
        optimized_schwartz_params = convert_to_schwartz_params_dict(result.x)
        all_optimized_schwartz_params.append(optimized_schwartz_params)
        #print result.fun, schwartzparamsdictfromlistx(result.x), result.x[9:]

        #schwartzftT, blackstdev, schwartzSDFT = calc_schwartz(observation_date, months, futures, impvols,
        #                                                      seasonal_factors, optimized_params)

        # Plot graphs
        # if plt is not None:
        #     plt.plot(futures, '-b')
        #     plt.plot(schwartzftT, '-r')
        #     plt.plot(blackstdev, '-b')
        #     plt.plot(schwartzSDFT, '-r')
        #     plt.show()

    correlation_matrix, all_rhos = fix_correlations(allData, all_optimized_schwartz_params)

    simulated_prices = simulate_prices(observation_date, months, all_optimized_schwartz_params,
                                         all_optimized_seasonal_params, all_rhos, path_count=path_count)

    # Estimate empirical correlation matrix
    spath0 = simulated_prices[0][0]
    spath0T = spath0.T
    sim_correlations = []
    for i in range(1, num_commodities):
        spath = simulated_prices[i][0]
        spathT = spath.T
        coefs = []
        num_observations = len(spathT)
        for t in range(num_observations):
            coef = np.corrcoef(spath0T[t], spathT[t])[0][1]
            coefs.append(coef)

        sim_correlations.append(np.mean(coefs))
    #
    # MarketCorrelations = cmatrix[0,1:]
    # SimCorrelations = empc[1:]


    return all_optimized_schwartz_params, all_optimized_seasonal_params, all_rhos, correlation_matrix, sim_correlations, simulated_prices


def scoreSchwartz(months, maturities, forwards, impvols, seasonalParams, schwartz_params):
    schwartzftT, blackstdev, schwartzSDFT = calc_schwartz(months, maturities, forwards, impvols, seasonalParams, schwartz_params)
    fwd_err = (forwards - schwartzftT)**2
    stdev_err = (blackstdev - schwartzSDFT)**2
    return fwd_err.sum() + stdev_err.sum()


def calc_schwartz(months, maturities, forwards, impvols, seasonal_params, schwartz_params):

    seasonal_weightings = get_seasonal_weightings(months, seasonal_params)

    kappa = schwartz_params['kappa']
    x0 = schwartz_params['x0']
    e0 = schwartz_params['e0']
    lambdax = schwartz_params['lambdax']
    mue = schwartz_params['mue']
    lambdae = schwartz_params['lambdae']
    sigmax = schwartz_params['sigmax']
    sigmae = schwartz_params['sigmae']
    pxe = schwartz_params['pxe']
    kappa_times_maturities = kappa * maturities
    exp_minus_kappa_times_maturities = np.exp(-kappa_times_maturities)
    sigmae_squared_times_maturities = sigmae ** 2 * maturities

    # Black Stdev
    blackstdev = np.sqrt(forwards * forwards * (np.exp(impvols * impvols * maturities) - 1))

    # Calc Schw E[ln(S(T))]
    schwartzES = exp_minus_kappa_times_maturities * x0 + e0 - (1 - exp_minus_kappa_times_maturities) * lambdax / kappa + (mue - lambdae) * maturities

    # Calc Schw V[ln(S(T))]
    schwartzVS = (1 - np.exp(-2 * kappa_times_maturities)) * (sigmax **2 / (2 * kappa)) \
                 + sigmae_squared_times_maturities \
                 + (2 * (1 - exp_minus_kappa_times_maturities) * pxe * sigmax * sigmae / kappa)

    # Calc Schw F(t,T)
    schwartzftT = seasonal_weightings * np.exp(schwartzES + 0.5 * schwartzVS)

    # Calc Schw V[ln(F(T))]
    # Todo: Find out why schwartzVFT is same expression as schwartzVS - is that correct?
    # schwartzVFT = (1 - np.exp(-2 * kappa_times_maturities)) * sigmax ** 2 / (2 * kappa) \
    #               + sigmae_squared_times_maturities \
    #               + 2 * (1 - exp_minus_kappa_times_maturities) * pxe * sigmax * sigmae / kappa
    schwartzVFT = schwartzVS

    schwartzSDFT = seasonal_weightings * np.sqrt((np.exp(schwartzVFT) - 1) * np.exp(2 * schwartzES + schwartzVS))

    return schwartzftT, blackstdev, schwartzSDFT


def calc_maturities(months, observation_date):
    # raise Exception(months, observation_date)
    convertMonthsToMaturities = np.vectorize(lambda month: round((month - observation_date).days / 365, 2))
    return convertMonthsToMaturities(months)


def get_seasonal_weightings(months, seasonalParams):
    convertMonthsToSeasonalWeightings = np.vectorize(lambda month: seasonalParams[month.month - 1])
    return convertMonthsToSeasonalWeightings(months)


def convert_to_schwartz_params_dict(x):
    x = x[:len(schwartzparamnames)]
    return dict(zip(schwartzparamnames, x))


def fix_correlations(allData, allOptimizedSchwartzParams):
    num_commodities = len(allData)

    fwds = []
    kappas = []
    sigmaxs = []
    sigmaes = []
    pxes = []
    Ts = []
    x0s = []
    e0s = []
    vxs = []
    ves = []
    covxes = []

    for c in range(num_commodities):
        commodity_data = allData[c]
        observation_date = commodity_data['observation_date']
        months = commodity_data['months']
        futures = commodity_data['futures']
        params = allOptimizedSchwartzParams[c]

        fwd = futures
        fwds.append(fwd)
        maturities = calc_maturities(months, observation_date)
        T = maturities
        Ts.append(T)

        kappa = params['kappa']
        kappas.append(kappa)
        sigmax = params['sigmax']
        sigmaxs.append(sigmax)
        sigmae = params['sigmae']
        sigmaes.append(sigmae)
        pxe = params['pxe']
        pxes.append(pxe)
        x0 = params['x0']
        x0s.append(x0)
        e0 = params['e0']
        e0s.append(e0)
        ve = sigmae**2 * T
        ves.append(ve)
        vx = 0.5 * sigmax**2 * (1 - np.exp(-2 * kappa * T)) / kappa
        vxs.append(vx)
        covxe = (1 - np.exp(-kappa * T)) * pxe * sigmax * sigmae / kappa
        covxes.append(covxe)

    # Compute Correlation matrix from observed forward curves
    all_futures = np.array([c['futures'] for c in allData])
    log_all_futures = np.log(all_futures)
    correlation_matrix = np.corrcoef(log_all_futures)

    import scipy.linalg

#    print "Cholesky:", scipy.linalg.cholesky(correlation_matrix)

    # Find correlation structures needed to keep observed correlation structure
    # during simulation
    num_observations = len(T)
    all_rhos = []
    all_rhos.append([1.0] * num_observations)
    for c in range(1, num_commodities):
        acorxys = []
        for t in range(num_observations):

            atop1 = (1 - np.exp(-(kappas[0] + kappas[c]) * Ts[c][t])) * sigmaxs[0] * sigmaxs[c] / (kappas[0] + kappas[c])

            atop2 = (1 - np.exp(-kappas[0] * Ts[c][t])) * pxes[c] * sigmaxs[0] * sigmaes[c] / kappas[0]
            atop3 = (1.- np.exp(-kappas[c] * Ts[c][t])) * pxes[0] * sigmaxs[c] * sigmaes[0] / kappas[c]

            atop4 = pxes[0] * pxes[c] * sigmaes[0] * sigmaes[c] * Ts[c][t]

            abot1 = np.sqrt(vxs[0][t] + ves[0][t] + 2 * covxes[0][t])
            abot2 = np.sqrt(vxs[c][t] + ves[c][t] + 2 * covxes[c][t])

            acorxys.append((atop1 + atop2 + atop3 + atop4) / (abot1 * abot2))

        rhos = correlation_matrix[c,0] / acorxys
        all_rhos.append(rhos)

    # Clip all the rhos to <= 1.0.
    all_rhos = np.array(all_rhos).clip(max=1.0)

    return correlation_matrix, all_rhos


def simulate_prices(observation_date, months, all_optimized_schwartz_params, all_optimized_seasonal_params, all_rhos, path_count):
    num_commodities = len(all_rhos)
    simulated_prices = []
    num_observations = len(months)
    maturities = calc_maturities(months, observation_date)

    rmatrix0 = np.random.standard_normal((num_observations, path_count))

    for c in range(num_commodities):
        rmatrix = np.zeros((num_observations, path_count))

        for t in range(num_observations):
            rmatrix[t] = all_rhos[c][t] * rmatrix0[t] +\
                         np.sqrt(1 - all_rhos[c][t]**2) * np.random.standard_normal(path_count)

        optimized_schwartz_params = all_optimized_schwartz_params[c]
        optimized_seasonal_params = all_optimized_seasonal_params[c]

        seasonal_weightings = get_seasonal_weightings(months, optimized_seasonal_params)

        (spath, tmpath, tstdpath, tefwd, testdfwd, xpath1, epath1, r1, r2) = simulate_single_schwartz_smith_process(
            optimized_schwartz_params,
            seasonal_weightings,
            maturities,
            rmatrix,
            path_count)

        spath = spath
        mpath = tmpath
        stdpath = tstdpath
        efwd = tefwd
        estdfwd = testdfwd
        epath = epath1
        xpath = xpath1
        lnpath = epath1+xpath1
        r1path = r1
        r2path = r2

        simulated_prices.append((spath, mpath, estdfwd, stdpath))

    return simulated_prices


def simulate_single_schwartz_smith_process(params, seasonal_weightings, maturities, rmatrix, path_count):
    kappa = params['kappa']
    mue = params['mue']
    sigmax = params['sigmax']
    sigmae = params['sigmae']
    lambdae = params['lambdae']
    lambdax = params['lambdax']
    pxe = params['pxe']
    x0 = params['x0']
    e0 = params['e0']

    kappa_times_maturities = kappa * maturities
    exp_minus_kappa_times_maturities = np.exp(-kappa_times_maturities)
    sigmae_squared_times_maturities = sigmae ** 2 * maturities
    sigmax_squared = sigmax ** 2

    explns = exp_minus_kappa_times_maturities * x0 \
             + e0 \
             - (1 - exp_minus_kappa_times_maturities) * lambdax / kappa \
             + (mue - lambdae) * maturities
    varlns = sigmae_squared_times_maturities \
             + (1 - np.exp(-2 * kappa_times_maturities)) * sigmax_squared / (2 * kappa) \
             + 2 * (1 - exp_minus_kappa_times_maturities) * pxe * sigmax * sigmae / kappa
    fwd = seasonal_weightings * np.exp(explns + 0.5 * varlns)

    # Todo: Find out why varlnfwd is same expression as varlns - is that correct?
    # varlnfwd = sigmae_squared_times_maturities \
    #            + (1 - np.exp(-2 * kappa_times_maturities)) * (sigmax_squared) / (2 * kappa) \
    #            + 2 * (1 - exp_minus_kappa_times_maturities) * pxe * sigmax * sigmae / kappa
    varlnfwd = varlns

    stdfwd = seasonal_weightings * np.sqrt((np.exp(varlnfwd) - 1) * np.exp(2 * explns + varlnfwd))

    r1 = rmatrix
    num_maturities = len(maturities)
    r2 = pxe * r1 + np.sqrt((1 - pxe**2)) * np.random.standard_normal((num_maturities, path_count))

    x = np.zeros((num_maturities, path_count))
    e = np.zeros((num_maturities, path_count))
    s = np.zeros((num_maturities, path_count))

    epath = np.zeros((path_count, num_maturities))
    xpath = np.zeros((path_count, num_maturities))
    spath = np.zeros((path_count, num_maturities))
    lnpath = np.zeros((path_count, num_maturities))

    x[0] = x0 - lambdax * maturities[0] - kappa * x0 * maturities[0] + sigmax * np.sqrt(maturities[0]) * r1[0]
    e[0] = e0 + (mue-lambdae) * maturities[0] + sigmae * np.sqrt(maturities[0]) * r2[0]
    s[0] = seasonal_weightings[0] * np.exp(e[0] + x[0])

    for t in range(num_maturities - 1):
        x[t+1] = x[t] - lambdax * (maturities[t+1] - maturities[t]) - kappa * x[t] * (maturities[t+1] - maturities[t]) + sigmax * np.sqrt(maturities[t+1] - maturities[t]) * r1[t+1]
        e[t+1] = e[t] + (mue-lambdae) * (maturities[t+1] - maturities[t]) + (sigmae * np.sqrt(maturities[t+1]-maturities[t])) * r2[t+1]
        s[t+1] = seasonal_weightings[t+1] * np.exp(e[t+1] + x[t+1])

    epath = e.T
    xpath = x.T
    spath = s.T
    lnpath = np.log(s.T)

    mpath = np.mean(spath, axis=0)
    varlnpath = np.var(lnpath, axis=0)
    mlnpath = np.mean(lnpath, axis=0)
    stdpath = np.sqrt((np.exp(varlnpath) - 1) * np.exp(2 * mlnpath + varlnpath))

    return [spath, mpath, stdpath, fwd, stdfwd, xpath, epath, r1, r2]


def plot_simulated_prices(all_data, all_simulated_prices):
    assert len(all_data) == len(all_simulated_prices)
    for i in range(len(all_simulated_prices)):
        commodity_data = all_data[i]
        spath, mpath, estdfwd, stdpath = all_simulated_prices[i]
        name = commodity_data['name']
        futures = commodity_data['futures']

        plt.subplot(len(all_data), 2, 2*i+1)
        plt.title('%s Market Fwd (blue) vs Sim Fwd (red)' % name)
        plt.plot(futures)
        plt.plot(mpath, 'r')

        plt.subplot(len(all_data), 2., 2*i+2)
        plt.title('%s Market Vol (blue) vs Sim Vol (red)' % name)
        plt.plot(estdfwd)
        plt.plot(stdpath, 'r')

    plt.show()
