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


import datetime
from matplotlib import pylab as plt, pylab
import numpy as np
from scipy.optimize import basinhopping
import xlrd

np.seterr(over='raise')

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

num_months = 12

enable_optimize_seasonal_factors = True


def main(alldata):
    allOptimizedParams, allOptimizedSeasonalFactors, allRhos = calibrate(alldata)

    months = [c['months'] for c in alldata]
    observationDate = alldata[0]['observationDate']

    simulatedPrices = simulatePrices(observationDate, months, allOptimizedParams, allOptimizedSeasonalFactors, allRhos, path_count=1000)

    plotSimulatedPrices(alldata, simulatedPrices)


def simulatePrices(observationDate, allMonths, allOptimizedParams, allOptimizedSeasonalFactors, rhoss, path_count):

    simulatedPrices = []

    num_observations = len(allMonths[0])
    rmatrix0 = np.random.standard_normal((num_observations, path_count))

    for i in range(len(rhoss)):
        rmatrix = np.zeros((num_observations, path_count))

        for k in range(num_observations):
            aaa = rhoss[i][k] * rmatrix0[k] + np.sqrt(1 - rhoss[i][k]**2) * np.random.standard_normal(path_count)
            rmatrix[k] = aaa

        months = allMonths[i]

        optimized_params = allOptimizedParams[i]
        optimized_seasonal_factors = allOptimizedSeasonalFactors[i]

        seasonal_weightings = get_seasonal_weightings(months, optimized_seasonal_factors)

        maturities = calc_maturities(months, observationDate)
        [tspath, tmpath, tstdpath, tefwd, testdfwd, xpath1, epath1, r1, r2] = schwartzsmithsim(
            optimized_params,
            seasonal_weightings,
            maturities,
            rmatrix,
            path_count)

        spath = tspath
        mpath = tmpath
        stdpath = tstdpath
        efwd = tefwd
        estdfwd = testdfwd
        epath = epath1
        xpath = xpath1
        lnpath = epath1+xpath1
        r1path = r1
        r2path = r2

        simulatedPrices.append((mpath, estdfwd, stdpath))

    return simulatedPrices


def plotSimulatedPrices(alldata, simulatedPrices):
    iter2 = 1.
    for i in range(len(alldata)):
        mpath, estdfwd, stdpath = simulatedPrices[i]
        commodity_data = alldata[i]
        name = commodity_data['name']
        futures = commodity_data['futures']

        plt.subplot(len(alldata), 2, iter2)
        plt.title('%s Market Fwd (blue) vs Sim Fwd (red)' % name)
        plt.plot(futures)
        plt.plot(mpath, 'r')
        plt.subplot(len(alldata), 2., (iter2+1))
        plt.title('%s Market Vol (blue) vs Sim Vol (red)' % name)
        plt.plot(estdfwd)
        plt.plot(stdpath, 'r')
        iter2 = iter2+2.

    plt.show()
    # Port this last bit later...
    # #%Estimate empirical correlation matrix
    # for i in range(1, num_commodities):
    #     for k in range(num_variables):
    #         c[k] = corr(np.log(spath.cell[0,:,k]()), np.log(spath.cell[i,:,k]()))
    #
    #     empc[i] = np.mean(c)
    #
    # MarketCorrelations = cmatrix[0,1:]
    # SimCorrelations = empc[1:]


def calibrate(alldata):
    results = []

    initial_seasonal_factors = [1.0] * 12

    for commodity_data in alldata:
        observation_date = commodity_data['observationDate']
        months = commodity_data['months']
        futures = commodity_data['futures']
        impvols = commodity_data['impliedAtmVols']


        def objective(x):
            # print "Calling objective function with x:", x
            params = schwartzparamsdictfromlistx(x)
            if len(x) - len(schwartzparamnames) == num_months:
                seasonal_factors = x[-num_months:]
            else:
                seasonal_factors = initial_seasonal_factors
            try:
                score = score_schwartz(observation_date, months, futures, impvols, seasonal_factors, params)
            except FloatingPointError:
                score = np.inf
            if np.isnan(score):
                score = np.inf
            # Constrain seasonal_weightings.
            penalty = ((1 + abs(1 - sum(seasonal_factors) / num_months))) ** 5
            score = score * penalty
            return score

        bounds = [
            (0.0000001, 100),  # kappa
            (-1, 1),  # mue
            (0, 5),  # sigmax
            (0, 5),  # sigmae
            (-5, 5),  # lambdae
            (-5, 5),  # lambdax
            (-5, 5),  # pxe
            (0, 100),  # x0
            (0, 100),  # e0
        ]

        if enable_optimize_seasonal_factors:
            bounds += [(0.2, 5)] * num_months

        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)

        x0 = [0.1 for name in schwartzparamnames]
        if enable_optimize_seasonal_factors:
            x0 += [1] * len(initial_seasonal_factors)

        def print_min(f, x, accept):
            print "Got a min:", f, x, accept

        # Run basin hopping.
        result = basinhopping(objective, x0, T=0.5, stepsize=0.1, niter=10, minimizer_kwargs=minimizer_kwargs)

        results.append(result)
        #print result.fun, schwartzparamsdictfromlistx(result.x), result.x[9:]
    allOptimizedParams = []
    allOptimizedSeasonalFactors = []
    for i, result in enumerate(results):
        commodity_data = alldata[i]
        observation_date = commodity_data['observationDate']
        months = commodity_data['months']
        futures = commodity_data['futures']
        impvols = commodity_data['impliedAtmVols']
        if len(result.x) - len(schwartzparamnames) == num_months:
            seasonal_factors = result.x[-num_months:]
        else:
            seasonal_factors = initial_seasonal_factors
        allOptimizedSeasonalFactors.append(seasonal_factors)
        optimized_params = schwartzparamsdictfromlistx(result.x)
        allOptimizedParams.append(optimized_params)
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

    allRhos = fixCorrelations(alldata, allOptimizedParams)

    return allOptimizedParams, allOptimizedSeasonalFactors, allRhos


def score_schwartz(observation_date, months, futures, impvols, seasonal_factors, params):

    schwartzftT, blackstdev, schwartzSDFT = calc_schwartz(observation_date, months, futures, impvols, seasonal_factors, params)

    # Calc Fwd Diffs
    forwardDiffs = np.abs(futures - schwartzftT)

    # Calc StDev Diffs
    stdevDiffs = np.abs(blackstdev - schwartzSDFT)

    # Calc overall SSE (sum all fwd diffs and all stdev diffs)
    totalDiff = forwardDiffs.sum() + stdevDiffs.sum()

    return totalDiff


def calc_schwartz(observation_date, months, futures, impvols, seasonal_factors, params):

    seasonal_weightings = get_seasonal_weightings(months, seasonal_factors)

    # Calc maturities
    maturities = calc_maturities(months, observation_date)

    # Black Stdev
    blackstdev = np.sqrt(futures * futures * (np.exp(impvols * impvols * maturities) - 1))

    # Calc Schw E[ln(S(T))]
    schwartzES = np.exp(-params['kappa'] * maturities) * params['x0'] + params['e0'] - (1 - np.exp(-params['kappa'] * maturities)) * params['lambdax'] / params['kappa'] + (params['mue'] - params['lambdae']) * maturities

    # Calc Schw V[ln(S(T))]
    schwartzVS = (1 - np.exp(-2 * params['kappa'] * maturities)) * (params['sigmax']**2 / (2 * params['kappa'])) \
             + params['sigmae']**2 * maturities \
             + (2 * (1 - np.exp(-params['kappa'] * maturities)) * params['pxe'] * params['sigmax'] * params['sigmae'] / params['kappa'])

    # Calc Schw F(t,T)
    schwartzftT = np.exp(np.log(seasonal_weightings) + schwartzES + 0.5*schwartzVS)

    # Calc Schw V[ln(F(T))]
    schwartzVFT = (1 - np.exp(-2 * params['kappa'] * maturities)) * params['sigmax'] ** 2 / (2 * params['kappa']) \
            + params['sigmae']**2 * maturities \
            + 2 * (1 - np.exp(-params['kappa'] * maturities)) * params['pxe'] * params['sigmax'] * params['sigmae'] / params['kappa']

    schwartzSDFT = seasonal_weightings * np.sqrt((np.exp(schwartzVFT) - 1) * np.exp( 2 * schwartzES + schwartzVS))

    return schwartzftT, blackstdev, schwartzSDFT


def calc_maturities(months, observation_date):
    return np.array([round((month - observation_date).days / 365, 2) for month in months])


def get_seasonal_weightings(months, seasonal_factors):
    try:
        seasonal_weighting = np.array([seasonal_factors[month.month - 1] for month in months])
    except AttributeError:
        pass
    return seasonal_weighting


def schwartzparamsdictfromlistx(x):
    x = x[:len(schwartzparamnames)]
    return dict(zip(schwartzparamnames, x))


def read_xl_doc():
    # We want the data in the first columns of the various sheets.
    alldata = []
    print_data = []
    with xlrd.open_workbook('SchwartzSmithMultiMarketExampleData.xls') as wb:
        for sheet_name in wb.sheet_names():
            sheet = wb.sheet_by_name(sheet_name)
            month_col_title = sheet.cell_value(1, 0)
            assert month_col_title == 'Month', month_col_title
            from xlrd import xldate_as_tuple
            months = [datetime.datetime(*xldate_as_tuple(c.value, wb.datemode)) for c in sheet.col_slice(0)[2:]]

            fo_col_title = sheet.cell_value(1, 1)
            assert fo_col_title == 'F0', fo_col_title
            futures = [c.value for c in sheet.col_slice(1)[2:]]

            iv_col_title = sheet.cell_value(1, 2)
            assert iv_col_title == "Vol", iv_col_title
            impvols = [c.value for c in sheet.col_slice(2)[2:]]

            def param(params, key, row, col):
                v = sheet.cell_value(row, col)
                if isinstance(key, basestring) and key.lower()[-4:] == 'date':
                    v = datetime.datetime(*xldate_as_tuple(v, wb.datemode))
                params[key] = v

            observation_date = datetime.datetime(*xldate_as_tuple(sheet.cell_value(21, 15), wb.datemode))

            seasonal_factors = [1] * num_months
            for month_int in range(num_months):
                param(seasonal_factors, month_int, month_int+24, 14)

            params = {}
            param(params, 'kappa', 18, 13)
            param(params, 'mue', 18, 14)
            param(params, 'sigmax', 18, 15)
            param(params, 'sigmae', 18, 16)
            param(params, 'lambdae', 18, 17)
            param(params, 'lambdax', 18, 18)
            param(params, 'pxe', 18, 19)
            param(params, 'x0', 18, 20)
            param(params, 'e0', 18, 21)

            alldata.append([observation_date, months, futures, impvols, seasonal_factors, params])
            idata = {
                'observationDate': "%04d-%02d-%02d" % (observation_date.year, observation_date.month, observation_date.day),
                'months': ["%04d-%02d-%02d" % (m.year, m.month, m.day) for m in months],
                'futures': [i for i in futures],
                'impvols': [i for i in impvols]
            }
            print_data.append(idata)

    import json
    print "import datetime"
    print "from numpy import array"
    print
    print json.dumps(print_data, indent=4)

    return alldata


def fixCorrelations(alldata, all_optimized_params):
    num_commodities = len(alldata)

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

    for k in range(num_commodities):
        commodity_data = alldata[k]
        observation_date = commodity_data['observationDate']
        months = commodity_data['months']
        futures = commodity_data['futures']
        params = all_optimized_params[k]

        maturities = calc_maturities(months, observation_date)
        fwd = futures
        fwds.append(fwd)
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
    all_futures = np.array([c['futures'] for c in alldata])
    log_all_futures = np.log(all_futures)
    cmatrix = np.corrcoef(log_all_futures)

    # Find correlation structures needed to keep observed correlation structure
    # during simulation
    numObservations = len(T)
    allRhos = []
    allRhos.append([1.0] * numObservations)
    for k in range(1, num_commodities):
        acorxys = []
        for i in range(numObservations):

            atop1 = (1 - np.exp(-(kappas[0] + kappas[k]) * Ts[k][i])) * sigmaxs[0] * sigmaxs[k] / (kappas[0] + kappas[k])

            atop2 = (1 - np.exp(-kappas[0] * Ts[k][i])) * pxes[k] * sigmaxs[0] * sigmaes[k] / kappas[0]
            atop3 = (1.- np.exp(-kappas[k] * Ts[k][i])) * pxes[0] * sigmaxs[k] * sigmaes[0] / kappas[k]

            atop4 = pxes[0] * pxes[k] * sigmaes[0] * sigmaes[k] * Ts[k][i]

            abot1 = np.sqrt(vxs[0][i] + ves[0][i] + 2 * covxes[0][i])
            abot2 = np.sqrt(vxs[k][i] + ves[k][i] + 2 * covxes[k][i])

            acorxys.append((atop1 + atop2 + atop3 + atop4) / (abot1 * abot2))

        rhos = cmatrix[k,0] / acorxys
        allRhos.append(rhos)

    # Clip all the rhos to <= 1.0.
    allRhos = np.array(allRhos).clip(max=1.0)

    return allRhos


def schwartzsmithsim(params, seasonal_weightings, maturities, rmatrix, path_count):

    T = maturities
    kappa = params['kappa']
    mue = params['mue']
    sigmax = params['sigmax']
    sigmae = params['sigmae']
    lambdae = params['lambdae']
    lambdax = params['lambdax']
    pxe = params['pxe']
    x0 = params['x0']
    e0 = params['e0']

    f = seasonal_weightings


    fwd = []
    stdfwd = []
    num_obs = len(T)

    for i in range(num_obs):
        explns = np.exp(-kappa * T[i]) * x0 \
                 + e0 \
                 - (1 - np.exp(-kappa * T[i])) * lambdax / kappa \
                 + (mue - lambdae) * T[i]
        varlns = ((1 - np.exp(-2 * kappa * T[i])) * sigmax**2 / (2 * kappa)) \
                 + sigmae**2 * T[i] \
                 + 2 * (1 - np.exp(-kappa * T[i])) * pxe * sigmax * sigmae / kappa
        fwd.append(f[i] * np.exp(explns + (5 * varlns)))

        varlnfwd = T[i] * sigmae**2 \
                   + (1 - np.exp(-2 * kappa * T[i])) * (sigmax**2) / (2 * kappa) \
                   + 2 * (1 - np.exp(-kappa * T[i])) * pxe * sigmax * sigmae / kappa

        stdfwd.append(f[i] * np.sqrt((np.exp(varlnfwd) - 1) * np.exp(2 * explns + varlnfwd)))

    m = num_obs
    r1 = rmatrix
    r2 = pxe * r1 + np.sqrt((1 - pxe**2)) * np.random.standard_normal((m, path_count))

    x = np.zeros(m)
    e = np.zeros(m)
    s = np.zeros(m)

    epath = np.zeros((path_count, num_obs))
    xpath = np.zeros((path_count, num_obs))
    spath = np.zeros((path_count, num_obs))
    lnpath = np.zeros((path_count, num_obs))

    for i in range(path_count):
        x[0] = x0 - lambdax * T[0] - kappa * x0 * T[0] + sigmax * np.sqrt(T[0]) * r1[0,i]
        e[0] = e0 + (mue-lambdae) * T[0] + sigmae * np.sqrt(T[0]) * r2[0,i]
        s[0] = f[0] * np.exp(e[0] + x[0])
        # for t=1:m-1
        for t in range(m - 1):
            x[t+1] = x[t] - lambdax * (T[t+1] - T[t]) - kappa * x[t] * (T[t+1] - T[t]) + sigmax * np.sqrt(T[t+1] - T[t]) * r1[t+1,i]
            e[t+1] = e[t] + (mue-lambdae) * (T[t+1] - T[t]) + (sigmae * np.sqrt(T[t+1]-T[t])) * r2[t+1,i]
            s[t+1] = f[t+1] * np.exp(e[t+1] + x[t+1])

        epath[i] = e
        xpath[i] = x
        spath[i] = s
        lnpath[i] = np.log(s)

    mpath = np.mean(spath, axis=0)
    varlnpath = np.var(lnpath, axis=0)
    mlnpath = np.mean(lnpath, axis=0)
    stdpath = np.sqrt((np.exp(varlnpath) - 1) * np.exp(2 * mlnpath + varlnpath))
    return [spath, mpath, stdpath, fwd, stdfwd, xpath, epath, r1, r2]

if __name__ == '__main__':
    main()

