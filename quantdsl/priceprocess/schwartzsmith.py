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

NUM_MONTHS_IN_YEAR = 12

enableOptimizeSeasonalParams = True


def calibrate(allData, niter=100, pathCount=1000):
    results = []

    numCommodities = len(allData)
    initialSeasonalParams = [1.0] * 12

    observationDate = allData[0]['observationDate']
    months = allData[0]['months']
    maturities = calcMaturities(months, observationDate)

    for commodityData in allData:
        commodityName = commodityData['name']
        futures = commodityData['futures']
        impvols = commodityData['impliedAtmVols']

        def objective(x):
            # print "Calling objective function with x:", x
            schwartzParams = convertToSchwartzParamsDict(x)
            if enableOptimizeSeasonalParams:
                seasonalParams = x[-NUM_MONTHS_IN_YEAR:]
            else:
                seasonalParams = initialSeasonalParams
            try:
                score = scoreSchwartz(months, maturities, futures, impvols, seasonalParams, schwartzParams)
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

        if enableOptimizeSeasonalParams:
            bounds += [(0.2, 5)] * NUM_MONTHS_IN_YEAR

        minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)

        x0 = [0.1] * len(schwartzparamnames)
        if enableOptimizeSeasonalParams:
            x0 += [1] * len(initialSeasonalParams)

        # Run the "basin hopping" routine.
        result = basinhopping(objective, x0, T=0.5, stepsize=0.1, niter=niter, minimizer_kwargs=minimizer_kwargs)

        results.append(result)
        print commodityName, result.fun, convertToSchwartzParamsDict(result.x), result.x[9:]
    allOptimizedSchwartzParams = []
    allOptimizedSeasonalParams = []
    for c in range(numCommodities):
        result = results[c]
        commodityData = allData[c]
        observationDate = commodityData['observationDate']
        months = commodityData['months']
        futures = commodityData['futures']
        impvols = commodityData['impliedAtmVols']
        if len(result.x) - len(schwartzparamnames) == NUM_MONTHS_IN_YEAR:
            seasonalParams = result.x[-NUM_MONTHS_IN_YEAR:]
        else:
            seasonalParams = initialSeasonalParams
        allOptimizedSeasonalParams.append(seasonalParams)
        optimizedSchwartzParams = convertToSchwartzParamsDict(result.x)
        allOptimizedSchwartzParams.append(optimizedSchwartzParams)
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

    correlationMatrix, allRhos = fixCorrelations(allData, allOptimizedSchwartzParams)

    simulatedPrices = simulatePrices(observationDate, months, allOptimizedSchwartzParams,
                                         allOptimizedSeasonalParams, allRhos, pathCount=pathCount)

    # Estimate empirical correlation matrix
    spath0 = simulatedPrices[0][0]
    spath0T = spath0.T
    simCorrelations = []
    for i in range(1, numCommodities):
        spath = simulatedPrices[i][0]
        spathT = spath.T
        coefs = []
        numObservations = len(spathT)
        for t in range(numObservations):
            coef = np.corrcoef(spath0T[t], spathT[t])[0][1]
            coefs.append(coef)

        simCorrelations.append(np.mean(coefs))
    #
    # MarketCorrelations = cmatrix[0,1:]
    # SimCorrelations = empc[1:]


    return allOptimizedSchwartzParams, allOptimizedSeasonalParams, allRhos, correlationMatrix, simCorrelations, simulatedPrices


def scoreSchwartz(months, maturities, forwards, impvols, seasonalParams, schwartzParams):
    schwartzftT, blackstdev, schwartzSDFT = calcSchwartz(months, maturities, forwards, impvols, seasonalParams, schwartzParams)
    fwdErr = (forwards - schwartzftT)**2
    stdevErr = (blackstdev - schwartzSDFT)**2
    return fwdErr.sum() + stdevErr.sum()


def calcSchwartz(months, maturities, forwards, impvols, seasonalParams, schwartzParams):

    seasonalWeightings = getSeasonalWeightings(months, seasonalParams)

    kappa = schwartzParams['kappa']
    x0 = schwartzParams['x0']
    e0 = schwartzParams['e0']
    lambdax = schwartzParams['lambdax']
    mue = schwartzParams['mue']
    lambdae = schwartzParams['lambdae']
    sigmax = schwartzParams['sigmax']
    sigmae = schwartzParams['sigmae']
    pxe = schwartzParams['pxe']
    kappaTimesMaturities = kappa * maturities
    expMinusKappaTimesMaturities = np.exp(-kappaTimesMaturities)
    sigmaeSquaredTimesMaturities = sigmae ** 2 * maturities

    # Black Stdev
    blackstdev = np.sqrt(forwards * forwards * (np.exp(impvols * impvols * maturities) - 1))

    # Calc Schw E[ln(S(T))]
    schwartzES = expMinusKappaTimesMaturities * x0 + e0 - (1 - expMinusKappaTimesMaturities) * lambdax / kappa + (mue - lambdae) * maturities

    # Calc Schw V[ln(S(T))]
    schwartzVS = (1 - np.exp(-2 * kappaTimesMaturities)) * (sigmax **2 / (2 * kappa)) \
                 + sigmaeSquaredTimesMaturities \
                 + (2 * (1 - expMinusKappaTimesMaturities) * pxe * sigmax * sigmae / kappa)

    # Calc Schw F(t,T)
    schwartzftT = seasonalWeightings * np.exp(schwartzES + 0.5 * schwartzVS)

    # Calc Schw V[ln(F(T))]
    # Todo: Find out why schwartzVFT is same expression as schwartzVS - is that correct?
    # schwartzVFT = (1 - np.exp(-2 * kappaTimesMaturities)) * sigmax ** 2 / (2 * kappa) \
    #               + sigmaeSquaredTimesMaturities \
    #               + 2 * (1 - expMinusKappaTimesMaturities) * pxe * sigmax * sigmae / kappa
    schwartzVFT = schwartzVS

    schwartzSDFT = seasonalWeightings * np.sqrt((np.exp(schwartzVFT) - 1) * np.exp(2 * schwartzES + schwartzVS))

    return schwartzftT, blackstdev, schwartzSDFT


def calcMaturities(months, observationDate):
    convertMonthsToMaturities = np.vectorize(lambda month: round((month - observationDate).days / 365, 2))
    return convertMonthsToMaturities(months)


def getSeasonalWeightings(months, seasonalParams):
    convertMonthsToSeasonalWeightings = np.vectorize(lambda month: seasonalParams[month.month - 1])
    return convertMonthsToSeasonalWeightings(months)


def convertToSchwartzParamsDict(x):
    x = x[:len(schwartzparamnames)]
    return dict(zip(schwartzparamnames, x))


def fixCorrelations(allData, allOptimizedSchwartzParams):
    numCommodities = len(allData)

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

    for c in range(numCommodities):
        commodityData = allData[c]
        observationDate = commodityData['observationDate']
        months = commodityData['months']
        futures = commodityData['futures']
        params = allOptimizedSchwartzParams[c]

        fwd = futures
        fwds.append(fwd)
        maturities = calcMaturities(months, observationDate)
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
    allFutures = np.array([c['futures'] for c in allData])
    logAllFutures = np.log(allFutures)
    correlationMatrix = np.corrcoef(logAllFutures)

    import scipy.linalg

#    print "Cholesky:", scipy.linalg.cholesky(correlationMatrix)

    # Find correlation structures needed to keep observed correlation structure
    # during simulation
    numObservations = len(T)
    allRhos = []
    allRhos.append([1.0] * numObservations)
    for c in range(1, numCommodities):
        acorxys = []
        for t in range(numObservations):

            atop1 = (1 - np.exp(-(kappas[0] + kappas[c]) * Ts[c][t])) * sigmaxs[0] * sigmaxs[c] / (kappas[0] + kappas[c])

            atop2 = (1 - np.exp(-kappas[0] * Ts[c][t])) * pxes[c] * sigmaxs[0] * sigmaes[c] / kappas[0]
            atop3 = (1.- np.exp(-kappas[c] * Ts[c][t])) * pxes[0] * sigmaxs[c] * sigmaes[0] / kappas[c]

            atop4 = pxes[0] * pxes[c] * sigmaes[0] * sigmaes[c] * Ts[c][t]

            abot1 = np.sqrt(vxs[0][t] + ves[0][t] + 2 * covxes[0][t])
            abot2 = np.sqrt(vxs[c][t] + ves[c][t] + 2 * covxes[c][t])

            acorxys.append((atop1 + atop2 + atop3 + atop4) / (abot1 * abot2))

        rhos = correlationMatrix[c,0] / acorxys
        allRhos.append(rhos)

    # Clip all the rhos to <= 1.0.
    allRhos = np.array(allRhos).clip(max=1.0)

    return correlationMatrix, allRhos


def simulatePrices(observationDate, months, allOptimizedSchwartzParams, allOptimizedSeasonalParams, allRhos, pathCount):
    numCommodities = len(allRhos)
    simulatedPrices = []
    numObservations = len(months)
    maturities = calcMaturities(months, observationDate)

    rmatrix0 = np.random.standard_normal((numObservations, pathCount))

    for c in range(numCommodities):
        rmatrix = np.zeros((numObservations, pathCount))

        for t in range(numObservations):
            rmatrix[t] = allRhos[c][t] * rmatrix0[t] +\
                         np.sqrt(1 - allRhos[c][t]**2) * np.random.standard_normal(pathCount)

        optimizedSchwartzParams = allOptimizedSchwartzParams[c]
        optimizedSeasonalParams = allOptimizedSeasonalParams[c]

        seasonalWeightings = getSeasonalWeightings(months, optimizedSeasonalParams)

        (spath, tmpath, tstdpath, tefwd, testdfwd, xpath1, epath1, r1, r2) = simulateSingleSchwartzSmithProcess(
            optimizedSchwartzParams,
            seasonalWeightings,
            maturities,
            rmatrix,
            pathCount)

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

        simulatedPrices.append((spath, mpath, estdfwd, stdpath))

    return simulatedPrices


def simulateSingleSchwartzSmithProcess(params, seasonalWeightings, maturities, rmatrix, pathCount):
    kappa = params['kappa']
    mue = params['mue']
    sigmax = params['sigmax']
    sigmae = params['sigmae']
    lambdae = params['lambdae']
    lambdax = params['lambdax']
    pxe = params['pxe']
    x0 = params['x0']
    e0 = params['e0']

    kappaTimesMaturities = kappa * maturities
    expMinusKappaTimesMaturities = np.exp(-kappaTimesMaturities)
    sigmaeSquaredTimesMaturities = sigmae ** 2 * maturities
    sigmaxSquared = sigmax ** 2

    explns = expMinusKappaTimesMaturities * x0 \
             + e0 \
             - (1 - expMinusKappaTimesMaturities) * lambdax / kappa \
             + (mue - lambdae) * maturities
    varlns = sigmaeSquaredTimesMaturities \
             + (1 - np.exp(-2 * kappaTimesMaturities)) * sigmaxSquared / (2 * kappa) \
             + 2 * (1 - expMinusKappaTimesMaturities) * pxe * sigmax * sigmae / kappa
    fwd = seasonalWeightings * np.exp(explns + 0.5 * varlns)

    # Todo: Find out why varlnfwd is same expression as varlns - is that correct?
    # varlnfwd = sigmaeSquaredTimesMaturities \
    #            + (1 - np.exp(-2 * kappaTimesMaturities)) * (sigmaxSquared) / (2 * kappa) \
    #            + 2 * (1 - expMinusKappaTimesMaturities) * pxe * sigmax * sigmae / kappa
    varlnfwd = varlns

    stdfwd = seasonalWeightings * np.sqrt((np.exp(varlnfwd) - 1) * np.exp(2 * explns + varlnfwd))

    r1 = rmatrix
    numMaturities = len(maturities)
    r2 = pxe * r1 + np.sqrt((1 - pxe**2)) * np.random.standard_normal((numMaturities, pathCount))

    x = np.zeros((numMaturities, pathCount))
    e = np.zeros((numMaturities, pathCount))
    s = np.zeros((numMaturities, pathCount))

    epath = np.zeros((pathCount, numMaturities))
    xpath = np.zeros((pathCount, numMaturities))
    spath = np.zeros((pathCount, numMaturities))
    lnpath = np.zeros((pathCount, numMaturities))

    x[0] = x0 - lambdax * maturities[0] - kappa * x0 * maturities[0] + sigmax * np.sqrt(maturities[0]) * r1[0]
    e[0] = e0 + (mue-lambdae) * maturities[0] + sigmae * np.sqrt(maturities[0]) * r2[0]
    s[0] = seasonalWeightings[0] * np.exp(e[0] + x[0])

    for t in range(numMaturities - 1):
        x[t+1] = x[t] - lambdax * (maturities[t+1] - maturities[t]) - kappa * x[t] * (maturities[t+1] - maturities[t]) + sigmax * np.sqrt(maturities[t+1] - maturities[t]) * r1[t+1]
        e[t+1] = e[t] + (mue-lambdae) * (maturities[t+1] - maturities[t]) + (sigmae * np.sqrt(maturities[t+1]-maturities[t])) * r2[t+1]
        s[t+1] = seasonalWeightings[t+1] * np.exp(e[t+1] + x[t+1])

    epath = e.T
    xpath = x.T
    spath = s.T
    lnpath = np.log(s.T)

    mpath = np.mean(spath, axis=0)
    varlnpath = np.var(lnpath, axis=0)
    mlnpath = np.mean(lnpath, axis=0)
    stdpath = np.sqrt((np.exp(varlnpath) - 1) * np.exp(2 * mlnpath + varlnpath))

    return [spath, mpath, stdpath, fwd, stdfwd, xpath, epath, r1, r2]


def plotSimulatedPrices(allData, allSimulatedPrices):
    assert len(allData) == len(allSimulatedPrices)
    for i in range(len(allSimulatedPrices)):
        commodityData = allData[i]
        spath, mpath, estdfwd, stdpath = allSimulatedPrices[i]
        name = commodityData['name']
        futures = commodityData['futures']

        plt.subplot(len(allData), 2, 2*i+1)
        plt.title('%s Market Fwd (blue) vs Sim Fwd (red)' % name)
        plt.plot(futures)
        plt.plot(mpath, 'r')

        plt.subplot(len(allData), 2., 2*i+2)
        plt.title('%s Market Vol (blue) vs Sim Vol (red)' % name)
        plt.plot(estdfwd)
        plt.plot(stdpath, 'r')

    plt.show()
