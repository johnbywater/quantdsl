from __future__ import division
import numpy as np
import os
try:
    import matplotlib.pylab as plt
    print("Imported matplotlib.pylab")
except ImportError:
    pass


"""
Based on matlab code by Prof. Moeti Ncube, "This code replicates the results for Figure 4, of the original
Schwartz-Smith 2000 paper (Short-Term Variations and Long-Term Dynamics in Commodity Prices)".

http://www.mathworks.co.uk/matlabcentral/fileexchange/29745-simulation-of-schwartz-smith-two-factor-model
"""


def main():
    futuresDataPath = 'crudeoil.txt'

    if not os.path.exists(futuresDataPath):
        msg = "File containing observations of futures not found: %s" % futuresDataPath
        raise Exception(msg)
    observations = np.loadtxt(futuresDataPath)

    k = 1.49
    sigmax = 0.286
    lambdax = 0.157
    mu = 0.03
    sigmae = 0.145
    rnmu = 0.0115
    pxe = 0.3
    s = np.array([[0.042], [0.006], [0.003], [0.0000], [0.004]])
    num_maturities = observations.shape[1]

    # Maturity times of futures contracts.
    matur = np.array([1/12, 5/12, 9/12, 13/12, 17/12])


    observation_interval = 7/360

    # Unobserved state variable equation:
    #
    #    x(t) = c + G*x(t-1) + w(t)  Equation (14)
    #

    control_input = np.array([[0], [mu * observation_interval]])

    state_transition_model = np.array([[np.exp(-k * observation_interval), 0], [0, 1]])

    xx = (1.- np.exp(-2 * k * observation_interval)) * sigmax**2 / (2 * k)
    xy = (1 - np.exp(-k * observation_interval)) * pxe * sigmax * sigmae / k
    yx = xy
    yy = sigmae**2 * observation_interval
    process_noise_covariance = np.array([
        [xx, xy],
        [yx, yy]
    ])

    # Observed state variable equation:
    #
    #   y(t) = d(t) + F(t)'*x(t) + v(t) Equation (15)
    #
    # p1, p2, and p3 are components of A(T) Equation 9
    time_variable_constant_A = np.zeros((int(num_maturities),1))
    observation_model = np.zeros((int(num_maturities),2))
    for z in np.arange(num_maturities):
        p1 = (1 - np.exp(-2 * k * matur[z])) * sigmax**2 / (2 * k)
        p2 = sigmae**2 * matur[z]
        p3 = 2 * (1 - np.exp(-k * matur[z])) * pxe * sigmax * sigmae / k
        time_variable_constant_A[z,0] = (rnmu * matur[z]) - (1.-np.exp(-k * matur[z])) * lambdax / k + 0.5 * (p1 + p2 + p3)
        observation_model[z,0] = np.exp(np.dot(-k, matur[z]))
        observation_model[z,1] = 1.

    # Initialise Kalman Filter
    # - measurement errors. Cov[v(t)]=V
    observation_noise_covariance = np.diagflat(s)
    # - state vector m(t)=E[xt;et]
    initial_state_estimate = np.array([[0.019], [2.957]])
    # - covariance matrix C(t)=cov[xt,et]
    initial_estimate_covariance = np.array([[0.1, 0.1], [0.1, 0.1]])

    state_estimate_history = kalman_filter(
        observations,
        observation_model,
        time_variable_constant_A,
        state_transition_model,
        control_input,
        initial_state_estimate,
        initial_estimate_covariance,
        observation_noise_covariance,
        process_noise_covariance
    )

    if plt:
        plot(state_estimate_history)
    else:
        print("Not showing plot...")
    print("Done")


def kalman_filter(observations, observation_model, time_variable_constant_A, state_transition_model, control_input, initial_state_estimate, initial_estimate_covariance, observation_noise_covariance, process_noise_covariance):

    num_observations = observations.shape[0]

    num_maturity_dates = observations.shape[1]

    state_estimate = initial_state_estimate
    estimate_covariance = initial_estimate_covariance

    log_likelihood_history = np.zeros((num_observations, num_maturity_dates, num_maturity_dates))
    m = observation_model.shape[1]

    #% placeholders
    predicted_state_estimate_history = np.zeros((num_observations, m))
    predicted_estimate_covariance_history = np.zeros((num_observations, m * m))
    predicted_observation_history = np.zeros((num_observations, num_maturity_dates))
    innovation_history = np.zeros((num_observations, num_maturity_dates))
    innovation_covariance_history = np.zeros((num_observations, num_maturity_dates * num_maturity_dates))
    state_estimate_history = np.zeros((num_observations, m))
    estimate_covariance_history = np.zeros((num_observations, m * m))

    for i in np.arange(0, num_observations):
        predicted_state_estimate = np.dot(state_transition_model, state_estimate) + control_input
        predicted_estimate_covariance = np.dot(np.dot(state_transition_model, estimate_covariance), state_transition_model.T) + process_noise_covariance
        predicted_observation = np.dot(observation_model, predicted_state_estimate) + time_variable_constant_A
        actual_observation = observations[i,:].reshape(num_maturity_dates, 1)
        innovation = actual_observation - predicted_observation
        innovation_covariance = np.dot(np.dot(observation_model, predicted_estimate_covariance), observation_model.T) + observation_noise_covariance
        inverse_innovation_covariance = np.linalg.inv(innovation_covariance)
        kalman_gain = np.dot(np.dot(predicted_estimate_covariance, observation_model.T), inverse_innovation_covariance)
        state_estimate = predicted_state_estimate + np.dot(kalman_gain, innovation)
        estimate_covariance = predicted_estimate_covariance - np.dot(np.dot(kalman_gain, observation_model), predicted_estimate_covariance)

        # Save results
        predicted_state_estimate_history[i] = predicted_state_estimate.T
        predicted_estimate_covariance_history[i] = predicted_estimate_covariance.flatten().T
        predicted_observation_history[i] = predicted_observation.T
        innovation_history[i] = innovation.T
        innovation_covariance_history[i] = innovation_covariance.flatten().T
        state_estimate_history[i] = state_estimate.T
        estimate_covariance_history[i] = estimate_covariance.flatten().T

        # Likelihood
        det_innovation_covariance = np.linalg.det(innovation_covariance)
        if det_innovation_covariance <= 0:
            det_innovation_covariance = 1e-10

        log_likelihood_history[i] = - (num_maturity_dates / 2) * np.log(2.* np.pi) - 0.5 * np.log(det_innovation_covariance) - np.dot(np.dot(0.5 * innovation.T, inverse_innovation_covariance), innovation)

    return state_estimate_history


def plot(state_estimate_history):
    spot_data_path = 'spotcrudeoil.txt'
    spot_data = np.loadtxt(spot_data_path)
    estimated_spot = np.exp(state_estimate_history.sum(axis=1))
    longrun_mean = np.exp(state_estimate_history[:, 1])
    # print(estimated_spot)
    print("Showing plot...")
    plt.plot(spot_data, '-k', label='Observed Price')
    # plt.plot(longrun_mean, '-b', label='Equilibrium Price')
    plt.plot(estimated_spot, '-r', label='Estimated Price')
    # plt.legend([plot_estimated_spot, plot_longrun_mean, plot_spot_data], ['Estimated Price', 'Equilibrium Price', 'Observed Price'])
    #plt.title('Schwartz-Smith Optimization Results')
    plt.show()

if __name__ == '__main__':
    main()