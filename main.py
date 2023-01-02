"""
Replicating 'Robust Calibration'

Here we'll replicate the algorithm described in (1) for calibrating sing-qubit gates

We'll do this using the 2-gate universal single-qubit gate-set of
Z_{pi/2}
X_{pi/2}
"""
import numpy as np
import matplotlib.pyplot as plt
import cirq
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.signal import find_peaks


class XrotGate(cirq.Gate):
    def __init__(self, phi, epsilon, theta):
        #super(MyGate, self)

        self.phi = phi
        self.epsilon = epsilon
        self.theta = theta

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        pauli_x = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])

        pauli_z = np.array([
            [1.0, 0.0],
            [0.0, -1.0]
        ])

        ident_coeff = np.cos(self.phi * (1 + self.epsilon) / 2)
        xz_coeff = -1j * np.sin(self.phi * (1 + self.epsilon) / 2)

        return np.identity(2) * ident_coeff + xz_coeff * (np.cos(theta) * pauli_x + np.sin(theta) * pauli_z)

    def _circuit_diagram_info_(self, args):
        return f"Xrot(phi = {self.phi}, epsilon = {self.epsilon}, theta = {self.theta})"


class robust_phase_estimation:
    def __init__(self, alpha, epsilon, theta):
        """
        Given the angle errors of alpha epsilon and theta, of
        Z(pi/2) and X(pi/4), identify these values using robust calibration.

        :param alpha:
        :param epsilon:
        :param theta:
        """

        self.alpha = alpha
        self.epsilon = epsilon
        self.theta = theta

        # define the rotations we'll be using

        # Z(pi/2) is directed along z, and has error propto (1+alpha)
        self.Zrot = cirq.rz(np.pi * (1 + alpha) / 2)

        # X(pi/4) is directed along the XZ plane, with an angle of theta from the x axis.
        # It shifts by an angle of (1 + epsilon) / 4 about this direction
        self.Xrot = XrotGate(np.pi / 4, epsilon, theta)

    @staticmethod
    def restrict_angle(theta):
        """
        map theta to an equivalent angle from -pi to +pi

        :param theta:
        :return:
        """
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def angle_in_range(self, theta, phi, delta_lower, delta_upper):
        """
        return True iff theta is in the range [phi-delta_lower, phi+delta_plus]

        :param theta1:
        :param theta2:
        :param delta:

        :return: Bool, True if the angles are within the accepted range
        """

        difference = self.restrict_angle(theta - phi)

        if difference >= 0:
            return difference < delta_upper

        if difference < 0:
            return abs(difference) < delta_lower

    @staticmethod
    def plot_angle_region(phi, delta_lower, delta_upper, radius, ax):
        """
        plot a circle with the region shaded in

        :param theta:
        :param delta:
        :return:
        """
        assert delta_lower > 0
        assert delta_upper > 0

        ax.fill_between(np.linspace(phi - delta_lower, phi + delta_upper, 100), 0, radius, alpha=.3)

    def z_zero_circuit(self, k, qubit):
        """
        Build a Z 0 circuit for its corresponding experiment

        :param k:
        :param qubit:
        :return:
        """

        k_sequence = [self.Zrot(qubit) for _ in range(k)]

        # 0 measurements
        circuit_list = [cirq.H(qubit)] + k_sequence + [cirq.H(qubit), cirq.measure(qubit)]

        return cirq.Circuit(circuit_list)

    def z_plus_circuit(self, k, qubit):
        """
        Build a Z + circuit for its corresponding experiment

        :param k:
        :param qubit:
        :return:
        """

        k_sequence = [self.Zrot(qubit) for _ in range(k)]

        # 0 measurements
        circuit_list = [cirq.H(qubit), cirq.S(qubit)] + k_sequence + [cirq.H(qubit), cirq.measure(qubit)]

        return cirq.Circuit(circuit_list)

    @staticmethod
    def measurement_function(j, K, delta = 0):
        """
        The function which tells us how many measurements to perform

        :param j:
        :param M_type:
        :return:
        """
        assert delta>=0

        if delta == 0:
            return 3 * (K - j) + 1

        else:
            Mj = 3 * (K - j) + 1

            numerator = np.log(
                (1 - np.sqrt(8) * delta)**(Mj) / 2
            )

            denominator = np.log(
                (1 - 0.5*(1 - np.sqrt(8) * delta) ** 2 )
            )

            integer_output = int(round(numerator / denominator))

            return integer_output

    def phase_estimation(self, zero_measurements, plus_measurements, ks, visualize = False):
        """
        Enact the phase estimation routine

        :param zero_measurements:
        :param plus_measurements:
        :param measurement_nums:
        :return:
        """

        Knum = len(ks)

        #the current estimate, its lower and upper bounds.
        estimate = [None, None, None]

        if visualize:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        for j in range(Knum):
            """
            get the estimate of A up to a factor of 2*pi*n/k
            """
            aplus = plus_measurements[j]
            azero = zero_measurements[j]
            kA = np.arctan2(aplus - 1 / 2, azero - 1 / 2)

            if estimate[0] is None:
                estimate[0] = self.restrict_angle(kA)
                estimate[1] = self.restrict_angle(np.pi / 2)
                estimate[2] = self.restrict_angle(np.pi / 2)

                current_A = estimate[0]

            else:
                # enumerate possible values for A based on this measurement
                all_A_values = [self.restrict_angle(kA / ks[j] + 2 * np.pi * n / (2 ** j)) for n in range(2 ** j)]

                if visualize:
                    for a in all_A_values:
                        ax.scatter(a, 1 + j / 10, color='b')
                #print("all a values", all_A_values)

                current_A = None
                for theta in all_A_values:
                    if self.angle_in_range(theta, *estimate):
                        current_A = theta
                #print(current_A)

                if visualize:
                    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                    self.plot_angle_region(estimate[0], estimate[1], estimate[2], 1, ax)
                    for theta in all_A_values:
                        ax.scatter([theta], [1], marker="x", s=100)
                    plt.show()

                    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
                    self.plot_angle_region(estimate[0], estimate[1], estimate[2], 1, ax1)

                #print("bounds before: ", estimate)
                # update the bounds
                estimate[1] = estimate[2] = np.pi / 2 ** (j + 1)

                # now update the estimate
                estimate[0] = current_A

        #return the final estimate
        return estimate[0]

    def alpha_calibration(self, simulator=cirq.Simulator(), Knum=5, visualize = False, bootstrap = False, delta_zero = 0, delta_plus = 0, delta = 0):
        """


        :param simulator:
        :param Knum:
        :param visualize:
        :param bootstrap: Bool, Whether to use an empirical bootstrap to estimate the variance of the estimator
        :return:
        """
        if delta==0:
            delta = max(abs(delta_zero), abs(delta_plus))
        #simulator = cirq.Simulator()

        """
        Z_{pi/2}
        
        now simulate |<+|Z_{pi/2}(alpha)^k|+>|^2 
        Do this by applying Z_{pi/2}(alpha)^k to |+>, and then measuring in the x basis
        Run this M times 
        """

        ks = [2 ** j for j in range(Knum)]
        #print("ks: ", ks)

        Ms = [self.measurement_function(j, Knum, delta = delta) for j in range(1, Knum+1)]
        #Ms = [10**3 for j in range(Knum)]

        # now set up the circuits and measure

        alpha_qubit = cirq.LineQubit.range(1)[0]

        zero_measurements = []
        plus_measurements = []

        #record the number of times Z is applied
        application_number = 0

        for j, k in enumerate(ks):
            # 0 measurements
            alpha_circuit_0 = self.z_zero_circuit(k, alpha_qubit)

            result = simulator.run(alpha_circuit_0, repetitions=Ms[j])

            application_number += Ms[j]*k

            #print(1 - result.data['0'].mean(), "correct", (1 + np.cos(-k * np.pi * (1 + self.alpha) / 2)) / 2)

            zero_measurements.append(1 - result.data['0'].mean())

            # plus measurements

            alpha_circuit_1 = self.z_plus_circuit(k, alpha_qubit)

            result = simulator.run(alpha_circuit_1, repetitions=Ms[j])

            application_number += Ms[j] * k

            #print(1 - result.data['0'].mean(), "correct", (1 + np.sin(-k * np.pi * (1 + self.alpha) / 2)) / 2)
            #print("k=", k)

            plus_measurements.append(1 - result.data['0'].mean())

        """
        Now lets modify the observed probailities using our deltas
        If delta is nonzero, resample with a shifted probability
        """
        if delta>0 and (delta_zero>0 or delta_plus>0):
            for i in range(Knum):
                p_zero = zero_measurements[i]
                p_plus = plus_measurements[i]

                p_zero += delta_zero
                p_plus += delta_plus

                p_zero = min(1-.001, p_zero)
                p_plus = min(1-.001, p_plus)

                p_zero = max(.001, p_zero)
                p_plus = max(.001, p_plus)

                #resample the probs to generate sample noise
                p_zero = np.random.binomial(Ms[i], p_zero)/Ms[i]
                p_plus = np.random.binomial(Ms[i], p_plus)/Ms[i]

                zero_measurements[i] = p_zero
                plus_measurements[i] = p_plus



        """
        next, we process these measurements to extract the value of alpha
        
        We do this iteratively
        """
        # Aj, delta_upper, delta_lower
        estimate = [None, None, None]

        if visualize:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        #now we allow for the possibility of bootstrapping
        if bootstrap:
            """
            If this is true, then resample according to the empirical probabilities in order to 
            get an estimate of the variance. 
            """

            reps = 10**3
            bootstrapped_alphas = []

            for _ in range(reps):
                """
                Resample according to the observed probabilities
                """

                new_zero_probs = []
                new_plus_probs = []

                for i, M in enumerate(Ms):
                    new_zero_prob = np.random.binomial(M, zero_measurements[i]) / M
                    new_plus_prob = np.random.binomial(M, plus_measurements[i]) / M

                    new_zero_probs.append(new_zero_prob)
                    new_plus_probs.append(new_plus_prob)

                """
                Now use these probabilities to do phase estimation
                """
                sampled_A = self.phase_estimation(new_zero_probs, new_plus_probs, ks)

                #sampled_alpha = -(1 + 2 * sampled_A / np.pi)

                #bootstrapped_alphas.append(sampled_alpha)

                bootstrapped_alphas.append(sampled_A)

            alpha = np.mean(bootstrapped_alphas)

            alpha_variance = np.var(bootstrapped_alphas, ddof=1)

            application_number /= 10**4

        else:
            """
            Evaluate a single repetition
            """
            A = self.phase_estimation(zero_measurements, plus_measurements, ks)

            #now extract alpha from A

            alpha = -(1 + 2 * A / np.pi)

            alpha_variance = 0

        print(alpha, "correct: ", -(1 + self.alpha) * np.pi / 2)

        return alpha, self.alpha, alpha_variance, application_number

def extract_heisenberg_scaling(A, delta_zero, delta_plus, delta, simulator=cirq.Simulator(), visualize=False):
    """
    Run some repetitions and extract the variance as the number of
    applications increases

    :param A:
    :return: the slope of 1/sigma vs T, the final std
    """

    reps = 100

    alpha = -(1 + 2 * A / np.pi)

    Knums = [3, 4, 5, 6, 7, 8, 9, 10]

    applications = []
    stds = []
    errors = []

    for knum in Knums:
        phase_est = robust_phase_estimation(alpha, 0, 0)

        estimates = []

        variances = []

        for rep in range(reps):
            estimate, correct, variance, application_number = phase_est.alpha_calibration(Knum=knum,
                                                                                          bootstrap=False,
                                                                                          delta_zero = delta_zero,
                                                                                          delta_plus = delta_plus,
                                                                                          delta = .1,
                                                                                          simulator=simulator)

            #convert from alpha to A and add to the data
            estimates.append(-(1 + estimate) * np.pi / 2)

            variances.append(variance)

        applications.append(application_number)
        stds.append(np.sqrt(np.mean(variances)))
        #stds.append(np.sqrt(np.var(estimates, ddof=1)))

        error = np.mean(estimate) - A

        errors.append(np.abs(error))

    """
    p = np.polyfit(Knums, np.log(stds), 1)

    plt.plot(Knums, np.log(stds))
    plt.plot(np.array(Knums), np.poly1d(p)(np.array(Knums)))
    plt.title(f"Log STD (slope = {p[0]})")
    plt.show()
    """

    p = np.polyfit(applications, 1/np.array(stds), 1)

    if visualize:
        plt.plot(applications, 1 / np.array(stds))
        plt.xlabel("T")
        plt.ylabel("$1/\sigma$")
        plt.title(f"T vs $1/\sigma$ - Slope = {p[0]}")
        plt.scatter(applications, 1 / np.array(stds))
        plt.show()

    return p[0], stds[-1]



def sweep_variances(delta_zero=0, delta_plus=0, visualize = True):
    """
    Here we'll sweep over all A values and record the variance, and the
    slope of the T vs 1/std plot

    :return: As, slopes, stds
    """

    As = np.linspace(-np.pi, np.pi, 20)
    slopes = []
    stds = []

    for A in As:
        s, std = extract_heisenberg_scaling(A, delta_zero, delta_plus, visualize=False)

        slopes.append(s)
        stds.append(std)

    slopes = np.array(slopes)
    stds = np.array(stds)

    if visualize:
        plt.plot(As, slopes)
        plt.xlabel("A values")
        plt.ylabel("T vs $1/\sigma$ slopes")
        plt.show()

        plt.plot(As, stds)
        plt.xlabel("A values")
        plt.ylabel("standard deviations")
        plt.show()

    return As, slopes, stds




class genericFsim(cirq.TwoQubitGate):
    def __init__(self, theta, phi, gamma, chi, zeta):
        self.theta = theta
        self.phi = phi
        self.gamma = gamma
        self.chi = chi
        self.zeta = zeta

    def _unitary_(self):
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.exp(-1j * (self.gamma + self.zeta)) * np.cos(self.theta),
             -1j * np.exp(-1j * (self.gamma - self.chi)) * np.sin(self.theta), 0.0],
            [0.0, -1j * np.exp(-1j * (self.gamma + self.chi)) * np.sin(self.theta),
             np.exp(-1j * (self.gamma - self.zeta)) * np.cos(self.theta), 0.0],
            [0.0, 0.0, 0.0, np.exp(-1j * (2 * self.gamma + self.phi))]
        ])

    def _circuit_diagram_info_(self, args):
        return "FSIM UP", "FSIM DOWN"


class TwoGateCalibration:
    """
    Here we'll apply both the phase method and the population method to estimate parameters

    :param alpha:
    :param gamma:
    :param chi:
    :param zeta:
    :return:
    """

    def __init__(self, theta, phi, gamma, chi, zeta):
        """

        :param theta:
        :param phi:
        :param gamma:
        :param chi:
        :param zeta:
        """

        # define the qubits we'll be using
        self.qubits = cirq.LineQubit.range(2)

        # define the relevant gates
        self.fsim = genericFsim(theta, phi, gamma, chi, zeta)

        # define the parameters
        self.theta = theta
        self.phi = phi
        self.gamma = gamma
        self.chi = chi
        self.zeta = zeta

    def build_phase_circuit_x(self, depth, qubits):
        """

        :param depth:
        :param qubits:
        :return:
        """
        assert len(qubits) == 2

        circuit_elements = cirq.Circuit([cirq.H(qubits[0])])

        for d in range(depth):
            circuit_elements.append([
                cirq.rz(-alpha / 2)(qubits[0]), cirq.rz(alpha / 2)(qubits[1])
            ])

            # im a little concerned about the orientation of this operator
            # check to make sure this works out
            circuit_elements.append(self.fsim(*qubits))

        """
        finally, add the measurements
        First, rotate to the x or y basis
        """
        circuit_elements.append(cirq.H(qubits[0]))

        circuit_elements.append(cirq.measure(qubits[0]))

        return circuit_elements

    def build_phase_circuit_y(self, depth, qubits, alpha):
        """

        :param depth:
        :param qubits:
        :return:
        """
        assert len(qubits) == 2

        circuit_elements = cirq.Circuit([cirq.H(qubits[0])])

        for d in range(depth):
            circuit_elements.append([
                cirq.rz(-alpha / 2)(qubits[0]), cirq.rz(alpha / 2)(qubits[1])
            ])

            # im a little concerned about the orientation of this operator
            # check to make sure this works out
            circuit_elements.append(self.fsim(*qubits))

        """
        finally, add the measurements
        First, rotate to the y basis
        """
        # rotating from y to -x
        circuit_elements.append([cirq.S(qubits[0])])
        # rotating from -x to -z
        circuit_elements.append(cirq.H(qubits[0]))

        circuit_elements.append(cirq.measure(qubits[0]))

        return circuit_elements

    def build_population_circuit(self, depth, qubits, alpha):
        """

        :param depth:
        :param qubits:
        :return:
        """
        assert len(qubits) == 2

        circuit_elements = cirq.Circuit([cirq.X(qubits[0])])

        for d in range(depth):
            circuit_elements.append([
                cirq.rz(-alpha / 2)(qubits[0]), cirq.rz(alpha / 2)(qubits[1])
            ])

            # im a little concerned about the orientation of this operator
            # check to make sure this works out
            circuit_elements.append(self.fsim(*qubits))

        """
        finally, add the measurements in the z basis
        """
        circuit_elements.append(cirq.measure(qubits[0]))

        return circuit_elements

    def run_phase_experiment(self, depths, measurement_num, alpha, simulator=cirq.Simulator()):

        x_measurements = []
        y_measurements = []

        # simulator = cirq.Simulator()

        for depth in depths:
            # run an x and y experiment for each depth
            x_circuit = self.build_phase_circuit_x(depth, self.qubits, alpha)

            results = simulator.run(x_circuit, repetitions=measurement_num)
            # here we care about the output of the first qubit

            x_measurement = results.data['0'].mean()

            x_measurements.append(x_measurement * 2 - 1)

            y_circuit = self.build_phase_circuit_y(depth, self.qubits, alpha)

            results = simulator.run(y_circuit, repetitions=measurement_num)
            # here we care about the output of the first qubit

            y_measurement = results.data['0'].mean()

            y_measurements.append(1 - y_measurement * 2)

        return x_measurements, y_measurements

    def run_population_experiment(self, depths, measurement_num, alpha, simulator=cirq.Simulator()):

        z_measurements = []

        # simulator = cirq.Simulator()

        for depth in depths:
            # run an x and y experiment for each depth
            z_circuit = self.build_population_circuit(depth, self.qubits, alpha)

            results = simulator.run(z_circuit, repetitions=measurement_num)
            # here we care about the output of the first qubit

            z_measurement = results.data['0'].mean()

            z_measurements.append(z_measurement * 2 - 1)

        return np.array(z_measurements)

    def obtain_decay_rate(self, xs, ys):
        """
        Fit to obtain an enveloping function for the xs and ys

        :param xs:
        :param ys:
        :return:
        """

        decay_xs = []
        decay_ys = []

        for i, v in enumerate(ys):
            if i == 0:
                neighbors = [i + 1]
            elif i == len(ys) - 1:
                neighbors = [i - 1]
            else:
                neighbors = [i - 1, i + 1]

            # is this a local max
            ismax = True
            for n in neighbors:
                if abs(ys[n]) > abs(v):
                    ismax = False

            if ismax:
                decay_ys.append(abs(v))
                decay_xs.append(xs[i])

        def func(t, a, b):
            return a * np.exp(-abs(b) * t)

        popt, pcov = curve_fit(func, decay_xs, decay_ys)

        plt.plot(xs, func(xs, *popt))

        plt.scatter(xs, ys)

        plt.show()

        return np.abs(popt[-1])

    @staticmethod
    def find_params(omegas, alphas, sigma):
        """
        Given values of omegas and alphas, identify zeta and theta
        using

        cos(omega) = cos(theta) cos(zeta + alpha)

        :param omegas:
        :param alphas:
        :param sigma:
        :return:
        """

        print(alphas, alphas[0], alphas[1])
        print(type(alphas), type(alphas[0]))

        def error(inputs):
            """
            the error of the fitting

            :param x:
            :param y:
            :return:
            """
            x, y = inputs

            alpha1 = float(alphas[0])
            alpha2 = float(alphas[1])

            omega1 = float(omegas[0])
            omega2 = float(omegas[1])

            loss1 = np.abs(np.cos(x) * np.cos(y + alpha1) - np.cos(omega1))
            loss2 = np.abs(np.cos(x) * np.cos(y + alpha2) - np.cos(omega2))

            return loss1 + loss2

        def upper_error(inputs):
            """
            the error of the fitting

            :param x:
            :param y:
            :return:
            """
            x, y = inputs

            loss1 = np.abs(np.cos(x) * np.cos(y + alphas[0]) - np.cos(omegas[0] + sigma))
            loss2 = np.abs(np.cos(x) * np.cos(y + alphas[1]) - np.cos(omegas[1] + sigma))

            return loss1 + loss2

        def lower_error(inputs):
            """
            the error of the fitting

            :param x:
            :param y:
            :return:
            """
            x, y = inputs

            loss1 = np.abs(np.cos(x) * np.cos(y + alphas[0]) - np.cos(omegas[0] - sigma))
            loss2 = np.abs(np.cos(x) * np.cos(y + alphas[1]) - np.cos(omegas[1] - sigma))

            return loss1 + loss2

        res = minimize(error, np.zeros(2), method='nelder-mead')
        params = res.x % (2 * np.pi)

        print("minimization result", res)

        # res = minimize(upper_error, params, method='nelder-mead')
        # upper_bound = res.x

        # res = minimize(lower_error, params, method='nelder-mead')
        # lower_bound = res.x

        return params, 0, 0  # lower_bound, upper_bound

    def full_population_experiment(self, depths, measurement_num, alphas, simulator=cirq.Simulator()):
        """
        run the population experiment at differing depths, and two different values of alpha,
        fourier transform to obtain the parameters.

        :param depths:
        :param measurement_num:
        :param alphas:
        :param simulator:
        :return:
        """

        # the results for both alpha values

        results_1 = self.run_population_experiment(depths, measurement_num, alphas[0], simulator)

        results_1 -= np.mean(results_1)

        results_2 = self.run_population_experiment(depths, measurement_num, alphas[1], simulator)

        results_2 -= np.mean(results_2)

        ft1 = lambda w: np.sum(results_1 * np.cos(w * depths))

        ft2 = lambda w: np.sum(results_2 * np.cos(w * depths))

        # the frequencies we sample with
        ws = np.linspace(0, np.pi, 10 ** 4)

        # identify peaks

        # first get rough peaks
        omega1 = ws[np.argmax(np.vectorize(ft1)(ws))] / 2



        omega2 = ws[np.argmax(np.vectorize(ft2)(ws))] / 2

        print(omega1)
        print(-ft1(np.array([omega1])))

        # omega1 = minimize(lambda w: -ft1(w), omega1).x[0]

        # omega2 = minimize(lambda w: -ft2(w), omega2).x[0]

        plt.plot(ws, np.vectorize(ft1)(ws) / np.max(np.vectorize(ft1)(ws)))
        plt.vlines(omega1*2, 0, 1, color = 'red')
        plt.show()

        plt.plot(ws, np.vectorize(ft2)(ws) / np.max(np.vectorize(ft2)(ws)))
        plt.vlines(omega2 * 2, 0, 1, color = 'red')
        plt.show()


        print("omegas: ", omega1, omega2)
        print("ideal omegas: ", np.arccos(np.cos(self.theta) * np.cos(self.zeta + alphas[0])),
              np.arccos(np.cos(self.theta) * np.cos(self.zeta + alphas[1])))

        """
        quantifying the noise
        I need to quantify the decay rate here 
        """

        gamma = self.obtain_decay_rate(depths, results_1)

        sigma = 1 / np.sqrt(np.sum([d ** 2 * np.exp(-gamma * d) for d in depths]) * measurement_num)

        plt.plot(ws, np.vectorize(ft1)(ws) / np.max(np.vectorize(ft1)(ws)))

        plt.vlines(2 * np.arccos(np.cos(self.theta) * np.cos(self.zeta + alphas[0])), 0, 1, color='red')

        plt.vlines(2 * omega1, 0, 1, color='green', linestyles='dashed')

        plt.show()

        plt.plot(ws, np.vectorize(ft2)(ws) / np.max(np.vectorize(ft2)(ws)))

        plt.vlines(2 * np.arccos(np.cos(self.theta) * np.cos(self.zeta + alphas[1])), 0, 1, color='red')

        plt.vlines(2 * omega2, 0, 1, color='green', linestyles='dashed')

        plt.show()

        """
        Now lets try to find the values of theta and gamma given the alpha values
        
        This should be possible exactly
        """

        if alphas == (0, 3 * np.pi / 2):
            zeta = np.arctan2(np.cos(omega2), np.cos(omega1))

            print(omega1, omega2)
            cos_theta1 = np.cos(omega1) / np.cos(zeta)
            cos_theta2 = np.cos(omega2) / np.sin(zeta)

            print(cos_theta1, cos_theta2)

            params = theta, zeta
        else:
            params, _, _ = self.find_params((omega1, omega2), alphas, sigma)

        print(f"theta: {params[0]}, zeta: {params[1]}")
        print(f"Ideal values: \n theta: {self.theta}, zeta: {self.zeta}")

    # Press the green button in the gutter to run the script.


if __name__ == '__main__':
    # plot_angle_region(theta=0, delta=.1, radius=2)
    # plt.show()
    # calibration_without_errors(alpha=.25, epsilon=.01, theta=.01)

    #run a variance test for a particular alpha by repreating individual measurements

    noisy_dsim = cirq.DensityMatrixSimulator(
        noise=cirq.generalized_amplitude_damp(p=0.01, gamma=0.1))


    #extract_heisenberg_scaling(A=-np.pi/2, delta_zero=0, delta_plus=0, visualize=True)

    #sweep_variances(delta_zero=0, delta_plus=0, visualize=True)

    extract_heisenberg_scaling(A=-np.pi/2, delta = .33, delta_zero=0, delta_plus=0, simulator=noisy_dsim, visualize=True)

    ghj
    alphas = (0, .1)
    theta, phi, gamma, chi, zeta = np.pi / 4, 0, np.pi / 6, np.pi / 12, np.pi / 24

    two_gate_exp = TwoGateCalibration(theta, phi, gamma, chi, zeta)

    ds = np.array(list(map(int, np.logspace(1, 3, 100))))

    print(ds)
    noisy_dsim = cirq.DensityMatrixSimulator(
        noise=cirq.generalized_amplitude_damp(p=0.0, gamma=0.0)
    )

    out = two_gate_exp.full_population_experiment(ds, 10 ** 4, alphas)#, noisy_dsim)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
