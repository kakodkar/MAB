from scipy.stats import beta
import json
import matplotlib
import numpy as np


class Util:
    debug = False

    def __init__(self):
        pass

    @staticmethod
    def set_debug():
        Util.debug = True

    @staticmethod
    def latexify(fig_width=None, fig_height=None, columns=1):
        """
            Copied from http://nipunbatra.github.io/2014/08/latexify/
            Set up matplotlib's RC params for LaTeX plotting.
            Call this before plotting a figure.

            Parameters
            ----------
            fig_width : float, optional, inches
            fig_height : float,  optional, inches
            columns : {1, 2}
            """

        # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

        # Width and max height in inches for IEEE journals taken from
        # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

        assert (columns in [1, 2])

        if fig_width is None:
            fig_width = 3.39 if columns == 1 else 6.9  # width in inches

        if fig_height is None:
            golden_mean = (np.math.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
            fig_height = fig_width * golden_mean  # height in inches

        max_height_inches = 8.0
        if fig_height > max_height_inches:
            print("WARNING: fig_height too large:" + fig_height +
                  "so will reduce to" + max_height_inches + "inches.")
            fig_height = max_height_inches

        params = {'backend': 'ps',
                  'text.latex.preamble': ['\usepackage{gensymb}'],
                  'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
                  'axes.titlesize': 8,
                  'text.fontsize': 8,  # was 10
                  'legend.fontsize': 8,  # was 10
                  'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'text.usetex': True,
                  'figure.figsize': [fig_width, fig_height],
                  'font.family': 'serif'
                  }

        matplotlib.rcParams.update(params)

    @staticmethod
    def dump_json_debug(results):
        if Util.debug:
            print json.dumps(results, default=lambda o: o.__dict__, indent=4, separators=(',', ': '))

    @staticmethod
    def dump_json(results):
        print json.dumps(results, default=lambda o: o.__dict__, indent=4, separators=(',', ': '))

    @staticmethod
    def get_json(results):
        return json.dumps(results, default=lambda o: o.__dict__, indent=4, separators=(',', ': '))


class Observations:
    def __init__(self, alpha, beta):
        self.alpha = int(alpha)
        self.beta = int(beta)
        self.s = 0
        self.f = 0
        self.mean_probability = 0.0
        self.update()
        pass

    def update(self):
        if self.alpha == 0 or self.beta == 0:
            raise ValueError('alpha and beta have to be greater than 0 and integers')
        self.mean_probability = float(self.alpha + self.s) / float(self.alpha + self.beta + self.s + self.f)

    def observe(self, s, f):
        self.s += s
        self.f += f
        self.update()

    def get_mean(self):
        return self.mean_probability

    def __str__(self):
        return Util.get_json(self)

    def __repr__(self):
        return self.__str__()


class LowRankMatrixFactorizer:
    def __init__(self):
        pass

    @staticmethod
    def factorize(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
        """
        Adapted from http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
        Adapted to use Probabilities instead of ratings.
        :param R: Beta[][]
        :param P:
        :param Q:
        :param K:
        :param steps:
        :param alpha:
        :param beta:
        :return:
        """
        Q = Q.T
        for step in xrange(steps):
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] is not None:
                        eij = R[i][j].get_mean() - np.dot(P[i, :], Q[:, j])
                        for k in xrange(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
            eR = np.dot(P, Q)
            e = 0
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] is not None:
                        e = e + pow(R[i][j].get_mean() - np.dot(P[i, :], Q[:, j]), 2)
                        for k in xrange(K):
                            e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
            if e < 0.001:
                break
        return P, Q.T


class Randomizer:
    def __init__(self):
        pass

    @staticmethod
    def generate_probability_matrix(user_count, item_count):
        """
        Returns a matrix whose rows represent the probabilities with which the user the row represents clicks on the item

        :return:
        """
        matrix = []
        for user in range(0, user_count):
            item_weights = []
            for item in range(0, item_count):
                item_weights.append(np.random.random())
            matrix.append(item_weights)
        return matrix

    @staticmethod
    def generate_user_sequence(user_count, iteration_count):
        iterations = []
        for i in range(0, iteration_count):
            iterations.append(np.random.permutation(range(0, user_count))[0])
        return iterations


class ContextualSimulator:

    def __init__(self, probability_matrix):
        self.underlying_matrix = probability_matrix
        self.user_count = probability_matrix.__len__()
        self.item_count = probability_matrix[0].__len__()
        pass

    def simulate(self, user, item):
        if not user < self.user_count:
            raise ValueError('invalid user')
        if not item < self.item_count:
            raise ValueError('invalid item')

        likelihood = self.underlying_matrix[user][item]
        return np.random.random() < likelihood


class Algorithm:
    def __init__(self):
        pass

    def record(self, arm, success, failure):
        pass

    def decide(self):
        pass


class Thompson:
    def __init__(self, arm_count):
        self.arm_count = arm_count
        self.observations = []
        for i in range(0,self.arm_count):
            self.observations.append(Observations(1,1))

    def record(self, arm, success, failure):
        self.observations[arm].observe(success,failure)

    def decide(self):
        winning_arm = None
        winning_ucb = None
        for arm in range(0,self.arm_count):
            obs = self.observations[arm]
            _alpha = obs.alpha + obs.s
            _beta = obs.beta + obs.f
            current_ucb = beta.rvs(_alpha, _beta)
            if winning_arm is None or current_ucb > winning_ucb:
                winning_arm = arm
                winning_ucb = current_ucb
        return winning_arm


class ContextualThompson:
    def __init__(self, arm_count):
        self.arm_count = arm_count
        self.observations = []
        for i in range(0,self.arm_count):
            self.observations.append(Observations(1,1))

    def record(self, arm, success, failure):
        self.observations[arm].observe(success,failure)

    def decide(self):
        winning_arm = None
        winning_ucb = None
        for arm in range(0,self.arm_count):
            obs = self.observations[arm]
            _alpha = obs.alpha + obs.s
            _beta = obs.beta + obs.f
            current_ucb = beta.rvs(_alpha, _beta)
            if winning_arm is None or current_ucb > winning_ucb:
                winning_arm = arm
                winning_ucb = current_ucb
        return arm


def run(algorithm, simulator, sequence):
    # type: (Thompson, ContextualSimulator, int[]) -> float
    reward = 0

    for user in sequence:
        arm = algorithm.decide()
        simulate = simulator.simulate(user, arm)
        s = f = 0
        if simulate :
            s = 1
            reward += 1
        else:
            f = 1
        algorithm.record(arm, s, f)

    return reward

Util.set_debug()
user_count = 20
item_count = 20
iteration_count = 1000

underlying_matrix = Randomizer.generate_probability_matrix(user_count, item_count)
Util.dump_json_debug(underlying_matrix)
simulator = ContextualSimulator(underlying_matrix)
sequence = Randomizer.generate_user_sequence(user_count, iteration_count)
print run(Thompson(item_count), simulator, sequence)
exit(0)
