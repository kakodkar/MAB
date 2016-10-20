import numpy as np
from scipy.stats import beta
import json
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime


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

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.math.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'text.fontsize': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)

def init_arm_wise_stats():
    arm_wise_stats = dict()
    for county in Simulator().get_counties():
        arm_wise_stats[county] = Measurement()
    return arm_wise_stats


def dump_json_debug(results):
    if debug:
        print json.dumps(results, default=lambda o: o.__dict__, indent=4, separators=(',', ': '))


def dump_json(results):
    print json.dumps(results, default=lambda o: o.__dict__, indent=4, separators=(',', ': '))


class Simulator:
    def __init__(self):
        pass

    probalilities = dict()
    probalilities["los_angeles_county_california"] = 0.091
    probalilities["cook_county_illinois"] = 0.085
    probalilities["maricopa_county_arizona"] = 0.09
    probalilities["harris_county_texas"] = 0.085
    probalilities["kings_county_new_york"] = 0.103
    probalilities["queens_county_new_york"] = 0.112
    probalilities["orange_county_california"] = 0.08
    probalilities["san_diego_county_california"] = 0.077
    probalilities["dallas_county_texas"] = 0.091
    probalilities["miami-Dade_county_florida"] = 0.079
    probalilities["wayne_county_michigan"] = 0.12
    probalilities["tarrant_county_texas"] = 0.11
    probalilities["riverside_county_california"] = 0.09
    probalilities["clark_county_nevada"] = 0.092
    probalilities["philadelphia_county_pennsylvania"] = 0.115
    probalilities["san_bernardino_county_california"] = 0.092
    probalilities["broward_county_florida"] = 0.095
    probalilities["bronx_county_new_york"] = 0.123
    probalilities["santa_clara_county_california"] = 0.084
    probalilities["bexar_county_texas"] = 0.089
    probalilities["cuyahoga_county_ohio"] = 0.12
    probalilities["palm_beach_county_florida"] = 0.105
    probalilities["king_county_washington"] = 0.068
    probalilities["alameda_county_california"] = 0.085

    def simulate(self, s):
        is_diabetic = np.random.random() < self.probalilities.get(s)
        if is_diabetic:
            return np.random.random() < buyProbability
        else:
            return False

    def get_counties(self):
        return self.probalilities.keys()


class Measurement:
    n = 0.0
    np = 0.0
    moneyTotal = 0.0
    p = v = sd = zs = -1.0

    def __init__(self):
        self.trials = []
        self.money = []
        self.invite_number = []
        self.arm_invite_number = []

    def process_variables(self):
        if self.n == 0 or self.np == 0 or self.n == self.np:
            self.p = 0
            self.v = 0
            self.sd = 0
            self.zs = 0
        else:
            self.p = self.np / self.n
            self.v = self.p * (1 - self.p) / self.n
            self.sd = np.math.sqrt(self.v)
            self.zs = self.p / self.sd

    def add_trial(self, downloaded, time):
        if downloaded:
            self.np += 1
        self.n += 1

        self.trials.append(downloaded)
        if downloaded:
            net_money = profit
        else:
            net_money = cost
        self.moneyTotal += net_money
        self.money.append(self.moneyTotal)

        self.invite_number.append(time)
        self.arm_invite_number.append(self.n)

        self.process_variables()
        return net_money


class Algorithm:
    def __init__(self):
        pass

    def decide(self, time):
        pass


class ABTesting:
    def __init__(self, m=205):
        self.m = m
        self.arms = Simulator().get_counties()
        self.armCount = self.arms.__len__()
        self.max_county = None
        self.algo = "AB Testing with M "+ str(m)

    def decide(self, time, current_revenue, arm_wise_stats):
        if time < self.m * self.armCount:
            return self.arms[time % self.armCount]
        else:
            if self.max_county is None:
                max_measurement = None
                for county in arm_wise_stats:
                    measurement = arm_wise_stats.get(county)
                    if measurement.zs > 1.7 and (max_measurement is None or measurement.p > max_measurement.p):
                        max_measurement = measurement
                        self.max_county = county
                dump_json(self.max_county)
            return self.max_county

class Epsilon:
    def __init__(self, delta_squared=0.01):
        self.delta_squared = delta_squared
        self.arms = Simulator().get_counties()
        self.armCount = self.arms.__len__()
        self.algo = "Epsilon with delta squared "+ str(delta_squared)

    def decide(self, time, current_revenue, arm_wise_stats):
        epsilon = np.minimum(1, 12 / (self.delta_squared * (1 + time)))

        if np.random.random() < epsilon :
            random_arm = self.arms[int(np.random.random() * self.armCount)]
            dump_json_debug("random : " + random_arm)
            return random_arm
        else:
            max_measurement = None
            max_county = None
            for county in arm_wise_stats:
                measurement = arm_wise_stats.get(county)
                if max_measurement is None or measurement.p > max_measurement.p:
                    max_measurement = measurement
                    max_county = county
            # dump_json(max_county)
            return max_county

class UCB1:
    def __init__(self):
        self.arms = Simulator().get_counties()
        self.armCount = self.arms.__len__()
        self.algo = "UCB 1"

    def decide(self, time, current_revenue, arm_wise_stats):
        max_ucb = None
        max_county = None
        for county in arm_wise_stats:
            m = arm_wise_stats.get(county)
            if m.n == 0 :
                return county
            current_ucb = m.np/m.n + np.math.sqrt(2*np.math.log(time) / m.n)
            if max_ucb is None or current_ucb > max_ucb:
                max_ucb = current_ucb
                max_county = county
        return max_county

class Thompson:
    def __init__(self, alpha, beta):
        self.arms = Simulator().get_counties()
        self.armCount = self.arms.__len__()
        self.alpha = alpha
        self.beta = beta
        self.algo = "Thompson with alpha " + str(alpha) + " and beta "+str(beta)

    def decide(self, time, current_revenue, arm_wise_stats):
        max_ucb = None
        max_county = None
        for county in arm_wise_stats:
            m = arm_wise_stats.get(county)
            _alpha = self.alpha + m.np
            _beta = self.beta + m.n - m.np
            current_ucb = beta.rvs(_alpha, _beta)
            if max_ucb is None or current_ucb > max_ucb:
                max_ucb = current_ucb
                max_county = county
        return max_county

def run(algorithm):
    simulator = Simulator()
    current_revenue = 1.0
    net_revenue = []
    arm_wise_stats = init_arm_wise_stats()

    for time in range(0, timeLimit):
        county = algorithm.decide(time, current_revenue, arm_wise_stats)

        simulate = simulator.simulate(county)

        measure = arm_wise_stats.get(county)
        current_revenue += measure.add_trial(simulate, time)
        net_revenue.append(current_revenue)
    dump_json(current_revenue)

    latexify()
    log_scale = 'log'
    extension = ".pdf"
    folder = '../plot/'

    for county in arm_wise_stats:
        plt.plot(arm_wise_stats[county].money)
        plt.title(algorithm.algo)
        plt.yscale(log_scale)
        # plt.xscale(log_scale)
        plt.ylabel('revenue per arm')
        plt.xlabel('invites per arm')
    plt.savefig(folder + algorithm.algo.replace(" ","") + "1" + extension)
    plt.close()

    for county in arm_wise_stats:
        plt.plot(arm_wise_stats[county].invite_number, arm_wise_stats[county].arm_invite_number)
        plt.title(algorithm.algo)
        # plt.yscale(log_scale)
        # plt.xscale(log_scale)
        plt.ylabel('invites per arm')
        plt.xlabel('total invites')
    plt.savefig(folder + algorithm.algo.replace(" ","") + "2" + extension)
    plt.close()

    plt.plot(net_revenue)
    plt.title(algorithm.algo)
    plt.yscale(log_scale)
    # plt.xscale(log_scale)
    plt.ylabel('total revenue')
    plt.xlabel('total invites')
    # plt.show()
    plt.savefig(folder + algorithm.algo.replace(" ","") + "3" + extension)
    plt.close()

cost = -0.001
profit = 2.99
timeLimit = 10000
buyProbability = 0.2
debug = False

for algorithm in [ABTesting(205), Epsilon(2.56), UCB1(), Thompson(5,5)]:
    print algorithm.algo
    run(algorithm)

# run(ABTesting(300))

# run(Thompson(1,2))