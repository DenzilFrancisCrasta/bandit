import numpy as np
import matplotlib.pyplot as plt
import random
import math

class Bandit:
    ''' Simulates a multi-arm Bandit 
        with true action values drawn from standard normal distribution '''

    def __init__(self, arm_count):
        self.arm_count = arm_count

        # initialize true action values by sampling standard normal distribution 
        self.true_action_values = np.random.randn(ARM_COUNT, 1)

    def pull_arm(self, arm_index):
        ''' generate reward in response to pulling the arm, 
            sampled from a normal distribution with 
            mean around the true_action_value with unit variance '''
        return np.random.standard_normal() + self.true_action_values[arm_index]

    def get_arm_count(self):
        return self.arm_count

    def get_optimal_arm(self):
        return np.argmax(self.true_action_values)

# STRATEGY PATTERN for action selectors 

class ActionSelector:
    ''' Abstract Action Selector that defines the interface for the strategy action selector methods '''
    def select_action(self, value_estimates, pull_counts, step):
        pass

# concrete strategies 
class EpsilonGreedySelector(ActionSelector):
    ''' Epsilon Greedy Selector''' 
    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon

    def select_action(self, value_estimates, pull_counts, step):
        # With probability epsilon EXPLORE choose an arm among all arms uniformly 
        if random.random() <= self.epsilon:
           action = random.randint(0, value_estimates.shape[0]-1)  

        # With probability 1-epsilon EXPLOIT choose the arm greedily
        else:
           action = np.argmax(value_estimates)
        return action 

    def __str__(self):
        return 'Epsilon ' + str(self.epsilon)

class UCB_selector(ActionSelector):
    def __init__(self, c):
        self.c = c

    def select_action(self, value_estimates, pull_counts, step):
        # if any of the arms has not yet been pulled then pull it
        # this ensures that all arms are tried first atleast once
        for i, count in enumerate(pull_counts):
            if count == 0:
                return i
        
        variance_factors = [ self.c * math.sqrt( math.log(step) / float(count)) for count in pull_counts]
        return np.argmax( value_estimates + np.asarray(variance_factors).reshape(value_estimates.shape))

    def __str__(self):
        return 'UCB c=' + str(self.c)

class SoftmaxSelector(ActionSelector):
    ''' Softmax Action Selector '''
    def __init__(self, temperature):
        self.temperature = temperature

    def _softmax(self, value_estimates):
        
        # subtract the max exponent from all values for numerical stability
        max_adjusted_values = np.exp( value_estimates - np.max(value_estimates)) 

        softmax_probabilities = max_adjusted_values / np.sum(max_adjusted_values)
        return softmax_probabilities

    def _inverse_transform_sample(self, pdf):
        ''' returns samples from the probability distribution given by pdf 
            using the technique of inverse transform sampling ''' 
        # cumulative distribution function of the given pdf
        cdf = np.cumsum(pdf)
        # uniform random number representing the probability we want 
        u = random.random()  
        return np.searchsorted(cdf, u)
        

    def select_action(self, value_estimates, pull_counts, step):
        tempered_values = value_estimates / float(self.temperature) 

        # calculate the softmax probabilites of each action 
        softmax_pdf = self._softmax(tempered_values)

        # return the action index of an action drawn from the above softmax_pdf 
        return self._inverse_transform_sample(softmax_pdf) 

    def __str__(self):
        return 'Temperature ' + str(self.temperature)


class ActionValueEstimator:
    ''' Estimates the action values of the arms of a mult-armed Bandit 
        using iterative sample-average method '''

    def __init__(self, bandit, action_selector, max_steps=1000):
        self.bandit = bandit
        self.max_steps = max_steps
        self.action_selector = action_selector


    def action_value_estimate_run(self):
        ''' A single run of max_steps to estimate action values 
            using iterative sample average method '''

        arm_count = self.bandit.get_arm_count()
            
        # initialize value estimates to zero for epsilon greedy methods
        value_estimates = np.zeros((arm_count, 1)) 
    
        # initialize the number of times each bandit arm has been pulled to zero
        pull_count = np.zeros((arm_count, 1))

        # tracks if optimal action was chosen in the time step
        is_optimal = [] 

        # track the rewards for each time step
        rewards = []

        for step in xrange(self.max_steps):
            # select an arm using the strategy action selection routine
            action = self.action_selector.select_action(value_estimates, pull_count, step)

            # get reward associated with the arm 
            reward = self.bandit.pull_arm(action) 
            #print("{0} action {1} reward {2}".format(self.action_selector, action ,reward))

            # store step-wide statistics to be plotted
            rewards.append(reward)
            is_optimal.append(int(action == self.bandit.get_optimal_arm()))

            # update the value estimates incrementally 
            pull_count[action] += 1 
            step_size = (1.0 / pull_count[action])
            value_estimates[action] += step_size * (reward - value_estimates[action])

        return (rewards, is_optimal)

class DataMongerer:
    ''' Collects data for the assignment questions  '''

    def get_mean_run_statistics(self, action_selector, arm_count=10, runs=1, max_steps=1000):
        ''' Utility method to calculate the average of rewards and 
            fraction of optimal actions per step across runs of the value estimator '''

        rewards_in_runs = []
        optimal_in_runs = []

        for i in xrange(runs):
            # get rewards and optimality of action selection during a "run" of the action_value_estimator
            bandit = Bandit(arm_count)
            estimator = ActionValueEstimator(bandit, action_selector, max_steps)
            rewards_per_step, optimal_action_fraction_per_step = estimator.action_value_estimate_run()

            # save the "run" statistics 
            rewards_in_runs.append(rewards_per_step)
            optimal_in_runs.append(optimal_action_fraction_per_step)

        # construct a 2D matrix of rewards where each row corresponds to the results of an individual run 
        reward_matrix     = np.array(rewards_in_runs) 
        optimality_matrix = np.array(optimal_in_runs) 

        # mean reward across runs
        avg_rewards    = np.mean(reward_matrix, axis=0)

        # mean of fraction of optimal action selection
        avg_optimality = np.mean(optimality_matrix, axis=0)
        avg_optimality = avg_optimality * 100 # convert fractions into percentages

        return (avg_rewards, avg_optimality) 

class Plotter:

    def plot_curves(self, x, y, labels):
        colors = ['k', 'r', 'g', 'y', 'm', 'b']
        for i in xrange(len(y)):
            plt.plot(x, y[i], colors[i % len(colors)], label=labels['legends'][i])
            plt.hold(True)
        plt.xlim([-20, 1001])
        plt.xlabel(labels['xlabel'])
        plt.ylabel(labels['ylabel'])
        plt.legend(loc='lower right')
        plt.show()

class Assignment:
    ''' A Facade that implements each assignment question coordinating with the objects
        required to solve each question of the assignment '''

    SAVE_DIR = 'plots/'

    # predicate "to save or not to save" run statistics on disk
    save_statistics = False 

    def evaluate_action_selectors(self, action_selectors, runs, max_steps, arm_count):  
        ''' collects performance statistics of each action_selector
            and plots the attributes avg reward and avg fraction of optimal actions '''

        rewards = [] 
        optimality = []

        data_monger = DataMongerer()

        for action_selector in action_selectors:

            # gather the avg reward and avg fraction of optimal actions of an action_selector
            avg_r, avg_o = data_monger.get_mean_run_statistics(action_selector, arm_count, runs, max_steps)

            rewards.append(avg_r)
            optimality.append(avg_o)

            if Assignment.save_statistics == True:
                np.savetxt(Assignment.SAVE_DIR + 'rewards'+ action_selector +'.txt', avg_r, fmt='%.2f')
                np.savetxt(Assignment.SAVE_DIR + 'optimality'+ action_selector +'.txt', avg_o, fmt='%.2f')

        plotter = Plotter()

        legends = [str(selector) for selector in action_selectors]

        labels = { 'title' : 'Average Reward','xlabel': 'Steps', 'ylabel' : 'Average Reward', 'legends': legends}
        plotter.plot_curves(np.arange(MAX_STEPS)+1, rewards, labels)

        labels = { 'title' : 'Optimal Action %','xlabel': 'Steps', 'ylabel' : 'Optimal Action %', 'legends': legends }
        plotter.plot_curves(np.arange(MAX_STEPS)+1, optimality, labels)



if __name__ == '__main__':

    # Assignment specific constants 
    RUNS      = 2000
    MAX_STEPS = 1000
    ARM_COUNT = 10
    EPSILONS  = [0.1, 0.01, 0]
    TEMPERTATURES = [0.001,0.1, 0.15, 0.2, 0.25]
    UCB_PARAMS = [0.4, 1, 2]

    # instantiate a list of epsilon greedy action selectors to be evaluated  
    eps_greedy_selectors = [EpsilonGreedySelector(epsilon) for epsilon in EPSILONS]
    softmax_selectors = [SoftmaxSelector(t) for t in TEMPERTATURES] 
    ucb_selectors = [UCB_selector(c) for c in UCB_PARAMS]

    driver = Assignment()
    driver.evaluate_action_selectors(ucb_selectors, RUNS, MAX_STEPS, ARM_COUNT) 


