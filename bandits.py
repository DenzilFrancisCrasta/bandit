import numpy as np
import matplotlib.pyplot as plt
import random

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
    def select_action(self, value_estimates):
        pass

# concrete strategies 
class EpsilonGreedySelector(ActionSelector):
    ''' Epsilon greedy selector''' 
    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon

    def select_action(self, value_estimates):
        # With probability epsilon EXPLORE choose an arm among all arms uniformly 
        if random.random() <= self.epsilon:
           action = random.randint(0, value_estimates.shape[0]-1)  

        # With probability 1-epsilon EXPLOIT choose the arm greedily
        else:
           action = np.argmax(value_estimates)
        return action 


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
            action = self.action_selector.select_action(value_estimates)

            # get reward associated with the arm 
            reward = self.bandit.pull_arm(action) 

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

    def plot_curves(self, x, y, labels, colors):
        for i in xrange(len(y)):
            plt.plot(x, y[i], colors[i])
            plt.hold(True)
        plt.xlabel(labels['xlabel'])
        plt.ylabel(labels['ylabel'])
        plt.show()


if __name__ == '__main__':

    # Assignment specific constants 
    RUNS      = 2000
    MAX_STEPS = 1000
    ARM_COUNT = 10
    SAVE_DIR = 'plots/'
    EPSILONS  = [0.1, 0.01, 0]


    rewards = [] 
    optimality = []

    data_monger = DataMongerer()

    for epsilon in EPSILONS:

        # gather the avg reward and avg fraction of optimal actions for an epsilon
        avg_r, avg_o = data_monger.get_mean_run_statistics(EpsilonGreedySelector(epsilon), ARM_COUNT, RUNS, MAX_STEPS)

        rewards.append(avg_r)
        optimality.append(avg_o)

        np.savetxt(SAVE_DIR + 'rewards'+ str(epsilon) +'.txt', avg_r, fmt='%.2f')
        np.savetxt(SAVE_DIR + 'optimality'+ str(epsilon) +'.txt', avg_o, fmt='%.2f')

    plotter = Plotter()

    colors = ['k', 'r', 'g']
    labels = { 'title' : 'Average Reward','xlabel': 'Steps', 'ylabel' : 'Average Reward' }
    plotter.plot_curves(np.arange(MAX_STEPS)+1, rewards, labels, colors)

    labels = { 'title' : 'Optimal Action %','xlabel': 'Steps', 'ylabel' : 'Optimal Action %' }
    plotter.plot_curves(np.arange(MAX_STEPS)+1, optimality, labels, colors)
