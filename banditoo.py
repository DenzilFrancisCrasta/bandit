import numpy as np
#import matplotlib.pyplot as plt
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


class ActionValueEstimator:
    ''' Estimates the action values of the arms of a mult-armed Bandit 
        using iterative sample-average method '''

    def __init__(self, bandit, epsilon=0.1, max_steps=1000):
        self.bandit = bandit
        self.epsilon = epsilon
        self.max_steps = max_steps

    def epsilon_greedy_choice(self, value_estimates):
        # With probability epsilon EXPLORE 
        if random.random() <= self.epsilon:
           # choose an arm among all arms uniformly 
           action = random.randint(0, value_estimates.shape[0]-1)  

        # With probability 1-epsilon EXPLOIT
        else:
           # choose the arm greedily
           action = np.argmax(value_estimates)
        return action 

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
            # select an arm using epsilon greedy approach
            action = self.epsilon_greedy_choice(value_estimates)

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

    def get_mean_run_statistics(self, epsilon=0.1, arm_count=10, runs=1, max_steps=1000):
        ''' Utility method to calculate the average of rewards and 
            fraction of optimal actions per step across runs of the value estimator '''

        rewards_in_runs = []
        optimal_in_runs = []

        for i in xrange(runs):
            # get rewards and optimality of action selection during a "run" of the action_value_estimator
            bandit = Bandit(arm_count)
            estimator = ActionValueEstimator(bandit, epsilon, max_steps)
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


if __name__ == '__main__':

    RUNS      = 2000
    MAX_STEPS = 1000
    ARM_COUNT = 10
    epsilons  = [0.1, 0.01, 0]

    data_monger = DataMongerer()

    to_plot_rewards = [] 
    to_plot_optimality = []

    for epsilon in epsilons:
        avg_rewards, avg_optimality = data_monger.get_mean_run_statistics(epsilon, ARM_COUNT, RUNS, MAX_STEPS)
        to_plot_rewards.append(avg_rewards)
        to_plot_optimality.append(avg_optimality)
        np.savetxt('oorewards'+ str(epsilon) +'.txt', avg_rewards, fmt='%.2f')
        np.savetxt('oooptimality'+ str(epsilon) +'.txt', avg_optimality, fmt='%.2f')


    '''
    plt.subplot(211)
    plt.plot(avg_rewards, 'r')
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')

    plt.subplot(212)
    plt.plot(avg_optimality, 'b')
    plt.ylabel('Optimal Action %')
    plt.xlabel('Steps')

    plt.show()
    '''
