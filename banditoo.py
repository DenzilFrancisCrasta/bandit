import numpy as np
import random

class Bandit:
    ''' Multi arm bandit with true action values drawn from standard
        normal distribution '''

    def __init__(self, arm_count):
        self.arm_count = arm_count

        # initialize true action values by sampling standard normal distribution 
        self.true_action_values = np.random.randn(ARM_COUNT, 1)

    def pull_arm(self, arm_index):
        # generate reward in response to pulling the arm which is 
        # sampled from a normal distribution with mean around the true_action_value with variance 1
        return np.random.standard_normal() + self.true_action_values[arm_index]

    def get_arm_count():
        return self.arm_count

    def get_optimal_arm():
        return np.argmax(self.true_action_values)


class ActionValueEstimator:

    def __init__(self, bandit, epsilon=0.1, max_steps=1000):
        self.max_steps = max_steps

    def epsilon_greedy_choice(self):
        # With probability epsilon EXPLORE 
        if random.random() <= self.epsilon:
           # choose an arm among all arms uniformly 
           action = random.randint(0, self.bandit.get_arm_count()-1)  

        # With probability 1-epsilon EXPLOIT
        else:
           # choose the arm greedily
           action = self.bandit.get_optimal_arm()
        return action 

    def action_value_estimate_run(self):
        ''' A single run of max_steps to estimate action values 
            using iterative sample average method '''

        ARM_COUNT = self.bandit.get_arm_count()
            
        # initialize value estimates to zero for epsilon greedy methods
        value_estimates = np.zeros((ARM_COUNT, 1)) 
    
        # initialize the number of times each bandit arm has been pulled to zero
        pull_count = np.zeros((ARM_COUNT, 1))

        # tracks if optimal action was chosen in the time step
        is_optimal = [] 

        # track the rewards for each time step
        rewards = []

        for step in xrange(self.max_steps):
            # select an arm using epsilon greedy approach
            action = self.epsilon_greedy_choice()

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



if __name__ == '__main__':

    RUNS      = 2000
    MAX_STEPS = 1000
    ARM_COUNT = 10

    # setup storage for data to be plotted
    rewards_in_runs = []
    optimal_in_runs = []

    bandit = Bandit(ARM_COUNT)
    estimator = ActionValueEstimator(bandit, epsilon=0.1, max_steps=MAX_STEPS)

    for i in xrange(RUNS):
        rewards, is_optimal = estimator.action_value_estimate_run()
        rewards_in_runs.append(rewards)
        optimal_in_runs.append(is_optimal)

    reward_matrix     = np.array(rewards_in_runs) 
    optimality_matrix = np.array(optimal_in_runs) 

    avg_rewards    = np.mean(reward_matrix, axis=0)
    avg_optimality = np.mean(optimality_matrix, axis=0)
    avg_optimality = avg_optimality * 100

    np.savetxt('rewards.txt', avg_rewards, fmt='%.2f')
    np.savetxt('optimality.txt', avg_optimality, fmt='%.2f')

    '''
    plt.subplot(211)
    plt.plot(avg_rewards, 'r--')
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')

    plt.subplot(212)
    plt.plot(avg_optimality, 'b--')
    plt.ylabel('Optimal Action %')
    plt.xlabel('Steps')

    plt.show()
    '''
