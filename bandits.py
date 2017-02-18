import numpy as np
import random 
import matplotlib.pyplot as plt

def bandit_run():
    ARM_COUNT = 10
    MAX_STEPS = 1000
    EPSILON = 0.1 # Epsilon [0,1) value to be used for epsilon-greedy strategy

    # initialize true action values by sampling standard normal distribution 
    action_values = np.random.randn(ARM_COUNT, 1)

    optimal_action = np.argmax(action_values)

    # initialize value estimates to zero for epsilon greedy methods
    value_estimates = np.zeros((ARM_COUNT, 1)) 

    # initialize the number of times each bandit arm has been pulled to zero
    pull_count = np.zeros((ARM_COUNT, 1))

    # tracks if optimal action was chosen in the time step
    is_optimal = [] 

    # track the rewards for each time step
    rewards = []

    for step in xrange(MAX_STEPS):

        # With probability epsilon EXPLORE 
        if random.random() <= EPSILON:
           # choose an arm among all arms uniformly 
           action = random.randint(0, ARM_COUNT-1)  

        # With probability 1-epsilon EXPLOIT
        else:
           # choose the arm greedily
           action = np.argmax(value_estimates)


        # generate reward which is sampled from a normal distribution
        # with mean around the action_value and variance 1
        reward = np.random.standard_normal() + action_values[action]

        # maintain per step statistics 
        is_optimal.append(int(action == optimal_action))
        rewards.append(reward)

        # update the value estimates incrementally 
        pull_count[action] += 1 
        step_size = (1.0 / pull_count[action])
        value_estimates[action] += step_size * (reward - value_estimates[action])

    return (rewards, is_optimal)

if __name__ == '__main__':

    results = []
    RUNS = 2000
    MAX_STEPS = 1000

    rewards_in_runs = []
    optimal_in_runs = []

    for i in xrange(RUNS):
        rewards, is_optimal = bandit_run()
        rewards_in_runs.append(rewards)
        optimal_in_runs.append(is_optimal)

    reward_matrix     = np.array(rewards_in_runs) 
    optimality_matrix = np.array(optimal_in_runs) 

    avg_rewards    = np.mean(reward_matrix, axis=0)
    avg_optimality = np.mean(optimality_matrix, axis=0)
    avg_optimality = avg_optimality * 100

    #np.savetxt('rewards.txt', avg_rewards, fmt='%.2f')
    #np.savetxt('optimality.txt', avg_optimality, fmt='%.2f')

    plt.subplot(211)
    plt.plot(avg_rewards, 'r--')
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')

    plt.subplot(212)
    plt.plot(avg_optimality, 'b--')
    plt.ylabel('Optimal Action %')
    plt.xlabel('Steps')

    plt.show()
