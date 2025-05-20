import sys
import math
import matplotlib.pyplot as plt

# NOTE THAT THESE TRY EXCEPTS ARE ONLY ADDED SO THAT YOU KNOW
# THAT YOU MUST INSTALL THESE LIBRARIES IF YOU DON"T ALREADY HAVE THEM

try:
    import gymnasium as gym
except:
    print("The gymnasium library is not installed!")
    print("Please install gymnasium in your python environment using:")
    print("\tpip install gymnasium")
    sys.exit(1)

try:
    import numpy as np
except:
    print("The numpy library is not installed!")
    print("Please install numpy in your python environment using:")
    print("\tpip install numpy")
    sys.exit(1)


def part_one():
    from tutorial import generate_random_policy, run_one_experiment, display_policy

    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="ansi")
    nS = env.observation_space.n # number of states -- 8x8=64
    nA = env.action_space.n # number of actions -- four directions; 0:left, 1:down, 2:right, 3:up

    # 10 different seeds for reproducibility of random policy
    seeds = [0, 1]
    num_experiments = 10 # number of experiments for each seed
    num_episodes = 30 # number of episodes for each 
    # seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # num_experiments = 100 # number of experiments for each seed
    # num_episodes = 10000 # number of episodes for each experiment

    # dictionary to store the statistics of each policy
    policy_stats = []

    # generate 10 different random policies using different seeds
    for seed in seeds:
        # generate a random policy for each seed
        policy = generate_random_policy(nA, nS, seed=seed)
        goals_list = [] # list to store the number of times the goal is reached
        steps_list = [] # list to store the number of steps taken to reach the goal for each experiment
        for _ in range(num_experiments):
            # run the experiment for each seed 10000 times
            print(f"Running experiment for seed {seed}...")
            goals, holes, total_rewards, total_goal_steps = run_one_experiment(env, policy, num_episodes)
            goals_list.append(goals) # append the number of times the goal is reached
            # preventing ZeroDivisionError when the goal is not reached
            avg_steps_to_goal = (total_goal_steps / goals) if (goals > 0) else 0
            # append the average number of steps taken to reach the goal
            steps_list.append(avg_steps_to_goal)
        mean_goals = np.mean(goals_list) # mean of the number of times the goal is reached
        std_goals = np.std(goals_list) # standard deviation of the number of times the goal is reached
        mean_steps = np.mean(steps_list) # mean of the average number of steps taken to reach the goal
        # append the statistics of the policy to the policy_stats dictionary for the chart
        policy_stats.append({
            'seed': seed,
            'policy': policy,
            'goals_list': goals_list,
            'steps_list': steps_list,
            'mean_goals': mean_goals,
            'std_goals': std_goals,
            'mean_steps': mean_steps
        })

    # Sort by mean_goals to find the top two policies
    top_two_policies = sorted(policy_stats, key=lambda x: x['mean_goals'], reverse=True)[:2]

    # print the top two policies
    for idx, policy in enumerate(top_two_policies, 1):
        print(f"\n *** Policy ***")
        print(display_policy(policy['policy'], nS))
        plt.figure()
        plt.hist(policy['goals_list'], bins=10, density=True)
        plt.title(
            f"Density Histogram of the Number of Episodes (out of 10,000) Reaching the Goal State\n\n"
            f"Policy {idx} (seed={policy['seed']})\n[mean: {policy['mean_goals']:.2f}, stdev: {policy['std_goals']:.2f}, mean#steps: {policy['mean_steps']:.1f}]", fontsize=8)
        plt.xlabel('Number of Episodes Reaching Goal (out of 10,000)', fontsize=8)
        plt.ylabel('Density', fontsize=8)
        plt.show()


def part_two():
    # TODO: your code here ...
    pass


def main():
    # TODO: feel free to change this as required
    # TODO: also, check tutorial.py for some hints on how to implement your experiments
    part_one()
    part_two()


if __name__ == "__main__":
    main()

