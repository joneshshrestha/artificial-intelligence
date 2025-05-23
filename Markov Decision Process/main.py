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
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_experiments = 100 # number of experiments for each seed
    num_episodes = 10000 # number of episodes for each experiment

    # dictionary to store the statistics of each policy
    policy_stats = []

    # generate 10 different random policies using different seeds (following tutorial.py pattern)
    for seed in seeds:
        print(f"\n*** EVALUATING POLICY WITH SEED {seed} ***")
        # generate a random policy for each seed
        policy = generate_random_policy(nA, nS, seed=seed)
        goals_list = [] # list to store the number of times the goal is reached
        steps_list = [] # list to store the number of steps taken to reach the goal for each experiment
        
        for experiment in range(num_experiments):
            if experiment % 20 == 0:  # Progress indicator (every 20 experiments)
                print(f"Running experiment {experiment+1}/{num_experiments} for seed {seed}...")
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

    # print the top two policies (following tutorial.py result format)
    for idx, policy in enumerate(top_two_policies, 1):
        print(f"\n*** TOP POLICY {idx} (SEED {policy['seed']}) RESULTS ***:")
        print(f"\tMean Goals: {policy['mean_goals']:>8.2f}/{num_episodes} episodes")
        print(f"\tStd Goals:  {policy['std_goals']:>8.2f}")
        print(f"\tMean Steps: {policy['mean_steps']:>8.2f} steps to goal")
        print("\n*** POLICY GRID ***")
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
    from tutorial import run_one_experiment, display_policy

    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="ansi")
    nS = env.observation_space.n
    nA = env.action_space.n
    discount_rate_gamma = 1.0
    convergence_threshold_theta = 1e-4

    # Initialize value function, current and previous using numpy arrays
    cur_value_function = np.zeros(nS)
    prev_value_function = np.zeros(nS)

    # Value Iteration
    iteration_count = 0
    while True:
        iteration_count += 1
        value_function_diff_delta = 0
        # for each state
        for s in range(nS):
            Q_sa = []
            # for each action
            for a in range(nA):
                q = 0
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    # calculate and append bellman optimality equation (q)
                    q += prob * (reward + discount_rate_gamma * prev_value_function[next_state])
                # append the calculated q values to sum
                Q_sa.append(q)
            # update the current value function with the max/best value
            cur_value_function[s] = max(Q_sa)
            # calculate the difference between the current and previous value function, and compare it to previous difference and get the max and store as new delta
            value_function_diff_delta = max(value_function_diff_delta, abs(cur_value_function[s] - prev_value_function[s]))
        # if the difference is less than the convergence threshold, break
        if value_function_diff_delta < convergence_threshold_theta:
            print(f"Value iteration converged after {iteration_count} iterations with delta = {value_function_diff_delta:.6f}")
            break
        # copy the current value function to previous value function, so that we don't overwrite it when we update the current value function
        prev_value_function = cur_value_function.copy()

    # next step is to extract optimal policy
    policy = np.zeros(nS, dtype=int) # stores integer values of actions (0,1,2,3)
    for s in range(nS):
        Q_sa = []
        # for each possible action in the current state, calculate the q value
        for a in range(nA):
            q = 0
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q += prob * (reward + discount_rate_gamma * cur_value_function[next_state])
            Q_sa.append(q)
        # store the action with the highest q value which is our policy
        policy[s] = np.argmax(Q_sa)

    # display policy and value function as 2D arrays
    print("\n*** Optimal Policy (Value Iteration) ***")
    print(display_policy(policy, nS))
    print("\n*** Converged V(s) Table ***")
    print(display_policy(cur_value_function, nS))

    print("\n Visual Simulation following the optimal policy")
    # using gymnasium's built-in FrozenLake
    sim_env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="human")
    sim_env.reset()  # Tutorial pattern: no unpacking
    done = False
    steps = 0

    while not done:
        state = sim_env.get_wrapper_attr("s")  # Tutorial pattern: consistent state access
        action = policy[state]
        action_names = ['Left', 'Down', 'Right', 'Up']
        print(f"Step {steps}: State={state}, Action={action} ({action_names[action]})")
        next_state, reward, done, info, p = sim_env.step(action)  # Tutorial pattern: 5 values
        steps += 1
    
    print(f"Visual simulation finished in {steps} steps with reward {reward}")
    sim_env.close()  # Close the visual window

    # Also show text version for comparison
    print("\n--- Text version of the simulation ---")
    sim_env_text = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="ansi")
    sim_env_text.reset()  # Tutorial pattern: no unpacking
    done = False
    steps = 0
    print("Initial state:")
    print(sim_env_text.render())
    
    while not done:
        state = sim_env_text.get_wrapper_attr("s")  # Tutorial pattern: consistent state access
        action = policy[state]
        next_state, reward, done, info, p = sim_env_text.step(action)  # Tutorial pattern: 5 values
        print(f"After step {steps}: Action={action}")
        print(sim_env_text.render())
        steps += 1
    print(f"Text simulation finished in {steps} steps with reward {reward}\n")

    # 5. Evaluate the policy (following tutorial.py pattern)
    print("\n*** EVALUATING OPTIMAL POLICY ***")
    num_experiments = 100
    num_episodes = 10000
    goals_list = []
    steps_list = []
    
    for experiment in range(num_experiments):
        if experiment % 10 == 0:  # Progress indicator
            print(f"Running experiment {experiment+1}/{num_experiments}...")
        goals, holes, total_rewards, total_goal_steps = run_one_experiment(env, policy, num_episodes)
        goals_list.append(goals)
        avg_steps_to_goal = (total_goal_steps / goals) if (goals > 0) else 0
        steps_list.append(avg_steps_to_goal)
    
    # Calculate statistics (following tutorial.py format)
    mean_goals = np.mean(goals_list)
    std_goals = np.std(goals_list)
    mean_steps = np.mean(steps_list)
    
    # Display results in tutorial.py format
    print(f"\n*** OPTIMAL POLICY RESULTS ***:")
    print(f"\tMean Goals: {mean_goals:>8.2f}/{num_episodes} episodes")
    print(f"\tStd Goals:  {std_goals:>8.2f}")
    print(f"\tMean Steps: {mean_steps:>8.2f} steps to goal")

    # 6. Plot histogram
    plt.figure()
    plt.hist(goals_list, bins=10, density=True)
    plt.title(
        f"Optimal Policy (Value Iteration)\n"
        f"[mean: {mean_goals:.2f}, stdev: {std_goals:.2f}, mean#steps: {mean_steps:.1f}]",
        fontsize=8)
    plt.xlabel('Number of Episodes Reaching Goal (out of 10,000)', fontsize=8)
    plt.ylabel('Density', fontsize=8)
    plt.show()


def main():
    # TODO: feel free to change this as required
    # TODO: also, check tutorial.py for some hints on how to implement your experiments
    # part_one()
    part_two()

if __name__ == "__main__":
    main() 