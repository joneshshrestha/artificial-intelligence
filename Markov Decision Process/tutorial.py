import sys
import math

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


"""
    Function to create a fixed deterministic policy
"""
def generate_random_policy(num_actions, num_states, seed=None):
    """
    A policy is a 1D array of length # of states, where each element is a
    number between 0 (inclusive) and # of actions (exclusive) randomly chosen.
    If a specific seed is passed, the same numbers are genereated, while
    if the seed is None, the numbers are unpredictable every time.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=num_actions, size=num_states)


"""
    Function to run one experiment for a given number of episodes
"""
def run_one_experiment(env, policy, num_episodes, display=False):
    """
    Run one experiment, when agent follows a policy, for a given number of episodes.
    """
    # Count the number of goals made and getting stuck in a hole
    goals = 0
    holes = 0
    # Total rewards and steps
    total_rewards = 0
    total_goal_steps = 0

    for _ in range(num_episodes):
        # For each time,
        env.reset()
        done = False
        rewards = 0
        steps = 0

        if display:
            episode = [(env.get_wrapper_attr("s"),)]  # initial state (in a tuple)
        else:
            episode = None

        while not done:
            # choose the action based on the policy
            state = env.get_wrapper_attr("s")
            action = policy[state]

            # take the action
            next_state, reward, done, info, p = env.step(action)
            steps += 1

            # extend the episode
            if display:
                episode.append(tuple([action, next_state]))

            # accumulate rewards
            rewards += reward

        # Calculate stats
        total_rewards += rewards
        if reward == 1.0:  # Goal, or env.s == 63
            goals += 1
            total_goal_steps += steps
        else:
            holes += 1

        # Display
        if display:
            print(episode)
            print(env.render())

    # One experiment finished,
    return goals, holes, total_rewards, total_goal_steps


"""
    A utility function to display a 1D array/policy in a 2D array/grid
"""
def display_policy(policy, n_states):
    side = int(math.sqrt(n_states))  # assuming a square
    policy = policy.reshape((side, side))
    return policy


def main():
    # Create a FrozenLake 8x8 environment using Gymnasium
    # (https://gymnasium.farama.org/environments/toy_text/frozen_lake/).
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="ansi")

    # Currently env is a wrapper for the environment (class = TimeLimit)
    print("type of env")
    print(type(env))
    # ... within that wrapper, there is another wrapper (class = OrderEnforcing)
    print("type of env.env")
    print(type(env.env))
    # ... within that one ... there is yet another layer (class = PassiveEnvChecker)
    print("type of env.env.env")
    print(type(env.env.env))
    # ... and then we get to our frozen lake env (class = FrozenLakeEnv)
    print("type of env.env.env.env")
    print(type(env.env.env.env))


    # Reset the environment and display it (in ansi ascii)
    env.reset()
    print(env.render())  # wrap render() in print()

    # Alternatively you can create a random map, as described in
    # the FrozenLake documentation page.  It's commented out for now.
    # from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    # env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8))

    # Make one (random) action
    action = env.action_space.sample()
    print("Sampling one action:")
    print(action)
    print("Applying the action:")
    env.step(action)
    print(env.render())
    print("Current state after one action:")
    print(env.get_wrapper_attr("s"))

    # general info about the environment ...
    nS = env.observation_space.n  # number of states -- 8x8=64
    nA = env.action_space.n  # number of actions -- four directions; 0:left, 1:down, 2:right, 3:up
    print(f"\nnumber of states: {nS}\nnumber of actions: {nA}")

    # Note that actions are 0 - based integers.You can check in the Gymnasium source code:
    # https://github.com/Farama-Foundation/Gymnasium/blob/d71a13588266256a4c900b5e0d72d10785816c3a/gymnasium/envs/toy_text/frozen_lake.py
    # 0: Move left
    # 1: Move down
    # 2: Move right
    # 3: Move up

    # All environment's probabilities are stored in 'P'.
    # It is a dictionary, keyed by the state index (e.g. env.P[0], env.P[1] etc.)
    # - Then for each state, the value is a dictionary, keyed by the actions (0-based).
    #   - Then for each action, the value is a list of tuples (p, s', r, T), showing:
    #      * p  = the probability of transitioning into the next state
    #      * s' = the index of the next state,
    #      * r  =  reward,
    #      * T  = is terminal, True/False (done=True if the next state is a Hole or the Goal).

    # we can access P[0] using
    print("\nProbabilities for each action from state 0")
    print("... using the wrapper function on the TimeLimit object directly ...")
    P_v1 = env.get_wrapper_attr("P")
    print(P_v1[0])
    # or we can simply do some unwrapping ...
    print("... unwrapping the FrozenLake Environment object...")
    frozen_lake_env = env.unwrapped
    P_v2 = frozen_lake_env.P
    print(P_v2[0])

    # Note on the environment (from the Gymnasium page)
    # is_slippery=True: If true the player will move in intended direction with probability of 1/3 else
    # will move in either perpendicular direction with equal probability of 1/3 in both directions.

    # For example, if action is left and is_slippery is True, then:

    # P(move left)=1/3
    # P(move up)=1/3
    # P(move down)=1/3

    # Probabilities from State 62 (left of the Goal state).
    # Notice some 'True' results (implying the goal is reached).
    print("\nProbabilities for each action from state 62")
    print(P_v1[62])

    # some auxiliary functions
    # 1) create a random policy
    policy = generate_random_policy(4, nS, seed=2024)
    # 2) display policy
    print("\n")
    print("Random policy")
    print(display_policy(policy, nS))
    # 3) Running one experiment ...
    print("\nRunning one experiment with this random policy (5 runs with display)")
    num_episodes = 5
    display = True
    goals, holes, total_rewards, total_goal_steps = run_one_experiment(env, policy, num_episodes, display)

    print("\nRunning one experiment with this random policy (1000 runs without display)")
    num_episodes = 1000
    display = False
    goals, holes, total_rewards, total_goal_steps = run_one_experiment(env, policy, num_episodes, display)

    percent_goal = goals / num_episodes
    percent_hole = holes / num_episodes
    mean_reward = total_rewards / num_episodes
    mean_goal_steps = 0.0 if (goals == 0) else (total_goal_steps / goals)

    print("\n*** RESULTS ***:")
    print(f"\tGoals: {goals:>5d}/{num_episodes} = {percent_goal:>7.3%}")
    print(f"\tHoles: {holes:>5d}/{num_episodes} = {percent_hole:>7.3%}")
    print(f"\tmean reward:          {mean_reward:.5f}")
    print(f"\tmean goal steps:     {mean_goal_steps:.2f}")



if __name__ == "__main__":
    main()