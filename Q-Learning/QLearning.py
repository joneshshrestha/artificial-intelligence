"""
Jonesh Shrestha (2222011)
Artificial Intelligence II (Winter 2026)
CSC 580 HW#2 "QLearning.py" -- Q-learning for the Snake Game
"""

import matplotlib
import SnakeEnv as snake_env
import Agent as agent_class

import numpy as np
import matplotlib.pyplot as plt
import random
import copy


def q_learning(agent, env, max_steps, train=True):
    """
    This function simulates a RL game, where the agent learns the (hopefully) optimal policy
    by Q-learning.  The parameters 'agent' and 'env' are created in the calling function and
    passed in, while 'max_step' specifies the maximum timesteps to play (Note: continuous
    after failing) and 'train' indicates the run is a training or otherwise (i.e., evaluation).
    Most lines are basic and general, calling functions in the environment or the agent.
    Details depend on the implementations of those components (and their functions).
    """
    # First reset the environment
    state = env.reset()
    agent.init_state(state)  # (A)

    # Initialize some housekeeping variables
    total_return, n_apples, n_stops, n_goodsteps = 0.0, 0, 0, 0
    done = False

    # Play continuously until max_steps.
    for i in range(max_steps):
        agent.init_state(state)  # initialize the state for the agent

        # Select the action to take at this state.
        if train:
            action = agent.select_action(state)  # (A) epsilon greedy selection
        else:
            # action = agent.select_greedy(state)  # (A) greedy selection
            action = agent.select_action(
                state
            )  # epsilon-greedy using learned Q-table for part 2

        # Environment executes the selected action.
        next_state, reward, done, _ = env.step(action)

        # Q-learning if training -- update the Q-table
        if train:
            agent.update_Qtable(state, action, reward, next_state)  # (A)

        # Update to prepare for the next iteration
        state = next_state

        # Accumulate the total return and other counts from this step
        total_return += pow(agent.gamma, i) * reward

        if reward == 10:
            n_apples += 1
        elif reward == 1:
            n_goodsteps += 1
        # The play is continuous, so this condition doesn't make the play terminate,
        # but an episode stops when a snake curls itself or hits a wall.
        elif reward == -100:  # i.e., done
            n_stops += 1
        #
    return (
        total_return,
        n_apples,
        n_stops,
        n_goodsteps,
        agent.num_states_visited(),
    )  # (A)


# Do q_learning for 'num_runs' times.  For each run, 'num_steps' steps is done.
def run_ql(max_runs, max_steps, in_params, qtable_file, display=False, train=False):
    """ """
    num_runs = max_runs
    num_steps = max_steps
    results_list = []
    best_return = float("-inf")
    best_qtable = None

    for run in range(num_runs):
        # reset params
        params = copy.deepcopy(in_params)  # reset the parameters

        # Create an environment and an agent
        env = snake_env.SnakeEnv()
        agent = agent_class.Agent(env, params)

        env.display = display  ## <== display True/False (on/off)

        # If evaluation, read in the given q-table (otherwise q_learning() initializes to small random numbers)
        if not train and qtable_file is not None:
            agent.read_qtable(qtable_file)

        ret = q_learning(
            agent, env, num_steps, train=train
        )  # training=False for evaluation
        results_list.append(ret)

        env.close()  # for each run
        print(
            "* Run {}: Return={:>8.3f}, #Apples={}, #Stops={}, #GoodSteps={}, #UniqueStatesVisited={}".format(
                run, ret[0], ret[1], ret[2], ret[3], ret[4]
            )
        )

        if train:
            if ret[0] > best_return:
                best_return = ret[0]
                best_qtable = agent.Q

    if train:
        agent.Q = best_qtable  # overwrite the agent's last Q table
        agent.write_qtable(qtable_file)  # so that this function can be used

    return results_list


##===================================================
## Call run_ql() for either/both training and evaluation
num_runs = 10  # 10
num_steps = 1000  # 300 #1000

params = dict()
params["gamma"] = 0.95
params["alpha"] = 0.7
params["epsilon"] = 0.2  # exploration probability at start
params["epsilon_min"] = 0.01  # minimum epsilon
params["epsilon_decay"] = 0.995  # exponential decay rate for epsilon

# qtable_file = "qtable_part1_exp3.csv"  # "qtable_true.csv" #None

# Call run_ql() for either training or evaluation
# results_list = run_ql(num_runs, num_steps, params, qtable_file, display=False, train=False)  # evaluation
# results_list = run_ql(num_runs, num_steps, params, qtable_file, display=False, train=True)  # training


# results = np.array(results_list)
# cmean = np.mean(results, axis=0)
# print(
#     "\n** Mean: Return={:>8.3f}, #Apples={}, #Stops={}, #GoodSteps={}, #UniqueStatesVisited={}".format(
#         cmean[0], cmean[1], cmean[2], cmean[3], cmean[4]
#     )
# )


# Part 2: Epsilon Experiment
epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9]

# Using learned Q-table from Part 1 Experiment 2 (num_steps=1000, epsilon=0.6)
qtable_file = "qtable_part1_exp2.csv"

means = []

for eps in epsilon_list:
    params["epsilon"] = eps

    print(f"Epsilon={eps}")

    # same as above, but with different epsilon values
    results_list = run_ql(
        num_runs, num_steps, params, qtable_file, display=False, train=False
    )

    results = np.array(results_list)
    cmean = np.mean(results, axis=0)
    means.append(cmean)

    print(
        "\n** Mean: Return={:>8.3f}, #Apples={}, #Stops={}, #GoodSteps={}, #UniqueStatesVisited={}".format(
            cmean[0], cmean[1], cmean[2], cmean[3], cmean[4]
        )
    )

means = np.array(means)

aspects = [
    "Returns",
    "# of Apples",
    "# of Stops",
    "# of Good Steps",
    "# of States Visited",
]

for i, name in enumerate(aspects):
    plt.figure()
    plt.plot(epsilon_list, means[:, i])
    plt.xlabel("epsilon")
    plt.ylabel(f"Mean {name}")
    plt.title(f"Mean {name} vs epsilon")
    plt.xticks(epsilon_list)
    plt.savefig(
        f"part2_{i}_{name}.png",
        bbox_inches="tight",
    )
    plt.close()

print("\nSaved plots")
