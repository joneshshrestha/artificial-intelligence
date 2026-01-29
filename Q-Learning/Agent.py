"""
Jonesh Shrestha (2222011)
CSC 580: Artificial Intelligence II (Winter 2026)
HW#2 "Agent.py" -- Class Agent, which performs Temporal Difference (TD) Q-Learning.
"""

import random
import numpy as np
import csv  # for writing and reading the q-table


class Agent:
    """
    An AI agent which controls the snake's movements.
    """

    def __init__(self, env, params):
        self.env = env
        self.action_space = env.action_space  # 4 actions for SnakeGame
        self.state_space = env.state_space  # 12 features for SnakeGame
        self.gamma = params["gamma"]
        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        # dictionary data structure to hold the Q table and initialize it
        self.Q = {}

    @staticmethod
    def state_to_int(state_list):
        """Map state as a list of binary digits, e.g. [0,1,0,0,1,1,1] to an integer."""
        return int("".join(str(x) for x in state_list), 2)

    @staticmethod
    def state_to_str(state_list):
        """Map state as a list of binary digits, e.g. [0,1,0,0,1,1,1], to a string e.g. '0100111'."""
        return "".join(str(x) for x in state_list)

    @staticmethod
    def binstr_to_int(state_str):
        """Map a state binary string, e.g. '0100111', to an integer."""
        return int(state_str, 2)

    # (A)
    def init_state(self, state):
        """Initialize the state's entry in state_table and Q, if anything needed at all."""
        s = self.state_to_str(state)
        if s not in self.Q:
            self.Q[s] = np.zeros(self.action_space)

    # (A)
    def select_action(self, state):
        """
        Do the epsilon-greedy action selection. Note: 'state' is an original list of binary digits.
        It should call the function select_greedy() for the greedy case.
        """
        s = self.state_to_str(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return self.select_greedy(state)

    # (A)
    def select_greedy(self, state):
        """
        Greedy choice of action based on the Q-table.
        """
        self.init_state(state)
        s = self.state_to_str(state)

        q_values = self.Q[s]
        max_q_value = np.max(q_values)
        max_q_actions = np.where(q_values == max_q_value)[0]
        return np.random.choice(max_q_actions)

    # (A)
    def update_Qtable(self, state, action, reward, next_state):
        """
        Update the Q-table (and anything else necessary) after an action is taken.
        Note that both 'state' and 'next_state' are an original list of binary digits.
        """
        # initialize the current and next states
        self.init_state(state)
        self.init_state(next_state)

        # convert the states to integers
        s = self.state_to_str(state)
        s_next = self.state_to_str(next_state)

        # get the q-value of the current state and the max q-value of the next state
        current_q_value = self.Q[s][action]
        next_max_q_value = np.max(self.Q[s_next])
        # calculate the target q-value
        target_q_value = reward + self.gamma * next_max_q_value

        # update the q-value of the current state and action using the Q-learning formula
        self.Q[s][action] = current_q_value + self.alpha * (
            target_q_value - current_q_value
        )

        # update the epsilon at the end
        self.adjust_epsilon()

    # (A)
    def num_states_visited(self):
        """Returns the number of unique states visited. Obtain from the Q table."""
        return len(self.Q)

    # (A)
    def write_qtable(self, filepath):
        """Write the content of the Q-table to an output file."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            for s in sorted(self.Q.keys()):
                for a in range(self.action_space):
                    writer.writerow([s, a, self.Q[s][a]])

    # (A)
    def read_qtable(self, filepath):
        """Read in the Q table saved in a csv file."""
        self.Q = {}
        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                s = row[0]
                a = int(row[1])
                q = float(row[2])
                if s not in self.Q:
                    self.Q[s] = np.zeros(self.action_space)
                self.Q[s][a] = q

    def adjust_epsilon(self):
        """Implements the epsilon decay."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
