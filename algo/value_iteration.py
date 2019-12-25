"""
Implementation of Value Iteration

Shinnosuke Usami, 2019
susami@andrew.cmu.edu
"""

from .rl import RL

import numpy as np

class ValueIteration(RL):
    def __init__(self, n_states, n_actions, trans_prob, discount):
        """Set up environment parameters for MDP

        Args:
            n_states (int): Number of states.
            n_actions (int): Number of actions.
            trans_prob ()
            discount (float): Discount factor.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.trans_prob = trans_prob
        self.discount = discount


    def optimal_value(self, reward, threshold=1e-2):
        """Find optimal value function given the reward function.

        Args:
            reward ([N]): Reward vector for each state.
            threshold (float): threshold to stop value iteration.

        Returns:
            value ([N]): N states vector of the value function.
        """

        value = np.zeros(self.n_states)
        n_iter = 0

        diff = float("inf")

        while diff > threshold:
            diff = 0
            for state in range(self.n_states):
                max_value = float("-inf")
                for action in range(self.n_actions):
                    tp = self.trans_prob[state, action, :]
                    max_value = max(max_value,
                                    np.dot(tp, reward + self.discount * value))
                new_diff = abs(value[state] - max_value)
                if new_diff > diff:
                    diff = new_diff
                value[state] = max_value
            n_iter += 1

        return value

    #TODO(Shin) understand better about this
    def optimal_policy(self, value, reward, stochastic=True, temperature=1.0):
        """Find optimal policy given the optimal value function.

        Args:
            value ([N]): N states vector of the value function.
            stochastic (bool): is the policy a stochastic policy?

        Returns:
            policy ([NxA]): N states x A actions matrix of the policy.
        """

        if stochastic:
            # Get Q using equation 9.2 from Ziebart's paper
            Q = np.zeros((self.n_states, self.n_actions))
            for i in range(self.n_states):
                for j in range(self.n_actions):
                    p = self.trans_prob[i, j, :]
                    Q[i, j] = p.dot(reward + self.discount * value)
            Q -= Q.max(axis=1).reshape((self.n_states, 1))
            Q /= temperature
            Q = np.exp(Q) / np.exp(Q).sum(axis=1).reshape((self.n_states, 1))
            return Q

        def _policy(s):
            return max(range(self.n_actions),
                       key = lambda a: reward[s]
                        + sum(trans_prob[s, a, k]
                            * discount * value[k] for k in range(self.n_states)))

        policy = np.array([_policy(s) for s in range(self.n_states)])
        return policy



