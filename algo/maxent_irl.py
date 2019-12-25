"""
Implementation of Maximum Entropy Inverse Reinforcement Learning

Shinnosuke Usami, 2019
susami@andrew.cmu.edu
"""

from itertools import product
import numpy as np

from .irl import IRL
from .value_iteration import ValueIteration

class MaxEntIRL(IRL):
    def __init__(self, n_states, n_actions, discount, trans_prob, feature_mat):
        """
        Args:
            n_states (int): number of states.
            n_actions (int): number of actions.
            discount (float): gamma discount factor.
            n_epochs (int): number of epochs (IRL parameter update).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.trans_prob = trans_prob
        self.feature_mat = feature_mat

        self.rl = ValueIteration(n_states, n_actions, trans_prob, discount)


    def expected_svf(self, reward, trajs, value=None):
        """Compute the expected state visitation frequency according to the
           given reward and trajectories.

        Args:
            reward:
            trajs (Numpy array (N, L, 2)): Trajectories of state, action.

        Return:
            expected state visitation frequency ():
        """
        n_trajs = trajs.shape[0]
        traj_len = trajs.shape[1]

        if value is None:
            value = self.rl.optimal_value(reward)
        policy = self.rl.optimal_policy(value, reward)

        start_state_count = np.zeros(self.n_states)
        for traj in trajs:
            start_state_count[traj[0, 0]] += 1
        p_start_state = start_state_count / n_trajs

        expected_svf = np.tile(p_start_state, (traj_len, 1)).T
        for t in range(1, traj_len):
            expected_svf[:, t] = 0
            for i, j, k in product(range(self.n_states),
                                   range(self.n_actions),
                                   range(self.n_states)):
                expected_svf[k, t] += (expected_svf[i, t - 1] *
                                      policy[i, j] *
                                      self.trans_prob[i, j, k])

        return expected_svf.sum(axis=1)


    def feature_expectation(self, trajs):
        """Compute the feature expectation using the given trajectories.

        Args:
            trajs (Numpy array (N, L, 2)): Trajectories of state, action.

        Returns:
            feature_expectation ():
        """

        feature_expectation = np.zeros(self.feature_mat.shape[1])

        for traj in trajs:
            for state, _, _ in traj:
                feature_expectation += self.feature_mat[state]

        feature_expectation /= traj.shape[0]

        return feature_expectation


    def recover_reward(self, trajs, n_epochs, learning_rate):
        """Compute the reward given trajectories of expert.

        Args:
            trajs (Numpy array (N, L, 2)): Trajectories of state, action.

        Returns:
            reward ():
        """

        # Compute feature expectation from expert trajectory
        feature_expectation = self.feature_expectation(trajs)
        # Init weights
        theta = np.random.normal(loc=0.5, scale=0.2,
                                 size=self.feature_mat.shape[1])

        for epoch in range(n_epochs):
            # 1-Compute parameterized reward function
            reward = self.feature_mat.dot(theta)

            # 2-Compute expected state visitation frequency
            expected_svf = self.expected_svf(reward, trajs)

            # 3-Compute gradient
            grad = feature_expectation - self.feature_mat.T.dot(expected_svf)

            # 4-Update weight
            theta += learning_rate * grad

        return self.feature_mat.dot(theta).reshape((self.n_states,))


