"""
Implementation of Conditional Choice Probability Inverse Reinforcement Learning

Shinnosuke Usami, 2019
susami@andrew.cmu.edu
"""

import numpy as np

def convert_trans_prob_to_per_action_matrix(trans_prob):
    """Convert transition probability to per action matrix of size NxN
       This method just transposes the matrix so that you can iterate through
       actions.

    Args:
        trans_prob ([NxAxN]): T(x, a, x') = p(x' | x, a)

    Returns:
        Numpy array of shape AxNxN
    """
    return np.transpose(trans_prob, [1, 0, 2])


def is_approx_equal(a, b, eps=1e-6):
    return a >= (b - eps) and a <= (b + eps)


