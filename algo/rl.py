"""
Abstract Class for Reinforcement Learning

Shinnosuke Usami, 2019
susami@andrew.cmu.edu
"""

from abc import ABCMeta, abstractmethod

class RL(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def optimal_value(self, reward, threshold=1e-2):
        pass

    @abstractmethod
    def optimal_policy(self, value, stochastic=True):
        pass

