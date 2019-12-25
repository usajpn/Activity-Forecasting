"""
Abstract class for inverse reinforcement learning

Shinnosuke Usami, 2019
susami@andrew.cmu.edu
"""

from abc import ABCMeta, abstractmethod

class IRL(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def recover_reward(self):
        pass

