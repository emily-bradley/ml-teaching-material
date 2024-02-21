"""
This is the Agent class.
Agents compete in Gamble rounds. Gamble is a game, involving a set of Agents and a
set of Bandits. Agents choose which bandit to pull each round, and are
rewarded according to the reward probability of the selected Bandit. The
Agent with the most rewards at the end of the game (when all rounds are
complete) is the winner.
"""
from abc import ABC, abstractmethod
import random


class Agent(ABC):
    """
    An agent does not know the number of other players (agents)
    Each Agent begins with a reward balance of 0 dollars. The objective is to finish
    the game with as high a balance as possible, or in other words, to minimize
    regret. Regret is the difference between all rewards possible in the game
    and the reward amount the agent received.

    :ivar list bandits: list of bandit objects
    :ivar int num_rounds: number of rounds in the game
    """

    def __init__(self, bandits, num_rounds):
        """ Constructor method
        """
        # list of bandit objects
        self.bandits = bandits
        # total number of rounds to be played
        self.num_rounds = num_rounds
        # Goal is to maximize rewards
        self.rewards = 0

    def __repr__(self):
        return str(self.rewards)

    def __str__(self):
        agent_str = ""
        for i in range(0, self.bandits):
            agent_str += f"Total Rewards: {self.rewards}, " \
                         f"Rounds Remaining: ${self.num_rounds})\n"
        return agent_str

    @property
    @abstractmethod
    def name(self):
        """
        The name of the agent instance. This string value is displayed in the competition leaderboard and graph

        :return: *string* that represents the name of the agent
        """
        raise NotImplementedError

    @abstractmethod
    def action(self):
        """
        The strategy the agent uses to take choose a bandit; Returns a bandit object.

        :return: *Bandit* choosen for this round
        """
        return NotImplementedError

    @abstractmethod
    def update(self, bandit, reward):
        """
        The method to update the agent's strategy, given a selected bandit and the reward amount.
        The update method is called following the action method.
        """
        return NotImplementedError


class ExampleAgent(Agent):
    """
    A simple example that pulls the same arm every time, selected randomly at initialization
    """
    name = "StaticSelectionAgent"

    def __init__(self, bandits, num_rounds):
        super().__init__(bandits, num_rounds)
        self.static_selection = random.choice(self.bandits)

    def action(self):
        return self.static_selection

    def update(self, bandit, reward):
        pass


class ExampleAgent2(Agent):
    """
    A simple example that pulls and arm at random
    """
    name = 'RandomSelectionAgent'

    def __init__(self, bandits, num_rounds):
        super().__init__(bandits, num_rounds)

    def action(self):
        bandit_choice = random.choice(self.bandits)
        return bandit_choice

    def update(self, bandit, reward):
        pass