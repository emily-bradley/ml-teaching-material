"""
This is the Auction module. This module contains the User Class and the Auction
class. An Auction is A game, involving a set of Bidders and a set of Users.
Bidders place bids and the winning bidder gets to show their ad to the User.
"""

import random
import numpy as np
import matplotlib.pyplot as plt


class User:
    """
    Each user has a user_id. Each user has a secret probability of clicking on
    an ad. The events of clicking on each ad are mutually independent. When a
    user is created, the secret probability is drawn from a uniform
    distribution from 0 to 1.
    Attributes
    ----------
    __probability : float
        probability attribute to represent the probability of clicking on
        an ad.
    Methods
    -------
    show_ad()
        Represents showing an ad to this User. Returns True if the user
        clicks on the ad.
    """

    def __init__(self):
        # probability attribute to represent the probability of clicking on
        # an ad.
        self.__probability = np.random.uniform(0, 1)

    def __repr__(self):
        return str(self.__probability)

    def __str__(self):
        return f"Probability of response: {self.__probability}"

    def show_ad(self):
        """
        Represents showing an ad to this User. Returns True if the user
        clicks on the ad.
        Returns
        -------
        bool
            True if a user clicks on the ad
        """
        # Returns true if random value is < the user's probability
        if np.random.uniform(0, 1) < self.__probability:
            return True
        return False


class Auction:
    """
    A game, involving a set of Bidders and a set of Users. Each round
    represents a user navigating to a site for an ad. Bidders will place bids
    and the winning bidder gets to show their ad to the User and see the
    results (clicked or did not click). The highest bidder wins but the price
    paid is the second-highest bid
    Attributes
    ----------
    users : list
        list of all User objects
    bidders : list
        list of all bidder objects
    balances : dict
        dictionary of the current balance of every Bidder; initially 0
    balances_per_round : list of lists
        The balances each round
    Methods
    -------
    execute_round()
        Executes all steps within a single round of the game
    plot_history()
        creates a visual representation of how the auction has proceeded
    """

    def __init__(self, users, bidders):
        # list of all User objects.
        self.users = users
        # list of all Bidder objects.
        self.bidders = bidders
        # dictionary of the current balance of every Bidder; initially 0
        self.balances = dict.fromkeys(np.arange(0, len(self.bidders)), 0)
        # list of the balances each round for plotting
        self.balances_per_round = [[] for i in self.bidders]

    def __repr__(self):
        return str(self.balances)

    def __str__(self):
        auction_str = "Auction balances: \n"
        for bidder, balance in self.balances.items():
            auction_str += f"Bidder {bidder}: ${balance}\n"
        return auction_str

    def execute_round(self):
        """
        The Auction occurs in rounds, and the total number of rounds is
        num_rounds . In each round, a second-price auction is conducted for a
        randomly chosen User.
        Each round represents an event in which a User navigates to a website
        with a space for an ad. Bidders place bids and the winner gets to show
        their ad to the user. The user may clock on the add or not click on
        the ad and the winning Bidder gets to observe the behavior.
        """
        # step 1: select a User at random
        round_user = random.choice(self.users)
        round_user_index = self.users.index(round_user)

        # step 2: call all bidders to bid with User and collect bids
        bids_list = []
        for bidder in self.bidders:
            bids_list.append(bidder.bid(user_id=round_user_index))

        # step 3: select auction winner
        # find the index at the highest bid
        winning_bidder = np.argmax(bids_list)
        # check for tie first place bids
        tie_winners = [i for i, e in enumerate(bids_list)
                            if e == bids_list[winning_bidder]
                            and i != winning_bidder]
        # if tie bids
        if tie_winners:
            # In the event that more than one Bidder ties for the highest bid,
            #  one of the highest Bidders is selected at random, each with
            #  equal probability.
            tie_winners.append(winning_bidder)
            winning_bidder = random.choice(tie_winners)

        # step 4: find the winning price (second highest bid)
        # take the max
        # In the event of a tie, highest price is max bid
        if tie_winners:
            winning_price = max(bids_list)
        # if there is only one bidder, winning_price is the bidder's bid
        elif len(bids_list) == 1:
            winning_price = bids_list[winning_bidder]
        # otherwise, the second place bid wins
        else:
            winning_price = max(n for n in bids_list
                                if n != bids_list[winning_bidder])

        # step 5: show the user the ad and retrieve results
        user_click = round_user.show_ad()

        # step 6: notify each bidder if they won or not and what the winning
        # price was
        for idx, bidder in enumerate(self.bidders):
            if idx == winning_bidder:
                # send the winning bidder the results
                bidder.notify(auction_winner=True, price=winning_price,
                              clicked=user_click)
            else:
                # otherwise, update other bidders
                bidder.notify(auction_winner=False, price=winning_price,
                              clicked=None)

        # step 7: update bidder balance
        # Balance is increased by 1 dollar if the User clicked (0 dollars if
        # the user did not click).
        if user_click:
            self.balances[winning_bidder] += 1
        # Balance is decreased by the winning price (whether or not clicked).
        self.balances[winning_bidder] -= winning_price

        # If a Bidder's balance goes below -1000 dollars, then the Bidder will
        # be disqualified from the Auction and further bidding.
        if self.balances[winning_bidder] < -1000:
            # remove the bidder
            del self.bidders[winning_bidder]
            # remove from balances as well
            del self.balances[winning_bidder]

        # step 8: update data for plotting
        for i in range(0, len(self.balances)):
            self.balances_per_round[i].append(self.balances[i])

    def plot_history(self):
        """
        # (optional) creates a visual representation of how the auction has
        # proceeded.
        # There is a problem with the autograder that it cannot import
        # matplotlib so comment out before submitting
        """
        # plot the data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # the x axis is the number of rounds
        rounds = np.arange(0, len(self.balances_per_round[0]))

        # plot
        for idx, balances in enumerate(self.balances_per_round):
            plt.plot(rounds, balances, label=f"Bidder {idx+1}")

        # include the legend
        plt.legend()

        # labels
        plt.xlabel("Number of Rounds")
        plt.ylabel("Bidder Balances")
        ax.set_title('Bidder Balances Over Time')

        # display the plot
        plt.show()
