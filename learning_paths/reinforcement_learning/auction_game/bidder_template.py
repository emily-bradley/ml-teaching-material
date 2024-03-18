"""
This is the Bidder module. This module contains the Bidder class. Bidders
compete in an Auction. An Auction is A game, involving a set of Bidders and a
set of Users. Bidders place bids and the winning bidder gets to show their
ad to the User.
"""
import numpy as np


class Bidder:
    """
    Each Bidder begins with a balance of 0 dollars. The objective is to finish
    the game with as high a balance as possible. At some points during the
    game, the Bidder 's balance may become negative. If you Bidder 's balance
    goes below -1000 dollars then your Bidder will be disqualified from the
    Auction and further bidding.
    Attributes
    ----------
    num_users : int
        The number of User objects in the game
    num_rounds : int
        The number of round to be played
    balance : float
        The current balance of the bidder
    rewards_by_user : list
        List of rewards (number of clicks) by user
    penalties_by_user : list
        List of penalties (missed bids) by user
    last_user : None or int
        The user_id of the most recent user bid on
    Methods
    -------
    bid(user_id)
        Generate a bid given a user_id
    notify(auction_winner, price, clicked)
        Information received after a bid has been placed
    """

    def __init__(self, num_users, num_rounds):
        # the number of User objects in the game
        self.num_users = num_users
        # contains the total number of rounds to be played
        self.num_rounds = num_rounds
        # goal is to maximize balance
        self.balance = 0

        # To begin with, all machines are assumed to have a uniform
        # distribution of the probability of success
        self.rewards_by_user = [.5] * num_users
        self.penalties_by_user = [.5] * num_users
        self.last_user = None

    def __repr__(self):
        return str(self.balance)

    def __str__(self):
        bidder_str = ""
        for i in range(0, self.num_users):
            current_bid = self.rewards_by_user[i] / \
                          (self.rewards_by_user[i] + self.penalties_by_user[i])
            attempts = self.rewards_by_user[i] + self.penalties_by_user[i]
            successes = self.rewards_by_user[i]
            bidder_str += f"User {i}: (Attempts: {attempts}, Successes: " \
                          f"{successes}, Current bid: ${current_bid})\n"
        return bidder_str

    def bid(self, user_id):
        """
        Returns a non-negative amount of money to bid, in dollars.
        Round to three decimal places.
        Params
        -------
        user_id : int
            The id of the user
        Returns
        -------
        float
            The non-negative amount to bid on the ad
        """
        # update performance based on user_id
        self.last_user = user_id

        if (self.rewards_by_user[user_id] + self.penalties_by_user[user_id]) \
                < 10 and self.num_rounds > 250:
            # pay for early information in high volume rounds
            return 1
        # For each observation, we will iterate through each machine and will
        # select the machine with the highest random beta distribution.
        bid_amount = self.rewards_by_user[user_id] \
                     / (self.rewards_by_user[user_id] +
                        self.penalties_by_user[user_id])

        return round(bid_amount, 3)

    def notify(self, auction_winner, price, clicked):
        """
        Update the bidder with information about what happened at the end of
        each Auction round.
        Params
        -------
        auction_winner : bool
            Represent whether the given Bidder won the auction ( True )
            or not ( False )
        price : float
            The amount of the second bid, which the winner pays.
        clicked : bool or None
            If the bidder won the auction, clicked will contain a boolean value
            to represent whether or not the user clicked on the ad. Otherwise
            None.
        """

        # if the bidder won
        if auction_winner:
            # update balance
            # Balance is increased by 1 if the User clicked (0 if not).
            if clicked:
                self.balance += 1
                # update last user
                self.rewards_by_user[self.last_user] += 1
            else:
                # update last user
                self.penalties_by_user[self.last_user] += 1
            # Balance is decreased by the winning price regardless
            self.balance -= price
