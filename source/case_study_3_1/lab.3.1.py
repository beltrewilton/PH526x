import numpy as np
import scipy.stats as stats
import random


def distance(p1, p2):
    """
    Find the distance between p1 & p2
    :param p1:
    :param p2:
    :return:
    """
    return np.sqrt(np.power(p2 - p1, 2))


p1 = np.array([3, 6])
p2 = np.array([9, 2])

r = distance(p1, p2)

# print(r)


def majority_vote(votes):
    """
    Return the most common element in votes.
    :param votes:
    :return:
    """
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1

    winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)

    return random.choice(winners)


def majority_vote_short(votes):
    """
    Rerturn the most commond element in votes (scipy version)
    :param votes:
    :return:
    """
    mode, count = stats.mode(votes)
    return mode


votes = [1,2,3,1,2,3,1,2,3,3,3,3,2,2,2]
r = majority_vote(votes)
r2 = majority_vote_short(votes)
print(r, r2)