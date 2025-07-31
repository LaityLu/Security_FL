import math

import numpy as np


def utility(client, beta):
    loss_list = client.get_loss()
    budget = client.remaining_budget
    # utility = \sqrt{\sum_{i=1}^{size} loss_i^2 /size}*size, loss_list is a tensor which shape is [size]
    utility = (1 - beta) * math.sqrt(sum([loss ** 2 for loss in loss_list]) / len(loss_list)) * len(loss_list) \
              + beta * budget

    return utility


class OortSelector:
    def __init__(self,
                 client_pools,
                 frac,
                 select_client_ratio,
                 explore_rate,
                 explore_rate_min,
                 decay_factor,
                 cut_off_util,
                 beta):
        self.client_pools = client_pools
        self.frac = select_client_ratio
        self.N = len(client_pools * frac)
        self.K = int(select_client_ratio * self.N)
        self.candidates = []
        self.score = {}
        self.utilities = []

        self.explore_rate = explore_rate
        self.explore_rate_min = explore_rate_min
        self.decay_factor = decay_factor
        self.cut_off_util = cut_off_util
        self.beta = beta

    def feedback(self, candidates):
        # count the oort utility of each winner
        self.score = {}
        self.utilities = []
        self.candidates = candidates
        for idx in self.candidates:
            u = utility(self.client_pools[idx], self.beta)
            self.score[idx] = u
            self.utilities.append([idx, u])

    def select(self):

        # update the explore_rate
        self.explore_rate = max(self.explore_rate_min, self.explore_rate * self.decay_factor)

        # sort the utilities ascending
        self.utilities.sort(key=lambda x: x[1], reverse=True)

        # -----------------exploit----------------------
        exploit_len = int(len(self.utilities) * (1 - self.explore_rate))

        # to ensure oort can convergence to the final set
        if exploit_len == self.K:
            return self.candidates

        # cut off the util
        cut_off_util = self.utilities[exploit_len][1] * self.cut_off_util

        picked_clients = []
        for idx, u in self.utilities:
            if u < cut_off_util:
                break
            picked_clients.append(idx)

        # sample with probability
        total_utility = max(1e-4, float(sum([self.score[index] for index in picked_clients])))
        picked_clients = list(
            np.random.choice(picked_clients, exploit_len,
                             p=[self.score[index] / total_utility for index in picked_clients], replace=False)
        )

        # -----------------explore----------------------
        exlore_len = self.K - exploit_len
        explore_clients = list(set(range(self.N)) - set(self.candidates))
        picked_clients.extend(list(np.random.choice(explore_clients, exlore_len, replace=False)))

        self.candidates = picked_clients
        return self.candidates
