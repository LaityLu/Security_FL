import numpy as np


class OortSelector(BaseSelector):
    def __init__(self, frac, explore_rate, explore_rate_min, decay_factor, cut_off_util):
        super(OortSelector, self).__init__(frac)
        self.winner = []
        self.round = 0

        self.explore_rate = explore_rate
        self.explore_rate_min = explore_rate_min
        self.decay_factor = decay_factor
        self.cut_off_util = cut_off_util

    def feedback(self):
        # count the oort utility of each winner
        self.score = {}
        self.utilities = []
        for idx in self.winner:
            u = oort_utility(self.client_pools[idx])
            self.score[idx] = u
            self.utilities.append([idx, u])

    def select(self):

        # update the round
        self.round = self.round + 1

        # update the explore_rate
        self.explore_rate = max(self.explore_rate_min, self.explore_rate * self.decay_factor)

        # sort the utilities ascending
        self.utilities.sort(key=lambda x: x[1], reverse=True)

        # -----------------exploit----------------------
        exploit_len = int(len(self.utilities) * (1 - self.explore_rate))

        # to ensure oort can convergence to the final set
        if exploit_len == self.K:
            return self.winner

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
        explore_clients = list(set(range(self.N)) - set(self.winner))
        picked_clients.extend(list(np.random.choice(explore_clients, exlore_len, replace=False)))

        self.winner = picked_clients
        return self.winner
