# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union
import numpy as np
import math

from .accountant import IAccountant
from .analysis import rdp as privacy_analysis


def generate_rdp_orders():
    dense = 1.07
    alpha_list = [int(dense ** i + 1) for i in range(int(math.floor(math.log(1000, dense))) + 1)]
    alpha_list = np.unique(alpha_list)
    return alpha_list


def F_dp(r, s, weights, values):
    # Initialize a DP table with (r+1) rows and (s+1) columns
    dp = [[0] * (s + 1) for _ in range(r + 1)]

    # Base case: when no items are selected, the value is 1
    for j in range(s + 1):
        dp[0][j] = 1

        # Fill the DP table
    for i in range(1, r + 1):
        ar = weights[i - 1]
        wr = values[i - 1]
        for j in range(s + 1):
            if ar <= j:
                dp[i][j] = dp[i - 1][j] + wr * dp[i - 1][j - ar]
            else:
                dp[i][j] = dp[i - 1][j]

    return dp[r][s]


def calculate_optcomp(k, epsilons, deltas, a_g, eps_0, delta_g, a_i):
    epsilon_g = a_g * eps_0

    # compute left_side_result
    # Calculate the product (1 + exp(epsilon_i))
    product_term = np.prod([1 + np.exp(e) for e in epsilons])

    # compute F(k,B)
    B = (np.sum(a_i) - a_g) // 2

    values_1 = [np.exp(-e) for e in epsilons]
    F_1 = F_dp(k, B, a_i, values_1)

    values_2 = [np.exp(e) for e in epsilons]
    F_2 = F_dp(k, B, a_i, values_2)

    sum_term = np.prod([np.exp(e) for e in epsilons]) * F_1 - np.exp(epsilon_g) * F_2

    # Divide by product term
    left_side_result = sum_term / product_term

    # compute right_side_result
    # Calculate the product (1 - delta)
    product_term_ = np.prod([1 - d for d in deltas])

    right_side_result = 1 - ((1 - delta_g) / product_term_)

    return left_side_result - right_side_result


def binary_search_epsilon_g(eps_0, k, epsilons, deltas, delta_g, a_i):
    a, b = 1, sum(a_i)  # Initial bounds
    i = 0
    while b >= a:
        i += 1
        m = (a + b) // 2
        result = calculate_optcomp(k, epsilons, deltas, m, eps_0, delta_g, a_i)
        if result < 0:
            b = m - 1
        elif result > 0:
            a = m + 1
        else:
            return m, m * eps_0
    return (a + b) // 2 + 1, ((a + b) // 2 + 1) * eps_0


def compute_privacy_cost_one_step(noise_multiplier, sample_rate, delta,
                                  alphas: Optional[List[Union[float, int]]] = None):
    """ compute the privacy cost of a step """
    if alphas is None:
        alphas = generate_rdp_orders()
    orders_vec = np.atleast_1d(alphas)
    inner_rdp = privacy_analysis.compute_rdp(q=sample_rate, noise_multiplier=noise_multiplier, steps=1,
                                             orders=alphas)
    eps_vec = (
            inner_rdp
            - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
            + np.log((orders_vec - 1) / orders_vec)
    )
    idx_opt = np.nanargmin(eps_vec)
    eps = eps_vec[idx_opt]
    eps = eps * sample_rate
    return eps, delta


class FedRDPAccountant(IAccountant):
    DEFAULT_ALPHAS = generate_rdp_orders()

    def __init__(self):
        super().__init__()

    def init(self,
             budget: float = None,
             total_budgets: List[List[float]] = None,
             sample_rate: float = 1.0,
             eta: float = 0.5,
             delta_g: float = 0.1,
             ):
        self.sample_rate = sample_rate
        self.budget = budget
        self.privacy_costs = []
        self.deltas = []
        # self.total_budgets = total_budgets
        self.eta = eta
        self.delta_g = delta_g

    def step(self, noise_multiplier: float, sample_rate: float):
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if (
                    last_noise_multiplier == noise_multiplier
                    and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))

        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def get_privacy_spent(
            self, *,
            delta: float = 0.001,
            alphas: Optional[List[Union[float, int]]] = None,
    ) -> Tuple[float, int]:

        # compute the privacy cost of a step
        if alphas is None:
            alphas = self.DEFAULT_ALPHAS
        for (noise_multiplier, sample_rate, num_steps) in self.history:
            eps, delta = compute_privacy_cost_one_step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                delta=delta,
                alphas=alphas
            )
            for step in range(num_steps):
                # save the historical privacy cost
                self.privacy_costs.append(eps)
                self.deltas.append(delta)
        self.history = []

        if len(self.privacy_costs) == 1:
            return self.privacy_costs[0], 0

        # compute the total privacy cost from the beginning
        eps_mean = sum(self.privacy_costs) / len(self.privacy_costs)
        beta = self.eta / (len(self.privacy_costs) * (1 + eps_mean) + 1)
        eps_0 = np.log(1 + beta)
        a = []
        eps_pie = []
        for eps_i in self.privacy_costs:
            a_i = math.ceil(eps_i * (1 / beta + 1))
            eps_i_pie = eps_0 * a_i
            a.append(a_i)
            eps_pie.append(eps_i_pie)
        a_g, epsilon_g = binary_search_epsilon_g(eps_0, len(self.privacy_costs), eps_pie, self.deltas, self.delta_g, a)

        return epsilon_g, a_g

    def get_epsilon(
            self, delta: float, alphas: Optional[List[Union[float, int]]] = None, **kwargs
    ):

        eps, _ = self.get_privacy_spent(delta=delta, alphas=alphas)
        return eps

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "fed_rdp"
