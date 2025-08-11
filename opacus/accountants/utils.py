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
import math
from typing import Optional

import numpy as np

from . import create_accountant
from .fedrdp import compute_privacy_cost_one_step, binary_search_epsilon_g

MAX_SIGMA = 1e6


def get_noise_multiplier(
        *,
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        epochs: Optional[int] = None,
        steps: Optional[int] = None,
        accountant: str = "rdp",
        epsilon_tolerance: float = 0.01,
        **kwargs,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    if (steps is None) == (epochs is None):
        raise ValueError(
            "get_noise_multiplier takes as input EITHER a number of steps or a number of epochs"
        )
    if steps is None:
        steps = int(epochs / sample_rate)

    eps_high = float("inf")
    accountant = create_accountant(mechanism=accountant)

    sigma_low, sigma_high = 0, 10
    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        accountant.history = [(sigma_high, sample_rate, steps)]
        eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        accountant.history = [(sigma, sample_rate, steps)]
        eps = accountant.get_epsilon(delta=target_delta, **kwargs)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high


def get_privacy_spent(privacy_costs, deltas, delta_g, eta):
    """ compute the total privacy cost from the beginning """
    eps_mean = sum(privacy_costs) / len(privacy_costs)
    beta = eta / (len(privacy_costs) * (1 + eps_mean) + 1)
    eps_0 = np.log(1 + beta)
    a = []
    eps_pie = []
    for eps_i in privacy_costs:
        a_i = math.ceil(eps_i * (1 / beta + 1))
        eps_i_pie = eps_0 * a_i
        a.append(a_i)
        eps_pie.append(eps_i_pie)
    a_g, epsilon_g = binary_search_epsilon_g(eps_0, len(privacy_costs),
                                             eps_pie, deltas, delta_g, a)
    return a_g, epsilon_g


def compute_privacy_cost_all_step(rounds,
                                  steps,
                                  recover_rounds,
                                  recover_steps,
                                  initial_sigma,
                                  sample_rate,
                                  delta,
                                  noise_config):
    privacy_costs = []
    deltas = []
    if noise_config['type'] == 'constant':
        eps, delta = compute_privacy_cost_one_step(initial_sigma, sample_rate, delta)
        privacy_costs.extend([eps] * (int(rounds * steps + recover_rounds * recover_steps)))
        deltas.extend([delta] * (int(rounds * steps + recover_rounds * recover_steps)))
    elif noise_config['type'] == 'step':
        eps, delta = compute_privacy_cost_one_step(initial_sigma, sample_rate, delta)
        privacy_costs.extend([eps] * (int(rounds * steps)))
        deltas.extend([delta] * (int(rounds * steps)))
        sigma = initial_sigma * noise_config['beta']
        eps, delta = compute_privacy_cost_one_step(sigma, sample_rate, delta)
        privacy_costs.extend([eps] * (int(recover_rounds * recover_steps)))
        deltas.extend([delta] * (int(recover_rounds * recover_steps)))
    elif noise_config['type'] == 'log':
        for i in range(int(rounds * steps + recover_rounds * recover_steps)):
            sigma = initial_sigma / (1 + noise_config['decay_rate'] * np.log(i + 1))
            eps, delta = compute_privacy_cost_one_step(sigma, sample_rate, delta)
            privacy_costs.append(eps)
            deltas.append(delta)
    elif noise_config['type'] == 'double_log':
        for i in range(int(rounds * steps + recover_rounds * recover_steps)):
            sigma = initial_sigma / (1 + noise_config['decay_rate'] * np.log(i + 1) * np.log(np.log(i + 2)))
            eps, delta = compute_privacy_cost_one_step(sigma, sample_rate, delta)
            privacy_costs.append(eps)
            deltas.append(delta)
    elif noise_config['type'] == 'inverse':
        for i in range(int(rounds * steps + recover_rounds * recover_steps)):
            sigma = initial_sigma / (1 + i) ** noise_config['decay_rate']
            eps, delta = compute_privacy_cost_one_step(sigma, sample_rate, delta)
            privacy_costs.append(eps)
            deltas.append(delta)
    else:
        raise ValueError("The noise type should be chosen from 'constant','step','log','double_log'")
    return privacy_costs, deltas


def get_noise_multiplier_with_fed_rdp(
        target_epsilon: float,
        rounds: int = 50,
        steps: int = 5,
        recover_rounds: int = 25,
        recover_steps: int = 2,
        sample_rate: float = 0.25,
        delta: float = 0.001,
        delta_g: float = 0.1,
        eta: float = 0.5,
        noise_config: dict = None,
        eps_tolerance: float = 0.1,
        noise_tolerance: float = 0.01
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget at the end of epochs, with a given sample_rate
    """

    eps_high = float("inf")

    sigma_low, sigma_high = 0, 2.5
    if rounds * steps + recover_rounds * recover_steps >= 100:
        sigma_high = 10

    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        privacy_costs, deltas = compute_privacy_cost_all_step(
            rounds=rounds,
            steps=steps,
            recover_rounds=recover_rounds,
            recover_steps=recover_steps,
            initial_sigma=sigma_high,
            sample_rate=sample_rate,
            delta=delta,
            noise_config=noise_config
        )
        _, eps_high = get_privacy_spent(privacy_costs, deltas, delta_g, eta)
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > eps_tolerance and sigma_high - sigma_low > noise_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        privacy_costs, deltas = compute_privacy_cost_all_step(
            rounds=rounds,
            steps=steps,
            recover_rounds=recover_rounds,
            recover_steps=recover_steps,
            initial_sigma=sigma,
            sample_rate=sample_rate,
            delta=delta,
            noise_config=noise_config
        )
        _, eps_g = get_privacy_spent(privacy_costs, deltas, delta_g, eta)

        if eps_g < target_epsilon:
            sigma_high = sigma
            eps_high = eps_g
        else:
            sigma_low = sigma

    return sigma_high


def get_noise_multiplier_with_fed_rdp_recover(
        target_epsilon: float,
        recover_rounds: int = 25,
        recover_steps: int = 2,
        sample_rate: float = 0.25,
        delta: float = 0.001,
        delta_g: float = 0.1,
        eta: float = 0.5,
        noise_config: dict = None,
        eps_tolerance: float = 0.1,
        noise_tolerance: float = 0.01,
        history_privacy_costs: list = None,
        history_deltas: list = None
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget at the end of epochs, with a given sample_rate
    """

    eps_high = float("inf")

    sigma_low, sigma_high = 0, 5

    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        privacy_costs, deltas = compute_privacy_cost_all_step(
            rounds=0,
            steps=0,
            recover_rounds=recover_rounds,
            recover_steps=recover_steps,
            initial_sigma=sigma_high,
            sample_rate=sample_rate,
            delta=delta,
            noise_config=noise_config
        )
        if history_privacy_costs is not None:
            privacy_costs = privacy_costs + history_privacy_costs
            deltas = deltas + history_deltas
        _, eps_high = get_privacy_spent(privacy_costs, deltas, delta_g, eta)
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > eps_tolerance and sigma_high - sigma_low > noise_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        privacy_costs, deltas = compute_privacy_cost_all_step(
            rounds=0,
            steps=0,
            recover_rounds=recover_rounds,
            recover_steps=recover_steps,
            initial_sigma=sigma,
            sample_rate=sample_rate,
            delta=delta,
            noise_config=noise_config
        )
        if history_privacy_costs is not None:
            privacy_costs = privacy_costs + history_privacy_costs
            deltas = deltas + history_deltas
        _, eps_g = get_privacy_spent(privacy_costs, deltas, delta_g, eta)

        if eps_g < target_epsilon:
            sigma_high = sigma
            eps_high = eps_g
        else:
            sigma_low = sigma

    return sigma_high


def MultiLevels(n_levels, ratios, values, size):
    assert abs(sum(ratios) - 1.0) < 1e-6, "the sum of `ratios` must equal to one."
    assert len(ratios) == len(values) and len(
        ratios) == n_levels, "both the size of `ratios` and the size of `values` must equal to the value of `n_levels`."

    target_epsilons = [0] * size
    pre = 0
    for i in range(n_levels - 1):
        l = int(size * ratios[i])
        target_epsilons[pre: pre + l] = [values[i]] * l
        pre = pre + l
    l = size - pre
    target_epsilons[pre:] = [values[-1]] * l
    return np.array(target_epsilons)


def MixGauss(ratios, means_and_stds, size):
    assert abs(sum(ratios) - 1.0) < 1e-6, "the sum of `ratios` must equal to one."
    assert len(ratios) == len(means_and_stds), "the size of `ratios` and `means_and_stds` must be equal."

    target_epsilons = []
    for i in range(size):
        dist_idx = np.argmax(np.random.multinomial(1, ratios))
        value = np.random.normal(loc=means_and_stds[dist_idx][0], scale=means_and_stds[dist_idx][1])
        target_epsilons.append(value)

    return np.array(target_epsilons)


def Gauss(mean_and_std, size):
    return np.random.normal(loc=mean_and_std[0], scale=mean_and_std[1], size=size)


def Pareto(shape, lower, size):
    return np.random.pareto(shape, size) + lower


GENERATE_EPSILONS_FUNC = {
    "ThreeLevels": lambda n, params: MultiLevels(3, *params, n),
    "BoundedPareto": lambda n, params: Pareto(*params, n),
    "BoundedMixGauss": lambda n, params: MixGauss(*params, n),
}
