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

from typing import Optional

import numpy as np

from . import create_accountant

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
    pre = 0
    for i in range(size):
        # random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]) # 掷骰子（多项式分布）
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
