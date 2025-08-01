#!/usr/bin/env python3
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

from typing import List

import numpy as np
import torch
from ..accountants import FedRDPAccountant
from torch.utils.data import Sampler


class UniformWithReplacementSampler(Sampler[List[int]]):
    r"""
    This sampler samples elements according to the Sampled Gaussian Mechanism.
    Each sample is selected with a probability equal to ``sample_rate``.
    The sampler generates ``steps`` number of batches, that defaults to 1/``sample_rate``.
    """

    def __init__(
            self, *, num_samples: int, sample_rate: float, generator=None, steps=None
    ):
        r"""
        Args:
            num_samples: number of samples to draw.
            sample_rate: probability used in sampling.
            generator: Generator used in sampling.
            steps: Number of steps (iterations of the Sampler)
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.generator = generator

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

        if steps is not None:
            self.steps = steps
        else:
            self.steps = int(1 / self.sample_rate)

    def __len__(self):
        return self.steps

    def __iter__(self):
        num_batches = self.steps
        while num_batches > 0:
            mask = (
                    torch.rand(self.num_samples, generator=self.generator)
                    < self.sample_rate
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            yield indices

            num_batches -= 1


class DistributedUniformWithReplacementSampler(Sampler):
    """
    Distributed batch sampler.

    Each batch is sampled as follows:
        1. Shuffle the dataset (enabled by default)
        2. Split the dataset among the replicas into chunks of equal size
           (plus or minus one sample)
        3. Each replica selects each sample of its chunk independently
           with probability `sample_rate`
        4. Each replica outputs the selected samples, which form a local batch

    The sum of the lengths of the local batches follows a Poisson distribution.
    In particular, the expected length of each local batch is:
    `sample_rate * total_size / num_replicas`
    """

    def __init__(
            self,
            *,
            total_size: int,
            sample_rate: float,
            shuffle: bool = True,
            shuffle_seed: int = 0,
            steps: int = None,
            generator=None,
    ):
        """

        Args:
            total_size: total number of samples to sample from
            sample_rate: number of samples to draw.
            shuffle: Flag indicating whether apply shuffle when dividing elements
                between workers
            shuffle_seed: Random seed used to shuffle when dividing elements across workers
            generator: torch.Generator() object used as a source of randomness
                when selecting items for the next round on a given worker
        """
        self.total_size = total_size
        self.sample_rate = sample_rate
        self.generator = generator
        self.num_replicas = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.epoch = 0
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        if self.total_size <= 0:
            raise ValueError(
                "total_size should be a positive integer "
                "value, but got total_size={}".format(self.total_size)
            )

        # Size of the local dataset specific to the current replica
        self.num_samples = self.total_size // self.num_replicas
        if self.rank < self.total_size % self.num_replicas:
            # The first replicas get an extra datapoint if necessary (balanced)
            self.num_samples += 1

        # Number of batches: same as non-distributed Poisson sampling, but each batch is smaller
        if steps is not None:
            self.num_batches = steps
        else:
            self.num_batches = int(1 / self.sample_rate)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.shuffle_seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g)  # type: ignore
        else:
            indices = torch.arange(self.total_size)  # type: ignore

        # Subset of the dataset assigned to this replica
        # NOTE: the first replicas might have 1 more sample.
        # (Different from the regular distributed loader that pads with more samples)
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        # Now, select a batch with Poisson subsampling
        for _ in range(self.num_batches):
            mask = (
                    torch.rand(self.num_samples, generator=self.generator)
                    < self.sample_rate
            )
            selected_examples = mask.nonzero(as_tuple=False).reshape(-1)
            if len(selected_examples) > 0:
                yield indices[selected_examples]

    def __len__(self) -> int:
        """
        Expected number of batches.
        """
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class NonUniformWithReplacementSampler(Sampler[List[int]]):
    r"""
    This sampler samples elements according to the Sampled Gaussian Mechanism.
    Each sample is selected with a probability equal to ``sample_rate_i``.
    """

    def __init__(
            self, *, num_samples: int, sample_rate: float, generator=None, steps=None
    ):
        r"""
        Args:
            num_samples: number of samples to draw.
            sample_rate: probability used in sampling.
            generator: Generator used in sampling, default: RandomGenerator in
            steps: Number of steps (iterations of the Sampler)
        """

        self.num_samples = num_samples
        self.sample_rate = np.full(num_samples, sample_rate)
        self.generator = generator

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

        if steps is not None:
            self.steps = steps
        else:
            self.steps = 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        num_batches = self.steps
        while num_batches > 0:
            mask = (
                    torch.rand(self.num_samples, generator=self.generator)
                    < torch.as_tensor(self.sample_rate)
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            yield indices

            num_batches -= 1

