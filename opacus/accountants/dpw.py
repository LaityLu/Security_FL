from typing import Any, List, Mapping, TypeVar, Optional, Union

from .accountant import IAccountant

T_state_dict = TypeVar("T_state_dict", bound=Mapping[str, Any])

"""modified by LuDai."""


class AccountantWrapper(IAccountant):

    def __init__(self, accountants: List[IAccountant], n_groups: int):
        """
        This is a wrapper around multiple accountants which are supposed to
        correspond to a privacy group (data points of training data who share
        the same privacy budget). The groups are supposed to be in ascending
        order in terms of their budgets.
        """
        self.n_groups = n_groups
        self.nm_scalars = [1.0] * n_groups
        self.sr_scalars = [1.0] * n_groups
        self.accountants = accountants
        self.history = []  # used to check if privacy histories were updated

    def step(self, *, noise_multiplier: float, sample_rate: float):
        pass

    def get_epsilon(self, delta: float,
                    **kwargs) -> List[float]:
        """
        Returns the expended privacy costs epsilon of all privacy groups.
        """
        return [accountant.get_epsilon(delta=delta, **kwargs)
                for group, accountant in enumerate(self.accountants)]

    def __len__(self) -> int:
        return len(self.accountants[0].history)

    @classmethod
    def mechanism(cls) -> str:
        return "dpw"
