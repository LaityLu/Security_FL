from typing import List

import numpy as np


class RandomSelector:
    def __init__(self, num_clients, client_rate):
        self.num_clients = num_clients
        self.client_rate = client_rate

    def select(self) -> List[int]:
        select_clients_id = np.random.choice(range(self.num_clients),
                                             int(self.client_rate * self.num_clients), replace=False).tolist()
        return select_clients_id
