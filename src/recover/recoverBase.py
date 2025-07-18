class RecoverBase:
    def __init__(self,
                 test_dataset,
                 global_model,
                 clients_pool,
                 old_global_models,
                 old_client_models,
                 select_info,
                 malicious_clients,
                 loss_function):
        self.test_dataset = test_dataset
        self.global_model = global_model
        self.clients_pool = clients_pool
        self.old_global_models = old_global_models
        self.old_client_models = old_client_models
        self.select_info = select_info
        self.malicious_clients = malicious_clients
        self.loss_function = loss_function
        self.time_cost = 0

    def remove_malicious_clients(self, clients_id: list, old_CM=None):
        remaining_clients_id = []
        remaining_clients_models = []
        for i, c_id in enumerate(clients_id):
            if c_id not in self.malicious_clients:
                remaining_clients_id.append(c_id)
                if old_CM is not None:
                    remaining_clients_models.append(old_CM[i])

        return remaining_clients_id, remaining_clients_models
