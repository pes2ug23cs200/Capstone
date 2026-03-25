import flwr as fl
import numpy as np

def krum(results, num_malicious=1):

    weights = [
        fl.common.parameters_to_ndarrays(fit_res.parameters)
        for _, fit_res in results
    ]

    num_clients = len(weights)
    scores = []

    for i in range(num_clients):
        distances = []

        for j in range(num_clients):
            if i != j:
                dist = sum(
                    np.linalg.norm(w1 - w2)
                    for w1, w2 in zip(weights[i], weights[j])
                )
                distances.append(dist)

        distances.sort()

        score = sum(distances[:num_clients - num_malicious - 2])
        scores.append(score)

    best_idx = int(np.argmin(scores))

    return weights[best_idx]


class KrumStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):

        if not results:
            return None, {}

        # apply Krum
        aggregated_weights = krum(results, num_malicious=1)

        return fl.common.ndarrays_to_parameters(aggregated_weights), {}


strategy = KrumStrategy()

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)