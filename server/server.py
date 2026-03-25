import flwr as fl
import numpy as np


def krum(results, num_malicious=1):
    """
    results: list of (parameters, num_examples, metrics)
    """

    weights = [fl.common.parameters_to_ndarrays(r[0]) for r in results]

    num_clients = len(weights)
    distances = np.zeros((num_clients, num_clients))

    # compute pairwise distances
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = sum(
                np.linalg.norm(w1 - w2) for w1, w2 in zip(weights[i], weights[j])
            )
            distances[i][j] = dist
            distances[j][i] = dist

    # score each client
    scores = []
    for i in range(num_clients):
        sorted_dist = np.sort(distances[i])
        score = np.sum(sorted_dist[: num_clients - num_malicious - 1])
        scores.append(score)

    # pick best client
    krum_index = np.argmin(scores)
    return weights[krum_index]


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