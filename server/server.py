import flwr as fl
import numpy as np
import tensorflow as tf
from shared.ids import detect

# Load model for detection
model = tf.keras.models.load_model("model/model.keras")

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

    return weights[best_idx], best_idx


class KrumStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):

        if not results:
            return None, {}

        # Check messages for malicious content
        for client_id, fit_res in results:
            if 'prediction' in fit_res.metrics and fit_res.metrics['prediction'] == "ATTACK":
                print(f"[Server] ALERT: Malicious message detected by client {client_id}!")
            if 'type' in fit_res.metrics and fit_res.metrics['type'] == "malicious":
                print(f"[Server] Attacker sent malicious message in this round")

        # apply Krum
        aggregated_weights, best_idx = krum(results, num_malicious=1)

        # Log if malicious detected (assuming benign is index 0)
        if best_idx != 0:
            print("[SERVER] Malicious update detected and filtered")

        # For now, log anomaly if scores indicate
        print("[SERVER] Krum aggregation completed")

        return fl.common.ndarrays_to_parameters(aggregated_weights), {}


strategy = KrumStrategy()

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
