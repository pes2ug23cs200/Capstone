import time
import sys
import flwr as fl
import numpy as np
import tensorflow as tf
import requests
from shared.synthetic_generator import generate_synthetic_benign_message, generate_synthetic_attack_message, load_stats

server = "fl_server:8080"

# load same model
model = tf.keras.models.load_model("model/model.keras")

# Load statistics for synthetic generation
stats = load_stats()
is_benign_round = True


class Attacker(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):

        global is_benign_round

        model.set_weights(parameters)

        # Alternate between benign and attack (synthetically generated)
        if is_benign_round:
            msg = generate_synthetic_benign_message(stats, method="normal")
            msg_type = "benign"
        else:
            msg = generate_synthetic_attack_message(stats, method="normal", perturbation_strength=0.5)
            msg_type = "malicious"

        is_benign_round = not is_benign_round  # Toggle for next round

        print(f"[Attacker] Sending {msg_type} dataset message")
        sys.stdout.flush()

        # Send message to vehicle
        try:
            response = requests.post('http://benign_vehicle:5000/receive_message', json={'message': msg.tolist()}, timeout=5)
            print(f"[Attacker] Sent message to vehicle: {response.status_code}")
            sys.stdout.flush()
        except Exception as e:
            print(f"[Attacker] Failed to send message: {e}")
            sys.stdout.flush()

        # Poison weights (slight perturbation)
        poisoned = [w + np.random.normal(0, 0.1, w.shape) for w in model.get_weights()]

        return poisoned, 1, {"type": msg_type}

    def evaluate(self, parameters, config):
        return 10.0, 10, {}


print("[Attacker] Connecting to server...")
sys.stdout.flush()

# retry loop
while True:
    try:
        fl.client.start_numpy_client(
            server_address=server,
            client=Attacker(),
        )
        break
    except Exception:
        print("[Attacker] Server not ready, retrying...")
        sys.stdout.flush()
        time.sleep(5)
