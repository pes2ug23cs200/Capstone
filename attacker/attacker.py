import time
import sys
import flwr as fl
import numpy as np
import tensorflow as tf
from shared.v2v import generate_message

server = "fl_server:8080"

# load same model
model = tf.keras.models.load_model("model/model.keras")


class Attacker(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):

        model.set_weights(parameters)

        print("[Attacker] Poisoning model")
        sys.stdout.flush()

        for _ in range(5):
            msg = generate_message(benign=False)
            print("[Attacker] Sending malicious message")
            sys.stdout.flush()

        # better poisoning
        poisoned = [w + np.random.normal(0, 5, w.shape) for w in model.get_weights()]

        return poisoned, 10, {}

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