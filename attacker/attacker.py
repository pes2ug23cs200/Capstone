import time
import flwr as fl
import numpy as np

from v2v import generate_message

server = "fl_server:8080"


class Attacker(fl.client.NumPyClient):

    def get_parameters(self, config):
        # must match model structure
        return parameters

    def fit(self, parameters, config):

        print("[Attacker] Poisoning model")

        # simulate malicious messages
        for _ in range(3):
            msg = generate_message(benign=False)
            print("[Attacker] Sending malicious message")

        # poison weights (same structure!)
        poisoned = [w * 50 for w in parameters]

        return poisoned, 10, {}

    def evaluate(self, parameters, config):
        return 10.0, 10, {}


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
        time.sleep(5)