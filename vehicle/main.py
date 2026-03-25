import os
import time
import flwr as fl
import numpy as np
import tensorflow as tf

from v2v import generate_message
from ids import detect

# load model
model = tf.keras.models.load_model("model/ids_model.keras")

server = os.getenv("SERVER_ADDRESS", "fl_server:8080")
vehicle_id = os.getenv("VEHICLE_ID", "1")

print(f"Vehicle {vehicle_id} connecting to {server}")


class VehicleClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):

    # load global weights
        model.set_weights(parameters)

        print("[Vehicle] Training locally")

        losses = []

        # simulate local training
        for _ in range(5):

            msg = generate_message(benign=True)

            # IMPORTANT: shape may need adjustment later
            label = np.array([[0]])  # benign

            loss = model.train_on_batch(msg, label)
            losses.append(loss)

            result = detect(model, msg)
            print(f"[Vehicle] Message classified as: {result}")

        avg_loss = float(np.mean(losses))

        print(f"[Vehicle] Avg loss: {avg_loss}")

        return model.get_weights(), 10, {"loss": avg_loss}

    def evaluate(self, parameters, config):

        model.set_weights(parameters)

        return 0.0, 10, {}


# retry loop
while True:
    try:
        fl.client.start_numpy_client(
            server_address=server,
            client=VehicleClient(),   # ✅ fixed
        )
        break
    except Exception:
        print("Server not ready yet, retrying in 5 seconds...")
        time.sleep(5)