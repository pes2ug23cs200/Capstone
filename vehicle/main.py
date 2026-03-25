import os
import time
import flwr as fl
import numpy as np
import tensorflow as tf

from shared.v2v import generate_message
from shared.ids import detect

# load model
model = tf.keras.models.load_model("model/model.keras")
print("MODEL INPUT SHAPE:", model.input_shape)
server = os.getenv("SERVER_ADDRESS", "fl_server:8080")
vehicle_id = os.getenv("VEHICLE_ID", "1")

print(f"Vehicle {vehicle_id} connecting to {server}")


class VehicleClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):

        print("=== ENTERING FIT ===")

        model.set_weights(parameters)

        for i in range(5):

            msg = generate_message(benign=True)

            print(f"[Vehicle] Message shape: {msg.shape}")

            try:
                result = detect(model, msg)
                print(f"[Vehicle] Prediction: {result}")
            except Exception as e:
                print("Prediction error:", e)

        print("=== EXITING FIT ===")

        return model.get_weights(), 10, {}

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