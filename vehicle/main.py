import os
import time
import sys
import flwr as fl
import numpy as np
import tensorflow as tf

from shared.ids import detect

# load model
model = tf.keras.models.load_model("model/model.keras")

print("MODEL INPUT SHAPE:", model.input_shape)
sys.stdout.flush()

server = os.getenv("SERVER_ADDRESS", "fl_server:8080")
vehicle_id = os.getenv("VEHICLE_ID", "1")

print(f"[Vehicle {vehicle_id}] Connecting to {server}")
sys.stdout.flush()


class VehicleClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):

        print("=== ENTERING FIT ===")
        sys.stdout.flush()

        model.set_weights(parameters)

        for i in range(5):

            # correct shape input
            msg = np.random.rand(1, 20, 17)

            print(f"[Vehicle] Message shape: {msg.shape}")
            sys.stdout.flush()

            # TRAIN (important for FL)
            model.train_on_batch(msg, np.array([[0]]))  # benign label

            try:
                # raw prediction
                pred = model.predict(msg, verbose=0)

                # IDS decision
                result = detect(model, msg)

                print(f"[Vehicle] Raw prediction: {pred}")
                print(f"[Vehicle] Classified as: {result}")
                sys.stdout.flush()

            except Exception as e:
                print("Prediction error:", e)
                sys.stdout.flush()

        print("=== EXITING FIT ===")
        sys.stdout.flush()

        return model.get_weights(), 10, {}

    def evaluate(self, parameters, config):

        model.set_weights(parameters)

        return 0.0, 10, {}


# retry loop
while True:
    try:
        fl.client.start_numpy_client(
            server_address=server,
            client=VehicleClient(),
        )
        break
    except Exception:
        print("[Vehicle] Server not ready, retrying in 5 seconds...")
        sys.stdout.flush()
        time.sleep(5)