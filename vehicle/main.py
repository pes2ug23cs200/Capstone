import os
import time
import sys
import flwr as fl
import numpy as np
import tensorflow as tf
from flask import Flask, request
import threading

from shared.ids import detect
from shared.synthetic_generator import generate_synthetic_benign_message, load_stats
import json

# load model
model = tf.keras.models.load_model("model/model.keras")

print("MODEL INPUT SHAPE:", model.input_shape)
sys.stdout.flush()

server = os.getenv("SERVER_ADDRESS", "fl_server:8080")
vehicle_id = os.getenv("VEHICLE_ID", "1")

print(f"[Vehicle {vehicle_id}] Connecting to {server}")
sys.stdout.flush()

# Load statistics for synthetic generation
stats = load_stats()

app = Flask(__name__)
latest_message = None

@app.route('/receive_message', methods=['POST'])
def receive_message():
    global latest_message
    data = request.get_json()
    latest_message = np.array(data['message'])
    print(f"[Vehicle] Received message: {latest_message.shape}")
    sys.stdout.flush()
    return {'status': 'received'}

def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# Start server in background
threading.Thread(target=run_server, daemon=True).start()


class VehicleClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):

        global latest_message

        print("=== ENTERING FIT ===")
        sys.stdout.flush()

        model.set_weights(parameters)

        # Generate synthetic benign message for training
        msg = generate_synthetic_benign_message(stats, method="normal")

        # Reshape to (1, 20, 17)
        msg = msg[np.newaxis, ...]

        # Train on benign data
        model.train_on_batch(msg, np.array([[0]]))

        # Detect the received message if any
        result = "NO_MESSAGE"
        if latest_message is not None:
            received_msg = latest_message[np.newaxis, ...]
            result = detect(model, received_msg)
            print(f"[Vehicle] Received message classified as: {result}")
            sys.stdout.flush()
            latest_message = None  # Reset

        print("=== EXITING FIT ===")
        sys.stdout.flush()

        return model.get_weights(), 1, {"prediction": result}

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