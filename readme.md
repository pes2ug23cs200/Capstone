# V2V Federated Learning with Intrusion Detection

## Project Overview

This repository implements a containerized simulation of federated learning for vehicle-to-vehicle (V2V) communication security. It models a benign vehicle client and a malicious attacker client training a shared intrusion detection model through the Flower framework. The central server uses the Krum aggregation strategy to reduce the impact of poisoned model updates.

## Repository Structure

- `docker-compose.yml` - orchestrates three services: `fl_server`, `benign_vehicle`, and `attacker_vehicle`
- `Dockerfile.server` - builds the Flower server image
- `Dockerfile.vehicle` - builds the benign vehicle client image
- `Dockerfile.attacker` - builds the attacker client image
- `server/server.py` - federated server logic and custom Krum aggregation strategy
- `vehicle/main.py` - benign vehicle client implementation
- `attacker/attacker.py` - malicious attacker client implementation
- `shared/data_loader.py` - loads and preprocesses the local dataset
- `shared/ids.py` - intrusion detection helper that classifies incoming messages
- `shared/v2v.py` - synthetic V2V message generator utilities
- `model/model.keras` - pre-trained Keras model loaded by all components
- `data/balanced_veremi_dataset.csv` - dataset used by the vehicle and attacker clients
- `data.csv` - additional dataset file currently untracked or under development

## Detailed Component Behavior

### Server (`server/server.py`)

- Starts a Flower server on `0.0.0.0:8080`
- Uses `KrumStrategy`, a subclass of `fl.server.strategy.FedAvg`
- In `aggregate_fit()`:
  - checks client metrics for malicious signals
  - applies the Krum aggregation algorithm to the client weight updates
  - returns the selected aggregation result to the Flower server
- Logs when a client emits suspicious data using request metrics
- Runs for 10 federated rounds by default via `ServerConfig(num_rounds=10)`

### Benign Vehicle (`vehicle/main.py`)

- Loads the same Keras model from `model/model.keras`
- Loads benign data from `data/balanced_veremi_dataset.csv`
- Exposes a Flask endpoint at `/receive_message` on port `5000`
- In `VehicleClient.fit()`:
  - receives current global model weights from the server
  - trains the model on one benign sample using `train_on_batch`
  - if a message was received from the attacker, classifies it with `shared.ids.detect()`
  - returns model weights and the detection result in `fit` metrics
- The client retries connection until the Flower server is available

### Attacker (`attacker/attacker.py`)

- Loads the same Keras model from `model/model.keras`
- Loads both benign and attack partitions from `data/balanced_veremi_dataset.csv`
- Alternates between sending a benign sample and a malicious sample to the benign vehicle service
- Sends HTTP POST requests to `http://benign_vehicle:5000/receive_message`
- In `Attacker.fit()`:
  - loads the provided model weights
  - selects either benign or attack data in alternating rounds
  - sends the payload to the benign vehicle for classification
  - poisons the model weights using Gaussian noise before returning them
  - returns metadata including `message` and `type` in `fit` metrics
- The client also retries until the Flower server is ready

### Shared Modules

#### `shared/data_loader.py`

- Loads `data/balanced_veremi_dataset.csv` using pandas
- Splits rows into benign samples and attack samples based on `AttackerType`
- Standardizes features with `StandardScaler`
- Pads data into shape `(N, 20, 17)` so it matches the model input
- Returns `benign_padded` and `attack_padded`

#### `shared/ids.py`

- Defines `detect(model, message)` to classify a single input message
- Uses the loaded Keras model to predict a score
- Returns:
  - `ATTACK` if score > 0.5
  - `BENIGN` otherwise

#### `shared/v2v.py`

- Provides a synthetic message generator for benign or malicious V2V traffic
- Message shapes are `(1, 20, 17)`
- Currently the main clients use the dataset loader instead of this generator

## Data and Model

- `model/model.keras` is the shared Keras model on disk
- `data/balanced_veremi_dataset.csv` is the main dataset used by both vehicle and attacker clients
- This dataset is large and may exceed GitHub limits; do not push it to GitHub without Git LFS or a storage alternative
- `data.csv` appears in the repository root but is separate from the dataset used by the Python clients

## Deployment

Start the whole system with Docker Compose:

```bash
cd /home/glenn/Desktop/project
docker-compose up --build
```

Service mapping:

- `fl_server` exposes port `8080`
- `benign_vehicle` exposes port `5000`
- `attacker_vehicle` does not expose ports externally

## Execution Flow

1. `fl_server` starts the Flower server and waits for clients.
2. `benign_vehicle` starts, loads the model, and registers as a Flower client.
3. `attacker_vehicle` starts, loads the model, and registers as a Flower client.
4. The server begins federated rounds and requests `fit()` from both clients.
5. `benign_vehicle` trains on benign data and may classify a received message.
6. `attacker_vehicle` alternates between benign and attack payloads and returns poisoned weights.
7. The server uses Krum to select the least anomalous weight update and broadcasts the aggregated model.

## Requirements

The Python components depend on:

- `flwr`
- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `flask`
- `requests`

If running outside Docker, install dependencies in a virtual environment.

## Notes and Troubleshooting

- The attacker sends its message payload to the benign vehicle over internal service DNS `benign_vehicle:5000`.
- The benign vehicle logs message reception and classification results during each training round.
- Both clients retry until the Flower server is available.
- If `data/balanced_veremi_dataset.csv` is too large for GitHub, use Git LFS or remove it from version control.
- The Krum implementation in `server/server.py` selects a best client update, but the current code includes `best_idx = ...` placeholder logic and logs that Krum aggregation completed.

## Recommended Improvements

- Replace the placeholder `best_idx = ...` line in `server/server.py` with a proper index return from `krum()`.
- Add explicit model evaluation metrics in the client `evaluate()` methods.
- Consolidate message generation logic so `shared/v2v.py` is used consistently across the project.
- Move large datasets out of the repository and load them from an external volume or remote storage.
