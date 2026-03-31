# V2V Federated Learning with Intrusion Detection

## Overview

This project simulates a federated learning system for vehicle-to-vehicle (V2V) communication security. The system includes benign vehicles that collaboratively train a machine learning model for intrusion detection, while defending against malicious attackers that attempt to poison the global model through adversarial updates.

The implementation uses the Flower framework for federated learning and demonstrates robust aggregation techniques to filter out malicious contributions.

## Architecture

The system consists of three main components running in Docker containers:

### Server (`server/server.py`)
- **Role**: Central federated learning coordinator
- **Technology**: Flower server with custom Krum aggregation strategy
- **Function**: Receives model updates from clients, applies Krum algorithm to filter malicious updates, and distributes aggregated global model
- **Key Class**: `KrumStrategy` - Extends Flower's `FedAvg` with Byzantine-robust aggregation

### Vehicle (`vehicle/main.py`)
- **Role**: Benign federated learning participant
- **Technology**: Flower client with TensorFlow/Keras
- **Function**: Generates benign V2V messages, runs intrusion detection training, participates in federated learning rounds
- **Key Class**: `VehicleClient` - Implements Flower's `NumPyClient` interface
- **Connections**:
  - Uses `shared.v2v.generate_message(benign=True)` to create training data
  - Uses `shared.ids.detect(model, message)` for intrusion detection evaluation

### Attacker (`attacker/attacker.py`)
- **Role**: Malicious federated learning participant
- **Technology**: Flower client with TensorFlow/Keras
- **Function**: Generates malicious V2V messages, poisons model weights with adversarial noise
- **Key Class**: `Attacker` - Implements Flower's `NumPyClient` interface
- **Connections**:
  - Uses `shared.v2v.generate_message(benign=False)` to create malicious data
  - Applies weight poisoning: `poisoned = [w + np.random.normal(0, 5, w.shape) for w in model.get_weights()]`

### Shared Components

#### V2V Message Generation (`shared/v2v.py`)
- **Function**: Generates synthetic V2V communication messages
- **Features**: 10-dimensional feature vectors
- **Distributions**:
  - Benign: Normal(0, 1)
  - Malicious: Normal(5, 2)

#### Intrusion Detection System (`shared/ids.py`)
- **Function**: Binary classification for attack detection
- **Model**: Pre-trained Keras model (`model/model.keras`)
- **Threshold**: 0.5 (scores > 0.5 classified as attacks)

## Class Connections and Data Flow

```
Vehicle Client ────→ Shared V2V (benign messages) ────→ Shared IDS (detection)
     │                       │                              │
     │                       │                              │
     └───── Model Updates ───┼──────────────────────────────┼─────→ Server (Krum Aggregation)
                             │                              │
Attacker Client ────→ Shared V2V (malicious messages) ────→ Server (filtered out)
```

1. **Training Phase**:
   - Vehicles generate benign messages via `shared.v2v.generate_message(benign=True)`
   - Vehicles evaluate messages using `shared.ids.detect(model, msg)`
   - Both vehicles and attackers send model updates to server

2. **Aggregation Phase**:
   - Server applies Krum algorithm in `KrumStrategy.aggregate_fit()`
   - Krum identifies and filters malicious weight updates
   - Clean aggregated model distributed to all clients

## Objectives

- **Security**: Demonstrate Byzantine-robust federated learning against poisoning attacks
- **Privacy**: Distributed training without sharing raw data
- **Scalability**: Horizontal scaling through containerized deployment
- **Realism**: Simulate V2V communication patterns and intrusion detection

## Technologies Used

- **Federated Learning**: Flower (flwr)
- **Machine Learning**: TensorFlow/Keras
- **Containerization**: Docker & Docker Compose
- **Language**: Python 3.10

## Running the System

```bash
# Build and start all services
docker-compose up --build

# Services will start in order:
# 1. fl_server (port 8080)
# 2. benign_vehicle
# 3. attacker_vehicle
```

## Configuration

- **Federated Rounds**: 10 rounds (configurable in `server.py`)
- **Malicious Clients**: 1 assumed in Krum algorithm
- **Training Samples**: 5 messages per round per client
- **Model**: Pre-trained Keras model in `model/model.keras`


sequence data (like time series)
→ 20 timesteps
→ each timestep has 17 features 

expected shape = (None, 20, 17)