import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_dataset(csv_path="data/balanced_veremi_dataset.csv"):
    """
    Load VeReMi dataset, extract features, normalize, return benign and attack data.
    
    Features: ["pos_0","pos_1","spd_0","spd_1","spd_noise_0","spd_noise_1","acl_0"]
    Return: benign_data (N_b, 20, 17), attack_data (N_a, 20, 17)
    """
    df = pd.read_csv(csv_path)
    
    features = ["pos_0","pos_1","spd_0","spd_1","spd_noise_0","spd_noise_1","acl_0"]
    
    # Separate benign and attack
    benign_df = df[df['AttackerType'] == 'Benign']
    attack_df = df[df['AttackerType'] != 'Benign']
    
    # Extract features
    benign_data = benign_df[features].values
    attack_data = attack_df[features].values
    
    # Normalize
    scaler = StandardScaler()
    benign_normalized = scaler.fit_transform(benign_data)
    attack_normalized = scaler.transform(attack_data)
    
    # Pad to match model input (20,17)
    def pad_data(data):
        padded = np.zeros((len(data), 20, 17))
        for i in range(len(data)):
            padded[i, 0, :7] = data[i]
        return padded
    
    benign_padded = pad_data(benign_normalized)
    attack_padded = pad_data(attack_normalized)
    
    return benign_padded, attack_padded