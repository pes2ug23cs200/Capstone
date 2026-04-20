import json
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Load statistics from stats.json
def load_stats(stats_path="shared/stats.json"):
    """Load statistics for each feature column."""
    if not os.path.exists(stats_path):
        # Try alternative path
        stats_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shared", "stats.json")
    
    with open(stats_path, 'r') as f:
        return json.load(f)


def generate_synthetic_feature_values(stats_dict, features_list, method="normal"):
    """
    Generate synthetic feature values based on statistics.
    
    Args:
        stats_dict: Dictionary loaded from stats.json
        features_list: List of feature names to generate (e.g., ["pos_0", "pos_1", ...])
        method: "normal" (normal distribution) or "uniform" (uniform within range)
    
    Returns:
        np.array of shape (len(features_list),) with synthetic values
    """
    values = []
    
    for feature in features_list:
        if feature not in stats_dict:
            print(f"Warning: {feature} not found in stats")
            values.append(0.0)
            continue
        
        stats = stats_dict[feature]
        min_val = stats["min"]
        max_val = stats["max"]
        mean = stats["mean"]
        std = stats["std"]
        
        if method == "normal":
            # Generate from normal distribution and clamp to [min, max]
            value = np.random.normal(mean, std)
            value = np.clip(value, min_val, max_val)
        else:  # uniform
            # Generate uniformly between min and max
            value = np.random.uniform(min_val, max_val)
        
        values.append(value)
    
    return np.array(values)


def generate_synthetic_message(stats_dict=None, features=None, scaler=None, method="normal"):
    """
    Generate a single synthetic message with the same format as dataset messages.
    
    Args:
        stats_dict: Statistics dictionary (loaded from stats.json if None)
        features: List of feature column names (default: from data_loader)
        scaler: StandardScaler instance fitted on benign data (if None, data won't be normalized)
        method: "normal" or "uniform" for value generation
    
    Returns:
        np.array of shape (20, 17) padded message, ready for model input
    """
    if stats_dict is None:
        stats_dict = load_stats()
    
    if features is None:
        features = ["pos_0", "pos_1", "spd_0", "spd_1", "spd_noise_0", "spd_noise_1", "acl_0"]
    
    # Generate raw feature values
    feature_values = generate_synthetic_feature_values(stats_dict, features, method=method)
    
    # Normalize if scaler provided
    if scaler is not None:
        feature_values = scaler.transform(feature_values.reshape(1, -1))[0]
    
    # Pad to (20, 17) format
    padded = np.zeros((20, 17))
    padded[0, :len(feature_values)] = feature_values
    
    return padded


def generate_synthetic_benign_message(stats_dict=None, scaler=None, method="normal"):
    """Generate a benign (normal) synthetic message."""
    return generate_synthetic_message(stats_dict, scaler=scaler, method=method)


def generate_synthetic_attack_message(stats_dict=None, scaler=None, method="normal", 
                                     perturbation_strength=0.5):
    """
    Generate an attack (malicious) synthetic message by perturbing benign values.
    
    Args:
        stats_dict: Statistics dictionary
        scaler: StandardScaler instance
        method: Value generation method
        perturbation_strength: Strength of perturbation (0-1), where 1 is max perturbation
    
    Returns:
        np.array of shape (20, 17) - perturbed message
    """
    benign_msg = generate_synthetic_benign_message(stats_dict, scaler, method)
    
    # Add perturbation to first 7 values (the feature columns)
    perturbation = np.random.normal(0, perturbation_strength, size=7)
    benign_msg[0, :7] += perturbation
    
    return benign_msg
