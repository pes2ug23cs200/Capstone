import pandas as pd
import numpy as np
import os

# Get the project root directory (parent of shared/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(project_root, "data", "balanced_veremi_dataset.csv")

COLUMNS = [
    "rcvTime",
    "pos_0","pos_1",
    "pos_noise_0","pos_noise_1",
    "spd_0","spd_1",
    "spd_noise_0","spd_noise_1",
    "acl_0","acl_1",
    "acl_noise_0","acl_noise_1",
    "hed_0","hed_1",
    "hed_noise_0","hed_noise_1"
]

def compute_stats(csv_path):
    df = pd.read_csv(csv_path)

    stats = {}
    for col in COLUMNS:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std())
        }

    return stats

import json

stats = compute_stats(csv_path)

output_path = os.path.join(project_root, "shared", "stats.json")
with open(output_path, "w") as f:
    json.dump(stats, f, indent=2)