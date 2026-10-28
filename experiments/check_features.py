import numpy as np
from data import load_and_split
from features import make_features
from model import build_features

train_data, validation_data, test_data = load_and_split()
sample_jobs = train_data.sample(2000, random_state=42)

new_features = make_features(sample_jobs).values

old_features = np.array([
    build_features(
        row["starttime"], row["core"], row["memory"],
        row["ssd"], row["nic"], row["priority"],
    )[0]
    for _, row in sample_jobs.iterrows()
])

print("same shape:", new_features.shape == old_features.shape)
print("same values:", np.allclose(new_features, old_features))
print("largest difference:", np.abs(new_features - old_features).max())