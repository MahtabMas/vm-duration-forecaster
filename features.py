import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_business", "is_weekend",
    "core", "memory", "ssd", "nic",
    "resource_load",
]

MODEL_FEATURES = [
    "hour_sin", "hour_cos", "is_business",
    "core", "memory", "ssd", "nic", "resource_load",
]


def make_features(jobs):
    # Turn a table of jobs into the 11 model inputs, all rows at once.
    hour = (jobs["starttime"] * 24) % 24
    day = jobs["starttime"].astype(int) % 7

    features = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_sin": np.sin(2 * np.pi * day / 7),
        "day_cos": np.cos(2 * np.pi * day / 7),
        "is_business": ((hour >= 9) & (hour <= 17)).astype(int),
        "is_weekend": (day >= 5).astype(int),
        "core": jobs["core"],
        "memory": jobs["memory"],
        "ssd": jobs["ssd"],
        "nic": jobs["nic"],
        "resource_load": (jobs["core"] + jobs["memory"] + jobs["ssd"] + jobs["nic"]) / 4,
    })
    return features[FEATURE_NAMES]