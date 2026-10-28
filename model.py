import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

from data import load_and_split, make_target
from features import make_features, MODEL_FEATURES

MODEL_PATH = Path("model.joblib")
LOWER_PATH = Path("model_lower.joblib")
UPPER_PATH = Path("model_upper.joblib")

# Settings from the round-2 search (see README)
GB_SETTINGS = {
    "n_estimators": 400,
    "max_depth": 7,
    "learning_rate": 0.02,
    "min_samples_leaf": 10,
    "subsample": 1.0,
}

# Quantile models for the 90% range: the same settings that were evaluated on the test days
INTERVAL_SETTINGS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
}


def train_model():
    print("Loading real Azure data...")
    train_data, _, _ = load_and_split()
    print(f"Training on {len(train_data)} jobs (days 0 to 9)")

    train_features = make_features(train_data)[MODEL_FEATURES]
    train_target = make_target(train_data)

    model = GradientBoostingRegressor(random_state=42, **GB_SETTINGS)
    model.fit(train_features, train_target)

    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved!")
    return model


def load_or_train():
    if MODEL_PATH.exists():
        print("Loading existing model...")
        return joblib.load(MODEL_PATH)
    print("Training new model...")
    return train_model()


def train_interval_models():
    print("Training the range models (5th and 95th percentile)...")
    train_data, _, _ = load_and_split()
    train_features = make_features(train_data)[MODEL_FEATURES]
    train_target = make_target(train_data)

    models = []
    for quantile, path in [(0.05, LOWER_PATH), (0.95, UPPER_PATH)]:
        quantile_model = GradientBoostingRegressor(
            loss="quantile", alpha=quantile, random_state=42, **INTERVAL_SETTINGS
        )
        quantile_model.fit(train_features, train_target)
        joblib.dump(quantile_model, path)
        models.append(quantile_model)
    print("Range models trained and saved!")
    return models[0], models[1]


def load_or_train_intervals():
    if LOWER_PATH.exists() and UPPER_PATH.exists():
        print("Loading existing range models...")
        return joblib.load(LOWER_PATH), joblib.load(UPPER_PATH)
    return train_interval_models()


def one_job(starttime, core, memory, ssd, nic):
    return pd.DataFrame({
        "starttime": [starttime],
        "core": [core],
        "memory": [memory],
        "ssd": [ssd],
        "nic": [nic],
    })


def to_minutes(log_prediction):
    return max(1.0, float(np.expm1(log_prediction)))


def predict_minutes(model, starttime, core, memory, ssd, nic):
    features = make_features(one_job(starttime, core, memory, ssd, nic))[MODEL_FEATURES]
    return to_minutes(model.predict(features)[0])


def predict(model, starttime, core, memory, ssd, nic, priority, interval_models=None):
    # priority is accepted for compatibility with main.py but is not a model input
    duration = predict_minutes(model, starttime, core, memory, ssd, nic)

    result = {"predicted_duration": round(duration, 1)}

    if interval_models is not None:
        lower_model, upper_model = interval_models
        features = make_features(one_job(starttime, core, memory, ssd, nic))[MODEL_FEATURES]
        lower = to_minutes(lower_model.predict(features)[0])
        upper = to_minutes(upper_model.predict(features)[0])
        result["range_low"] = round(min(lower, upper), 1)
        result["range_high"] = round(max(lower, upper), 1)

    return result

# Original row-by-row feature builder. Kept only so that
# experiments/check_features.py can verify the fast version against it.
def build_features(starttime, core, memory, ssd, nic, priority):
    hour = (starttime * 24) % 24
    day = int(starttime) % 7

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day / 7)
    day_cos = np.cos(2 * np.pi * day / 7)

    is_business = 1 if 9 <= hour <= 17 else 0
    is_weekend = 1 if day >= 5 else 0

    resource_load = (core + memory + ssd + nic) / 4

    return np.array([[
        hour_sin, hour_cos, day_sin, day_cos,
        is_business, is_weekend,
        core, memory, ssd, nic,
        resource_load
    ]])


if __name__ == "__main__":
    for path in [MODEL_PATH, LOWER_PATH, UPPER_PATH]:
        if path.exists():
            path.unlink()
            print(f"Deleted {path}")

    model = load_or_train()
    interval_models = load_or_train_intervals()

    result = predict(model, starttime=0.479, core=0.0208, memory=0.0208,
                     ssd=0.0157, nic=0.025, priority=0, interval_models=interval_models)
    print("Test prediction:")
    print(f"  Predicted duration: {result['predicted_duration']} min")
    print(f"  90% range:          {result['range_low']} to {result['range_high']} min")