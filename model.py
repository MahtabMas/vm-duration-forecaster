import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

from data import load_and_split, make_target
from features import make_features, MODEL_FEATURES

MODEL_PATH = Path("model.joblib")

# Settings from the round-2 search (see README)
GB_SETTINGS = {
    "n_estimators": 400,
    "max_depth": 7,
    "learning_rate": 0.02,
    "min_samples_leaf": 10,
    "subsample": 1.0,
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


def one_job(starttime, core, memory, ssd, nic):
    return pd.DataFrame({
        "starttime": [starttime],
        "core": [core],
        "memory": [memory],
        "ssd": [ssd],
        "nic": [nic],
    })


def predict_minutes(model, starttime, core, memory, ssd, nic):
    features = make_features(one_job(starttime, core, memory, ssd, nic))[MODEL_FEATURES]
    duration = float(np.expm1(model.predict(features)[0]))
    return max(1.0, duration)


def predict(model, starttime, core, memory, ssd, nic, priority):
    # priority is accepted for compatibility with main.py but is not a model input
    duration = predict_minutes(model, starttime, core, memory, ssd, nic)

    # Counterfactual: the same job submitted at 3am
    off_peak_starttime = int(starttime) + 0.125
    off_peak_duration = predict_minutes(model, off_peak_starttime, core, memory, ssd, nic)

    savings = round(duration - off_peak_duration, 1)

    return {
        "predicted_duration": round(duration, 1),
        "off_peak_duration": round(off_peak_duration, 1),
        "savings": savings,
    }


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
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
        print("Deleted old model")

    model = load_or_train()

    result = predict(model, starttime=0.479, core=0.0208, memory=0.0208,
                     ssd=0.0157, nic=0.025, priority=0)
    print("Test prediction:")
    print(f"  Predicted duration: {result['predicted_duration']} min")
    print(f"  Off-peak duration:  {result['off_peak_duration']} min")
    print(f"  Savings:            {result['savings']} min")