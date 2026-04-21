import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

MODEL_PATH = Path("model.joblib")

def build_features(starttime, core, memory, ssd, nic, priority):
    # Extract time features from starttime (in days)
    hour = (starttime * 24) % 24
    day  = int(starttime) % 7

    # Cyclic encoding for time
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin  = np.sin(2 * np.pi * day / 7)
    day_cos  = np.cos(2 * np.pi * day / 7)

    # Business hours and weekend flags
    is_business = 1 if 9 <= hour <= 17 else 0
    is_weekend  = 1 if day >= 5 else 0


    # Combined resource load
    resource_load = (core + memory + ssd + nic) / 4

    return np.array([[
        hour_sin, hour_cos, day_sin, day_cos,
        is_business, is_weekend,
        core, memory, ssd, nic,
        resource_load, priority
    ]])

def train_model():
    print("Loading real Azure data...")
    df = pd.read_csv("/Users/m_masoor/Desktop/Data/Azure_Data0.csv")

    # Calculate duration in minutes
    df['duration'] = (df['endtime'] - df['starttime']) * 24 * 60

    # Remove negative or zero durations
    df = df[df['duration'] > 0].copy()
    print(f"Training on {len(df)} jobs")

    # Log transform duration (consistent with published research)
    y = np.log1p(df['duration'].values)

    # Build feature matrix
    X = []
    for _, row in df.iterrows():
        features = build_features(
            row['starttime'],
            row['core'],
            row['memory'],
            row['ssd'],
            row['nic'],
            row['priority']
        )
        X.append(features[0])

    X = np.array(X)

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X, y)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved!")

    return model

def load_or_train():
    if MODEL_PATH.exists():
        print("Loading existing model...")
        return joblib.load(MODEL_PATH)
    else:
        print("Training new model...")
        return train_model()

def predict(model, starttime, core, memory, ssd, nic, priority):
    # Build feature array
    features = build_features(starttime, core, memory, ssd, nic, priority)

    # Make prediction
    duration = float(np.expm1(model.predict(features)[0]))
    duration = max(1.0, duration)

    # Counterfactual: what if submitted at 3am?
    # 3am = 3/24 = 0.125 days offset
    off_peak_starttime = int(starttime) + 0.125
    off_peak = build_features(off_peak_starttime, core, memory, ssd, nic, priority)
    off_peak_duration = float(np.expm1(model.predict(off_peak)[0]))
    off_peak_duration = max(1.0, off_peak_duration)

    savings = round(duration - off_peak_duration, 1)

    return {
        "predicted_duration": round(duration, 1),
        "off_peak_duration":  round(off_peak_duration, 1),
        "savings":            savings
    }


if __name__ == "__main__":
    # Delete old model to force retraining on real data
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
        print("Deleted old model")

    # Train on real data
    model = load_or_train()

    # Test prediction (starttime=0.479 ≈ 11am on day 0)
    result = predict(model,
                     starttime=0.479,
                     core=0.083,
                     memory=0.083,
                     ssd=0.056,
                     nic=0.1,
                     priority=0)

    print("Test prediction:")
    print(f"  Predicted duration: {result['predicted_duration']} min")
    print(f"  Off-peak duration:  {result['off_peak_duration']} min")
    print(f"  Savings:            {result['savings']} min")


    
