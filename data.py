import numpy as np
import pandas as pd
from features import make_features, MODEL_FEATURES

DATA_PATH = "data/Azure_Data0.csv"


def load_and_split():
    # Load the CSV and split it by time into train, validation, and test.
    df = pd.read_csv(DATA_PATH)

    # Day number of each job: 0 to 13
    df["day"] = df["starttime"].astype(int)

    train_data = df[df["day"] <= 9].copy()
    validation_data = df[(df["day"] >= 10) & (df["day"] <= 11)].copy()
    test_data = df[df["day"] >= 12].copy()

    return train_data, validation_data, test_data



def make_target(jobs):
    # Job duration in minutes, then log1p. This is what the model predicts.
    duration_minutes = (jobs["endtime"] - jobs["starttime"]) * 24 * 60
    return np.log1p(duration_minutes)


def get_model_data():
    # Return features and target for train, validation and test.
    train_data, validation_data, test_data = load_and_split()

    train_features = make_features(train_data)[MODEL_FEATURES]
    validation_features = make_features(validation_data)[MODEL_FEATURES]
    test_features = make_features(test_data)[MODEL_FEATURES]

    train_target = make_target(train_data)
    validation_target = make_target(validation_data)
    test_target = make_target(test_data)

    return (train_features, train_target,
            validation_features, validation_target,
            test_features, test_target)


def standardize(train_features, validation_features, test_features):
    # Rescale every feature using the training rows only.
    training_average = train_features.mean()
    training_spread = train_features.std()

    train_scaled = (train_features - training_average) / training_spread
    validation_scaled = (validation_features - training_average) / training_spread
    test_scaled = (test_features - training_average) / training_spread

    return train_scaled, validation_scaled, test_scaled