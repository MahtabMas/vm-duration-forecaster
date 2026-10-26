import pandas as pd

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