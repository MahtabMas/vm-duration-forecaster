import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor
from data import load_and_split, make_target
from features import make_features, MODEL_FEATURES
from mlp import train_mlp
from metrics import score_log_scale

train_data, validation_data, test_data = load_and_split()
all_jobs = pd.concat([train_data, validation_data, test_data])
all_features = make_features(all_jobs)[MODEL_FEATURES]
all_target = make_target(all_jobs)
days = all_jobs["day"].values

GB_SETTINGS = {
    "n_estimators": 400, "max_depth": 7, "learning_rate": 0.02,
    "min_samples_leaf": 10, "subsample": 1.0,
}
MLP_SETTINGS = {
    "hidden_sizes": (128, 64, 32), "learning_rate": 0.01,
    "weight_decay": 0.1, "dropout_rate": 0.1, "batch_size": 256,
}
SEEDS = [0, 1, 2]


def to_tensor(values):
    return torch.tensor(np.asarray(values), dtype=torch.float32)


validation_mask = (days >= 10) & (days <= 11)
test_mask = (days >= 12) & (days <= 13)

print("test days 12-13 and validation days 10-11 fixed; every training window ends at day 9")
print("training days | training jobs | gb mae  mlp mae (mean of 3 seeds)  difference (gb minus mlp)")
for first_train_day in [6, 4, 2, 0]:
    train_mask = (days >= first_train_day) & (days <= 9)

    train_features = all_features[train_mask]
    validation_features = all_features[validation_mask]
    test_features = all_features[test_mask]
    train_target = all_target[train_mask]
    validation_target = all_target[validation_mask]
    test_target = all_target[test_mask]

    gb_model = GradientBoostingRegressor(random_state=42, **GB_SETTINGS).fit(train_features, train_target)
    gb_mae = score_log_scale(test_target, gb_model.predict(test_features))["mae"]

    training_average = train_features.mean()
    training_spread = train_features.std()
    train_inputs = to_tensor(((train_features - training_average) / training_spread).values)
    validation_inputs = to_tensor(((validation_features - training_average) / training_spread).values)
    test_inputs = to_tensor(((test_features - training_average) / training_spread).values)
    train_targets = to_tensor(train_target.values)
    validation_targets = to_tensor(validation_target.values)

    mlp_maes = []
    for seed in SEEDS:
        model, _ = train_mlp(
            train_inputs, train_targets, validation_inputs, validation_targets,
            seed=seed, **MLP_SETTINGS,
        )
        with torch.no_grad():
            predictions = model(test_inputs).numpy()
        mlp_maes.append(score_log_scale(test_target, predictions)["mae"])

    print(f"{first_train_day}-9           | {train_mask.sum():12d} | "
          f"{gb_mae:.4f}  {np.mean(mlp_maes):.4f}  {gb_mae - np.mean(mlp_maes):+.4f}")