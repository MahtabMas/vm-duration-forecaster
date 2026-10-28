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
FOLDS = [(6, 7), (8, 9), (10, 11), (12, 13)]


def to_tensor(values):
    return torch.tensor(np.asarray(values), dtype=torch.float32)


print("fold (test days) | train days | gb mae  mlp mae (mean of 3 seeds)  difference | gb rmse  mlp rmse")
for first_test_day, last_test_day in FOLDS:
    validation_start = first_test_day - 2
    train_mask = days < validation_start
    validation_mask = (days >= validation_start) & (days < first_test_day)
    test_mask = (days >= first_test_day) & (days <= last_test_day)

    train_features = all_features[train_mask]
    validation_features = all_features[validation_mask]
    test_features = all_features[test_mask]
    train_target = all_target[train_mask]
    validation_target = all_target[validation_mask]
    test_target = all_target[test_mask]

    gb_model = GradientBoostingRegressor(random_state=42, **GB_SETTINGS).fit(train_features, train_target)
    gb_scores = score_log_scale(test_target, gb_model.predict(test_features))

    training_average = train_features.mean()
    training_spread = train_features.std()
    train_inputs = to_tensor(((train_features - training_average) / training_spread).values)
    validation_inputs = to_tensor(((validation_features - training_average) / training_spread).values)
    test_inputs = to_tensor(((test_features - training_average) / training_spread).values)
    train_targets = to_tensor(train_target.values)
    validation_targets = to_tensor(validation_target.values)

    mlp_maes, mlp_rmses = [], []
    for seed in SEEDS:
        model, _ = train_mlp(
            train_inputs, train_targets, validation_inputs, validation_targets,
            seed=seed, **MLP_SETTINGS,
        )
        with torch.no_grad():
            predictions = model(test_inputs).numpy()
        scores = score_log_scale(test_target, predictions)
        mlp_maes.append(scores["mae"])
        mlp_rmses.append(scores["rmse"])

    difference = gb_scores["mae"] - np.mean(mlp_maes)
    print(f"days {first_test_day:2d}-{last_test_day:2d} | train 0-{validation_start - 1:2d} | "
          f"{gb_scores['mae']:.4f}  {np.mean(mlp_maes):.4f}  {difference:+.4f} | "
          f"{gb_scores['rmse']:.4f}  {np.mean(mlp_rmses):.4f}   (jobs: train {train_mask.sum()}, test {test_mask.sum()})")