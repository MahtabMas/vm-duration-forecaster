from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from data import get_model_data, standardize
from mlp import train_mlp
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

train_scaled, validation_scaled, test_scaled = standardize(
    train_features, validation_features, test_features
)

train_inputs = torch.tensor(train_scaled.values, dtype=torch.float32)
train_targets = torch.tensor(train_target.values, dtype=torch.float32)
validation_inputs = torch.tensor(validation_scaled.values, dtype=torch.float32)
validation_targets = torch.tensor(validation_target.values, dtype=torch.float32)
test_inputs = torch.tensor(test_scaled.values, dtype=torch.float32)

test_predictions = {}

test_predictions["median_baseline"] = np.full(len(test_target), np.median(train_target))

linear_model = LinearRegression().fit(train_features, train_target)
test_predictions["linear_regression"] = linear_model.predict(test_features)

gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
).fit(train_features, train_target)
test_predictions["gradient_boosting"] = gb_model.predict(test_features)

TUNED_SETTINGS = {
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "dropout_rate": 0.0,
    "batch_size": 1024,
}

mlp_scores = []
for seed in [0, 1, 2, 3, 4]:
    model, best_epoch = train_mlp(
        train_inputs, train_targets, validation_inputs, validation_targets,
        seed=seed, **TUNED_SETTINGS,
    )
    with torch.no_grad():
        predictions = model(test_inputs).numpy()
    test_predictions[f"mlp_seed_{seed}"] = predictions
    mlp_scores.append(score_log_scale(test_target, predictions))
    print(f"mlp seed {seed}: epoch {best_epoch}  mae {mlp_scores[-1]['mae']:.4f}  "
          f"rmse {mlp_scores[-1]['rmse']:.4f}  r2 {mlp_scores[-1]['r2']:.4f}")

print()
print("TEST SET, log scale")
for name in ["median_baseline", "linear_regression", "gradient_boosting"]:
    scores = score_log_scale(test_target, test_predictions[name])
    print(f"{name}: mae {scores['mae']:.4f}  rmse {scores['rmse']:.4f}  r2 {scores['r2']:.4f}")

for metric in ["mae", "rmse", "r2"]:
    values = [row[metric] for row in mlp_scores]
    print(f"mlp over 5 seeds, {metric}: mean {np.mean(values):.4f}  spread {np.std(values, ddof=1):.4f}")

Path("results").mkdir(exist_ok=True)
np.savez("results/test_predictions.npz",
         true_target=test_target.values, **test_predictions)
print()
print("saved results/test_predictions.npz")