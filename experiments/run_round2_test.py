from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from data import get_model_data, standardize
from mlp import train_mlp
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()
print("number of input features:", train_features.shape[1])

train_scaled, validation_scaled, test_scaled = standardize(
    train_features, validation_features, test_features
)
train_inputs = torch.tensor(train_scaled.values, dtype=torch.float32)
train_targets = torch.tensor(train_target.values, dtype=torch.float32)
validation_inputs = torch.tensor(validation_scaled.values, dtype=torch.float32)
validation_targets = torch.tensor(validation_target.values, dtype=torch.float32)
test_inputs = torch.tensor(test_scaled.values, dtype=torch.float32)

GB_SETTINGS = {
    "n_estimators": 400, "max_depth": 7, "learning_rate": 0.02,
    "min_samples_leaf": 10, "subsample": 1.0,
}
MLP_SETTINGS = {
    "hidden_sizes": (128, 64, 32), "learning_rate": 0.01,
    "weight_decay": 0.1, "dropout_rate": 0.1, "batch_size": 256,
}

test_predictions = {}

gb_model = GradientBoostingRegressor(random_state=42, **GB_SETTINGS).fit(train_features, train_target)
test_predictions["gradient_boosting"] = gb_model.predict(test_features)
gb_scores = score_log_scale(test_target, test_predictions["gradient_boosting"])

mlp_scores = []
for seed in [0, 1, 2, 3, 4]:
    model, best_epoch = train_mlp(
        train_inputs, train_targets, validation_inputs, validation_targets,
        seed=seed, **MLP_SETTINGS,
    )
    with torch.no_grad():
        predictions = model(test_inputs).numpy()
    test_predictions[f"mlp_seed_{seed}"] = predictions
    mlp_scores.append(score_log_scale(test_target, predictions))
    print(f"mlp seed {seed}: epoch {best_epoch}  mae {mlp_scores[-1]['mae']:.4f}  "
          f"rmse {mlp_scores[-1]['rmse']:.4f}  r2 {mlp_scores[-1]['r2']:.4f}")

print()
print("TEST SET, log scale, round 2")
print(f"gradient boosting: mae {gb_scores['mae']:.4f}  rmse {gb_scores['rmse']:.4f}  r2 {gb_scores['r2']:.4f}")
for metric in ["mae", "rmse", "r2"]:
    values = [row[metric] for row in mlp_scores]
    print(f"mlp over 5 seeds, {metric}: mean {np.mean(values):.4f}  spread {np.std(values, ddof=1):.4f}")

print()
print("round 1, for comparison (test set, log scale):")
print("  gradient boosting: mae 1.2027  rmse 1.6912  r2 0.3431")
print("  mlp over 5 seeds:  mae 1.2265  rmse 1.7193  r2 0.3210")

Path("results").mkdir(exist_ok=True)
np.savez("results/test_predictions_round2.npz",
         true_target=test_target.values, **test_predictions)
print()
print("saved results/test_predictions_round2.npz")