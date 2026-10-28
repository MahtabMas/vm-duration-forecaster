from pathlib import Path
import numpy as np
import pandas as pd
from data import get_model_data
from metrics import score_log_scale, score_minutes

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

saved = np.load("results/test_predictions.npz")
true_target = saved["true_target"]
assert np.allclose(true_target, test_target.values), "saved predictions do not match the test rows"

# Lookup baseline: one average log duration per VM type, learned on the training days
resource_columns = ["core", "memory", "ssd", "nic"]


def make_keys(features):
    return features[resource_columns].round(6).astype(str).apply("-".join, axis=1)


average_by_type = pd.DataFrame(
    {"key": make_keys(train_features), "target": train_target}
).groupby("key")["target"].mean()
lookup_predictions = (
    make_keys(test_features).map(average_by_type).fillna(np.median(train_target)).values
)


def all_scores(predictions):
    log_scores = score_log_scale(true_target, predictions)
    minute_scores = score_minutes(true_target, predictions)
    return [log_scores["mae"], log_scores["rmse"], log_scores["r2"],
            minute_scores["mae_minutes"], minute_scores["median_abs_error_minutes"]]


rows = [
    ("Median baseline", all_scores(saved["median_baseline"]), None),
    ("Linear regression", all_scores(saved["linear_regression"]), None),
    ("Lookup by VM type (added after the test run)", all_scores(lookup_predictions), None),
    ("Gradient Boosting (8 features)", all_scores(saved["gradient_boosting"]), None),
]

mlp_scores = np.array([all_scores(saved[f"mlp_seed_{seed}"]) for seed in range(5)])
rows.append(("MLP, tuned (mean of 5 seeds)", mlp_scores.mean(axis=0), mlp_scores.std(axis=0, ddof=1)))

lines = [
    "| Model | MAE (log) | RMSE (log) | R2 (log) | MAE (min) | Median abs error (min) |",
    "|---|---|---|---|---|---|",
]
for name, values, spreads in rows:
    cells = []
    for index, value in enumerate(values):
        decimals = 1 if index >= 3 else 4
        text = f"{value:.{decimals}f}"
        if spreads is not None:
            text += f" (sd {spreads[index]:.{decimals}f})"
        cells.append(text)
    lines.append(f"| {name} | " + " | ".join(cells) + " |")

table = "\n".join(lines)
print(table)

Path("results").mkdir(exist_ok=True)
Path("results/results_table.md").write_text(table + "\n")
print()
print("saved results/results_table.md")
