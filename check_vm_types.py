import numpy as np
import pandas as pd
from data import get_model_data
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

resource_columns = ["core", "memory", "ssd", "nic"]


def make_keys(features):
    rounded = features[resource_columns].round(6)
    return rounded.astype(str).apply("-".join, axis=1)


train_keys = make_keys(train_features)
validation_keys = make_keys(validation_features)

table = pd.DataFrame({"key": train_keys, "target": train_target})
group_stats = table.groupby("key")["target"].agg(["size", "mean"])
group_stats = group_stats.sort_values("size", ascending=False)

cumulative_share = group_stats["size"].cumsum() / group_stats["size"].sum()
print("distinct resource combinations in training days:", len(group_stats))
print("combinations covering 50% of training jobs:", int((cumulative_share < 0.5).sum()) + 1)
print("combinations covering 90% of training jobs:", int((cumulative_share < 0.9).sum()) + 1)

print()
print("five most common combinations (key = core-memory-ssd-nic):")
for key, row in group_stats.head(5).iterrows():
    share = row["size"] / group_stats["size"].sum()
    print(f"  {key}: {int(row['size'])} jobs ({share:.1%}), mean log duration {row['mean']:.2f}")

seen = validation_keys.isin(group_stats.index)
print()
print(f"validation jobs whose combination appears in training: {seen.mean():.1%}")

lookup_predictions = validation_keys.map(group_stats["mean"]).fillna(np.median(train_target))
scores = score_log_scale(validation_target, lookup_predictions.values)
print("lookup baseline on validation (no model, one average per combination):")
for name, value in scores.items():
    print(" ", name, round(value, 4))
print("for comparison, gradient boosting on validation: mae 1.1602  rmse 1.6441  r2 0.4279")