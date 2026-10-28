import numpy as np
import pandas as pd
from data import get_model_data
from metrics import score_log_scale, score_minutes

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

resource_columns = ["core", "memory", "ssd", "nic"]


def make_keys(features):
    return features[resource_columns].round(6).astype(str).apply("-".join, axis=1)


train_keys = make_keys(train_features)
test_keys = make_keys(test_features)

average_by_type = pd.DataFrame({"key": train_keys, "target": train_target}).groupby("key")["target"].mean()

seen = test_keys.isin(average_by_type.index)
print(f"test jobs whose resource combination appears in training: {seen.mean():.1%} "
      f"({int((~seen).sum())} jobs fall back to the training median)")

lookup_predictions = test_keys.map(average_by_type).fillna(np.median(train_target)).values

log_scores = score_log_scale(test_target, lookup_predictions)
minute_scores = score_minutes(test_target, lookup_predictions)

print()
print("lookup baseline (one average per VM type, no model) on the TEST days:")
print(f"  log scale: mae {log_scores['mae']:.4f}  rmse {log_scores['rmse']:.4f}  r2 {log_scores['r2']:.4f}")
print(f"  minutes:   mae {minute_scores['mae_minutes']:.1f}  "
      f"median abs error {minute_scores['median_abs_error_minutes']:.1f}")

print()
print("for comparison, on the test days (log scale):")
print("  gradient boosting, 8 features:           mae 1.2027  rmse 1.6912  r2 0.3431")
print("  gradient boosting, without time features: mae 1.2352  rmse 1.7207  r2 0.3199")