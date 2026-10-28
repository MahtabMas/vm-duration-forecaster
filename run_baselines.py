import numpy as np
from data import get_model_data
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

# Baseline 1: predict the training median for every job
median_value = np.median(train_target)
median_predictions = np.full(len(validation_target), median_value)

scores = score_log_scale(validation_target, median_predictions)

print("median of training target (log scale):", round(median_value, 4))
print("median baseline on validation:")
for name, value in scores.items():
    print(" ", name, round(value, 4))