import numpy as np
from data import get_model_data
from metrics import score_log_scale
from sklearn.linear_model import LinearRegression

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




# Baseline 2: linear regression on the 11 features
linear_model = LinearRegression()
linear_model.fit(train_features, train_target)
linear_predictions = linear_model.predict(validation_features)

linear_scores = score_log_scale(validation_target, linear_predictions)

print("linear regression on validation:")
for name, value in linear_scores.items():
    print(" ", name, round(value, 4))