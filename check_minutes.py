import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from data import get_model_data
from metrics import score_minutes

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

predictions = {}

predictions["median baseline"] = np.full(
    len(validation_target), np.median(train_target)
)

linear_model = LinearRegression().fit(train_features, train_target)
predictions["linear regression"] = linear_model.predict(validation_features)

gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
).fit(train_features, train_target)
predictions["gradient boosting"] = gb_model.predict(validation_features)

for name, predicted in predictions.items():
    scores = score_minutes(validation_target, predicted)
    print(f"{name}:")
    print("  mae (minutes):", round(scores["mae_minutes"], 1))
    print("  median absolute error (minutes):", round(scores["median_abs_error_minutes"], 1))