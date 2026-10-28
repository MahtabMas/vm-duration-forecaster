import time
from sklearn.ensemble import GradientBoostingRegressor
from data import get_model_data
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

# The original settings from model.py
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
)

start = time.time()
gb_model.fit(train_features, train_target)
train_seconds = time.time() - start

gb_predictions = gb_model.predict(validation_features)
gb_scores = score_log_scale(validation_target, gb_predictions)

print("gradient boosting (original settings) on validation:")
for name, value in gb_scores.items():
    print(" ", name, round(value, 4))
print("training time (seconds):", round(train_seconds, 1))