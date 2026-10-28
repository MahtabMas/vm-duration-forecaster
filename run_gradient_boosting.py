import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ParameterSampler
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

# Tuning: 15 random combinations, scored on validation
search_space = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "min_samples_leaf": [1, 10, 50],
}

N_TRIALS = 15
trials = list(ParameterSampler(search_space, n_iter=N_TRIALS, random_state=42))

results = []
for trial_number, settings in enumerate(trials, start=1):
    start = time.time()
    tuned_model = GradientBoostingRegressor(random_state=42, **settings)
    tuned_model.fit(train_features, train_target)
    seconds = time.time() - start

    predictions = tuned_model.predict(validation_features)
    scores = score_log_scale(validation_target, predictions)

    results.append((scores["mae"], scores["rmse"], scores["r2"], seconds, settings))
    print(f"trial {trial_number}/{N_TRIALS}: mae {scores['mae']:.4f}  ({seconds:.0f}s)  {settings}")

results.sort(key=lambda row: row[0])
best_mae, best_rmse, best_r2, best_seconds, best_settings = results[0]

print()
print("best settings:", best_settings)
print("best validation mae:", round(best_mae, 4))
print("best validation rmse:", round(best_rmse, 4))
print("best validation r2:", round(best_r2, 4))