import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ParameterSampler
from data import get_model_data
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()
print("number of input features:", train_features.shape[1])

# Reference: the original settings on these inputs
reference_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
).fit(train_features, train_target)
reference = score_log_scale(validation_target, reference_model.predict(validation_features))
print(f"original settings (reference): mae {reference['mae']:.4f}  "
      f"rmse {reference['rmse']:.4f}  r2 {reference['r2']:.4f}")
print()

search_space = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.02, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "min_samples_leaf": [1, 10, 50, 100],
}

N_TRIALS = 25
trials = list(ParameterSampler(search_space, n_iter=N_TRIALS, random_state=42))

results = []
for trial_number, settings in enumerate(trials, start=1):
    start = time.time()
    model = GradientBoostingRegressor(random_state=42, **settings)
    model.fit(train_features, train_target)
    seconds = time.time() - start

    scores = score_log_scale(validation_target, model.predict(validation_features))
    results.append((scores["mae"], scores["rmse"], scores["r2"], settings))
    print(f"trial {trial_number:2d}/{N_TRIALS}: mae {scores['mae']:.4f}  ({seconds:.0f}s)  {settings}")

results.sort(key=lambda row: row[0])
best_mae, best_rmse, best_r2, best_settings = results[0]

print()
print("best settings:", best_settings)
print("best validation mae:", round(best_mae, 4))
print("best validation rmse:", round(best_rmse, 4))
print("best validation r2:", round(best_r2, 4))