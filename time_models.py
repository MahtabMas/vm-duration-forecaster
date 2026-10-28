import time
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from data import get_model_data, standardize
from mlp import train_mlp

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

train_scaled, validation_scaled, test_scaled = standardize(
    train_features, validation_features, test_features
)
train_inputs = torch.tensor(train_scaled.values, dtype=torch.float32)
train_targets = torch.tensor(train_target.values, dtype=torch.float32)
validation_inputs = torch.tensor(validation_scaled.values, dtype=torch.float32)
validation_targets = torch.tensor(validation_target.values, dtype=torch.float32)
test_inputs = torch.tensor(test_scaled.values, dtype=torch.float32)

TUNED_SETTINGS = {
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "dropout_rate": 0.0,
    "batch_size": 1024,
}


def average_seconds(function, repeats=20):
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - start) / repeats


rows = []

start = time.perf_counter()
linear_model = LinearRegression().fit(train_features, train_target)
training_seconds = time.perf_counter() - start
rows.append(("linear regression", training_seconds,
             average_seconds(lambda: linear_model.predict(test_features))))

start = time.perf_counter()
gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
).fit(train_features, train_target)
training_seconds = time.perf_counter() - start
rows.append(("gradient boosting", training_seconds,
             average_seconds(lambda: gb_model.predict(test_features))))

start = time.perf_counter()
mlp_model, best_epoch = train_mlp(
    train_inputs, train_targets, validation_inputs, validation_targets,
    seed=0, **TUNED_SETTINGS,
)
training_seconds = time.perf_counter() - start


def mlp_predict():
    with torch.no_grad():
        mlp_model(test_inputs)


rows.append(("mlp (tuned, seed 0)", training_seconds, average_seconds(mlp_predict)))

print(f"prediction time is for all {len(test_features)} test jobs together")
print("model                 training (s)   prediction (ms)")
for name, training_seconds, prediction_seconds in rows:
    print(f"{name:20s} {training_seconds:10.1f} {prediction_seconds * 1000:14.1f}")