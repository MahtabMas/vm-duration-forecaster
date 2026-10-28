import time
import torch
from data import get_model_data, standardize
from mlp import train_mlp
from metrics import score_log_scale

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

start_time = time.time()
model, best_epoch = train_mlp(
    train_inputs, train_targets, validation_inputs, validation_targets
)
training_seconds = time.time() - start_time

with torch.no_grad():
    validation_predictions = model(validation_inputs).numpy()
scores = score_log_scale(validation_target, validation_predictions)

print("best epoch:", best_epoch)
print("mlp with early stopping on validation:")
for name, value in scores.items():
    print(" ", name, round(value, 4))
print("training time (seconds):", round(training_seconds, 1))