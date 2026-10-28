import time
import torch
from sklearn.model_selection import ParameterSampler
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

search_space = {
    "learning_rate": [0.0003, 0.001, 0.003, 0.01],
    "weight_decay": [0.0, 0.01, 0.1],
    "dropout_rate": [0.0, 0.1, 0.2, 0.3],
    "batch_size": [128, 256, 512, 1024],
}

N_TRIALS = 15
trials = list(ParameterSampler(search_space, n_iter=N_TRIALS, random_state=42))

results = []
for trial_number, settings in enumerate(trials, start=1):
    start = time.time()
    model, best_epoch = train_mlp(
        train_inputs, train_targets, validation_inputs, validation_targets,
        **settings,
    )
    seconds = time.time() - start

    with torch.no_grad():
        predictions = model(validation_inputs).numpy()
    scores = score_log_scale(validation_target, predictions)

    results.append((scores["mae"], scores["rmse"], scores["r2"], best_epoch, settings))
    print(f"trial {trial_number}/{N_TRIALS}: mae {scores['mae']:.4f}  "
          f"rmse {scores['rmse']:.4f}  epoch {best_epoch}  ({seconds:.0f}s)  {settings}")

results.sort(key=lambda row: row[0])
best_mae, best_rmse, best_r2, best_epoch, best_settings = results[0]

print()
print("best settings:", best_settings)
print("best epoch:", best_epoch)
print("best validation mae:", round(best_mae, 4))
print("best validation rmse:", round(best_rmse, 4))
print("best validation r2:", round(best_r2, 4))