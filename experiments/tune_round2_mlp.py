import time
import numpy as np
import torch
from sklearn.model_selection import ParameterSampler
from data import get_model_data, standardize
from mlp import train_mlp
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()
print("number of input features:", train_features.shape[1])

train_scaled, validation_scaled, test_scaled = standardize(
    train_features, validation_features, test_features
)
train_inputs = torch.tensor(train_scaled.values, dtype=torch.float32)
train_targets = torch.tensor(train_target.values, dtype=torch.float32)
validation_inputs = torch.tensor(validation_scaled.values, dtype=torch.float32)
validation_targets = torch.tensor(validation_target.values, dtype=torch.float32)

SEEDS = [0, 1, 2]


def score_configuration(settings):
    maes, rmses, r2s, epochs = [], [], [], []
    for seed in SEEDS:
        model, best_epoch = train_mlp(
            train_inputs, train_targets, validation_inputs, validation_targets,
            seed=seed, **settings,
        )
        with torch.no_grad():
            predictions = model(validation_inputs).numpy()
        scores = score_log_scale(validation_target, predictions)
        maes.append(scores["mae"])
        rmses.append(scores["rmse"])
        r2s.append(scores["r2"])
        epochs.append(best_epoch)
    return np.mean(maes), np.std(maes, ddof=1), np.mean(rmses), np.mean(r2s), epochs


# Reference: the round-1 tuned settings, same three seeds
round1_settings = {"learning_rate": 0.01, "weight_decay": 0.0, "dropout_rate": 0.0, "batch_size": 1024}
mae, spread, rmse, r2, epochs = score_configuration(round1_settings)
print(f"round-1 tuned settings (reference): mean mae {mae:.4f} (spread {spread:.4f})  "
      f"rmse {rmse:.4f}  r2 {r2:.4f}")
print()

search_space = {
    "hidden_sizes": [(32, 16), (64, 32), (128, 64), (64, 64, 32), (128, 64, 32), (256, 128)],
    "learning_rate": [0.0003, 0.001, 0.003, 0.01],
    "weight_decay": [0.0, 0.01, 0.1],
    "dropout_rate": [0.0, 0.1, 0.2],
    "batch_size": [256, 512, 1024],
}

N_TRIALS = 25
trials = list(ParameterSampler(search_space, n_iter=N_TRIALS, random_state=42))

results = []
for trial_number, settings in enumerate(trials, start=1):
    start = time.time()
    mae, spread, rmse, r2, epochs = score_configuration(settings)
    seconds = time.time() - start
    results.append((mae, spread, rmse, r2, settings))
    print(f"trial {trial_number:2d}/{N_TRIALS}: mean mae {mae:.4f} (spread {spread:.4f})  "
          f"epochs {epochs}  ({seconds:.0f}s)  {settings}")

results.sort(key=lambda row: row[0])
best_mae, best_spread, best_rmse, best_r2, best_settings = results[0]

print()
print("best settings:", best_settings)
print(f"best validation mean mae: {best_mae:.4f} (spread {best_spread:.4f})")
print("best validation mean rmse:", round(best_rmse, 4))
print("best validation mean r2:", round(best_r2, 4))