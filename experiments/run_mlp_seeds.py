import numpy as np
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

SEEDS = [0, 1, 2, 3, 4]

configurations = {
    "original settings": {},
    "tuned settings": {
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "dropout_rate": 0.0,
        "batch_size": 1024,
    },
}

for configuration_name, settings in configurations.items():
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
        print(f"  {configuration_name}, seed {seed}: mae {scores['mae']:.4f}  "
              f"rmse {scores['rmse']:.4f}  epoch {best_epoch}")

    print(f"{configuration_name} over {len(SEEDS)} seeds (mean, spread):")
    print("  mae :", round(np.mean(maes), 4), round(np.std(maes, ddof=1), 4))
    print("  rmse:", round(np.mean(rmses), 4), round(np.std(rmses, ddof=1), 4))
    print("  r2  :", round(np.mean(r2s), 4), round(np.std(r2s, ddof=1), 4))
    print()
    