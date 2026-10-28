import copy
import time
import torch
from torch import nn
from data import get_model_data, standardize
from mlp import VMDurationNet
from metrics import score_log_scale

BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 10

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

torch.manual_seed(42)
model = VMDurationNet()
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

best_validation_loss = float("inf")
best_epoch = 0
best_weights = None
epochs_without_improvement = 0

start_time = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    shuffled_order = torch.randperm(len(train_inputs))
    total_loss = 0.0

    for start in range(0, len(train_inputs), BATCH_SIZE):
        batch_rows = shuffled_order[start:start + BATCH_SIZE]
        batch_inputs = train_inputs[batch_rows]
        batch_targets = train_targets[batch_rows]

        predictions = model(batch_inputs)
        loss = loss_function(predictions, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_rows)

    train_loss = total_loss / len(train_inputs)

    model.eval()
    with torch.no_grad():
        validation_predictions = model(validation_inputs)
        validation_loss = loss_function(validation_predictions, validation_targets).item()
    scores = score_log_scale(validation_target, validation_predictions.numpy())

    print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  "
          f"validation loss {validation_loss:.4f}  mae {scores['mae']:.4f}")

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_epoch = epoch
        best_weights = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"no improvement for {PATIENCE} epochs, stopping")
            break

training_seconds = time.time() - start_time

# Go back to the best weights, not the last ones
model.load_state_dict(best_weights)
model.eval()
with torch.no_grad():
    validation_predictions = model(validation_inputs).numpy()
final_scores = score_log_scale(validation_target, validation_predictions)

print()
print("best epoch:", best_epoch)
print("mlp with early stopping on validation:")
for name, value in final_scores.items():
    print(" ", name, round(value, 4))
print("training time (seconds):", round(training_seconds, 1))