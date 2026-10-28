import time
import torch
from torch import nn
from data import get_model_data, standardize
from mlp import VMDurationNet
from metrics import score_log_scale

BATCH_SIZE = 256
NUMBER_OF_EPOCHS = 15

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

train_scaled, validation_scaled, test_scaled = standardize(
    train_features, validation_features, test_features
)

train_inputs = torch.tensor(train_scaled.values, dtype=torch.float32)
train_targets = torch.tensor(train_target.values, dtype=torch.float32)
validation_inputs = torch.tensor(validation_scaled.values, dtype=torch.float32)

torch.manual_seed(42)
model = VMDurationNet()
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

start_time = time.time()

for epoch in range(1, NUMBER_OF_EPOCHS + 1):
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
        validation_predictions = model(validation_inputs).numpy()
    scores = score_log_scale(validation_target, validation_predictions)

    print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  "
          f"validation mae {scores['mae']:.4f}  rmse {scores['rmse']:.4f}  r2 {scores['r2']:.4f}")

print("training time (seconds):", round(time.time() - start_time, 1))