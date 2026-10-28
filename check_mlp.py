import torch
from data import get_model_data, standardize
from mlp import VMDurationNet

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

train_scaled, validation_scaled, test_scaled = standardize(
    train_features, validation_features, test_features
)

train_inputs = torch.tensor(train_scaled.values, dtype=torch.float32)

torch.manual_seed(42)
model = VMDurationNet()

first_jobs = train_inputs[:5]
predictions = model(first_jobs)

print("input shape:", first_jobs.shape)
print("output shape:", predictions.shape)
print("predictions (random, untrained):", predictions)
print("parameters:", sum(parameter.numel() for parameter in model.parameters()))