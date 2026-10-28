import copy
import torch
from torch import nn


class VMDurationNet(nn.Module):
    def __init__(self, number_of_features=8, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(number_of_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, job_features):
        return self.layers(job_features).squeeze(1)


def train_mlp(train_inputs, train_targets, validation_inputs, validation_targets,
              seed=42, learning_rate=0.001, weight_decay=0.01, batch_size=256,
              dropout_rate=0.1, max_epochs=100, patience=10):
    # Train one network with early stopping.
    # Returns the network with its best weights loaded, and the best epoch.
    torch.manual_seed(seed)
    model = VMDurationNet(dropout_rate=dropout_rate)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        shuffled_order = torch.randperm(len(train_inputs))

        for start in range(0, len(train_inputs), batch_size):
            batch_rows = shuffled_order[start:start + batch_size]
            predictions = model(train_inputs[batch_rows])
            loss = loss_function(predictions, train_targets[batch_rows])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(validation_inputs), validation_targets
            ).item()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    return model, best_epoch