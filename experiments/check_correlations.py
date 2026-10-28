from data import get_model_data

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

resource_columns = ["core", "memory", "ssd", "nic", "resource_load"]
print("correlation between resource features (training days):")
print(train_features[resource_columns].corr().round(3))

print()
print("correlation of each resource feature with the target (training days):")
for name in resource_columns:
    print(f"  {name:15s} {train_features[name].corr(train_target):+.3f}")