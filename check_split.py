from data import get_model_data

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

print("train:", train_features.shape, train_target.shape)
print("validation:", validation_features.shape, validation_target.shape)
print("test:", test_features.shape, test_target.shape)
print("any missing values:", train_features.isna().any().any(), train_target.isna().any())