from data import get_model_data, standardize

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

train_scaled, validation_scaled, test_scaled = standardize(
    train_features, validation_features, test_features
)

print("train average (should be about 0):")
print(train_scaled.mean().round(3))
print("train spread (should be about 1):")
print(train_scaled.std().round(3))
print("validation average (close to 0, but not exactly):")
print(validation_scaled.mean().round(3))