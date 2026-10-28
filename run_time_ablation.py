from sklearn.ensemble import GradientBoostingRegressor
from data import get_model_data
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

TIME_FEATURES = ["hour_sin", "hour_cos", "is_business"]
all_columns = list(train_features.columns)
columns_without_time = [name for name in all_columns if name not in TIME_FEATURES]

results = {}
for version_name, columns in [("with time features (8)", all_columns),
                              ("without time features (5)", columns_without_time)]:
    gb_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
    )
    gb_model.fit(train_features[columns], train_target)

    validation_scores = score_log_scale(validation_target, gb_model.predict(validation_features[columns]))
    test_scores = score_log_scale(test_target, gb_model.predict(test_features[columns]))
    results[version_name] = (validation_scores, test_scores)

    print(version_name + ":")
    print(f"  validation: mae {validation_scores['mae']:.4f}  rmse {validation_scores['rmse']:.4f}  r2 {validation_scores['r2']:.4f}")
    print(f"  test:       mae {test_scores['mae']:.4f}  rmse {test_scores['rmse']:.4f}  r2 {test_scores['r2']:.4f}")

print()
with_time = results["with time features (8)"]
without_time = results["without time features (5)"]
for index, set_name in enumerate(["validation", "test"]):
    r2_with = with_time[index]["r2"]
    r2_without = without_time[index]["r2"]
    share_lost = (r2_with - r2_without) / r2_with
    print(f"{set_name}: r2 drops from {r2_with:.4f} to {r2_without:.4f}, "
          f"share of r2 lost without time features: {share_lost:.1%}")