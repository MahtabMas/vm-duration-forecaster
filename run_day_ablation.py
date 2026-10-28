from sklearn.ensemble import GradientBoostingRegressor
from data import get_model_data
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

DAY_FEATURES = ["day_sin", "day_cos", "is_weekend"]
all_columns = list(train_features.columns)
columns_without_day = [name for name in all_columns if name not in DAY_FEATURES]

feature_sets = {
    "all 11 features": all_columns,
    "without day-of-week features (8)": columns_without_day,
}

for name, columns in feature_sets.items():
    gb_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
    )
    gb_model.fit(train_features[columns], train_target)
    predictions = gb_model.predict(validation_features[columns])
    scores = score_log_scale(validation_target, predictions)

    print(name + ":")
    for metric_name, value in scores.items():
        print(" ", metric_name, round(value, 4))
        