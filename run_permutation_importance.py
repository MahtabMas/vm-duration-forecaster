from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from data import get_model_data

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
).fit(train_features, train_target)

importance = permutation_importance(
    gb_model, validation_features, validation_target,
    scoring="neg_mean_absolute_error", n_repeats=10, random_state=42,
)

print("rise in validation MAE when one feature is shuffled (bigger = more important):")
order = importance.importances_mean.argsort()[::-1]
for position in order:
    name = validation_features.columns[position]
    print(f"  {name:15s} {importance.importances_mean[position]:+.4f}  "
          f"(spread {importance.importances_std[position]:.4f})")