import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from data import get_model_data

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()


def fit_quantile_model(quantile):
    model = GradientBoostingRegressor(
        loss="quantile", alpha=quantile,
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
    )
    return model.fit(train_features, train_target)


lower_model = fit_quantile_model(0.05)
upper_model = fit_quantile_model(0.95)


def check_interval(name, features, target):
    lower = lower_model.predict(features)
    upper = upper_model.predict(features)
    true_values = np.asarray(target)

    inside = (true_values >= lower) & (true_values <= upper)
    below = true_values < lower
    above = true_values > upper
    width = upper - lower

    print(f"{name}:")
    print(f"  coverage (target 90%): {inside.mean():.1%}")
    print(f"  below the lower bound (target 5%): {below.mean():.1%}")
    print(f"  above the upper bound (target 5%): {above.mean():.1%}")
    print(f"  median width, log scale: {np.median(width):.2f}"
          f"  (upper bound is roughly {np.exp(np.median(width)):.0f} times the lower)")
    print(f"  jobs where lower bound is above upper bound: {int((lower > upper).sum())}")


check_interval("validation days", validation_features, validation_target)
check_interval("test days", test_features, test_target)