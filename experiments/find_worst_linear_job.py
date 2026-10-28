import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from data import get_model_data, load_and_split
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()
_, validation_data, _ = load_and_split()

linear_model = LinearRegression().fit(train_features, train_target)
predictions = linear_model.predict(validation_features)
worst = int(np.argmax(predictions))

print("intercept:", round(float(linear_model.intercept_), 3))
print("feature   coefficient   training min   training max   this job")
for name, coefficient in zip(train_features.columns, linear_model.coef_):
    print(f"{name:14s} {coefficient:+11.2f}   {train_features[name].min():10.4f}   "
          f"{train_features[name].max():10.4f}   {validation_features[name].iloc[worst]:.4f}")

gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
).fit(train_features, train_target)
job_row = validation_features.iloc[[worst]]

print()
print("raw starttime of this job:", validation_data["starttime"].iloc[worst])
print("true duration (minutes):", round(float(np.expm1(validation_target.iloc[worst])), 1))
print("linear prediction (minutes):", round(float(np.expm1(predictions[worst])), 0))
print("gradient boosting prediction (minutes):", round(float(np.expm1(gb_model.predict(job_row)[0])), 1))

keep = np.arange(len(predictions)) != worst
all_scores = score_log_scale(validation_target, predictions)
without_scores = score_log_scale(validation_target[keep], predictions[keep])
print()
print("linear regression on validation, log scale:")
print("  with this job:    mae {:.4f}  rmse {:.4f}  r2 {:.4f}".format(
    all_scores["mae"], all_scores["rmse"], all_scores["r2"]))
print("  without this job: mae {:.4f}  rmse {:.4f}  r2 {:.4f}".format(
    without_scores["mae"], without_scores["rmse"], without_scores["r2"]))