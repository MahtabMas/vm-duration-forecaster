import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from data import get_model_data, load_and_split

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

_, validation_data, test_data = load_and_split()

gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
).fit(train_features, train_target)

held_out_features = pd.concat([validation_features, test_features])
held_out_target = np.concatenate([validation_target, test_target])
held_out_days = np.concatenate([validation_data["day"].values, test_data["day"].values])
assert len(held_out_days) == len(held_out_features)

predictions = gb_model.predict(held_out_features)

print("day  jobs   true mean   predicted mean   mae")
for day in range(10, 14):
    mask = held_out_days == day
    print(f"{day:3d}  {mask.sum():5d}   {held_out_target[mask].mean():8.3f}   "
          f"{predictions[mask].mean():13.3f}   {np.abs(predictions[mask] - held_out_target[mask]).mean():.4f}")