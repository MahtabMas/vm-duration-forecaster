import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from data import get_model_data, load_and_split

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

_, validation_data, test_data = load_and_split()

resource_columns = ["core", "memory", "ssd", "nic"]


def make_keys(features):
    return features[resource_columns].round(6).astype(str).apply("-".join, axis=1)


gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
).fit(train_features, train_target)

held_out_features = pd.concat([validation_features, test_features])
held_out_target = np.concatenate([validation_target, test_target])
held_out_days = np.concatenate([validation_data["day"].values, test_data["day"].values])
held_out_keys = make_keys(held_out_features).values
predictions = gb_model.predict(held_out_features)

train_keys = make_keys(train_features)
top_types = train_keys.value_counts().head(4).index

for key in top_types:
    print(f"type {key}  (share of training jobs {(train_keys == key).mean():.1%})")
    print("  day  share of day   true mean   predicted mean")
    for day in range(10, 14):
        day_mask = held_out_days == day
        type_mask = day_mask & (held_out_keys == key)
        if type_mask.sum() == 0:
            continue
        print(f"  {day:3d}  {type_mask.sum() / day_mask.sum():11.1%}   "
              f"{held_out_target[type_mask].mean():9.3f}   {predictions[type_mask].mean():13.3f}")
    print()