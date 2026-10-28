import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from data import get_model_data
from metrics import score_log_scale

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

# Pool the days 0 to 11 rows only. The test rows stay untouched.
pool_features = pd.concat([train_features, validation_features])
pool_target = pd.concat([train_target, validation_target])

random_generator = np.random.default_rng(42)
shuffled_order = random_generator.permutation(len(pool_features))

number_of_train_rows = len(train_features)
random_train_rows = shuffled_order[:number_of_train_rows]
random_validation_rows = shuffled_order[number_of_train_rows:]

gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
)
gb_model.fit(pool_features.iloc[random_train_rows], pool_target.iloc[random_train_rows])

predictions = gb_model.predict(pool_features.iloc[random_validation_rows])
scores = score_log_scale(pool_target.iloc[random_validation_rows], predictions)

print("train rows:", len(random_train_rows), " validation rows:", len(random_validation_rows))
print("gradient boosting, RANDOM split:")
for name, value in scores.items():
    print(" ", name, round(value, 4))
print("gradient boosting, TIME-BASED split (from before): mae 1.1646  rmse 1.6439  r2 0.428")