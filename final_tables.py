import numpy as np
from metrics import score_minutes, score_by_bucket

saved = np.load("results/test_predictions.npz")
true_target = saved["true_target"]

single_models = ["median_baseline", "linear_regression", "gradient_boosting"]
mlp_names = [f"mlp_seed_{seed}" for seed in range(5)]

print("TEST SET, errors in minutes")
for name in single_models:
    scores = score_minutes(true_target, saved[name])
    print(f"{name}: mae {scores['mae_minutes']:.1f} min, "
          f"median abs error {scores['median_abs_error_minutes']:.1f} min")

mlp_scores = [score_minutes(true_target, saved[name]) for name in mlp_names]
print("mlp (mean of 5 seeds): mae {:.1f} min, median abs error {:.1f} min".format(
    np.mean([s["mae_minutes"] for s in mlp_scores]),
    np.mean([s["median_abs_error_minutes"] for s in mlp_scores]),
))


def print_buckets(name, rows):
    print(name + ":")
    for bucket_name, row in rows.items():
        print(f"  {bucket_name}: jobs {row['jobs']}, log mae {row['log_mae']:.3f}, "
              f"bias {row['bias']:+.3f}, "
              f"median abs error {row['median_abs_error_minutes']:.1f} min")


print()
print("TEST SET, by duration bucket")
for name in single_models:
    print_buckets(name, score_by_bucket(true_target, saved[name]))

mlp_rows = [score_by_bucket(true_target, saved[name]) for name in mlp_names]
averaged = {}
for bucket_name in mlp_rows[0]:
    averaged[bucket_name] = {
        "jobs": mlp_rows[0][bucket_name]["jobs"],
        "log_mae": np.mean([rows[bucket_name]["log_mae"] for rows in mlp_rows]),
        "bias": np.mean([rows[bucket_name]["bias"] for rows in mlp_rows]),
        "median_abs_error_minutes": np.mean(
            [rows[bucket_name]["median_abs_error_minutes"] for rows in mlp_rows]
        ),
    }
print_buckets("mlp (mean of 5 seeds)", averaged)

print()
print("five largest predictions on the test days, in minutes:")
for name in single_models:
    largest = np.sort(np.expm1(saved[name]))[-5:]
    print(f"  {name}: {largest.round(0)}")
print("  for reference, the largest true duration:", round(np.expm1(true_target).max(), 0))
