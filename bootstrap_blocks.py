import numpy as np
from data import load_and_split, make_target

saved = np.load("results/test_predictions.npz")
true_target = saved["true_target"]
gb_predictions = saved["gradient_boosting"]
mlp_predictions = [saved[f"mlp_seed_{seed}"] for seed in range(5)]

# Rebuild each test job's hour and check the order matches the saved file
_, _, test_data = load_and_split()
assert np.allclose(make_target(test_data).values, true_target), "job order does not match"

hour_of_trace = (test_data["starttime"] * 24).astype(int).values
block_index = hour_of_trace - hour_of_trace.min()
number_of_blocks = block_index.max() + 1


def per_block_sums(values):
    return np.bincount(block_index, weights=values, minlength=number_of_blocks)


jobs_per_block = np.bincount(block_index, minlength=number_of_blocks)
gb_abs_sums = per_block_sums(np.abs(gb_predictions - true_target))
gb_sq_sums = per_block_sums((gb_predictions - true_target) ** 2)
mlp_abs_sums = [per_block_sums(np.abs(p - true_target)) for p in mlp_predictions]
mlp_sq_sums = [per_block_sums((p - true_target) ** 2) for p in mlp_predictions]

print("hour blocks:", number_of_blocks, " empty blocks:", int((jobs_per_block == 0).sum()))

NUMBER_OF_RESAMPLES = 2000
random_generator = np.random.default_rng(42)


def differences_for(picked_blocks):
    total_jobs = jobs_per_block[picked_blocks].sum()
    gb_mae = gb_abs_sums[picked_blocks].sum() / total_jobs
    mlp_mae = np.mean([s[picked_blocks].sum() / total_jobs for s in mlp_abs_sums])
    gb_rmse = np.sqrt(gb_sq_sums[picked_blocks].sum() / total_jobs)
    mlp_rmse = np.mean([np.sqrt(s[picked_blocks].sum() / total_jobs) for s in mlp_sq_sums])
    return gb_mae - mlp_mae, gb_rmse - mlp_rmse


observed_mae, observed_rmse = differences_for(np.arange(number_of_blocks))

mae_differences, rmse_differences = [], []
for _ in range(NUMBER_OF_RESAMPLES):
    picked_blocks = random_generator.integers(0, number_of_blocks, number_of_blocks)
    mae_difference, rmse_difference = differences_for(picked_blocks)
    mae_differences.append(mae_difference)
    rmse_differences.append(rmse_difference)


def show(name, differences, observed_difference):
    differences = np.array(differences)
    low, high = np.percentile(differences, [2.5, 97.5])
    share_gb_better = (differences < 0).mean()
    print(f"{name}: observed difference {observed_difference:+.4f}, "
          f"95% interval [{low:+.4f}, {high:+.4f}], "
          f"gradient boosting better in {share_gb_better:.1%} of resamples")


print("difference = gradient boosting minus MLP (negative means gradient boosting is better)")
show("MAE ", mae_differences, observed_mae)
show("RMSE", rmse_differences, observed_rmse)