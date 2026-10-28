import numpy as np

saved = np.load("results/test_predictions.npz")
true_target = saved["true_target"]
gb_predictions = saved["gradient_boosting"]
mlp_predictions = [saved[f"mlp_seed_{seed}"] for seed in range(5)]

gb_abs_error = np.abs(gb_predictions - true_target)
mlp_abs_errors = [np.abs(p - true_target) for p in mlp_predictions]
gb_squared_error = (gb_predictions - true_target) ** 2
mlp_squared_errors = [(p - true_target) ** 2 for p in mlp_predictions]

number_of_jobs = len(true_target)
NUMBER_OF_RESAMPLES = 2000
random_generator = np.random.default_rng(42)

mae_differences = []
rmse_differences = []
for _ in range(NUMBER_OF_RESAMPLES):
    picked_jobs = random_generator.integers(0, number_of_jobs, number_of_jobs)

    gb_mae = gb_abs_error[picked_jobs].mean()
    mlp_mae = np.mean([errors[picked_jobs].mean() for errors in mlp_abs_errors])
    mae_differences.append(gb_mae - mlp_mae)

    gb_rmse = np.sqrt(gb_squared_error[picked_jobs].mean())
    mlp_rmse = np.mean([np.sqrt(errors[picked_jobs].mean()) for errors in mlp_squared_errors])
    rmse_differences.append(gb_rmse - mlp_rmse)


def show(name, differences, observed_difference):
    differences = np.array(differences)
    low, high = np.percentile(differences, [2.5, 97.5])
    share_gb_better = (differences < 0).mean()
    print(f"{name}: observed difference {observed_difference:+.4f}, "
          f"95% interval [{low:+.4f}, {high:+.4f}], "
          f"gradient boosting better in {share_gb_better:.1%} of resamples")


observed_mae = gb_abs_error.mean() - np.mean([e.mean() for e in mlp_abs_errors])
observed_rmse = np.sqrt(gb_squared_error.mean()) - np.mean(
    [np.sqrt(e.mean()) for e in mlp_squared_errors]
)

print("difference = gradient boosting minus MLP (negative means gradient boosting is better)")
show("MAE ", mae_differences, observed_mae)
show("RMSE", rmse_differences, observed_rmse)