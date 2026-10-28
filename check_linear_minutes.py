import numpy as np
from sklearn.linear_model import LinearRegression
from data import get_model_data

(train_features, train_target,
 validation_features, validation_target,
 test_features, test_target) = get_model_data()

linear_model = LinearRegression().fit(train_features, train_target)
baseline_minutes = np.expm1(np.median(train_target))

for set_name, features, target in [("validation", validation_features, validation_target),
                                   ("test", test_features, test_target)]:
    true_minutes = np.expm1(np.asarray(target))
    predicted_minutes = np.expm1(linear_model.predict(features))
    linear_error = np.abs(true_minutes - predicted_minutes)
    baseline_error = np.abs(true_minutes - baseline_minutes)
    high = predicted_minutes > 300
    number_of_jobs = len(true_minutes)

    print(f"{set_name} days:")
    print("  overall mae in minutes: linear {:.1f}, baseline {:.1f}".format(
        linear_error.mean(), baseline_error.mean()))
    print("  five largest linear predictions:", np.sort(predicted_minutes)[-5:].round(0))
    print(f"  jobs predicted above 300 min: {high.sum()} ({high.mean():.1%} of jobs)")
    if high.sum() > 0:
        print("    their true duration: median {:.0f} min, mean {:.0f} min".format(
            np.median(true_minutes[high]), true_minutes[high].mean()))
    print("  share of the mae that comes from those jobs: linear {:.1f}, baseline {:.1f}".format(
        linear_error[high].sum() / number_of_jobs, baseline_error[high].sum() / number_of_jobs))
    print("  share of the mae that comes from all other jobs: linear {:.1f}, baseline {:.1f}".format(
        linear_error[~high].sum() / number_of_jobs, baseline_error[~high].sum() / number_of_jobs))
    print()