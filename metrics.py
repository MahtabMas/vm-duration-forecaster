import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score_log_scale(true_target, predicted_target):
    # MAE, RMSE and R2 on the log scale.
    return {
        "mae": mean_absolute_error(true_target, predicted_target),
        "rmse": np.sqrt(mean_squared_error(true_target, predicted_target)),
        "r2": r2_score(true_target, predicted_target),
    }

def score_minutes(true_target, predicted_target):
    # Errors in minutes: convert back from the log scale first.
    true_minutes = np.expm1(true_target)
    predicted_minutes = np.expm1(predicted_target)
    errors = np.abs(true_minutes - predicted_minutes)
    return {
        "mae_minutes": errors.mean(),
        "median_abs_error_minutes": np.median(errors),
    }

def score_by_bucket(true_target, predicted_target):
    # Scores for short, medium and long jobs (grouped by true duration).
    true_target = np.asarray(true_target)
    predicted_target = np.asarray(predicted_target)
    true_minutes = np.expm1(true_target)
    predicted_minutes = np.expm1(predicted_target)
    log_error = predicted_target - true_target

    buckets = {
        "short (under 30 min)": true_minutes < 30,
        "medium (30 min to 12 h)": (true_minutes >= 30) & (true_minutes < 720),
        "long (12 h or more)": true_minutes >= 720,
    }

    rows = {}
    for name, mask in buckets.items():
        rows[name] = {
            "jobs": int(mask.sum()),
            "log_mae": np.abs(log_error[mask]).mean(),
            "bias": log_error[mask].mean(),
            "median_abs_error_minutes": np.median(
                np.abs(true_minutes[mask] - predicted_minutes[mask])
            ),
        }
    return rows