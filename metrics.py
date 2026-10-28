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