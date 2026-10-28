# Experiments

Run from the project folder with the environment active, for example `python -m experiments.run_baselines`. Scripts that read saved predictions need `run_final_test` and `run_round2_test` run first.

**Main results**
- `run_baselines`: median and linear baselines (validation)
- `run_final_test`, `run_round2_test`: test-day results for round 1 and round 2, and the saved predictions
- `make_results_table`, `final_tables`: results tables from the saved predictions
- `bootstrap_compare`, `bootstrap_blocks`: confidence intervals for Gradient Boosting vs the MLP, for single jobs and for whole hours

**Tuning**
- `run_gradient_boosting`, `tune_mlp`: round-1 searches
- `tune_round2_gb`, `tune_round2_mlp`: round-2 searches
- `run_mlp_seeds`: the MLP over five seeds

**Analysis**
- `run_time_ablation`, `run_day_ablation`: what the time-of-day and day-of-week features add
- `run_random_split`: random split vs time-based split
- `run_prediction_intervals`: the 90% interval and its coverage
- `run_rolling_origin`, `run_training_size`: how the ranking changes across periods and training sizes

**Checks**
- `check_features`: the fast feature code against the original row-by-row code
- `check_vm_types`: the 12 distinct resource combinations