| Model | MAE (log) | RMSE (log) | R2 (log) | MAE (min) | Median abs error (min) |
|---|---|---|---|---|---|
| Median baseline | 1.6346 | 2.1136 | -0.0262 | 406.8 | 34.3 |
| Linear regression | 1.2403 | 1.7226 | 0.3184 | 409.9 | 35.4 |
| Lookup by VM type (added after the test run) | 1.2352 | 1.7207 | 0.3199 | 411.3 | 31.3 |
| Gradient Boosting (8 features) | 1.2027 | 1.6912 | 0.3431 | 409.5 | 29.6 |
| MLP, tuned (mean of 5 seeds) | 1.2265 (sd 0.0106) | 1.7193 (sd 0.0113) | 0.3210 (sd 0.0089) | 412.7 (sd 4.3) | 28.5 (sd 0.8) |
