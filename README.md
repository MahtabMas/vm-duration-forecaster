# VM Duration Forecaster

## What this project does

When a job is submitted to a cloud provider, a scheduler would like to know how long it will run, but that is not known in advance. This project predicts the duration of Azure VM jobs from the resources a job requests and the hour it is submitted. It compares two kinds of model, Gradient Boosting and a neural network (a PyTorch MLP), and tests them carefully: on days they never trained on, with several random seeds, and with confidence intervals for the difference. It extends my doctoral research on cloud scheduling at Concordia University.

## Data

112,552 jobs from the Azure workload trace (Hadary et al., OSDI 2020). I kept one machine and priority 0 jobs, and only jobs that finished within the trace (302 jobs, 0.27%, were still running at the end and were removed, so the longest jobs are missing). The CSV is not in the repository. The subset was selected with the query in `sql/build_subset.sql`, which keeps one machine, priority 0 jobs, and jobs that finished within the trace.

- **Target:** the job duration in minutes, on a log scale. Durations are very skewed: the median is 38.8 minutes and the mean is 481.8.
- **Inputs:** the four resource sizes (CPU, memory, SSD, network), their average, and three time-of-day features.
- **Split by time:** train on days 0 to 9, tune on days 10 and 11, and test on days 12 and 13, so every model is tested on jobs that arrive after the ones it learned from. A random split of the same days scored higher for Gradient Boosting (R2 0.484 against 0.428), so a random split would have flattered the results.

The inputs are known when a job is submitted. The end time is used only to build the target, and the MLP's inputs are standardized with training-day statistics only.

## Models

- **Baselines:** always predicting the training median, linear regression, and a lookup that predicts the training average for each distinct resource combination.
- **Gradient Boosting:** scikit-learn, 400 trees of depth 7.
- **MLP:** a PyTorch network with three hidden layers (128, 64 and 32 neurons) and a training loop I wrote myself, with early stopping on the tuning days. It is trained with five different seeds, and I report the average.

Both models were tuned with the same number of random configurations (25 each, and each MLP configuration was run with 3 seeds), scored on the tuning days and not on the test days.

## Results (test days)

| Model | MAE (log scale) | R2 |
|---|---|---|
| Median baseline | 1.6346 | -0.0262 |
| Linear regression | 1.2403 | 0.3184 |
| Lookup by resource combination | 1.2352 | 0.3199 |
| MLP (mean of 5 seeds) | 1.2200 | 0.3281 |
| Gradient Boosting | 1.2025 | 0.3396 |

MAE is the average error on the log scale (lower is better). R2 is the share of the variation that a model explains.

## What I found

- **Gradient Boosting has the lowest error, but the MLP is close.** The bootstrap interval for the MAE difference stays below zero, though only just when resampling whole hours.
- **Most of what the models know is the VM size.** The data has only 12 distinct resource combinations, and the lookup that predicts the average for each one scores almost as well as the models.
- **Time of day helps only a little.** It adds about 0.02 of R2 on top of the resource sizes, and explains under 2% of the variance on its own.
- **The ranking depends on the amount of training data.** With about 52,000 training jobs or fewer, the MLP was ahead, and with 65,000 or more, Gradient Boosting was.
- **Long jobs are out of reach.** All models explain about a third of the variation, and every one of them misses jobs of 12 hours or more by about 28 hours.
- **A 90% prediction interval** covered 87.8% of the test jobs, but it is very wide.
- **On this data I would use Gradient Boosting.** The main way to improve predictions is probably better inputs, not a different model.

## Limitations

- One machine, priority 0 jobs, and 12 resource combinations. Other settings were not tested.
- The app accepts any resource values, but the model was trained and tested on only 12 distinct resource combinations. For a job far from all of them, the prediction is an untested extrapolation, and the app does not warn about it.
- Only two test days. Some checks were run after I first looked at the test days (the second round of tuning, the lookup baseline and the training-size comparison). None of them changed a model or a setting.
- The models were tuned and compared at about 81,000 training jobs, and not re-tuned at smaller sizes.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Put your key in a file named .env: ANTHROPIC_API_KEY=your-key-here
# Place the CSV at data/Azure_Data0.csv, then train the models:
python model.py

uvicorn main:app --port 8003
```

This starts a small web app that predicts a job's duration and its 90% range. The text explanation in the app is written by a language model and can be wrong. The experiments are in `experiments/`, and run as modules, for example `python -m experiments.run_baselines`.

Python 3.13.9, packages pinned in `requirements.txt`, Gradient Boosting seed 42, MLP seeds 0 to 4.

## Related publication

Masoori, Narayanan, Pankratov. *Renting Servers in the Cloud: Empirical Study on Real-World Data.* ALGOCLOUD 2025, Springer. Best Paper Award.

Mahtab Masoori, PhD in Computer Science