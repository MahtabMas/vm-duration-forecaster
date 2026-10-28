# VM Duration Forecaster

A web application that predicts how long a Microsoft Azure VM job will take, and explains the prediction in plain English.

## Background

This project is an extension of my doctoral research in cloud scheduling at Concordia University. My PhD work included an empirical study of VM job behavior on real Microsoft Azure data, which revealed that temporal features, particularly time of day, are strong predictors of job duration. This tool applies those findings in a practical setting.

## Dataset

The model is trained on **112,552 real Microsoft Azure VM jobs** from the publicly available Azure workload dataset.

## How It Works

1. User configures a VM job (CPU, memory, storage, network, submission time)
2. A Gradient Boosting model predicts the job duration in minutes
3. A language model explains why the prediction is what it is, in plain English
4. The app also estimates whether submitting at a quieter time would make the job faster

## Technical Details

**Machine Learning:**
- Gradient Boosting Regressor (scikit-learn)
- Log-transformed duration to handle the skewed distribution in real Azure data
- Time features encoded as sine/cosine to capture the cyclical nature of hours and days
- 4 resource dimensions: CPU, memory, SSD, NIC
- Finding from my research: time of day alone accounts for ~65% of the variance in job duration

**Language Model:**
- Used to translate the model's numerical output into a readable explanation
- Takes the prediction, job configuration, and timing context as input
- Returns a short paragraph explaining the likely causes and one practical suggestion

**Stack:**
- Python, FastAPI, scikit-learn, pandas, numpy
- HTML and JavaScript for the interface

## Project Structure

```
project3_forecaster/
├── main.py            # web server and language model integration
├── model.py           # model training, feature engineering, prediction
└── templates/
    └── index.html     # user interface
```

## Running Locally

```bash
# Install dependencies
pip install fastapi uvicorn scikit-learn pandas numpy joblib anthropic jinja2

# Set your LLM API key
export ANTHROPIC_API_KEY="your-key-here"

# Place your Azure CSV at the path in model.py

# Start the server
uvicorn main:app --reload --port 8003

# Open in browser
http://127.0.0.1:8003
```

## Related Publications

**Dataset source:**
Hadary et al. *Protean: VM Allocation Service at Scale.* Microsoft Azure and Microsoft Research.

**Research using this dataset:**
Masoori, Narayanan, Pankratov. *Renting Servers in the Cloud: Empirical Study on Real-World Data.* ALGOCLOUD 2025, Springer. Best Paper Award. doi:10.1007/978-3-032-13744-9

## Author

Mahtab Masoori  
PhD, Computer Science  



# VM Duration Forecaster: validating Gradient Boosting and a PyTorch MLP

This project predicts how long an Azure VM job will run, using the resources it requests and the time it is submitted. It compares Gradient Boosting with a PyTorch MLP using time-based validation, multiple seeds, bootstrap confidence intervals, prediction intervals, and ablations.

It extends my doctoral research on cloud scheduling at Concordia University, which used real Azure data (see Related Publications). Prior work reports that jobs arriving at the same time tend to have similar durations, which motivated testing time of day as a predictor. The validation in this repo found its effect to be small.

## 1. Data

**Source.** The Azure workload trace released with Hadary et al., "Protean: VM Allocation Service at Scale" (OSDI 2020). I selected a subset with this query, which joins the `vm` and `vmType` tables:

```sql
SELECT vm.vmId, vm.vmTypeId, vmType.machineID, vm.starttime, vm.endtime,
       vmType.core, vmType.memory, vmType.ssd, vmType.nic, vm.Priority
FROM vm
JOIN vmType ON vm.vmTypeId = vmType.vmTypeId
WHERE vmType.machineID = 0
  AND vm.Priority = 0
  AND vm.starttime > 0
  AND vm.endtime IS NOT NULL;
```

**What the filters mean**
- Machine 0 only. A machine id is a pool of homogeneous servers, not a single server.
- Priority 0 only. Low-priority VMs can be evicted, which makes their durations unreliable. Priority 0 is 89.51% of the jobs on machine 0 (Masoori et al., ALGOCLOUD 2025, Section 4).
- `starttime > 0` removes jobs that started before the collection window.
- `endtime IS NOT NULL` removes jobs still running at the end of the trace. Arrivals span 14 days and jobs are observed for 76 more days. This removed 302 of 112,854 jobs (0.27%). They are the longest jobs, so the models have no examples of them.

**Result.** 112,552 jobs. The CSV is not in this repository. To reproduce it, run the query above on the public trace and save the result as `data/Azure_Data0.csv`.

**Target.** Duration in minutes, `(endtime - starttime) * 24 * 60`, modeled as `log1p(duration)`. Durations are strongly right-skewed: median 38.8 minutes, mean 481.8, 90th percentile 701, 99th percentile 9,675, maximum 117,305. The log transform keeps a few multi-day jobs from dominating training.

**Inputs.** The four resource sizes (`core`, `memory`, `ssd`, `nic`), their mean (`resource_load`), and three time-of-day features (`hour_sin`, `hour_cos`, `is_business`).

**The resource features are close to a VM-type identifier.** The training days contain only 12 distinct resource combinations. One covers 59.8% of jobs and four cover 90%. The four resource columns are 96% to 99.9% correlated with each other, so the models see little beyond the VM type plus the hour of day.

## 2. Method

**Split by time.** Jobs are split by submission day, so every model is tested on jobs that arrive after the ones it trained on.

| Set | Days | Jobs | Used for |
|---|---|---|---|
| Train | 0 to 9 | 80,729 | fitting models |
| Validation | 10 and 11 | 15,341 | early stopping and choosing settings |
| Test | 12 and 13 | 16,482 | final scores, opened once |

All choices (features, settings, which settings to carry forward) were made on the validation days before the test days were scored. A random split of the same data scores about 0.05 higher in R2 for Gradient Boosting (0.484 vs 0.428 on validation), so a random split would have flattered the results.

**No leakage.** The inputs use only the submission time and the requested resources, which are known when a job arrives. The end time is used only to build the target. Standardization for the MLP uses statistics from the training days only.

**Models**
- **Median baseline:** predicts the training median for every job.
- **Linear regression:** scikit-learn, on the 8 inputs.
- **Lookup by VM type:** predicts the training average for each resource combination. It has no settings and was added after the test run.
- **Gradient Boosting:** scikit-learn, 200 trees, depth 5, learning rate 0.05, seed 42.
- **MLP:** PyTorch, written from scratch: 8 inputs, hidden layers of 64 and 32 with ReLU, one output. AdamW optimizer, mean squared error loss, early stopping on validation loss with patience 10. Final settings: learning rate 0.01, weight decay 0, dropout 0, batch size 1024. Reported as the mean and standard deviation over seeds 0 to 4.

**Equal tuning budget.** Both models got a 15-trial random search, scored by validation MAE. For Gradient Boosting the search found nothing better than the original settings, so those were kept. For the MLP, the best of 15 trials looked better than it was: over five seeds the tuned settings beat the original ones by less than the seed-to-seed spread. They were kept because their mean validation MAE was lower. Both searches ran before the final 8-input set was fixed, so the settings were not re-tuned for it. The MLP search did not vary the network's width or depth.

**Metrics.** MAE, RMSE and R2 on the log scale, MAE and median absolute error in minutes, and errors by duration bucket (short under 30 minutes, medium 30 minutes to 12 hours, long 12 hours or more).

**Comparing models.** A bootstrap with 2,000 resamples of the test jobs gives a 95% confidence interval for the difference in error between Gradient Boosting and the MLP (averaged over its five seeds). It was run twice, resampling single jobs and resampling whole hours, since jobs submitted in the same hour tend to behave alike.

**Prediction intervals.** Two Gradient Boosting models trained with a quantile loss (5th and 95th percentiles) give a 90% interval. Coverage is the share of test jobs that fall inside it.


