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
