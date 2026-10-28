from dotenv import load_dotenv
load_dotenv()

import os
import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model import load_or_train, predict, build_features



app = FastAPI(title="VM Duration Forecaster")

#sSet up templates folder
templates = Jinja2Templates(directory="templates")

client = anthropic.Anthropic()

model = load_or_train()


def get_explanation(job_data, result):
    starttime = job_data["starttime"]
    core      = job_data["core"]
    memory    = job_data["memory"]
    nic       = job_data["nic"]
    priority  = job_data["priority"]
    duration  = result["predicted_duration"]
    savings   = result["savings"]
    off_peak  = result["off_peak_duration"]

    # Extract hour from starttime
    hour = (starttime * 24) % 24
    day  = int(starttime) % 7

    is_business  = 9 <= hour <= 17
    is_weekend   = day >= 5
    time_context = "business hours" if is_business else "off-peak hours"
    day_context  = "weekend" if is_weekend else "weekday"
    priority_str = "high" if priority == 1 else "low"

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=250,
        system="""You explain the output of a small machine learning model that predicts Azure VM job durations.
                Be specific, concise and honest. Maximum 3 sentences.
                Facts about the model you must respect: it is driven mostly by the VM size (the resource request), and time of day adds only a small effect. Its predictions are uncertain, often off by a factor of three or more, and it is least reliable for jobs that run for many hours or days. Never promise that submitting at a different time will make a job finish sooner. If you mention the 3am comparison, describe it as a model estimate, and say so when it points the other way.""",
        messages=[{
            "role": "user",
            "content": f"""A Gradient Boosting model trained on 80,729 Microsoft Azure VM jobs predicted this job will take {duration} minutes.

            Job details:
            - Submitted around hour {hour:.0f} of the day ({time_context})
            - CPU (normalized): {core}
            - Memory (normalized): {memory}
            - NIC (normalized): {nic}
            - Model estimate for the same job submitted at 3am: {off_peak} minutes (difference: {savings} minutes, where a negative number means the model expects the 3am job to take longer)

            Explain what mainly drives this prediction and how uncertain it is, in plain words."""
                    }]
                )

    return response.content[0].text

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/predict")
async def predict_duration(request: Request):
    data = await request.json()

    starttime = data["starttime"]
    core      = data["core"]
    memory    = data["memory"]
    ssd       = data["ssd"]
    nic       = data["nic"]
    priority  = data["priority"]

    result = predict(model, starttime, core, memory, ssd, nic, priority)
    explanation = get_explanation(data, result)

    return {
        "predicted_duration": result["predicted_duration"],
        "off_peak_duration":  result["off_peak_duration"],
        "savings":            result["savings"],
        "explanation":        explanation
    }
