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
        max_tokens=200,
        system="""You are an expert in cloud computing and VM workload analysis.
                Explain ML model predictions in clear practical terms.
                Be specific, concise, and helpful. Maximum 3 sentences.""",
        messages=[{
            "role": "user",
            "content": f"""A Gradient Boosting model trained on 112,552 real Microsoft Azure VM jobs predicted this job will take {duration} minutes.

            Job details:
            - Submitted at: {hour:.1f}:00 ({time_context}, {day_context})
            - CPU (normalized): {core} 
            - Memory (normalized): {memory}
            - NIC (normalized): {nic}
            - Priority: {priority_str}
            - Potential savings if submitted at 3am: {savings} minutes

            Explain why the model predicted this duration and give one practical recommendation."""
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
