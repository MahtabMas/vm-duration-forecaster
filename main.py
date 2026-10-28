from dotenv import load_dotenv
load_dotenv()

import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model import load_or_train, load_or_train_intervals, predict


app = FastAPI(title="VM Duration Forecaster")

# Set up templates folder
templates = Jinja2Templates(directory="templates")

client = anthropic.Anthropic()

model = load_or_train()
interval_models = load_or_train_intervals()


def get_explanation(job_data, result, custom_size=False):
    starttime = job_data["starttime"]
    core      = job_data["core"]
    memory    = job_data["memory"]
    ssd       = job_data["ssd"]
    nic       = job_data["nic"]
    duration  = result["predicted_duration"]
    low       = result["range_low"]
    high      = result["range_high"]

    hour = round(starttime * 24) % 24
    time_context = "business hours" if 9 <= hour <= 17 else "off-peak hours"

    if custom_size:
        size_note = (
            "This resource combination is NOT one of the 12 sizes in the training data, "
            "so the prediction is an extrapolation the model was never tested on. "
            "Say clearly that it is a guess. Do not say which resource drives the prediction."
        )
    else:
        size_note = (
            "This resource combination is one of the 12 sizes in the training data. "
            "The model's predictions are mostly driven by this size."
        )

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=250,
        system="""You explain the output of a small machine learning model that predicts Azure VM job durations.
Be specific, concise and honest. Maximum 3 sentences. Use plain text only, with no markdown, no asterisks and no bullet points.
Facts about the model you must respect: it is driven mostly by the VM size (the resource request), and time of day adds only a small effect. Its predictions are uncertain, often off by a factor of three or more, and it is least reliable for jobs that run for many hours or days. The 90% range given to you is meant to contain about 9 in 10 similar jobs, and on held-out days it contained about 88%.
Rules: never recommend a submission time, never say that one time looks better or more favorable than another, and never promise that submitting at a different time will make a job finish sooner.""",
        messages=[{
            "role": "user",
            "content": f"""A Gradient Boosting model trained on 80,729 Microsoft Azure VM jobs predicted this job will take {duration} minutes, with a 90% range of {low} to {high} minutes.

{size_note}

Job details:
- Submitted around hour {hour} of the day ({time_context})
- CPU (fraction of a server): {core}
- Memory (fraction of a server): {memory}
- SSD (fraction of a server): {ssd}
- NIC (fraction of a server): {nic}

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
    custom    = bool(data.get("custom", False))

    result = predict(model, starttime, core, memory, ssd, nic, priority,
                     interval_models=interval_models)
    explanation = get_explanation(data, result, custom_size=custom)

    return {
        "predicted_duration": result["predicted_duration"],
        "range_low":          result["range_low"],
        "range_high":         result["range_high"],
        "explanation":        explanation,
    }