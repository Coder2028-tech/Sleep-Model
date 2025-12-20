from flask import Flask, render_template, request
import numpy as np
from sleep_model import rf, mapping  # your trained model & mapping dict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            age = int(request.form.get("age", 15))
            activity = mapping.get(request.form.get("activity"), 1)
            energy = mapping.get(request.form.get("energy"), 1)
            anxiety = mapping.get(request.form.get("anxiety"), 1)
            brain = mapping.get(request.form.get("brain"), 1)

            features = np.array([age, activity, energy, anxiety, brain]).reshape(1, -1)
            predicted_sleep = rf.predict(features)
            hours = int(predicted_sleep[0])
            minutes = int((predicted_sleep[0] - hours) * 60)
            result = f"{hours} hours and {minutes} minutes"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result)
