from flask import Flask, render_template, request
import numpy as np
from sleep_model import rf, mapping  # your trained model & mapping dict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        age = int(request.form.get("age"))
        activity = mapping[request.form.get("activity")]
        energy = mapping[request.form.get("energy")]
        anxiety = mapping[request.form.get("anxiety")]
        brain = mapping[request.form.get("brain")]

        user_features = np.array([age, activity, energy, anxiety, brain]).reshape(1, -1)
        predicted_sleep = rf.predict(user_features)
        hours = int(predicted_sleep[0])
        minutes = int((predicted_sleep[0] - hours) * 60)
        result = f"{hours} hours and {minutes} minutes"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
