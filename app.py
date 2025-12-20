from flask import Flask, render_template, request
from sleep_model import predict_sleep

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        age = int(request.form["age"])
        activity = request.form["activity"]
        energy = request.form["energy"]
        anxiety = request.form["anxiety"]
        brain = request.form["brain"]

        hours, minutes = predict_sleep(age, activity, energy, anxiety, brain)
        result = f"{hours} hours and {minutes} minutes"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
