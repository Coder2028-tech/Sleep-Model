AI Sleep Predictor
A machine learning web app that predicts optimal sleep duration based on personal lifestyle factors.
What it does
Users input five factors: Age, physical activity level, energy level, anxiety level, and mental activity, and the app returns a personalized sleep duration recommendation in hours and minutes.
Tech Stack

Python: Core language
Flask: Web framework
scikit-learn: Random Forest model
NumPy: Feature processing
HTML/CSS: Frontend interface

How it works

User submits lifestyle inputs via web form
Inputs are mapped to numerical values and passed to a trained Random Forest model
Model predicts optimal sleep hours
Result is displayed on the same page

Files
app.py: Flask application and routing
sleep_model.py: Trained Random Forest model and input mappings
templates/index.html: Frontend interface

How to run
bashpip install flask numpy scikit-learn
python app.py
Then open http://localhost:5000 in your browser.

Background
Built as part of MIT BWSI Aspiring Engineers Spring 2025. Trained on real-world sleep data using random forest modeling with feature selection and performance analysis.
