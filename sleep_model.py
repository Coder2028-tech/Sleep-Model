import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# Create dataset
# -------------------------
data = [
    {'age': 15, 'activity': 'high', 'energy': 'moderate', 'anxiety': 'low', 'brain': 'high', 'sleep': 9},
    {'age': 16, 'activity': 'low', 'energy': 'low', 'anxiety': 'high', 'brain': 'low', 'sleep': 9.5},
    {'age': 14, 'activity': 'moderate', 'energy': 'moderate', 'anxiety': 'moderate', 'brain': 'moderate', 'sleep': 9.25},
    {'age': 18, 'activity': 'high', 'energy': 'high', 'anxiety': 'low', 'brain': 'high', 'sleep': 7.5},
    {'age': 12, 'activity': 'low', 'energy': 'low', 'anxiety': 'high', 'brain': 'low', 'sleep': 11},
    {'age': 13, 'activity': 'moderate', 'energy': 'moderate', 'anxiety': 'moderate', 'brain': 'moderate', 'sleep': 9.25},
    {'age': 15, 'activity': 'high', 'energy': 'low', 'anxiety': 'high', 'brain': 'moderate', 'sleep': 10.5},
]

df = pd.DataFrame(data)

# Encode categorical values
mapping = {'low': 0, 'moderate': 1, 'high': 2}
for col in ['activity', 'energy', 'anxiety', 'brain']:
    df[col] = df[col].map(mapping)

X = df[['age', 'activity', 'energy', 'anxiety', 'brain']]
y = df['sleep']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Prediction function
# -------------------------
def predict_sleep(age, activity, energy, anxiety, brain):
    user_data = np.array([
        age,
        mapping.get(activity, 1),
        mapping.get(energy, 1),
        mapping.get(anxiety, 1),
        mapping.get(brain, 1)
    ]).reshape(1, -1)

    prediction = model.predict(user_data)[0]
    hours = int(prediction)
    minutes = int((prediction - hours) * 60)
    return hours, minutes
