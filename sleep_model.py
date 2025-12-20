# sleep_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --------------------------
# 1. Raw data
# --------------------------
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

# --------------------------
# 2. Mapping categorical features
# --------------------------
mapping = {'low': 0, 'moderate': 1, 'high': 2}

X = df[['age', 'activity', 'energy', 'anxiety', 'brain']].replace(mapping)
y = df['sleep']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
