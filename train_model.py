import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import random
import os

print("Generating synthetic dataset...")
# Define typical ranges for each crop
crop_conditions = {
    'Rice': {'N': (60, 100), 'P': (35, 60), 'K': (35, 45), 'temperature': (20, 40), 'humidity': (80, 85), 'ph': (5.5, 7.5), 'rainfall': (150, 300)},
    'Maize': {'N': (60, 100), 'P': (35, 60), 'K': (15, 25), 'temperature': (18, 28), 'humidity': (50, 75), 'ph': (5.5, 7.5), 'rainfall': (50, 150)},
    'Cotton': {'N': (100, 140), 'P': (35, 60), 'K': (15, 25), 'temperature': (22, 32), 'humidity': (75, 85), 'ph': (5.8, 7.8), 'rainfall': (60, 110)},
    'Soybean': {'N': (20, 60), 'P': (55, 80), 'K': (15, 25), 'temperature': (20, 30), 'humidity': (60, 70), 'ph': (6.0, 7.5), 'rainfall': (40, 90)},
    'Wheat': {'N': (80, 120), 'P': (40, 60), 'K': (20, 30), 'temperature': (15, 25), 'humidity': (40, 60), 'ph': (6.0, 7.5), 'rainfall': (50, 100)},
    'Sugarcane': {'N': (100, 150), 'P': (40, 60), 'K': (30, 50), 'temperature': (25, 35), 'humidity': (75, 85), 'ph': (6.5, 8.0), 'rainfall': (150, 250)},
    'Coffee': {'N': (80, 120), 'P': (20, 40), 'K': (25, 35), 'temperature': (20, 30), 'humidity': (50, 70), 'ph': (5.5, 7.0), 'rainfall': (100, 200)},
    'Apple': {'N': (10, 40), 'P': (120, 145), 'K': (195, 205), 'temperature': (20, 25), 'humidity': (90, 95), 'ph': (5.5, 6.5), 'rainfall': (100, 120)},
    'Mango': {'N': (10, 40), 'P': (15, 40), 'K': (25, 35), 'temperature': (27, 36), 'humidity': (45, 55), 'ph': (4.5, 7.0), 'rainfall': (80, 100)},
    'Grapes': {'N': (10, 40), 'P': (120, 145), 'K': (195, 205), 'temperature': (8, 40), 'humidity': (80, 85), 'ph': (5.5, 6.5), 'rainfall': (60, 75)}
}

data = []
for crop, conditions in crop_conditions.items():
    for _ in range(200): # 200 samples per crop
        n = random.uniform(*conditions['N'])
        p = random.uniform(*conditions['P'])
        k = random.uniform(*conditions['K'])
        temp = random.uniform(*conditions['temperature'])
        hum = random.uniform(*conditions['humidity'])
        ph = random.uniform(*conditions['ph'])
        rain = random.uniform(*conditions['rainfall'])
        data.append([n, p, k, temp, hum, ph, rain, crop])

df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])

# Save dataset just in case
df.to_csv('crop_dataset.csv', index=False)
print("Dataset saved to crop_dataset.csv")

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'model.pkl')
print("Model successfully saved to model.pkl")
