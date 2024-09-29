import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load or create dataset (replace with actual data)
def load_data():
    data = {
        'age': [63, 37, 41, 56, 57],
        'gender': [1, 1, 0, 1, 0],
        'blood_pressure': [145, 130, 130, 120, 120],
        'cholesterol': [233, 250, 204, 236, 354],
        'resting_heart_rate': [150, 187, 172, 178, 163],
        'smoking': [1, 0, 1, 0, 0],
        'diabetes': [0, 1, 0, 0, 0],
        'physical_activity': [2, 1, 2, 0, 0],
        'heart_disease': [1, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    X = df.drop('heart_disease', axis=1)
    y = df['heart_disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train and save the model
model = train_model()
joblib.dump(model, 'heart_disease_model.pkl')
