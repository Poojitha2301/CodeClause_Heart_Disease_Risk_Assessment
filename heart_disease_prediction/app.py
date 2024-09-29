from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model (Random Forest in this case)
model = joblib.load('heart_disease_model.pkl')

def predict_heart_disease(input_data):
    # Reshape input data for prediction
    input_data_reshaped = [input_data]
    
    # Get the probability of being in the class '1' (Heart disease)
    probability = model.predict_proba(input_data_reshaped)[0][1]  # The second element is the probability of class '1'
    
    # Return the risk rate as a percentage
    return probability * 100

@app.route('/', methods=['GET', 'POST'])
def index():
    risk_rate = None
    if request.method == 'POST':
        # Get data from form
        age = float(request.form.get('age'))
        gender = float(request.form.get('gender'))
        blood_pressure = float(request.form.get('blood_pressure'))
        cholesterol = float(request.form.get('cholesterol'))
        resting_heart_rate = float(request.form.get('resting_heart_rate'))
        smoking = float(request.form.get('smoking'))
        diabetes = float(request.form.get('diabetes'))
        physical_activity = float(request.form.get('physical_activity'))

        # Prepare input data
        input_data = [
            age,
            gender,
            blood_pressure,
            cholesterol,
            resting_heart_rate,
            smoking,
            diabetes,
            physical_activity
        ]

        # Predict risk
        risk_rate = predict_heart_disease(input_data)
        
    return render_template('index.html', risk_rate=risk_rate)

if __name__ == '__main__':
    app.run(debug=True)
