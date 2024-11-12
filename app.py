import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load MinMaxScaler and RandomForestClassifier objects
with open('min_max_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('random_forest_classifier.pkl', 'rb') as classifier_file:
    classifier = pickle.load(classifier_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    features = []
    for key in request.form:
        features.append(int(request.form[key]))
    
    # Convert features into pandas DataFrame
    columns = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 
               'Genital thrush', 'visual blurring', 'Irritability', 'partial paresis', 'muscle stiffness', 
               'Alopecia', 'Obesity']
    input_data = pd.DataFrame([features], columns=columns)
    
    # Scale input data
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = classifier.predict(scaled_data)
    
    # Prepare prediction result
    if prediction[0] == 1:
        result = "Positive"
    else:
        result = "Negative"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
