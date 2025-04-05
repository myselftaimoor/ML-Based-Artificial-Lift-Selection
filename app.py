from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
df = pd.read_excel('DATA_ALMODEL.xlsx')
categorical_cols = ['COUNTRY', 'FORMATION', 'FIELD ', 'PROD PATH']
le = LabelEncoder()

# Encode categorical columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Load the trained model
with open('pipe_svc.pkl', 'rb') as file:
    model = pickle.load(file)

# Prepare input data for prediction
X = df.drop('PROD PATH', axis=1)
y = df['PROD PATH']

# Store well IDs separately and remove from X
well_id_col = X['WELL ID']
X = X.drop('WELL ID', axis=1)

# Store the scaler and fit it
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define an API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the POST request
    data = request.get_json(force=True)

    # Extract well_id from the input
    well_id = data.get('WELL ID')
    
    if well_id is None:
        return jsonify({"error": "WELL ID is required"}), 400
    
    # Convert the remaining input data to a pandas DataFrame
    input_data = pd.DataFrame([data])
    
    # Rename columns to match the training data
    input_data = input_data.rename(columns={
        'CUM WATER PROD, RB': 'CUM WATER PROD, RB',
        'DAILY WATER PROD, RB/D': 'DAILY WATER PROD , RB/D',
    })

    # Ensure all required columns are present
    required_columns = X.columns  # Since 'WELL ID' is not in X anymore
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Set default value for missing columns

    # Reorder columns to match the training data
    input_data = input_data[required_columns]

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction_proba = model.predict_proba(input_scaled)

    # Check if the input matches any row in the dataset
    actual_output = None
    matching_row = X[X.eq(input_data.iloc[0]).all(axis=1)]
    if not matching_row.empty:
        actual_output = y.iloc[matching_row.index[0]]

    # Return the predicted probabilities and actual output as JSON response
    result = {
        'well_id': well_id,
        'predicted_probabilities': prediction_proba.tolist(),
        'actual_output': int(actual_output) if actual_output is not None else None
    }
    return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
