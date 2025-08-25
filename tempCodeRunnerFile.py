
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form inputs
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = int(request.form['fuel_type'])  # Categorical: Petrol=0, Diesel=1, CNG=2
        owner = int(request.form['owner'])
        seller_type_individual = int(request.form['seller_type_individual'])  # Binary: 1=Individual, 0=Dealer
        transmission_manual = int(request.form['transmission_manual'])  # Binary: 1=Manual, 0=Automatic

        # Prepare data for prediction
        input_data = np.array([[year, present_price, kms_driven, fuel_type, owner, seller_type_individual, transmission_manual]])

        # Make prediction
        prediction = model.predict(input_data)

        # Format the result
        result = f"Estimated Selling Price: ₹{prediction[0]:,.2f} Lakhs"
        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
