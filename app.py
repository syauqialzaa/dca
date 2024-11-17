import json
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import math

app = Flask(__name__)
# Allow cross-origin requests
CORS(app)

# Load historical data from JSON file
with open('historical_data.json') as f:
    raw_data = json.load(f)

# Format data for chart (Date -> timestamp, Production -> y)
historical_data = [{"x": entry["Date"], "y": entry["Production"]} for entry in raw_data]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['GET'])
def get_data():
    return jsonify(historical_data)

@app.route('/calculate_dca', methods=['POST'])
def calculate_dca():
    selected_area = request.json.get('selected_area')

    if not selected_area or len(selected_area) < 2:
        return jsonify({"error": "Insufficient data points in selected_area"}), 400

    # Parse input data
    selected_area = sorted(selected_area, key=lambda point: point['x'])  # Ensure data is sorted by x-axis

    # Get the last date and calculate the interval
    last_date = datetime.fromtimestamp(selected_area[-1]['x'] / 1000)  # Convert ms to seconds
    first_date = datetime.fromtimestamp(selected_area[0]['x'] / 1000)
    interval = (last_date - first_date).days // (len(selected_area) - 1)  # Calculate average interval

    # Get last y value to use as the starting point for forecasting
    last_y_value = selected_area[-1]['y']

    # Decline parameters (can be made dynamic or calculated based on data)
    decline_rate = 0.1  # Nominal decline rate per time unit
    hyperbolic_b = 0.5  # Hyperbolic exponent

    # Forecasted range (e.g., day 11-20 based on interval)
    forecasted_data = []
    for i in range(1, len(selected_area) + 1):  # Extend by the same number of days as the selected range
        forecast_date = last_date + timedelta(days=i * interval)
        forecast_timestamp = int(forecast_date.timestamp() * 1000)  # Convert back to ms

        # Time since start of forecast (in the same units as interval)
        t = i * interval

        # Apply decline models
        exponential_y = last_y_value * math.exp(-decline_rate * t)
        harmonic_y = last_y_value / (1 + decline_rate * t)
        hyperbolic_y = last_y_value / ((1 + hyperbolic_b * decline_rate * t) ** (1 / hyperbolic_b))

        forecasted_data.append({
            "x": forecast_timestamp,
            "exponential": exponential_y,
            "harmonic": harmonic_y,
            "hyperbolic": hyperbolic_y
        })

    # Format DCA results (only forecasted data)
    dca_results = {
        "exponential": [{"x": point["x"], "y": point["exponential"]} for point in forecasted_data],
        "harmonic": [{"x": point["x"], "y": point["harmonic"]} for point in forecasted_data],
        "hyperbolic": [{"x": point["x"], "y": point["hyperbolic"]} for point in forecasted_data],
    }

    # time.sleep(2)  # Simulate processing time
    return jsonify(dca_results)

if __name__ == '__main__':
    app.run(debug=True)
