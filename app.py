import json
import time
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

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
    # Simulate DCA calculation for three models
    time.sleep(2)  # Simulate processing time
    dca_results = {
        "exponential": [{"x": point["x"], "y": point["y"] * 0.95} for point in selected_area],
        "harmonic": [{"x": point["x"], "y": point["y"] * 0.9} for point in selected_area],
        "hyperbolic": [{"x": point["x"], "y": point["y"] * 0.85} for point in selected_area],
    }
    return jsonify(dca_results)

if __name__ == '__main__':
    app.run(debug=True)
