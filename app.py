import json
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import io
import pickle
from xgboost import DMatrix
import pandas as pd
import numpy as np
from dca_model import analyze_dca
import logging
from scipy.interpolate import interp1d
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
# Allow cross-origin requests
# CORS(app, resources={r"/generate": {"origins": "http://localhost:63342"}})
CORS(app)

# Load trained models
depth_model = joblib.load('depth_model.pkl')
material_model = joblib.load('material_model.pkl')

# Load historical data from JSON file
with open('historical_actual_preprocessed.json') as f:
    raw_data = json.load(f)

# Load the trained model
with open('final_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Format data for chart (Date -> timestamp, Production -> y)
historical_data = [{"x": entry["Date"], "y": entry["Production"]} for entry in raw_data]

# Load and preprocess dataset
file_path = 'Dataset DCA.xlsx'

@app.route('/')
def index():
    return render_template('index.html')

def load_data():
    data = pd.read_excel(file_path)
    data['TEST_DATE'] = pd.to_datetime(data['TEST_DATE'])
    data_sorted = data.sort_values(by="TEST_DATE")
    data_sorted = data_sorted.groupby("TEST_DATE")["TSTOIL"].sum().reset_index()
    data_sorted.columns = ['Date', 'Production']
    data_sorted['Production'] = data_sorted['Production'].interpolate(method='linear')

    # Remove outliers using IQR
    Q1 = data_sorted['Production'].quantile(0.25)
    Q3 = data_sorted['Production'].quantile(0.75)
    IQR = Q3 - Q1
    data_sorted = data_sorted[
        ~((data_sorted['Production'] < (Q1 - 1.5 * IQR)) |
          (data_sorted['Production'] > (Q3 + 1.5 * IQR)))
    ]
    return data_sorted

data_sorted = load_data()

@app.route('/get_data', methods=['GET'])
def get_data():
    return jsonify(historical_data)


@app.route('/get_wells', methods=['GET'])
def get_wells():
    try:
        # Read data from Excel file
        df = pd.read_excel(file_path)
        df = df.sort_values(by=['STRING_CODE'])  # Sort wells by ascending order
        unique_wells = df['STRING_CODE'].drop_duplicates().tolist()  # Get unique well codes
        return jsonify({"wells": unique_wells})  # Return well codes as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_history', methods=['POST'])
def get_history():
    # Receive request data from frontend
    request_data = request.get_json()
    selected_well = request_data.get('well', None)
    start_date = request_data.get('start_date', None)
    end_date = request_data.get('end_date', None)

    # Load and preprocess data
    data = pd.read_excel(file_path)
    data['TEST_DATE'] = pd.to_datetime(data['TEST_DATE'])
    # Pastikan TSTFLUID ada agar bisa ditampilkan juga
    data = data.dropna(subset=['TSTOIL', 'TSTFLUID'])

    # Filter by selected well
    if selected_well:
        data = data[data['STRING_CODE'] == selected_well]

    # Filter by date range
    if start_date:
        start_date = pd.to_datetime(start_date)
        data = data[data['TEST_DATE'] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        data = data[data['TEST_DATE'] <= end_date]

    # Default to last 12 months if no filters provided
    if not selected_well and not start_date and not end_date:
        max_date = data['TEST_DATE'].max()
        min_date = max_date - pd.DateOffset(months=12)
        data = data[(data['TEST_DATE'] >= min_date) & (data['TEST_DATE'] <= max_date)]

    # Sort data
    data = data.sort_values(by='TEST_DATE')

    # Convert to JSON-friendly format
    data_json = data[['TEST_DATE', 'TSTOIL', 'TSTFLUID']].rename(
        columns={'TEST_DATE': 'Date', 'TSTOIL': 'Production', 'TSTFLUID': 'Fluid'}
    )
    data_json['Date'] = data_json['Date'].dt.strftime('%Y-%m-%d')
    history = data_json.to_dict(orient='records')

    # Filter to only include points where production changes (grouping based on Production)
    filtered_history = []
    prev_value = None
    for record in history:
        current_prod = record['Production']
        if prev_value is None or current_prod != prev_value:
            filtered_history.append(record)
        prev_value = current_prod

    return jsonify(filtered_history)


@app.route('/calculate_dca3', methods=['POST'])
def calculate_dca3():
    try:
        # Parse request
        data = request.get_json()
        well = data.get('well')
        start_date_input = data.get('start_date', None)
        end_date_input = data.get('end_date', None)

        # Load dataset
        df = pd.read_excel(file_path)
        df['TEST_DATE'] = pd.to_datetime(df['TEST_DATE'])

        # Filter by well
        df = df[df['STRING_CODE'] == well]

        # Ensure required columns exist
        required_columns = ['STRING_CODE', 'TEST_DATE', 'TSTOIL', 'JOB_CODE']
        for col in required_columns:
            if col not in df.columns:
                return jsonify({"error": f"Column {col} not found in dataset"}), 400

        # Determine last production date
        last_production_date = df[df['TSTOIL'] > 0]['TEST_DATE'].max()
        if last_production_date is None:
            return jsonify({"error": "No production data available for the selected well"}), 400

        # Identify latest job code
        latest_job = df.dropna(subset=['JOB_CODE']).sort_values(by='TEST_DATE', ascending=False)
        if not latest_job.empty:
            latest_job_date = latest_job['TEST_DATE'].iloc[0]
            # Determine if latest_job_date is within 12 months of the last production date
            if latest_job_date >= last_production_date - pd.DateOffset(months=12):
                start_date = latest_job_date
            else:
                # Use 12 months from last production date
                start_date = last_production_date - pd.DateOffset(months=12)
        else:
            # Use 12 months from last production date if no job code exists
            start_date = last_production_date - pd.DateOffset(months=12)

        # Filter data based on calculated start_date
        filtered_data = df[df['TEST_DATE'] >= start_date]

        # Filter by user-specified end_date if provided
        if end_date_input:
            end_date_input = pd.to_datetime(end_date_input)
            filtered_data = filtered_data[filtered_data['TEST_DATE'] <= end_date_input]

        # Sort and clean data
        filtered_data = filtered_data.sort_values(by='TEST_DATE')

        # Handling outliers using IQR
        Q1 = filtered_data['TSTOIL'].quantile(0.25)
        Q3 = filtered_data['TSTOIL'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_data = filtered_data[(filtered_data['TSTOIL'] >= lower_bound) & (filtered_data['TSTOIL'] <= upper_bound)]

        # Filter significant production changes for visualization
        filtered_data['TSTOIL_diff'] = filtered_data['TSTOIL'].diff().fillna(0)
        visualization_data = filtered_data[filtered_data['TSTOIL_diff'].abs() > 1e-2]

        # Calculate mid-point based on slope
        filtered_data['TSTOIL_slope'] = filtered_data['TSTOIL'].diff() / filtered_data['TEST_DATE'].diff().dt.days
        filtered_data['TSTOIL_slope'] = filtered_data['TSTOIL_slope'].rolling(window=5, center=True).mean()
        significant_change = filtered_data['TSTOIL_slope'].abs() > 50
        if significant_change.any():
            mid_index = significant_change.idxmax()
            mid_date = filtered_data.loc[mid_index, 'TEST_DATE']
        else:
            mid_date = start_date + (filtered_data['TEST_DATE'].max() - start_date) / 2

        # Identify end-point based on economic limit rate
        elr = 10  # Economic Limit Rate (BOPD)
        end_data = filtered_data[filtered_data['TSTOIL'] >= elr]
        end_date = end_data['TEST_DATE'].iloc[-1] if not end_data.empty else filtered_data['TEST_DATE'].iloc[-1]

        # Prepare data for DCA fitting
        analysis_data = filtered_data[(filtered_data['TEST_DATE'] >= start_date) & (filtered_data['TEST_DATE'] <= end_date)]
        analysis_data['days'] = (analysis_data['TEST_DATE'] - start_date).dt.days
        t = analysis_data['days'].values
        q = analysis_data['TSTOIL'].values

        # Define the three DCA models
        def exponential_decline(t, qi, d):
            return qi * np.exp(-d * t)

        def harmonic_decline(t, qi, b):
            return qi / (1 + b * t)

        def hyperbolic_decline(t, qi, b, d):
            return qi / (1 + b * d * t) ** (1 / b)

        # Calculate DCA predictions
        t_fit = np.linspace(0, t[-1] + 365, len(t) + 30)
        exp_fit_opt = exponential_decline(t_fit, q[0], 0.001)
        harm_fit_opt = harmonic_decline(t_fit, q[0], 0.002)
        hyper_fit_opt = hyperbolic_decline(t_fit, q[0], 1.0, 0.001)
        extended_dates = [start_date + pd.Timedelta(days=int(day)) for day in t_fit]

        # Build response
        response = {
            "historical": [{"Date": d.strftime('%Y-%m-%d'), "Production": p} for d, p in zip(visualization_data['TEST_DATE'], visualization_data['TSTOIL'])],
            "exponential": [{"Date": d.strftime('%Y-%m-%d'), "Production": p} for d, p in zip(extended_dates, exp_fit_opt)],
            "harmonic": [{"Date": d.strftime('%Y-%m-%d'), "Production": p} for d, p in zip(extended_dates, harm_fit_opt)],
            "hyperbolic": [{"Date": d.strftime('%Y-%m-%d'), "Production": p} for d, p in zip(extended_dates, hyper_fit_opt)],
            "start_date": start_date.strftime('%Y-%m-%d'),
            "mid_date": mid_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in calculate_dca3: {e}")
        return jsonify({"error": str(e)}), 500


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

def predict_future(model, data, n_days, n_lags):
    if len(data) < n_lags:
        raise ValueError(f"Insufficient data: Need at least {n_lags} rows, but only {len(data)} available.")

    future_predictions = []
    latest_data = data.tail(n_lags).copy()  # Ambil data terakhir untuk prediksi

    for _ in range(n_days):
        # Generate lagged features
        lagged_features = latest_data['Production'].values[-n_lags:].reshape(1, -1)
        dmatrix_features = DMatrix(lagged_features)

        # Predict next value
        next_prediction = float(model.predict(dmatrix_features)[0])  # Convert to float
        future_predictions.append(next_prediction)

        # Add new prediction to data
        new_row = pd.DataFrame({'Production': [next_prediction]})
        latest_data = pd.concat([latest_data, new_row], ignore_index=True)

    return future_predictions


@app.route('/calculate_ml', methods=['POST'])
def calculate_ml():
    try:
        global best_model, data_sorted
        n_days = 30
        n_lags = 30

        # Load the model if not already loaded
        if best_model is None:
            with open('final_model.pkl', 'rb') as file:
                best_model = pickle.load(file)

        # Generate predictions
        predictions = predict_future(best_model, data_sorted, n_days, n_lags)

        # Convert predictions to float
        predictions = [float(pred) for pred in predictions]

        # Prepare response data
        last_date = data_sorted['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
        prediction_data = [{"x": date.isoformat(), "y": pred} for date, pred in zip(future_dates, predictions)]

        return jsonify(prediction_data)

    except Exception as e:
        # Log error to the console
        print(f"Error in /calculate_ml: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/calculate_dca2', methods=['POST'])
def calculate_dca_endpoint():
    try:
        data = request.get_json()
        well = data['well']
        start_date = data['start_date']
        end_date = data['end_date']

        # Load the dataset and filter based on well and date
        filtered_data = load_filtered_data(well, start_date, end_date)

        # Calculate DCA
        results = analyze_dca(filtered_data)

        # Convert NumPy arrays to lists for JSON serialization
        results['exp_fit'] = results['exp_fit'].tolist()
        results['harm_fit'] = results['harm_fit'].tolist()
        results['hyper_fit'] = results['hyper_fit'].tolist()

        # Convert dates to strings
        results['start_date'] = str(results['start_date'])
        results['mid_date'] = str(results['mid_date'])
        results['end_date'] = str(results['end_date'])

        return jsonify(results), 200

    except Exception as e:
        print(e)
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500


def generate_wellbore_diagram(parameters):
    fig, ax = plt.subplots(figsize=(4, 8))  # Perbesar width untuk memberi ruang pada legenda

    max_depth = 0  # Track the maximum depth for scaling

    for param in parameters:
        if param["type"] == "casing":
            ax.add_patch(patches.Rectangle(
                (param["x"], param["top_depth"]),
                param["width"],
                param["bottom_depth"] - param["top_depth"],
                hatch=param.get("hatch", ""),
                facecolor=param.get("color", "none"),
                edgecolor="blue",
                label=param["label"]
            ))
        elif param["type"] == "completion":
            ax.plot(param["x"], param["depth"], marker="o", markersize=8, color=param.get("color", "orange"), label=param["label"])

        # Update max depth
        max_depth = max(max_depth, param["bottom_depth"] if "bottom_depth" in param else param["depth"])

    # Add depth labels on the left
    step = 100  # Interval for depth labels
    for depth in range(0, int(max_depth) + step, step):
        ax.text(0.4, depth, f"{depth} ft", fontsize=8, verticalalignment="center", horizontalalignment="right")

    # Styling the plot
    ax.set_xlim(0.2, 4.5)  # Adjust x-axis to leave space for labels and legend
    ax.set_ylim(0, max_depth + 50)
    ax.invert_yaxis()  # Reverse the depth axis for wellbore style
    ax.axis("off")

    # Add legend to the right
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8, frameon=False)

    # Save the figure to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Ensure the diagram is properly cropped
    buf.seek(0)
    plt.close(fig)
    return buf



@app.route('/generate', methods=['POST'])
def generate():
    parameters = []

    # Integrate predicted casing/liner based on geologic input
    if 'gamma_ray' in request.form:
        input_data = {
            'Gamma Ray (GR)': float(request.form['gamma_ray']),
            'Resistivity (ohm.m)': float(request.form['resistivity']),
            'Formation Pressure (psi)': float(request.form['pressure']),
            'Formation Temperature (°F)': float(request.form['temperature'])
        }
        input_df = pd.DataFrame([input_data])  # Create input DataFrame
        predicted_depth = depth_model.predict(input_df)[0]
        predicted_material = material_model.predict(input_df)[0]

        # Add predicted casing to parameters FIRST
        parameters.append({
            "type": "casing",
            "x": 1.25,
            "top_depth": 0,
            "bottom_depth": int(predicted_depth),
            "width": 1.5,
            "color": "cyan",  # Highlight predicted casing with a distinct color
            "label": f"Predicted Casing ({predicted_material})"
        })

    # Add default parameters AFTER predicted casing
    parameters.append({
        "type": "casing",
        "x": 0.5,
        "top_depth": 0,
        "bottom_depth": int(request.form['surface_casing_depth']),
        "width": 3,
        "hatch": "o",
        "color": "none",
        "label": "Surface Casing"
    })
    parameters.append({
        "type": "casing",
        "x": 1,
        "top_depth": int(request.form['surface_casing_depth']),
        "bottom_depth": int(request.form['production_casing_depth']),
        "width": 2,
        "hatch": "..",
        "color": "none",
        "label": "Production Casing"
    })
    parameters.append({
        "type": "casing",
        "x": 1.5,
        "top_depth": int(request.form['production_casing_depth']),
        "bottom_depth": int(request.form['production_liner_depth']),
        "width": 1,
        "color": "lightgreen",
        "label": "Production Liner"
    })
    parameters.append({
        "type": "casing",
        "x": 1.75,
        "top_depth": 0,
        "bottom_depth": int(request.form['tubing_depth']),
        "width": 0.5,
        "color": "red",
        "label": "Tubing"
    })
    parameters.append({
        "type": "completion",
        "x": 2,
        "depth": int(request.form['tubing_pump_depth']),
        "color": "orange",
        "label": "Tubing Pump"
    })

    # Generate diagram
    img = generate_wellbore_diagram(parameters)
    return send_file(img, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)
