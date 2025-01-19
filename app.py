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
from scipy.optimize import curve_fit
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
# Allow cross-origin requests
# CORS(app, resources={r"/generate": {"origins": "http://localhost:63342"}})
CORS(app)

# # Load trained models
# depth_model = joblib.load('depth_model.pkl')
# material_model = joblib.load('material_model.pkl')
#
# # Load historical data from JSON file
# with open('historical_actual_preprocessed.json') as f:
#     raw_data = json.load(f)
#
# # Load the trained model
# with open('final_model.pkl', 'rb') as file:
#     best_model = pickle.load(file)
#
# # Format data for chart (Date -> timestamp, Production -> y)
# historical_data = [{"x": entry["Date"], "y": entry["Production"]} for entry in raw_data]

# Load and preprocess dataset
file_path = 'DCA1.xlsx'

#updateed 28/12/2024
# Define DCA models
def exponential_decline(t, qi, b):
    return qi * np.exp(-b * t)

def harmonic_decline(t, qi, b):
    return qi / (1 + b * t)

def hyperbolic_decline(t, qi, b, n):
    return qi * (1 + b * t) ** (-1 / n)

# Load the adjusted dataset
adjusted_data = pd.read_excel(file_path)
adjusted_data['TEST_DATE'] = pd.to_datetime(adjusted_data['TEST_DATE'], format='%d/%m/%Y')

# Handle outliers
def handle_outliers(data, column):
    """Remove outliers using the IQR method."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Clean TSTOIL and TSTFLUID columns
adjusted_data = handle_outliers(adjusted_data, 'TSTOIL')
adjusted_data = handle_outliers(adjusted_data, 'TSTFLUID')

# Determine starting points for DCA based on data after the last JOB_CODE
grouped_wells_dca = adjusted_data.groupby('STRING_CODE')
dca_start_points_after_jobcode = {}

for well, data in grouped_wells_dca:
    data_sorted = data.sort_values(by='TEST_DATE')
    last_jobcode_date = data_sorted[data_sorted['JOB_CODE'].notnull()]['TEST_DATE'].max()
    data_after_jobcode = data_sorted[data_sorted['TEST_DATE'] > last_jobcode_date]

    if not data_after_jobcode.empty:
        oil_diff = data_after_jobcode['TSTOIL'].diff().fillna(0)
        stable_points = data_after_jobcode[oil_diff <= 0]
        if not stable_points.empty:
            dca_start_points_after_jobcode[well] = stable_points.iloc[0]

# Adjust fitting with better initial guesses and bounds
exp_initial = [60, 0.01]  # [qi, b]
harm_initial = [60, 0.01]  # [qi, b]
hyper_initial = [60, 0.01, 1.0]  # [qi, b, n]
hyper_bounds = ([0, 0, 0.5], [np.inf, 0.1, 2])

def validate_dca_data(well_data):
    if len(well_data) < 2:
        return "Data terlalu sedikit untuk analisis DCA."
    if (well_data['TSTOIL'] <= 0).any():
        return "Terdapat nilai produksi minyak (TSTOIL) yang nol atau negatif."
#     if well_data['TSTOIL'].diff().fillna(0).gt(0).any():
#         return "Produksi minyak mengalami peningkatan pada beberapa titik, tidak sesuai untuk DCA."
    return None

def predict_to_economic_limit(model, params, economic_limit):
    t = 0
    while model(t, *params) > economic_limit:
        t += 1
    return t

@app.route('/')
def index():
    return render_template('index.html')

# def load_data():
#     data = pd.read_excel(file_path)
#     data['TEST_DATE'] = pd.to_datetime(data['TEST_DATE'])
#     data_sorted = data.sort_values(by="TEST_DATE")
#     data_sorted = data_sorted.groupby("TEST_DATE")["TSTOIL"].sum().reset_index()
#     data_sorted.columns = ['Date', 'Production']
#     data_sorted['Production'] = data_sorted['Production'].interpolate(method='linear')
#
#     # Remove outliers using IQR
#     Q1 = data_sorted['Production'].quantile(0.25)
#     Q3 = data_sorted['Production'].quantile(0.75)
#     IQR = Q3 - Q1
#     data_sorted = data_sorted[
#         ~((data_sorted['Production'] < (Q1 - 1.5 * IQR)) |
#           (data_sorted['Production'] > (Q3 + 1.5 * IQR)))
#     ]
#     return data_sorted
#
# data_sorted = load_data()

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
    if not start_date and not end_date:
        max_date = data['TEST_DATE'].max()
        min_date = max_date - pd.DateOffset(months=24)
        data = data[(data['TEST_DATE'] >= min_date) & (data['TEST_DATE'] <= max_date)]

    # Sort data
    data = data.sort_values(by='TEST_DATE')
    data['JOB_CODE'] = data['JOB_CODE'].astype(str).replace('nan','')
#     data['JOB_CODE'] = data['JOB_CODE'].apply(lambda x: x if pd.notnull(x) else '')
#     update if JObCode is null / empty / undefined then set to empty string
    # Convert to JSON-friendly format
    data_json = data[['TEST_DATE', 'TSTOIL', 'TSTFLUID','JOB_CODE']].rename(
        columns={'TEST_DATE': 'Date', 'TSTOIL': 'Production', 'TSTFLUID': 'Fluid','JOB_CODE': 'JobCode'}
    )
    data_json['Date'] = data_json['Date'].dt.strftime('%Y-%m-%d')
    history = data_json.to_dict(orient='records')

    # Filter to only include points where production changes (grouping based on Production)
    filtered_history = []
    prev_value = None
    for record in history:
        current_prod = record['Production']
        current_job_code = record.get('JobCode', '').strip()  # Get JobCode, default to empty string if missing
        if prev_value is None or current_prod != prev_value:
          filtered_history.append(record)
        elif current_prod == prev_value and current_job_code:
          filtered_history[-1] = record
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
        t_fit = np.linspace(0, t[-1] + 120, len(t) + 30)
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


# @app.route('/calculate_dca2', methods=['POST'])
# def calculate_dca_endpoint():
#     try:
#         data = request.get_json()
#         well = data['well']
#         start_date = data['start_date']
#         end_date = data['end_date']
#
#         # Load the dataset and filter based on well and date
#         filtered_data = load_filtered_data(well, start_date, end_date)
#
#         # Calculate DCA
#         results = analyze_dca(filtered_data)
#
#         # Convert NumPy arrays to lists for JSON serialization
#         results['exp_fit'] = results['exp_fit'].tolist()
#         results['harm_fit'] = results['harm_fit'].tolist()
#         results['hyper_fit'] = results['hyper_fit'].tolist()
#
#         # Convert dates to strings
#         results['start_date'] = str(results['start_date'])
#         results['mid_date'] = str(results['mid_date'])
#         results['end_date'] = str(results['end_date'])
#
#         return jsonify(results), 200
#
#     except Exception as e:
#         print(e)
#         return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500


@app.route('/automatic_dca', methods=['POST'])
def automatic_dca_analysis():
    data = request.get_json()
    well = data.get('well')

    if well not in adjusted_data['STRING_CODE'].unique():
        return jsonify({"error": f"Well {well} not found in dataset."}), 404

    if well not in dca_start_points_after_jobcode:
        return jsonify({"error": f"No valid starting points for well {well}."}), 404

    start_point = dca_start_points_after_jobcode[well]
    well_data = adjusted_data[
        (adjusted_data['STRING_CODE'] == well) &
        (adjusted_data['TEST_DATE'] >= start_point['TEST_DATE'])
    ].sort_values(by='TEST_DATE')

    # Filter hanya data ketika ada perubahan produksi
    well_data = well_data[well_data['TSTOIL'].diff().fillna(0) != 0]

    validation_error = validate_dca_data(well_data)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    try:
        t = (well_data['TEST_DATE'] - well_data['TEST_DATE'].min()).dt.days
        q = well_data['TSTOIL']

        exp_params, _ = curve_fit(exponential_decline, t, q, p0=exp_initial, maxfev=10000)
        harm_params, _ = curve_fit(harmonic_decline, t, q, p0=harm_initial, maxfev=10000)
        hyper_params, _ = curve_fit(hyperbolic_decline, t, q, p0=hyper_initial, bounds=hyper_bounds, maxfev=10000)

        # Data historis (hanya data perubahan produksi)
        actual_data = [
            {"date": date.strftime('%Y-%m-%d'), "value": value, "fluid": fluid}
            for date, value, fluid in zip(well_data['TEST_DATE'], well_data['TSTOIL'], well_data['TSTFLUID'])
        ]

        return jsonify({
            "Exponential": exp_params.tolist(),
            "Harmonic": harm_params.tolist(),
            "Hyperbolic": hyper_params.tolist(),
            "StartDate": well_data['TEST_DATE'].min().strftime('%Y-%m-%d'),
            "EndDate": well_data['TEST_DATE'].max().strftime('%Y-%m-%d'),
            "ActualData": actual_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_production', methods=['POST'])
def predict_production():
    try:
        request_data = request.get_json()
        well = request_data.get('well')
        start_date = request_data.get('start_date', '2024-07-09')
        selected_data = request_data.get('selected_data')  # Data yang dipilih
        economic_limit = request_data.get('elr', 5)

        # Debugging
        if not well:
            return jsonify({"error": "Well parameter is missing"}), 400
        if not selected_data:
            print("Selected data is empty")
        print(f"Start date: {start_date}, Economic limit: {economic_limit}")

        # Validasi selected_data
        if selected_data:
            well_data = pd.DataFrame(selected_data)
            well_data['Date'] = pd.to_datetime(well_data['Date'])
            well_data = well_data.sort_values(by='Date')
        else:
            # Proses data dari start_date jika selected_data tidak ada
            well_data = adjusted_data[(adjusted_data['STRING_CODE'] == well) & (adjusted_data['TEST_DATE'] >= start_date)]
            if well_data.empty:
                return jsonify({"error": f"No data available for well {well} starting from {start_date}."}), 404

        # Prediksi DCA
        t = (well_data['Date'] - well_data['Date'].min()).dt.days
        q = well_data['Production']

        exp_params, _ = curve_fit(exponential_decline, t, q, p0=exp_initial, maxfev=10000)
        harm_params, _ = curve_fit(harmonic_decline, t, q, p0=harm_initial, maxfev=10000)
        hyper_params, _ = curve_fit(hyperbolic_decline, t, q, p0=hyper_initial, bounds=hyper_bounds, maxfev=10000)

        # Prediksi ke depan
        predictions = {"Exponential": [], "Harmonic": [], "Hyperbolic": []}
        future_days = np.arange(0, predict_to_economic_limit(exponential_decline, exp_params, economic_limit))

        for day in future_days:
            predictions["Exponential"].append({
                "date": (well_data['Date'].min() + pd.Timedelta(days=day)).strftime('%Y-%m-%d'),
                "value": exponential_decline(day, *exp_params)
            })
            predictions["Harmonic"].append({
                "date": (well_data['Date'].min() + pd.Timedelta(days=day)).strftime('%Y-%m-%d'),
                "value": harmonic_decline(day, *harm_params)
            })
            predictions["Hyperbolic"].append({
                "date": (well_data['Date'].min() + pd.Timedelta(days=day)).strftime('%Y-%m-%d'),
                "value": hyperbolic_decline(day, *hyper_params)
            })

        return jsonify({"Predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
