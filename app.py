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

import pandas as pd
import numpy as np
from dca_model import analyze_dca
import logging
import tensorflow as tf
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
# Allow cross-origin requests
# CORS(app, resources={r"/generate": {"origins": "http://localhost:63342"}})
CORS(app)

# Load model once on startup
model = tf.keras.models.load_model("hybrid_tft_lstm_model.h5")

# Load dataset
file_path = 'DCA1.xlsx'
# df = pd.ExcelFile(file_path)
# adjusted_data = df.parse('A002')
adjusted_data = pd.read_excel(file_path)
adjusted_data['TEST_DATE'] = pd.to_datetime(adjusted_data['TEST_DATE'], format='%d/%m/%Y')

# Handle outliers
def handle_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Clean TSTOIL and TSTFLUID columns
adjusted_data = handle_outliers(adjusted_data, 'TSTOIL')
adjusted_data = handle_outliers(adjusted_data, 'TSTFLUID')

# Define DCA models using curve_fit
def exponential_decline(t, qi, b):
    return qi * np.exp(-b * t)

def harmonic_decline(t, qi, b):
    return qi / (1 + b * t)

def hyperbolic_decline(t, qi, b, n):
    return qi * (1 + b * t) ** (-1 / n)

# Define DCA models using Regresi Linear
def fit_exponential_linear(t, q):
    ln_q = np.log(q)
    slope, intercept = np.polyfit(t, ln_q, 1)
    d = -slope
    qi = np.exp(intercept)
    return [qi, d]

def fit_harmonic_linear(t, q):
    inv_q = 1 / q
    slope, intercept = np.polyfit(t, inv_q, 1)
    qi = 1 / intercept
    # b = slope*qi
    b = slope
    return [qi, b]

def fit_hyperbolic_linear(t, q, b_values=[0.001, 0.005, 0.01, 0.05, 0.1]):
  best_params = None
  best_mse = float('inf')

  for b_init in b_values:
    # Linearize hyperbolic equation
    ln_q = np.log(q)
    ln_term = np.log(1 + b_init * t)

    # Linear regression
    slope, intercept = np.polyfit(ln_term, ln_q, 1)

    # Extract parameters
    b = -1 / slope
    qi = np.exp(intercept)

    # Calculate predictions and evaluate error
    params = [qi, b, b_init]
    prediction = hyperbolic_decline(t, qi, b_init, b)
    mse = np.mean((prediction - q) ** 2)

    # Check if best fit
    if mse < best_mse:
      best_mse = mse
      best_params = [qi, b_init, b]  # qi, initial guess b, exponent b

  return best_params


# Adjust fitting with better initial guesses and bounds
fixed_dca_results = {}

# Initialize global variable to store the latest DCA result
latest_dca_result = None

# Determine starting points for DCA based on data after the last JOB_CODE (considering ignored JOB_CODE)
def determine_dca_start_points(data, ignored_job_codes):
    grouped_wells_dca = data.groupby('STRING_CODE')
    dca_start_points_after_jobcode = {}

    for well, well_data in grouped_wells_dca:
        well_data_sorted = well_data.sort_values(by='TEST_DATE')

        # Filter JOB_CODEs
        ignored_job_data = well_data_sorted[well_data_sorted['JOB_CODE'].isin(ignored_job_codes)]
        valid_job_data = well_data_sorted[~well_data_sorted['JOB_CODE'].isin(ignored_job_codes) & well_data_sorted['JOB_CODE'].notnull()]

        # Determine the last relevant date
        if not valid_job_data.empty:
            last_job_date = valid_job_data['TEST_DATE'].max()
        elif not ignored_job_data.empty:
            last_job_date = ignored_job_data['TEST_DATE'].max()
        else:
            last_job_date = None

        # Get data after the determined last JOB_CODE date
        if last_job_date is not None:
            data_after_jobcode = well_data_sorted[well_data_sorted['TEST_DATE'] > last_job_date]
        else:
            data_after_jobcode = well_data_sorted  # Use all data if no JOB_CODE exists

        # Identify stable points for DCA starting point
        if not data_after_jobcode.empty:
            oil_diff = data_after_jobcode['TSTOIL'].diff().fillna(0)
            fluid_diff = data_after_jobcode['TSTFLUID'].diff().fillna(0)
            stable_points = data_after_jobcode[(oil_diff <= 0) & (fluid_diff <= 0)]
            if not stable_points.empty:
                dca_start_points_after_jobcode[well] = stable_points.iloc[0]

    return dca_start_points_after_jobcode

ignored_job_codes = [
            'PMP14', 'PMP29', 'PMP43', 'PMP45', 'PMP01', 'PMP02', 'PMP03',
            'PMP04', 'PMP05', 'PMP31', 'PMP32', 'PMP33', 'PMP34', 'PMP35',
            'PMP36', 'PMP37', 'PMP38', 'PMP39'
        ]
dca_start_points_after_jobcode = determine_dca_start_points(adjusted_data, ignored_job_codes)

# Validate DCA data
def validate_dca_data(well_data):
    if len(well_data) < 2:
        return "Data terlalu sedikit untuk analisis DCA."
    if (well_data['TSTOIL'] <= 0).any():
        return "Terdapat nilai produksi minyak (TSTOIL) yang nol atau negatif."
    return None

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/get_data', methods=['GET'])
# def get_data():
#     return jsonify(historical_data)

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


def safe_curve_fit(model_func, t, q, param_grid, bounds=None, maxfev=10000):
    best_params = None
    lowest_error = float('inf')

    for p0 in param_grid:
        try:
            if bounds:
                params, _ = curve_fit(model_func, t, q, p0=p0, bounds=bounds, maxfev=maxfev)
            else:
                params, _ = curve_fit(model_func, t, q, p0=p0, maxfev=maxfev)

            prediction = model_func(t, *params)
            mse = np.mean((prediction - q) ** 2)

            if mse < lowest_error:
                lowest_error = mse
                best_params = params
        except:
            continue  # skip jika gagal konvergen

    return best_params

from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(t, q_actual, model_func, params):
    q_pred = model_func(t, *params)
    mse = mean_squared_error(q_actual, q_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(q_actual, q_pred)
    return {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "prediction": q_pred
    }


@app.route('/automatic_dca', methods=['POST'])
def automatic_dca_analysis():
    global latest_dca_result

    try:
        data = request.get_json()
        well = data.get('well')
        selected_data = data.get('selected_data')
        custom_filter = data.get('custom_filter')

        if selected_data:
            well_data_all = pd.DataFrame(selected_data)
            well_data_all['Date'] = pd.to_datetime(well_data_all['Date'])
            well_data_all.rename(columns={'Date': 'TEST_DATE', 'Production': 'TSTOIL','Fluid': 'TSTFLUID'}, inplace=True)
        else:
            if well not in dca_start_points_after_jobcode:
                return jsonify({"error": f"Well {well} not found in dataset."}), 404

            well_data_all = adjusted_data[adjusted_data['STRING_CODE'] == well]

            if custom_filter:
                if 'date_range' in custom_filter:
                    start_date, end_date = custom_filter['date_range']
                    well_data_all = well_data_all[(well_data_all['TEST_DATE'] >= start_date) & (well_data_all['TEST_DATE'] <= end_date)]
                if 'production_range' in custom_filter:
                    min_prod, max_prod = custom_filter['production_range']
                    well_data_all = well_data_all[(well_data_all['TSTOIL'] >= min_prod) & (well_data_all['TSTOIL'] <= max_prod)]
            else:
                two_years_ago = pd.Timestamp.now() - timedelta(days=2*365)
                last_job_date = well_data_all[
                    (well_data_all['JOB_CODE'].notnull()) &
                    (~well_data_all['JOB_CODE'].isin(ignored_job_codes)) &
                    (well_data_all['TEST_DATE'] >= two_years_ago)
                ]['TEST_DATE'].max()

                if pd.notnull(last_job_date):
                    start_date = last_job_date
                else:
                    start_date = two_years_ago

                well_data_all = well_data_all[well_data_all['TEST_DATE'] >= start_date]

        # print("Selected Data oi: ", selected_data)
        well_data_all = well_data_all.sort_values(by='TEST_DATE')

        validation_error = validate_dca_data(well_data_all)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        t = (well_data_all['TEST_DATE'] - well_data_all['TEST_DATE'].min()).dt.days
        q = well_data_all['TSTOIL']
        qi_initial = well_data_all['TSTOIL'].iloc[0]
        # print("qi data : ", qi_initial)
        exp_initial = [qi_initial, 0.01]
        harm_initial = [qi_initial, 0.01]
        hyper_initial = [qi_initial, 0.01, 1.0]
        hyper_bounds = ([0, 0, 0], [np.inf, 1.0, 2])

        # DCA using curve fit
#         exp_params, _ = curve_fit(exponential_decline, t, q, p0=exp_initial, maxfev=10000)
#         harm_params, _ = curve_fit(harmonic_decline, t, q, p0=harm_initial, maxfev=10000)
#         hyper_params, _ = curve_fit(hyperbolic_decline, t, q, p0=hyper_initial, bounds=hyper_bounds, maxfev=10000)

        # Parameter awal alternatif untuk grid search
        qi = qi_initial
        d_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        b_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        n_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

        # --- Exponential ---
        exp_grid = [[qi, d] for d in d_values]
        exp_params_cf = safe_curve_fit(exponential_decline, t, q, exp_grid)

        # --- Harmonic ---
        harm_grid = [[qi, b] for b in b_values]
        harm_params_cf = safe_curve_fit(harmonic_decline, t, q, harm_grid)

        # --- Hyperbolic ---
        hyper_grid = [[qi, d, n] for d in d_values for n in n_values]
        hyper_bounds = ([0, 0, 0], [np.inf, 1.0, 2])
        hyper_params_cf = safe_curve_fit(hyperbolic_decline, t, q, hyper_grid, bounds=hyper_bounds)

        # DCA using Regresi Linear
        exp_params_excel = fit_exponential_linear(t, q)
        harm_params_excel = fit_harmonic_linear(t, q)
        hyper_params_excel = fit_hyperbolic_linear(t, q, b_values=[0.001, 0.005, 0.01])

        exp_params = exp_params_excel
        harm_params = harm_params_excel
        hyper_params = hyper_params_excel

        latest_dca_result = (well_data_all, exp_params, harm_params, hyper_params)

        historical_data = [
            {"date": date.strftime('%Y-%m-%d'), "value": value, "fluid": fluid}
            for date, value, fluid in zip(well_data_all['TEST_DATE'], well_data_all['TSTOIL'], well_data_all['TSTFLUID'])
        ]

        # Inisialisasi daftar yang disaring dengan titik data pertama
        filtered_data = []
        if historical_data:
            filtered_data.append(historical_data[0])  # Selalu sertakan titik data pertama

            # Iterasi melalui titik-titik data yang tersisa
            for i in range(1, len(historical_data)):
                prev = historical_data[i - 1]
                current = historical_data[i]

                # Periksa apakah ada perubahan pada 'value' atau 'fluid'
                if current['value'] != prev['value'] or current['fluid'] != prev['fluid']:
                    filtered_data.append(current)

        start_date = well_data_all['TEST_DATE'].min().strftime('%Y-%m-%d')
        end_date = well_data_all['TEST_DATE'].max().strftime('%Y-%m-%d')

        # Faktor konversi dari per hari ke per tahun
        DAYS_PER_YEAR = 365
        # Faktor konversi ke persentase
        PERCENTAGE_FACTOR = 100

        # exp_eval = evaluate_model(t, q, exponential_decline, exp_params)
        # harm_eval = evaluate_model(t, q, harmonic_decline, harm_params)
        # hyper_eval = evaluate_model(t, q, hyperbolic_decline, hyper_params)

        exp_eval_excel = evaluate_model(t, q, exponential_decline, exp_params_excel)
        harm_eval_excel = evaluate_model(t, q, harmonic_decline, harm_params_excel)
        hyper_eval_excel = evaluate_model(t, q, hyperbolic_decline, hyper_params_excel)

        print("Exponential Eval : ", exp_eval_excel)
        print("Harmonic Eval : ", harm_eval_excel)
        print("Hyperbolic Eval : ", hyper_eval_excel)

        # Tampilkan secara eksplisit tanggal dan produksi minyak
        print("\nData yang digunakan untuk perhitungan DCA:")
        for date, oil in zip(well_data_all['TEST_DATE'], well_data_all['TSTOIL']):
          print(f"{date.strftime('%Y-%m-%d')}, TSTOIL: {oil}")

        return jsonify({
            "Exponential": exp_params,
            "Harmonic": harm_params,
            "Hyperbolic": hyper_params,
            # "Exponential": [round(value, 4) for value in exp_params.tolist()],
            # "Harmonic": [round(value, 4) for value in harm_params.tolist()],
            # "Hyperbolic": [round(value, 4) for value in hyper_params.tolist()],
            "DeclineRate": {
                            "Exponential": round(exp_params[1] * DAYS_PER_YEAR * PERCENTAGE_FACTOR, 2),
                            "Harmonic": round(harm_params[1] * DAYS_PER_YEAR * PERCENTAGE_FACTOR, 2),
                            "Hyperbolic": round(hyper_params[1] * DAYS_PER_YEAR * PERCENTAGE_FACTOR, 2)
                        },
            "ActualData": filtered_data,
            "StartDate": start_date,
            "EndDate": end_date
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/predict_production', methods=['POST'])
def predict_production():
    global latest_dca_result

    try:
        data = request.get_json()
        well = data.get('well')
        economic_limit = float(data.get('economic_limit', 5))
        selected_data = data.get('selected_data')
        print("Request Data:", data)
        print("Selected Data:", data.get("selected_data"))
        print("Economic Limit:", data.get("economic_limit"))

        if latest_dca_result is None:
            return jsonify({"error": "Run 'Model Automate DCA' first to generate DCA Prediction."}), 400


        well_data, exp_params, harm_params, hyper_params = latest_dca_result
        # Update parameter model dengan nilai terakhir historis
        last_q = well_data['TSTOIL'].iloc[-1]
        exp_params = [last_q, exp_params[1]]
        harm_params = [last_q, harm_params[1]]
        hyper_params = [last_q, hyper_params[1], hyper_params[2]]

        if selected_data:
            if isinstance(selected_data, dict) and 'Date' in selected_data and 'Production' in selected_data:
                start_date = pd.to_datetime(selected_data['Date'])
                start_production = selected_data['Production']

                # Update parameter model berdasarkan selected_data
                exp_params = [start_production, exp_params[1]]
                harm_params = [start_production, harm_params[1]]
                hyper_params = [start_production, hyper_params[1], hyper_params[2]]
            else:
                return jsonify({"error": "Invalid selected_data format. Must contain 'Date' and 'Production'."}), 400
        else:
            start_date = well_data['TEST_DATE'].max()

        def predict_to_economic_limit(model, params, economic_limit, start_date):
            t = 0
            predicted_dates = []
            predicted_values = []

            while model(t, *params) > economic_limit:
                predicted_dates.append(start_date + timedelta(days=t))
                predicted_values.append(model(t, *params))
                t += 1

            return predicted_dates, predicted_values

        exp_pred_dates, exp_pred_values = predict_to_economic_limit(
            exponential_decline, exp_params, economic_limit, start_date
        )
        harm_pred_dates, harm_pred_values = predict_to_economic_limit(
            harmonic_decline, harm_params, economic_limit, start_date
        )
        hyper_pred_dates, hyper_pred_values = predict_to_economic_limit(
            hyperbolic_decline, hyper_params, economic_limit, start_date
        )

        exp_predictions = [
            {"date": date.strftime('%Y-%m-%d'), "value": round(value,2)}
            for date, value in zip(exp_pred_dates, exp_pred_values)
        ]
        harm_predictions = [
            {"date": date.strftime('%Y-%m-%d'), "value": round(value,2)}
            for date, value in zip(harm_pred_dates, harm_pred_values)
        ]
        hyper_predictions = [
            {"date": date.strftime('%Y-%m-%d'), "value": round(value,2)}
            for date, value in zip(hyper_pred_dates, hyper_pred_values)
        ]

        return jsonify({
            "ExponentialPrediction": exp_predictions,
            "HarmonicPrediction": harm_predictions,
            "HyperbolicPrediction": hyper_predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_ml", methods=["POST"])
def predictml():
    data = request.get_json()
    elr = data.get("elr", 10.0)

    # Validasi ELR
    if not isinstance(elr, (int, float)) or elr <= 0:
        return jsonify({"error": "ELR must be a positive number"}), 400

    df = adjusted_data.copy()
    df.dropna(subset=['TSTFLUID', 'TSTOIL'], inplace=True)
    df = df.sort_values(by='TEST_DATE')
    df['days'] = (df['TEST_DATE'] - df['TEST_DATE'].min()).dt.days

    df['decline_rate'] = df['TSTOIL'].pct_change().fillna(0)
    df['moving_avg'] = df['TSTOIL'].rolling(window=5, min_periods=1).mean()
    df['exp_decline'] = df['TSTOIL'].ewm(span=10, adjust=False).mean()

    # Outlier removal
    def remove_outliers(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]

    for col in ['TSTOIL', 'TSTFLUID', 'decline_rate', 'moving_avg', 'exp_decline']:
        df = remove_outliers(df, col)

    X = df[['days', 'TSTFLUID', 'decline_rate', 'moving_avg', 'exp_decline']].values
    y = df['TSTOIL'].values

    # Reshape untuk prediksi
    X_seq = X.reshape((X.shape[0], 1, X.shape[1]))
    y_pred = model.predict(X_seq).flatten()

    # Extrapolation
    last_known = X[-1]
    future_days = []
    future_preds = []
    last_day = int(last_known[0])
    features = last_known[1:]
    batch_size = 10
    max_days = 100  # Batas maksimum hari prediksi

    max_iterations = 10  # Batas maksimum iterasi
    iteration = 5
    while iteration < max_iterations:
        batch_days = np.arange(last_day + 1, last_day + 1 + batch_size)
        batch_input = np.tile(features, (batch_size, 1))
        batch_data = np.column_stack([batch_days, batch_input]).reshape(batch_size, 1, -1)
        batch_preds = model.predict(batch_data).flatten()

        future_days.extend(batch_days.tolist())
        future_preds.extend(batch_preds.tolist())

        last_day = batch_days[-1]
        if batch_preds[-1] <= elr or (last_day - int(last_known[0])) > max_days:
            break

    # Format tanggal
    last_date = df['TEST_DATE'].iloc[-1]
    future_dates = [(last_date + timedelta(days=int(d - last_known[0]))).strftime('%Y-%m-%d') for d in future_days]
    history_dates = df['TEST_DATE'].dt.strftime('%Y-%m-%d').tolist()

    return jsonify({
        "dates_actual": history_dates,
        "actual": y.tolist(),
        "predicted": y_pred.tolist(),
        "dates_extended": future_dates,
        "extended_prediction": future_preds,
        "elr_threshold": elr
    })

if __name__ == '__main__':
    app.run(debug=True)
