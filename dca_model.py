import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Define the three DCA models
def exponential_decline(t, qi, d):
    return qi * np.exp(-d * t)

def harmonic_decline(t, qi, b):
    return qi / (1 + b * t)

def hyperbolic_decline(t, qi, b, d):
    return qi / (1 + b * d * t) ** (1 / b)

# Preprocessing function
def preprocess_data(file_path, string_code):
    # Load dataset
    df = pd.ExcelFile(file_path)
    df_select = df.parse('Select select')

    # Data cleaning and sorting
    df_cleaned = df_select.dropna(subset=['TSTFLUID', 'TSTOIL'])
    df_cleaned['TEST_DATE'] = pd.to_datetime(df_cleaned['TEST_DATE'])
    df_cleaned = df_cleaned.sort_values(by=['STRING_CODE', 'TEST_DATE'])

    # Filter data for a specific well
    well_data = df_cleaned[df_cleaned['STRING_CODE'] == string_code]

    if well_data.empty:
        raise ValueError(f"No data found for well: {string_code}")

    # Identify latest job code
    latest_job = well_data.dropna(subset=['JOB_CODE']).sort_values(by='TEST_DATE', ascending=False)
    if not latest_job.empty:
        latest_job_date = latest_job['TEST_DATE'].iloc[0]
        filtered_data = well_data[well_data['TEST_DATE'] >= latest_job_date]
    else:
        filtered_data = well_data[well_data['TEST_DATE'] >= well_data['TEST_DATE'].max() - pd.DateOffset(months=12)]

    if filtered_data.empty:
        raise ValueError("Filtered data is empty after applying date range.")

    # Handling outliers using 1.5 * IQR rule
    Q1 = filtered_data['TSTOIL'].quantile(0.25)
    Q3 = filtered_data['TSTOIL'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_data = filtered_data[(filtered_data['TSTOIL'] >= lower_bound) & (filtered_data['TSTOIL'] <= upper_bound)]
    if filtered_data.empty:
        raise ValueError("Filtered data is empty after removing outliers.")

    return filtered_data


def determine_points(filtered_data, elr=10, threshold=50):
    if filtered_data.empty:
        raise ValueError("Filtered data is empty, cannot determine points.")

    # Calculate stabilization window for start point
    filtered_data['TSTOIL_diff'] = filtered_data['TSTOIL'].diff()
    stabilization_window = filtered_data['TSTOIL_diff'].abs().rolling(window=5).mean() < 5
    start_index = stabilization_window.idxmax()

    if start_index not in filtered_data.index:
        raise ValueError("Start point could not be determined.")

    start_date = filtered_data.loc[start_index, 'TEST_DATE']

    # Identify mid-point based on significant trend changes
    filtered_data['TSTOIL_slope'] = filtered_data['TSTOIL'].diff() / filtered_data['TEST_DATE'].diff().dt.days
    significant_change = filtered_data['TSTOIL_slope'].abs() > threshold
    mid_index = significant_change.idxmax()

    if mid_index not in filtered_data.index:
        mid_date = filtered_data['TEST_DATE'].iloc[len(filtered_data) // 2]
    else:
        mid_date = filtered_data.loc[mid_index, 'TEST_DATE']

    # Identify end-point based on economic limit or stabilization
    end_data = filtered_data[filtered_data['TSTOIL'] >= elr]
    end_date = end_data['TEST_DATE'].iloc[-1] if not end_data.empty else filtered_data['TEST_DATE'].iloc[-1]

    return start_date, mid_date, end_date


# Use pre-optimized parameters to fit models and predict
def fit_models_with_optimized_params(t, t_future, q):
    if len(t) == 0 or len(q) == 0:
        raise ValueError("Input data for model fitting is empty.")

    opt_exp_params = [68.78, 0.0017]
    opt_harm_params = [72.24, 0.0026]
    opt_hyper_params = [72.06, 0.87, 0.0025]

    exp_forecast = exponential_decline(t_future, *opt_exp_params)
    harm_forecast = harmonic_decline(t_future, *opt_harm_params)
    hyper_forecast = hyperbolic_decline(t_future, *opt_hyper_params)

    exp_mae = mean_absolute_error(q, exponential_decline(t, *opt_exp_params))
    harm_mae = mean_absolute_error(q, harmonic_decline(t, *opt_harm_params))
    hyper_mae = mean_absolute_error(q, hyperbolic_decline(t, *opt_hyper_params))

    return {
        "params": {
            "exponential": opt_exp_params,
            "harmonic": opt_harm_params,
            "hyperbolic": opt_hyper_params
        },
        "forecast": {
            "exponential": exp_forecast.tolist(),
            "harmonic": harm_forecast.tolist(),
            "hyperbolic": hyper_forecast.tolist()
        },
        "mae": {
            "exponential": exp_mae,
            "harmonic": harm_mae,
            "hyperbolic": hyper_mae
        }
    }


# Full analysis pipeline
def analyze_dca(file_path, string_code, future_days=365):
    filtered_data = preprocess_data(file_path, string_code)

    if filtered_data.empty:
        raise ValueError(f"No data available for analysis after preprocessing for well: {string_code}")

    start_date, mid_date, end_date = determine_points(filtered_data)

    analysis_data = filtered_data[(filtered_data['TEST_DATE'] >= start_date) & (filtered_data['TEST_DATE'] <= end_date)]
    analysis_data['days'] = (analysis_data['TEST_DATE'] - start_date).dt.days
    t = analysis_data['days'].values
    q = analysis_data['TSTOIL'].values

    if len(t) == 0 or len(q) == 0:
        raise ValueError("Insufficient data points for analysis.")

    t_future = np.linspace(0, t[-1] + future_days, len(t) + future_days)

    results = fit_models_with_optimized_params(t, t_future, q)

    return {
        "start_date": str(start_date),
        "mid_date": str(mid_date),
        "end_date": str(end_date),
        "results": results
    }

