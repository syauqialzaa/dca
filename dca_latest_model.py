import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load dataset
file_path = 'Dataset DCA.xlsx'  # Replace with the correct path
df = pd.ExcelFile(file_path)
df_select = df.parse('Select select')

# Preprocessing
df_cleaned = df_select.dropna(subset=['TSTFLUID', 'TSTOIL'])  # Drop rows with missing data
df_cleaned['TEST_DATE'] = pd.to_datetime(df_cleaned['TEST_DATE'])  # Convert dates
df_cleaned = df_cleaned.sort_values(by=['STRING_CODE', 'TEST_DATE'])  # Sort data

# Filter data for a specific well
well_data = df_cleaned[df_cleaned['STRING_CODE'] == 'PKU00001-01']

# Cek data produksi terakhir
last_production_date = well_data[well_data['TSTOIL'] > 0]['TEST_DATE'].max()
print(f"Data produksi terakhir: {last_production_date}")

# Cek latest job code
latest_job = well_data.dropna(subset=['JOB_CODE']).sort_values(by='TEST_DATE', ascending=False)
if not latest_job.empty:
    latest_job_date = latest_job['TEST_DATE'].iloc[0]
    print(f"Latest job code date: {latest_job_date}")
else:
    print("Tidak ada job code untuk sumur ini.")

# Validasi start point
if not latest_job.empty and latest_job_date >= last_production_date - pd.DateOffset(months=12):
    calculated_start_date = latest_job_date
else:
    calculated_start_date = last_production_date - pd.DateOffset(months=12)

print(f"Start point berdasarkan logika: {calculated_start_date}")

# Tentukan start_date untuk analisis
if not latest_job.empty:
    if latest_job_date >= last_production_date - pd.DateOffset(months=12):
        start_date = latest_job_date
    else:
        start_date = last_production_date - pd.DateOffset(months=12)
else:
    start_date = last_production_date - pd.DateOffset(months=12)

print(f"Start point yang divisualisasikan: {start_date}")

# Filter data berdasarkan start_date
filtered_data = well_data[well_data['TEST_DATE'] >= start_date]

# Handling outliers using 1.5 * IQR rule
Q1 = filtered_data['TSTOIL'].quantile(0.25)
Q3 = filtered_data['TSTOIL'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove rows with production values outside the bounds
filtered_data = filtered_data[(filtered_data['TSTOIL'] >= lower_bound) & (filtered_data['TSTOIL'] <= upper_bound)]

# Remove plateau data for visualization (only include changes in production)
threshold = 1e-2  # Define a small threshold for detecting significant changes
filtered_data['TSTOIL_diff'] = filtered_data['TSTOIL'].diff().fillna(0)
visualization_data = filtered_data[filtered_data['TSTOIL_diff'].abs() > threshold]

# Calculate mid-point based on significant trend changes
filtered_data['TSTOIL_slope'] = filtered_data['TSTOIL'].diff() / filtered_data['TEST_DATE'].diff().dt.days
filtered_data['TSTOIL_slope'] = filtered_data['TSTOIL_slope'].rolling(window=5, center=True).mean()  # Smoothing
significant_change = filtered_data['TSTOIL_slope'].abs() > 50  # Example threshold for significant slope change
if significant_change.any():
    mid_index = significant_change.idxmax()
    mid_date = filtered_data.loc[mid_index, 'TEST_DATE']
    # Ensure mid-point is within start and end date
    if mid_date < start_date or mid_date > filtered_data['TEST_DATE'].max():
        mid_date = start_date + (filtered_data['TEST_DATE'].max() - start_date) / 2
else:
    mid_date = start_date + (filtered_data['TEST_DATE'].max() - start_date) / 2

# Identify end-point based on economic limit or stabilization
elr = 10  # Economic limit rate (BOPD)
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

# Optimized parameters (replace with your optimization results)
opt_exp_params = [q[0], 0.001]  # Initial guess for Exponential optimization
opt_harm_params = [q[0], 0.002]  # Initial guess for Harmonic optimization
opt_hyper_params = [q[0], 1.0, 0.001]  # Initial guess for Hyperbolic optimization

# Generate predictions for optimized fitting and extend 4 months into the future
t_fit = np.linspace(0, t[-1] + 365, len(t) + 30)  # Extend 120 days (4 months)
exp_fit_opt = exponential_decline(t_fit, *opt_exp_params)
harm_fit_opt = harmonic_decline(t_fit, *opt_harm_params)
hyper_fit_opt = hyperbolic_decline(t_fit, *opt_hyper_params)

# Create extended date range for prediction
extended_dates = [start_date + pd.Timedelta(days=int(day)) for day in t_fit]

# Calculate MAE for historical data only
historical_t = np.linspace(0, t[-1], len(t))
historical_exp_fit = exponential_decline(historical_t, *opt_exp_params)
historical_harm_fit = harmonic_decline(historical_t, *opt_harm_params)
historical_hyper_fit = hyperbolic_decline(historical_t, *opt_hyper_params)
mae_exp_opt = mean_absolute_error(q, historical_exp_fit)
mae_harm_opt = mean_absolute_error(q, historical_harm_fit)
mae_hyper_opt = mean_absolute_error(q, historical_hyper_fit)

# Visualize results
plt.figure(figsize=(12, 6))

# Plot actual data with significant changes only
plt.plot(visualization_data['TEST_DATE'], visualization_data['TSTOIL'], label='Actual Data (Significant Changes)', color='orange', marker='o', linestyle='-', linewidth=1)

# Generate predictions and plot DCA models (optimized)
plt.plot(extended_dates, exp_fit_opt, label='Exponential Decline (Optimized)', linestyle='--', color='green')
plt.plot(extended_dates, harm_fit_opt, label='Harmonic Decline (Optimized)', linestyle='--', color='blue')
plt.plot(extended_dates, hyper_fit_opt, label='Hyperbolic Decline (Optimized)', linestyle='--', color='red')

# Highlight start, mid, and end points
plt.axvline(start_date, color='green', linestyle='--', label=f'Start Date ({start_date.date()})')
plt.axvline(mid_date, color='blue', linestyle='--', label=f'Mid-Point ({mid_date.date()})')
plt.axvline(end_date, color='red', linestyle='--', label=f'End Date ({end_date.date()})')

# Add labels, legends, and grid
plt.title('Optimized Decline Curve Analysis (DCA) with Future Prediction')
plt.xlabel('Date')
plt.ylabel('Production Rate (BOPD)')
plt.legend()
plt.grid(True)
plt.show()

# Print optimized parameters and MAE
print(f"Start Date: {start_date}")
print(f"Mid-Point: {mid_date}")
print(f"End Date: {end_date}\n")

print("Mean Absolute Error (MAE) for Optimized Models (Historical Data):")
print(f"  Exponential Decline Model: {mae_exp_opt:.2f} BOPD")
print(f"  Harmonic Decline Model: {mae_harm_opt:.2f} BOPD")
print(f"  Hyperbolic Decline Model: {mae_hyper_opt:.2f} BOPD")
