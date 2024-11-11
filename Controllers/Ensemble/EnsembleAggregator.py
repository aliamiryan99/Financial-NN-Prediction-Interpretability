# Import necessary libraries
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

# Define the path to your results folder
results_folder = 'Results\\ForexData\\XAUUSD_H1'  # Replace with your actual path

# Collect all CSV files
csv_files = glob.glob(os.path.join(results_folder, '*.csv'))

# Initialize an empty list to store DataFrames
df_list = []

# Loop through each file, read the VolumeForecast column, and append it to the list
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df['VolumeForecast'])

# Create a combined DataFrame with all VolumeForecast columns side by side
forecast_df = pd.concat(df_list, axis=1)

# Name the columns based on the model names extracted from file names
model_names = [os.path.splitext(os.path.basename(file))[0] for file in csv_files]
forecast_df.columns = model_names

# Calculate statistical parameters for ensemble forecast
forecast_df['VolumeForecast_Avg'] = forecast_df.mean(axis=1)
forecast_df['VolumeForecast_Min'] = forecast_df.min(axis=1)
forecast_df['VolumeForecast_Max'] = forecast_df.max(axis=1)
forecast_df['VolumeForecast_Var'] = forecast_df.var(axis=1)
forecast_df['VolumeForecast_STD'] = forecast_df.std(axis=1)

# Calculate the uncertainty percentage
forecast_df['Uncertainty_Percentage'] = (forecast_df['VolumeForecast_STD'] / forecast_df['VolumeForecast_Avg']) * 100

# Read one of the original CSV files to get the additional columns (e.g., timestamps)
original_df = pd.read_csv(csv_files[0])

# Replace the VolumeForecast column with the ensemble average
original_df['VolumeForecast'] = forecast_df['VolumeForecast_Avg']

# Add the new statistical columns to the original DataFrame
original_df['VolumeForecast_Min'] = forecast_df['VolumeForecast_Min']
original_df['VolumeForecast_Max'] = forecast_df['VolumeForecast_Max']
original_df['VolumeForecast_Var'] = forecast_df['VolumeForecast_Var']
original_df['Uncertainty_Percentage'] = forecast_df['Uncertainty_Percentage']

# Save the new ensemble CSV file
output_file = os.path.join(results_folder, 'EnsembleAggregator.csv')
original_df.to_csv(output_file, index=False)

print(f"Ensemble forecast saved as: {output_file}")
