# Import Necessary Libraries
import time
import warnings

import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

warnings.filterwarnings("ignore")  # Suppress warnings

# Config fields
input_data_path = "Data/ForexData/XAUUSD_H1.csv"
output_data_path = "Results/SARIMA.csv"

# Step 1: Load the Data
start_time = time.time()
print("Step 1: Loading the Data")
data = pd.read_csv(input_data_path, parse_dates=['Time'], index_col='Time')
end_time = time.time()
print(f"Step 1 completed in {end_time - start_time:.2f} seconds\n")

# Step 2: Preprocess the Data
start_time = time.time()
print("Step 2: Preprocessing the Data")
data.sort_index(inplace=True)  # Ensure data is sorted by Time
data = data.asfreq('H')        # Set the frequency to hourly

# Handle missing values by forward-filling
data['Volume'].fillna(method='ffill', inplace=True)
end_time = time.time()
print(f"Step 2 completed in {end_time - start_time:.2f} seconds\n")

# Optional: Reduce data size for faster computation (e.g., last 90 days)
# data = data.last('90D')

# Step 3: Use auto_arima to Find the Best Hyperparameters
start_time = time.time()
print("Step 3: Finding the Best Hyperparameters")
stepwise_fit = auto_arima(
    data['Volume'][:1000],
    seasonal=True,
    m=24,
    stepwise=True,
    max_p=2, max_d=1, max_q=2,
    max_P=1, max_D=1, max_Q=1,
    suppress_warnings=True
)
end_time = time.time()
print(f"Step 3 completed in {end_time - start_time:.2f} seconds\n")

# Step 4: Split the Data into Training and Testing Sets
start_time = time.time()
print("Step 4: Splitting the Data")
train_size = int(0.8 * len(data))
train = data.iloc[:train_size]
test = data.iloc[train_size:]
end_time = time.time()
print(f"Step 4 completed in {end_time - start_time:.2f} seconds\n")

# Step 5: Fit the SARIMA Model Using the Best Parameters
start_time = time.time()
print("Step 5: Fitting the SARIMA Model")
best_order = stepwise_fit.order
best_seasonal_order = stepwise_fit.seasonal_order

model = ARIMA(
    train['Volume'],
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit()
end_time = time.time()
print(f"Step 5 completed in {end_time - start_time:.2f} seconds\n")


# Step 6: Forecast the Test Data
print("Step 6: Forecasting the Test Data")
# Generate dynamic forecasts
forecast = model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=False)

# Add forecast to test dataframe
test['VolumeForecast'] = forecast.values

####################### slow alternative
# predictions = pd.Series(index=test.index)

# # Real-time prediction loop with tqdm
# for time_point in tqdm(test.index, desc="Real-time Prediction Progress"):
#     # Forecast the next time step
#     forecast = model_fit.forecast(steps=1)
    
#     # Store the prediction
#     predictions[time_point] = forecast.iloc[0]
    
#     # Add forecast to test dataframe
#     test.loc[time_point, 'VolumeForecast'] = forecast.iloc[0]
    
#     # Get the new data point
#     new_value = test.loc[time_point, 'Volume']
    
#     # Update the model with the new data point
#     model_fit = model_fit.append([new_value], refit=False)

# Save the test data with forecast to a CSV file
#####################################

test.to_csv(output_data_path)

print(f"Results Saved to {output_data_path}")
