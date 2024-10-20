# Import Necessary Libraries
import warnings

import pandas as pd
from prophet import Prophet  # Import Prophet model

from Configs.config_schema import Config
from Controllers.ModelModules.modules import (
    preprocess_data,
    split_data
)
from Utils.io import load_data, save_results

warnings.filterwarnings("ignore")  # Suppress warnings

# ============================
# Global Prophet Parameters
# ============================

PROPHET_GROWTH = 'linear'  # Can be 'linear' or 'logistic'
PROPHET_DAILY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = False
PROPHET_YEARLY_SEASONALITY = False
PROPHET_SEASONALITY_MODE = 'additive'  # Can be 'additive' or 'multiplicative'
PROPHET_CHANGEOINT_PRIOR_SCALE = 0.05
PROPHET_SEASONALITY_PRIOR_SCALE = 10.0
PROPHET_HOLIDAYS_PRIOR_SCALE = 10.0

# ============================


def prepare_data_for_prophet(data, target_column):
    """
    Prepare the data for Prophet model.
    Prophet requires the dataframe to have two columns: 'ds' and 'y'.
    """
    df = data.copy()
    df.rename(columns={target_column: 'y'}, inplace=True)
    # Remove timezone information from the index
    df.index = df.index.tz_localize(None)
    df['ds'] = df.index  # Assuming the index is datetime
    df = df[['ds', 'y']]
    return df


def build_model():
    """Build the Prophet model using global parameters."""
    model = Prophet(
        growth=PROPHET_GROWTH,
        daily_seasonality=PROPHET_DAILY_SEASONALITY,
        weekly_seasonality=PROPHET_WEEKLY_SEASONALITY,
        yearly_seasonality=PROPHET_YEARLY_SEASONALITY,
        seasonality_mode=PROPHET_SEASONALITY_MODE,
        changepoint_prior_scale=PROPHET_CHANGEOINT_PRIOR_SCALE,
        seasonality_prior_scale=PROPHET_SEASONALITY_PRIOR_SCALE,
        holidays_prior_scale=PROPHET_HOLIDAYS_PRIOR_SCALE
    )
    return model


def run(config: Config):
    model_parameters = config.model_parameters

    # Load data
    print("Step 1: Loading the Data")
    data = load_data(config.data.in_path)

    # Preprocess data
    print("Step 2: Preprocessing the Data")
    data = preprocess_data(
        data,
        model_parameters.feature_columns + [model_parameters.target_column],
        filter_holidays=config.preprocess_parameters
    )

    # Ensure that the index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.set_index('Time', inplace=True)
    data.index = pd.to_datetime(data.index)

    # Remove timezone information from the index
    data.index = data.index.tz_localize(None)

    # Prepare data for Prophet
    print("Step 3: Preparing the Data for Prophet")
    df_prophet = prepare_data_for_prophet(data, model_parameters.target_column)

    # Split data
    print("Step 4: Splitting the Data")
    train_size = int(len(df_prophet) * model_parameters.train_ratio)
    train_data = df_prophet.iloc[:train_size]
    test_data = df_prophet.iloc[train_size:]

    # Build model
    print("Step 5: Building the Prophet Model")
    model = build_model()

    # Train model
    print("Step 6: Training the Model")
    model.fit(train_data)

    # Forecast
    print("Step 7: Forecasting the Test Data")
    # Use the 'ds' column from test data as future dataframe
    future = test_data[['ds']].copy()
    forecast = model.predict(future)

    # Merge actual and predicted values
    print("Step 8: Merging Actual and Predicted Values")
    results = test_data[['ds', 'y']].merge(forecast[['ds', 'yhat']], on='ds', how='left')

    # Save results
    print("Step 9: Saving the Results")
    save_results(
        data.reset_index(),
        results['yhat'].values,
        train_size,
        0,  # Prophet does not use sequence length
        config.data.out_path
    )

    # Optionally evaluate the model
    print("Step 10: Evaluating the Model")
    evaluate_model(results['y'].values, results['yhat'].values)


# Evaluation function
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def evaluate_model(y_true, y_pred):
    """Evaluate the model's performance."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
