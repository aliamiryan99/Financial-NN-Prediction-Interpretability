import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from Configs.ConfigSchema import Config
from Controllers.ModelModules.modules import (preprocess_data, scale_data,
                                              split_data)
from Utils.io import load_data, save_results

# Define ARIMA parameters (p, d, q)
arima_order = (5, 1, 3)

def fit_arima_model(train_data, target_column, order):
    """
    Fit an ARIMA model to the training data.

    Parameters:
    - train_data (pd.DataFrame): The training dataset.
    - target_column (str): The column to forecast.
    - order (tuple): The (p, d, q) order of the ARIMA model.

    Returns:
    - model_fit: The fitted ARIMA model.
    """
    y_train = train_data[target_column]

    # Fit the ARIMA model
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def forecast(model_fit, steps):
    """
    Make predictions using the trained ARIMA model.

    Parameters:
    - model_fit: The fitted ARIMA model.
    - steps (int): Number of periods to forecast.

    Returns:
    - y_pred (np.ndarray): Forecasted values.
    """
    y_pred = model_fit.forecast(steps=steps)
    return y_pred

def run(config: Config):
    """
    Orchestrate the ARIMA modeling process.

    Parameters:
    - config (Config): Configuration object loaded from Config.yaml.
    """
    model_parameters = config.model_parameters

    # Load data
    print("Step 1: Loading the Data")
    data = load_data(config.data.in_path)

    # Preprocess data
    print("Step 2: Preprocessing the Data")
    data = preprocess_data(data, model_parameters.feature_columns, filter_holidays=True)

    # Scale data
    print("Step 3: Scaling the Data")
    scaled_data, scalers = scale_data(data, model_parameters.feature_columns)

    # Split data into train/test
    print("Step 4: Splitting the Data")
    train, test = split_data(scaled_data, model_parameters.train_ratio)

    # Fit ARIMA model
    print("Step 5: Fitting the ARIMA Model")
    model_fit = fit_arima_model(
        train_data=train,
        target_column=model_parameters.target_column,
        order=arima_order
    )

    # Forecast
    print("Step 6: Forecasting the Test Data")
    n_periods = len(test)
    y_pred = forecast(model_fit, steps=n_periods)
    y_pred = np.array(y_pred)

    # Inverse scale predictions back to original scale
    print("Step 7: Inversing the Scale of Predictions")
    volume_scaler = scalers['Volume']
    y_pred_inv = volume_scaler.inverse_transform(y_pred.reshape(-1, 1))

    # Save results
    print("Step 8: Saving the Results")
    save_results(
        data=data,
        predictions=y_pred_inv.flatten(),
        train_size=len(train),
        offset=0,  # No sequence length offset for ARIMA
        out_path=config.data.out_path
    )
