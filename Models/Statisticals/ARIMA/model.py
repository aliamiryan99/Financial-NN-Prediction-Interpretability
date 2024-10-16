import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from Configs.ConfigSchema import Config
from Controllers.ModelModules.modules import (preprocess_data, scale_data,
                                              split_data)
from Utils.io import load_data, save_results
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings

# Define ARIMA parameters (p, d, q)
ARIMA_ORDER = (2, 1, 2)

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

def forecast(model_fit, test_data):
    """
    Make predictions using the trained ARIMA model.

    Parameters:
    - model_fit: The fitted ARIMA model.
    - test (series): test data to forcast.

    Returns:
    - y_pred (np.ndarray): Forecasted values.
    """
    # Prepare a DataFrame to store predictions
    predictions = pd.Series(index=test_data.index)

    # Real-time prediction loop
    for time_point in test_data.index:
        # Forecast the next time step
        forecast = model_fit.forecast(steps=1)
        
        # Store the prediction
        predictions[time_point] = forecast.values[0]

        # Get the new data point
        new_value = test_data.loc[time_point, 'Volume']
        
        # Append the new data point to the existing data
        model_fit = model_fit.apply(endog=pd.Series([new_value]), refit=False)

    return predictions

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
    data = preprocess_data(data, model_parameters.feature_columns, filter_holidays=config.preprocess_parameters)

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
        order=ARIMA_ORDER
    )

    # Forecast
    print("Step 6: Forecasting the Test Data")
    y_pred = forecast(model_fit, test)
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
