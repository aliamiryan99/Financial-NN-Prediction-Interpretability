# Import Necessary Libraries
import warnings

import numpy as np
import pandas as pd
from arch import arch_model  # Import GARCH model from the 'arch' library

from Configs.config_schema import Config
from Modules.ModelModules.modules import (
    preprocess_data,
    scale_data,
    split_data
)
from Utils.io import load_data, save_forecasting_results

warnings.filterwarnings("ignore")  # Suppress warnings

# ============================
# Global GARCH Parameters
# ============================

GARCH_P = 1        # Order of the autoregressive term
GARCH_Q = 1        # Order of the moving average term
GARCH_VOL = 'Garch'  # Type of volatility process
GARCH_DIST = 'normal'  # Error distribution
GARCH_MEAN = 'Constant'  # Mean model

# ============================


def build_model(train_data, model_parameters):
    """Build the GARCH model using training data."""
    # Extract the target variable
    y_train = train_data[model_parameters.target_column]

    # Initialize variables
    exog_vars = []
    X_train = None
    use_exog = False

    # Check if target_column is in feature_columns before attempting to remove it
    if model_parameters.target_column in model_parameters.feature_columns:
        exog_vars = model_parameters.feature_columns.copy()
        exog_vars.remove(model_parameters.target_column)
    else:
        exog_vars = model_parameters.feature_columns.copy()

    # If there are exogenous variables after removal, set X_train
    if len(exog_vars) > 0:
        X_train = train_data[exog_vars]
        use_exog = True
        print(f"Exogenous variables detected: {exog_vars}")
    else:
        print("No exogenous variables detected.")

    # Initialize the GARCH model
    model = arch_model(
        y_train,
        x=X_train,
        mean=GARCH_MEAN,
        vol=GARCH_VOL,
        p=GARCH_P,
        q=GARCH_Q,
        dist=GARCH_DIST,
        rescale=False
    )
    return model, use_exog


def train_model(model):
    """Train the GARCH model."""
    model_results = model.fit(disp='off')
    return model_results


def forecast(model_results, steps, test_data, model_parameters, use_exog):
    """Make forecasts using the trained GARCH model."""
    if use_exog:
        # Ensure exogenous variables are present in test_data
        exog_vars = model_parameters.feature_columns.copy()
        if model_parameters.target_column in exog_vars:
            exog_vars.remove(model_parameters.target_column)
        X_test = test_data[exog_vars]
        print(f"Using exogenous variables for forecasting: {exog_vars}")
        # Forecast future values with exogenous variables
        forecasts = model_results.forecast(horizon=steps, x=X_test, reindex=False)
    else:
        print("Forecasting without exogenous variables.")
        # Forecast future values without exogenous variables
        forecasts = model_results.forecast(horizon=steps, reindex=False)
    # Extract the mean forecast
    # The forecasts object contains forecasts for each step; to align with test data, extract step-wise forecasts
    # Here, we assume multi-step forecasting matching the test set length
    y_pred = forecasts.mean.iloc[-1]
    # y_pred will be a Series; ensure it's aligned with test set
    y_pred = y_pred[:steps].values
    return y_pred


def run(config: Config):
    model_parameters = config.model_parameters

    # Load data
    print("Step 1: Loading the Data")
    data = load_data(config.data.in_path)

    # Preprocess data
    print("Step 2: Preprocessing the Data")
    data = preprocess_data(
        data,
        model_parameters.feature_columns,
        filter_holidays=config.preprocess_parameters
    )

    # Scale data
    print("Step 3: Scaling the Data")
    scaled_data, scalers = scale_data(
        data,
        model_parameters.feature_columns
    )

    # Split data
    print("Step 4: Splitting the Data")
    train, test = split_data(
        scaled_data,
        model_parameters.train_ratio
    )

    # Build model
    print("Step 5: Building the GARCH Model")
    model, use_exog = build_model(train, model_parameters)

    # Train model
    print("Step 6: Training the Model")
    model_results = train_model(model)

    # Forecast
    print("Step 7: Forecasting the Test Data")
    steps = len(test)
    y_pred_scaled = forecast(model_results, steps, test, model_parameters, use_exog)

    # Inverse transform the predictions and actual values
    target_scaler = scalers[model_parameters.target_column]
    y_pred_inv = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_inv = target_scaler.inverse_transform(test[model_parameters.target_column].values.reshape(-1, 1))

    # Save results
    train_size = int(model_parameters.train_ratio * len(scaled_data))
    save_forecasting_results(
        data,
        y_pred_inv.flatten(),
        train_size,
        seq_length=0,  # GARCH model does not use sequence length
        out_path=config.data.out_path
    )

    print("Forecasting completed and results saved.")


