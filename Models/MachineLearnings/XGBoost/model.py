# Import Necessary Libraries
import warnings

import numpy as np
from xgboost import XGBRegressor  # Import XGBoost regressor

from Configs.config_schema import Config
from Controllers.ModelModules.modules import (preprocess_data, scale_data,
                                              split_data)
from Utils.io import load_data, save_results

warnings.filterwarnings("ignore")  # Suppress warnings


def create_sequences(features, target, seq_length):
    """Prepare sequences for XGBoost model."""
    X = []
    y = []
    for i in range(len(features) - seq_length):
        # Flatten the sequence for XGBoost
        X.append(features[i:(i + seq_length)].flatten())
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def build_model():
    """Build the XGBoost model."""
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    return model


def train_model(model, X_train, y_train):
    """Train the XGBoost model."""
    model.fit(X_train, y_train)
    return model


def forecast(model, X_test):
    """Make predictions using the trained XGBoost model."""
    y_pred = model.predict(X_test)
    return y_pred


def run(config: Config):
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

    # Split data
    print("Step 4: Splitting the Data")
    train, test = split_data(scaled_data, model_parameters.train_ratio)

    # Prepare sequences
    print("Step 5: Preparing the Data for XGBoost")
    X_train, y_train = create_sequences(train[model_parameters.feature_columns].values, train[model_parameters.target_column].values, model_parameters.seq_length)
    X_test, y_test = create_sequences(test[model_parameters.feature_columns].values, test[model_parameters.target_column].values, model_parameters.seq_length)

    # Build model
    print("Step 6: Building the XGBoost Model")
    model = build_model()

    # Train model
    print("Step 7: Training the Model")
    model = train_model(model, X_train, y_train)

    # Forecast
    print("Step 8: Forecasting the Test Data")
    y_pred = forecast(model, X_test)

    # Inverse transform the predictions and actual values
    volume_scaler = scalers['Volume']
    y_pred_inv = volume_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = volume_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Save results
    train_size = int(model_parameters.train_ratio * len(scaled_data))
    save_results(data, y_pred_inv.flatten(), train_size, model_parameters.seq_length, config.data.out_path)
