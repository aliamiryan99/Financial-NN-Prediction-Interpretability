# Import Necessary Libraries
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import \
    RandomForestRegressor  # Import RandomForest regressor
from sklearn.preprocessing import MinMaxScaler

from Configs.ConfigSchema import Config
from Controllers.ModelModules.modules import (preprocess_data, scale_data,
                                              split_data)
from Utils.io import load_data, save_results

warnings.filterwarnings("ignore")  # Suppress warnings


def create_sequences(features, target, seq_length):
    """Prepare sequences for RandomForest."""
    print("Step 5: Preparing the Data for RandomForest")
    X = []
    y = []
    for i in range(len(features) - seq_length):
        # Flatten the sequence for RandomForest
        X.append(features[i:(i + seq_length)].flatten())
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)


def build_model():
    """Build the RandomForest model."""
    print("Step 6: Building the RandomForest Model")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )
    return model


def train_model(model, X_train, y_train):
    """Train the RandomForest model."""
    print("Step 7: Training the Model")
    model.fit(X_train, y_train)
    return model


def forecast(model, X_test):
    """Make predictions using the trained RandomForest model."""
    print("Step 8: Forecasting the Test Data")
    y_pred = model.predict(X_test)
    return y_pred


def run(config: Config):
    model_parameters = config.model_parameters

    # Load data
    data = load_data(config.data.in_path)

    # Preprocess data
    data = preprocess_data(data, model_parameters.feature_columns)

    # Scale data
    scaled_data, scalers = scale_data(data, model_parameters.feature_columns)

    # Split data
    train, test = split_data(scaled_data, model_parameters.train_ratio)

    # Prepare sequences
    X_train, y_train = create_sequences(train[model_parameters.feature_columns].values, train[model_parameters.target_column].values, model_parameters.seq_length)
    X_test, y_test = create_sequences(test[model_parameters.feature_columns].values, test[model_parameters.target_column].values, model_parameters.seq_length)

    # Build model
    model = build_model()

    # Train model
    model = train_model(model, X_train, y_train)

    # Forecast
    y_pred = forecast(model, X_test)

    # Inverse transform the predictions and actual values
    volume_scaler = scalers['Volume']
    y_pred_inv = volume_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = volume_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Save results
    train_size = int(model_parameters.train_ratio * len(scaled_data))
    save_results(data, y_pred_inv.flatten(), train_size, model_parameters.seq_length, config.data.out_path)
