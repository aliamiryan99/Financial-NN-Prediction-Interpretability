# Import Necessary Libraries
import numpy as np
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.models import Sequential

from Configs.config_schema import Config
from Controllers.ModelModules.modules import (preprocess_data, scale_data,
                                              split_data, create_sequences)
from Utils.io import load_data, save_results


def build_model(seq_length, num_features):
    """Build the GRU model."""
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(seq_length, num_features)))
    model.add(GRU(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs, batch_size):
    """Train the GRU model."""
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def forecast(model, X_test):
    """Make predictions using the trained model."""
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
    print("Step 5: Preparing the Data for GRU")
    X_train, y_train = create_sequences(
        train[model_parameters.feature_columns].values, 
        train[model_parameters.target_column].values, 
        model_parameters.seq_length,
        reshape=True
    )
    X_test, y_test = create_sequences(
        test[model_parameters.feature_columns].values, 
        test[model_parameters.target_column].values, 
        model_parameters.seq_length,
        reshape=True
    )
    
    # Build model
    print("Step 6: Building the GRU Model")
    model = build_model(model_parameters.seq_length, len(model_parameters.feature_columns))
    
    # Train model
    print("Step 7: Training the Model")
    model = train_model(
        model, X_train, y_train, 
        model_parameters.epochs, model_parameters.batch_size
    )
    
    # Forecast
    print("Step 8: Forecasting the Test Data")
    y_pred = forecast(model, X_test)
    
    # Inverse transform the predictions and actual values
    volume_scaler = scalers[model_parameters.target_column]
    y_pred_inv = volume_scaler.inverse_transform(y_pred)
    y_test_inv = volume_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Save results
    train_size = int(model_parameters.train_ratio * len(scaled_data))
    save_results(
        data, y_pred_inv.flatten(), train_size, 
        model_parameters.seq_length, config.data.out_path
    )
