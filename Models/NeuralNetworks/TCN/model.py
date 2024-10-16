# Import Necessary Libraries
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tcn import TCN

from Configs.ConfigSchema import Config
from Controllers.ModelModules.modules import (preprocess_data, scale_data,
                                              split_data)
from Utils.io import load_data, save_results


def create_sequences(features, target, seq_length):
    """Prepare sequences for the TCN model."""
    X = []
    y = []
    for i in range(len(features) - seq_length):
        X.append(features[i:(i + seq_length)])
        y.append(target[i + seq_length])
    X = np.array(X)
    y = np.array(y)
    # Reshape input to be [samples, time steps, features]
    X = X.reshape((X.shape[0], seq_length, features.shape[1]))
    return X, y

def build_model(seq_length, num_features):
    """Build the TCN model."""
    model = Sequential()
    model.add(TCN(input_shape=(seq_length, num_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs, batch_size):
    """Train the TCN model."""
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
    data = preprocess_data(data, model_parameters.feature_columns, filter_holidays=True)

    # Scale data
    print("Step 3: Scaling the Data")
    scaled_data, scalers = scale_data(data, model_parameters.feature_columns)

    # Split data
    print("Step 4: Splitting the Data")
    train, test = split_data(scaled_data, model_parameters.train_ratio)

    # Prepare sequences
    print("Step 5: Preparing the Data for TCN")
    X_train, y_train = create_sequences(
        train[model_parameters.feature_columns].values,
        train[model_parameters.target_column].values,
        model_parameters.seq_length
    )
    X_test, y_test = create_sequences(
        test[model_parameters.feature_columns].values,
        test[model_parameters.target_column].values,
        model_parameters.seq_length
    )

    # Build model
    print("Step 6: Building the TCN Model")
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
