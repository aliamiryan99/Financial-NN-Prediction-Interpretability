# Scripts/interpret_lstm_core.py

import importlib
import os
import sys
import warnings
import joblib
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")  # Suppress warnings

def main():
    # Get the script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Add project root to sys.path
    sys.path.insert(0, project_root)
    from Configs.config_schema import Config, load_config

    # Read config.yaml
    config: Config = load_config()

    # Extract model path
    model_path = config.model  # e.g., 'NeuralNetworks/LSTM'

    # Import the forecasting model class
    try:
        model_module = importlib.import_module('Models.' + model_path.replace('/', '.') + '.model')
        ForecastingModel = getattr(model_module, 'ForecastingModel')
    except (ImportError, AttributeError) as e:
        print(f"Error importing ForecastingModel: {e}")
        return

    # Define the experiments path
    experiments_path = config.data.exp_path
    if not os.path.exists(experiments_path):
        raise Exception(f"Experiment path '{experiments_path}' does not exist. Make sure to run the model training first.")

    # Load model, scalers, and config
    print("Loading the Model, Scalers, and Config File...")
    model_file_path = os.path.join(experiments_path, 'model.keras')
    scaler_file_path = os.path.join(experiments_path, 'scalers.pkl')
    config_file_path = os.path.join(experiments_path, 'config.pkl')

    model = load_model(model_file_path)
    with open(scaler_file_path, 'rb') as scaler_file:
        scalers = joblib.load(scaler_file)
    with open(config_file_path, 'rb') as config_file:
        experiment_config = pickle.load(config_file)

    # Instantiate the forecasting model
    forecasting_model = ForecastingModel(config=experiment_config)
    forecasting_model.model = model
    forecasting_model.scalers = scalers

    # Load test data
    print("Loading and preparing test data for interpretation...")
    from Modules.ModelModules.modules import preprocess_data, scale_data, split_data
    from Utils.io import load_data

    model_parameters = config.model_parameters

    # Load data
    data = load_data(config.data.in_path)

    # Preprocess data
    data = preprocess_data(
        data,
        model_parameters.feature_columns,
        filter_holidays=config.preprocess_parameters.filter_holidays
    )

    # Scale data
    scaled_data, _ = scale_data(data, model_parameters.feature_columns)

    # Split data
    train, test = split_data(scaled_data, model_parameters.train_ratio)

    # Prepare the data using the forecasting_model's logic
    X_train, y_train, X_test, y_test = forecasting_model.prepare_data(train, test)

    # Import the LSTMCoreInterpreter
    try:
        interpreter_module = importlib.import_module('Interpretations.Internal.lstm_core_interpretation')
        LSTMCoreInterpreter = getattr(interpreter_module, 'LSTMCoreInterpreter')
    except (ImportError, AttributeError) as e:
        print(f"Error importing LSTMCoreInterpreter: {e}")
        return

    # Instantiate the interpreter
    interpreter = LSTMCoreInterpreter(model=forecasting_model.model,
                                      X_test=X_test,
                                      config=config)

    # Run the interpretation
    interpreter.run()


if __name__ == "__main__":
    main()
