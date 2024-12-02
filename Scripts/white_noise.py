import importlib
import os
import sys
import warnings
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Get the script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add project root to sys.path
sys.path.insert(0, project_root)

# Import necessary modules for data preparation
from Controllers.ModelModules.modules import preprocess_data, scale_data, split_data
from Utils.io import load_data
from Configs.config_schema import Config, load_config

warnings.filterwarnings("ignore")  # Suppress warnings

def load_trained_model(config: Config):
    """Load the trained forecasting model and scalers."""
    model_path = config.model  # e.g., 'NeuralNetworks.LSTM'
    full_model_path = 'Models.' + model_path + ".model"
    
    # Import the model
    try:
        model_module = importlib.import_module(full_model_path)
        ForecastingModel = getattr(model_module, 'ForecastingModel')
    except (ImportError, AttributeError) as e:
        print(f"Error importing forecasting model class: {e}")
        return None

    # Define the experiment path
    experiments_path = config.data.exp_path
    if not os.path.exists(experiments_path):
        raise Exception(f"Experiment path '{experiments_path}' does not exist. Make sure to run the model training first.")

    # Load the model, scalers, and config files
    model_file_path = os.path.join(experiments_path, 'model.keras')
    scaler_file_path = os.path.join(experiments_path, 'scalers.pkl')
    config_file_path = os.path.join(experiments_path, 'config.pkl')

    # Load the trained model and scalers
    model = load_model(model_file_path)
    with open(scaler_file_path, 'rb') as scaler_file:
        scalers = joblib.load(scaler_file)
    with open(config_file_path, 'rb') as config_file:
        experiment_config = pickle.load(config_file)
    
    # Instantiate the forecasting model
    forecasting_model = ForecastingModel(config=experiment_config)
    forecasting_model.model = model
    forecasting_model.scalers = scalers

    return forecasting_model

def generate_white_noise_input(data: pd.DataFrame):
    """Generate white noise signals for each input feature as a DataFrame."""
    # White noise is generated with the same shape as the input features
    white_noise = np.random.normal(0, 1, size=data.shape)
    white_noise_df = pd.DataFrame(white_noise, columns=data.columns)  # Use data's columns
    return white_noise_df

def plot_spectra(data, predictions, white_noise, white_noise_predictions):
    """Plot the spectral analysis of actual data, predictions, and white noise signals."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    # Actual Data Spectra
    for i in range(data.shape[1]):
        axes[0].psd(data.iloc[:, i], NFFT=1024, Fs=1, label=f"Actual {data.columns[i]}")
    axes[0].set_title('Spectra of Actual Data Channels')
    axes[0].legend()  # Add legend to the actual data plot
    
    # Actual Predictions Spectra
    for i in range(predictions.shape[1]):
        axes[1].psd(predictions[:, i], NFFT=1024, Fs=1, label=f"Prediction")
    axes[1].set_title('Spectra of Actual Predictions')
    axes[1].legend()  # Add legend to the predictions plot
    
    # White Noise Channels Spectra
    for i in range(white_noise.shape[1]):
        axes[2].psd(white_noise.iloc[:, i], NFFT=1024, Fs=1, label=f"White Noise {white_noise.columns[i]}")
    axes[2].set_title('Spectra of White Noise Channels')
    axes[2].legend()  # Add legend to the white noise plot
    
    # White Noise Predictions Spectra
    for i in range(white_noise_predictions.shape[1]):
        axes[3].psd(white_noise_predictions[:, i], NFFT=1024, Fs=1, label=f"White Noise Prediction")
    axes[3].set_title('Spectra of White Noise Predictions')
    axes[3].legend()  # Add legend to the white noise predictions plot

    # Adjust vertical margin and spacing between subplots
    plt.subplots_adjust(hspace=0.4)  # hspace controls the vertical spacing
    
    plt.tight_layout()
    plt.show()

def plot_difference_and_division_spectra(actual_predictions, white_noise_predictions):
    """Plot the difference and division of the spectra between actual predictions and white noise predictions."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Compute the difference (Actual Predictions - White Noise Predictions)
    difference = actual_predictions - white_noise_predictions
    for i in range(difference.shape[1]):
        axes[0].psd(difference[:, i], NFFT=1024, Fs=1, label=f"Difference")
    axes[0].set_title('Spectra of Difference between Actual and White Noise Predictions')
    axes[0].legend()  # Add legend to the difference plot
    
    # Compute the division (Actual Predictions / White Noise Predictions)
    # Avoid division by zero errors by adding a small epsilon value
    epsilon = 1e-6
    division = actual_predictions / (white_noise_predictions + epsilon)
    for i in range(division.shape[1]):
        axes[1].psd(division[:, i], NFFT=1024, Fs=1, label=f"Division")
    axes[1].set_title('Spectra of Division (Actual / White Noise Predictions)')
    axes[1].legend()  # Add legend to the division plot

    # Adjust vertical margin and spacing between subplots
    plt.subplots_adjust(hspace=0.4)  # hspace controls the vertical spacing
    
    # Show the plots
    plt.tight_layout()
    plt.show()

def main():
    # Load the configuration from the config file
    config: Config = load_config()
    model_parameters = config.model_parameters
    feature_columns = model_parameters.feature_columns
    num_features = len(feature_columns)

    # Load the actual data using the load_data function from the Utils module
    data = load_data(config.data.in_path)

    # Preprocess the data using the preprocess_data function from the Controller
    data = preprocess_data(data, feature_columns, config.preprocess_parameters.filter_holidays)

    # Scale the data using the scale_data function from the Controller
    scaled_data, scalers = scale_data(data, feature_columns)

    # Split the data into training and testing sets using the split_data function from the Controller
    train_data, test_data = split_data(scaled_data, model_parameters.train_ratio)
    actual_data = test_data

    # Load the trained forecasting model
    forecasting_model = load_trained_model(config)
    if forecasting_model is None:
        return
    
    # Prepare data according to the forecasting model
    X_train, y_train, X_test, y_test = forecasting_model.prepare_data(train_data, test_data)

    # Forecast using the actual data
    predictions = forecasting_model.model.predict(X_test)

    # Generate white noise for all input channels
    white_noise_input = generate_white_noise_input(actual_data)

    # Prepare data according to the forecasting model for white noise
    _, _, X_white_noise, y_white_noise = forecasting_model.prepare_data(train_data, white_noise_input)

    # Forecast using the white noise inputs
    white_noise_predictions = forecasting_model.model.predict(X_white_noise)

    # Plot the spectral analysis of the actual data, white noise, and predictions
    plot_spectra(actual_data, predictions, white_noise_input, white_noise_predictions)

    # Plot the difference and division spectra
    plot_difference_and_division_spectra(predictions, white_noise_predictions)

    # Save the predictions
    result_path = os.path.join(os.path.dirname(config.data.out_path), "white_noise", os.path.basename(config.data.out_path))
    np.savetxt(result_path, predictions, delimiter=",")
    print(f"Predictions saved at {result_path}")

if __name__ == "__main__":
    main()
