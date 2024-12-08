# Interpretation Script
import importlib
import os
import sys
import warnings
import joblib
import pickle
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")  # Suppress warnings

def main():
    # Get the script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Add project root to sys.path
    sys.path.insert(0, project_root)
    from Configs.config_schema import Config, load_config

    # Read config.json
    config: Config = load_config()
    model_path = config.model  # e.g., 'NeuralNetworks.LSTM'

    # Log the model path name
    print(f"#########  Interpreting {model_path} with {config.interpretability_class} ( Type : {config.interpretation_type}) #########")

    # Full model path
    full_model_path = 'Models.' + model_path + ".model"

    # Import the model class
    try:
        model_module = importlib.import_module(full_model_path)
        ForecastingModel = getattr(model_module, 'ForecastingModel')
    except (ImportError, AttributeError) as e:
        print(f"Error importing forecasting model class: {e}")
        return

    # Define the experiments path
    experiments_path = config.data.exp_path
    if not os.path.exists(experiments_path):
        raise Exception(f"Experiment path '{experiments_path}' does not exist. Make sure to run the model training first.")

    # Load model, scalers, and config
    print("Loading the Model, Scalers, and Config File")
    model_file_path = os.path.join(experiments_path, 'model.keras')
    scaler_file_path = os.path.join(experiments_path, 'scalers.pkl')
    config_file_path = os.path.join(experiments_path, 'config.pkl')

    model = load_model(model_file_path)
    with open(scaler_file_path, 'rb') as scaler_file:
        scalers = joblib.load(scaler_file)
    with open(config_file_path, 'rb') as config_file:
        experiment_config = pickle.load(config_file)

    # Instantiate the model
    forecasting_model = ForecastingModel(config=experiment_config)
    forecasting_model.model = model
    forecasting_model.scalers = scalers

    # Import the interpretation class
    interpretation_class, interpretation_type = config.interpretability_class, config.interpretation_type
    interpretation_class_path = 'Interpretations.' + interpretation_type + "." + interpretation_class + '.model'
    try:
        interpretation_module = importlib.import_module(interpretation_class_path)
        InterpretationModel = getattr(interpretation_module, 'InterpretationModel')
    except (ImportError, AttributeError) as e:
        print(f"Error importing interpretation class: {e}")
        return

    # Instantiate the interpretation class
    interpretation_model = InterpretationModel(config=config, forecasting_model=forecasting_model)

    # Run the interpretation
    interpretation_model.run()

    print("#########  Interpretation Process Completed Successfully  #########")

if __name__ == "__main__":
    main()