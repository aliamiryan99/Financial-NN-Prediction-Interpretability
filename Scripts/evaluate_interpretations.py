# Scripts/evaluate_interpretations.py

import importlib
import os
import sys
import warnings
import joblib
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")  # Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages

def main():
    # Get the script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Add project root to sys.path
    sys.path.insert(0, project_root)
    from Configs.config_schema import Config, load_config

    # Read config.yaml
    config: Config = load_config()
    model_path = config.model  # e.g., 'NeuralNetworks.LSTM'

    # Extract the interpretation class name and type
    time_interpretation_class = config.time_interpretability_class
    spectral_interpretation_class = config.spectral_interpretability_class
    interpretation_type = config.interpretability_type

    interpretation_class = time_interpretation_class if interpretation_type == "Time" else spectral_interpretation_class

    # Log the model path name
    print(f"#########  Evaluating Interpretability of {model_path} with {interpretation_class} ( Type : {interpretation_type}) #########")

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
    print("Step 1 : Loading the Model, Scalers, and Config File")
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

    # Import the interpretation class
    interpretation_class_path = 'Interpretations.' + interpretation_type + "." + interpretation_class + '.model'
    try:
        interpretation_module = importlib.import_module(interpretation_class_path)
        InterpretationModel = getattr(interpretation_module, 'InterpretationModel')
    except (ImportError, AttributeError) as e:
        print(f"Error importing interpretation class: {e}")
        return

    # Instantiate the interpretation class
    interpretation_model = InterpretationModel(config=config, forecasting_model=forecasting_model)

    # Load test data
    print("Step 2 : Loading and preparing test data for evaluation...")
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
    interpretation_model.X_train = X_train
    interpretation_model.y_train = y_train
    interpretation_model.X_test = X_test
    interpretation_model.y_test = y_test

    # Define reference inputs for iAUC
    # Here, define reference_input as the same shape as X_test, but masked (e.g., zeros)
    reference_input = np.zeros_like(X_test)

    # Define parameters for iAUC
    # No class_c needed for regression

    # Define parameters for Max-Sensitivity
    r = 0.01  # Perturbation range
    num_perturbations = 5  # Number of perturbations per input

    # Define perturb function
    def perturb_function(X: np.ndarray, r: float) -> np.ndarray:
        """
        Generates perturbed inputs by adding uniform noise within range r.

        Parameters:
        - X: np.ndarray of shape (N, T, F)
        - r: float, perturbation range

        Returns:
        - perturbed_X: np.ndarray of shape (N, T, F)
        """
        noise = np.random.uniform(-r, r, size=X.shape)
        perturbed_X = X + noise
        return perturbed_X

    # Import the evaluation functions
    from Modules.Evaluations.interpretability_evaluations import compute_iAUC, compute_max_sensitivity

    # Compute iAUC scores
    print("Step 3 : Computing iAUC scores...")
    iAUC_scores = compute_iAUC(
        model=model,
        interpretability_method=interpretation_model,
        X_test=X_test,
        reference_input=reference_input,
        num_features=len(model_parameters.feature_columns)  # Number of features F
    )

    # Compute Max-Sensitivity scores
    print("Step 4 : Computing Max-Sensitivity scores...")
    max_sensitivity_scores = compute_max_sensitivity(
        model=model,
        interpretability_method=interpretation_model,
        X_test=X_test,
        perturb_function=perturb_function,
        r=r,
        num_perturbations=num_perturbations
    )

    # Assemble scores into DataFrame
    scores_df = pd.DataFrame({
        'iAUC': iAUC_scores,
        'Max_Sensitivity': max_sensitivity_scores
    })

    # Save the scores
    interpret_results_path = config.data.interpret_evaluation_path  # e.g., 'Results/Interpretability/evaluation_scores.csv'
    # Ensure the directory exists
    os.makedirs(os.path.dirname(interpret_results_path), exist_ok=True)
    scores_df.to_csv(interpret_results_path, index=False)
    print(f"Step 5 : Interpretability evaluation scores saved at {interpret_results_path}")

    print("#########  Interpretability Evaluation Completed Successfully  #########")

if __name__ == "__main__":
    main()
