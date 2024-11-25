import importlib
import os
import sys
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings


def main():
    # Get the script directory and project root
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Add project root to sys.path
    sys.path.insert(0, project_root)
    from Configs.config_schema import Config, load_config

    # Read config.json
    config: Config = load_config()
    model_path = config.model  # e.g., 'NeuralNetworks.LSTM'

    # Log the model path name
    print(f"#########  {model_path}  #########")

    # Full model path
    full_model_path = 'Models.' + model_path + ".model"

    # Import the model
    try:
        model_module = importlib.import_module(full_model_path)
    except ImportError as e:
        print(f"Error importing model: {e}")
        return

    # Find the ForecastingModel class and call its run method
    if hasattr(model_module, 'ForecastingModel'):
        ForecastingModel = getattr(model_module, 'ForecastingModel')
        model_instance = ForecastingModel(config)
        if hasattr(model_instance, 'run'):
            model_instance.run()
        else:
            print(f"No run method found in ForecastingModel in model {full_model_path}")
    else:
        print(f"No ForecastingModel class found in model {full_model_path}")

    print("#########  Process Completed Successfully  #########")
    
if __name__ == "__main__":
    main()
