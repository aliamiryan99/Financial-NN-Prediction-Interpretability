import importlib
import os
import sys
import warnings

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

    # Full model path
    full_model_path = 'Models.' + model_path + ".model"

    # Import the model
    try:
        model = importlib.import_module(full_model_path)
    except ImportError as e:
        print(f"Error importing model: {e}")

    # Run the run function
    if hasattr(model, 'run'):
        model.run(config)
    else:
        print(f"No run function found in model {full_model_path}")

if __name__ == "__main__":
    main()
