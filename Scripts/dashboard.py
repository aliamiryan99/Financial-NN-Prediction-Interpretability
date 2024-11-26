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

    # Run dashboard
    from Views.dashboard import Dashboard
    dashboard = Dashboard(config)
    

main()
