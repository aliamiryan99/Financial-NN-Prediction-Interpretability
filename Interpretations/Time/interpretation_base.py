import os
import numpy as np
from abc import ABC, abstractmethod

from Models.model_base import ModelBase
from Configs.config_schema import Config
from Modules.ModelModules.modules import preprocess_data, scale_data, split_data
from Utils.io import load_data, save_interpretability_results

class InterpretationBase(ABC):
    def __init__(self, config: Config, forecasting_model: ModelBase):
        self.config = config
        self.forecasting_model = forecasting_model
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    @abstractmethod
    def interpret(self, X_test: np.ndarray) -> np.ndarray:
        """
        Abstract method that should be implemented by derived classes 
        for computing an importance map shaped like X_test itself:
            (num_samples, seq_length, num_features).

        Returns:
            importance_map: np.ndarray of shape (N, T, F),
            where importance_map[i, t, f] is the importance 
            value for the i-th sample, at timestep t, feature f.
        """
        pass

    def run(self):
        """
        High-level steps:
        1. Load and preprocess data
        2. Prepare data with the forecasting model
        3. Call child's interpret() -> returns shape (N, T, F)
        4. Sum over time -> feature importances (N, F)
        5. Sum over features -> timestep importances (N, T)
        6. Normalize, concatenate (N, F + T), and save
        """
        print("Step 1: Loading and Preparing the Data and Forecasting Model")
        experiments_path = self.config.data.exp_path
        if not os.path.exists(experiments_path):
            raise Exception(
                f"Experiment path '{experiments_path}' does not exist. "
                f"Make sure to run the model first."
            )

        model_parameters = self.config.model_parameters

        # Step 2: Load data
        data = load_data(self.config.data.in_path)

        # Step 3: Preprocess data
        data = preprocess_data(
            data,
            model_parameters.feature_columns,
            filter_holidays=self.config.preprocess_parameters
        )

        # Step 4: Scale data
        scaled_data, _ = scale_data(data, model_parameters.feature_columns)

        # Step 5: Split data
        train, test = split_data(scaled_data, model_parameters.train_ratio)

        # Step 6: Prepare the data using the forecasting_model's logic
        self.X_train, self.y_train, self.X_test, self.y_test = self.forecasting_model.prepare_data(train, test)

        # Step 7: Call child interpretation => (N, T, F)
        print("Step 2: Interpreting the Model Predictions => 3D importance map")
        importance_map_3d = self.interpret(self.X_test)  # shape: (N, T, F)

        # --- Summation and Normalization ---
        # Step 8: Summation => feature-level and timestep-level
        # Feature importances: sum over timesteps => shape = (N, F)
        print("Step 3: Calculating feature/timestep importance from 3D map...")
        feature_importances = np.sum(np.abs(importance_map_3d), axis=1)  # sum over axis=1 => timesteps

        # Timestep importances: sum over features => shape = (N, T)
        timestep_importances = np.sum(np.abs(importance_map_3d), axis=2) # sum over axis=2 => features

        # Step 9: Normalize each
        normalized_features = self._normalize(feature_importances)
        normalized_timesteps = self._normalize(timestep_importances)

        # Step 10: Concatenate => shape = (N, F + T)
        final_importances = np.hstack((normalized_features, normalized_timesteps))

        # Step 11: Save interpretation results
        print("Step 4: Saving the Results...")
        interpret_results_path = self.config.data.interpret_path
        save_interpretability_results(
            data,
            final_importances,
            interpret_results_path,
            model_parameters.feature_columns
        )

    def _normalize(self, array: np.ndarray) -> np.ndarray:
        """
        Normalize a 2D ndarray row-wise.
        If row sum = 0, we replace it with a tiny constant to avoid div-by-zero.
        """
        row_sums = array.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-9
        return array / row_sums
