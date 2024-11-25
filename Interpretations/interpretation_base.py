# Import Necessary Libraries
import os
import numpy as np
from abc import ABC, abstractmethod

from Models.model_base import ModelBase
from Configs.config_schema import Config
from Controllers.ModelModules.modules import preprocess_data, scale_data, split_data
from Utils.io import load_data, save_forecasting_results, save_interpretability_results

class InterpretationBase(ABC):
    def __init__(self, config: Config, forecasting_model: ModelBase):
        self.config = config
        self.forecasting_model = forecasting_model

    @abstractmethod
    def interpret(self, X_test):
        """
        Abstract method that should be implemented by derived classes for interpreting model predictions.
        :param X_test: Input data to be predicted and interpret it's result
        :return: A pandas dataframe that each row belong to a single prediction and each column represent the importance of a feature or a timestep.
        """
        pass

    def run(self):
        print("Step 1: Loading and Preparing the Data and Forecasting Model")
        # Define the experiments path
        experiments_path = self.config.data.exp_path
        if not os.path.exists(experiments_path):
            raise Exception(f"Experiment path '{experiments_path}' does not exist. Make sure to run the model first.")

        model_parameters = self.config.model_parameters
        # Step 2: Load data
        data = load_data(self.config.data.in_path)

        # Step 3: Preprocess data
        data = preprocess_data(data, model_parameters.feature_columns, filter_holidays=self.config.preprocess_parameters)

        # Step 4: Scale data
        scaled_data, _ = scale_data(data, model_parameters.feature_columns)

        # Step 5: Split data
        train, test = split_data(scaled_data, model_parameters.train_ratio)
        X_train, y_train, X_test, y_test = self.forecasting_model.prepare_data(train, test)

        # Step 6: Interpret model predictions
        print("Step 2: Interpreting the Model Predictions")
        interpretation_results = self.interpret(X_test)

        # Step 7: Normalize the interpretation results
        print("Step 3: Normalizing the Interpretation Results")
        num_features = len(model_parameters.feature_columns)
        normalized_results = self.normalize_interpretations(interpretation_results, num_features)

        # Step 7: Save interpretation results
        print("Step 4: Saving The Results")
        interpret_results_path = self.config.data.interpret_path
        # Use the save_interpretability_results function
        save_interpretability_results(data, normalized_results , interpret_results_path, model_parameters.feature_columns)

    def normalize(self, array):
        """
        Normalize a 2D ndarray along each row.
        Rows with a sum of zero will remain unchanged.
        
        :param array: 2D ndarray to be normalized
        :return: Normalized 2D ndarray
        """
        row_sums = array.sum(axis=1, keepdims=True)
        # Avoid division by zero by setting zero sums to 1 (no change to array values in such rows)
        row_sums[row_sums == 0] = 1
        return array / row_sums

    def normalize_interpretations(self, interpretation_results, num_features):
        """
        Normalize feature importances and timestep importances separately in a 2D ndarray.
        
        :param interpretation_results: A 2D ndarray where each row corresponds to a single prediction.
                                    The first `num_features` columns are feature importances,
                                    and the remaining columns are timestep importances.
        :param num_features: Number of feature importance columns at the start of the array.
        :return: A 2D ndarray with normalized feature and timestep importances.
        """
        # Split into feature and timestep importances
        feature_importances = interpretation_results[:, :num_features]
        timestep_importances = interpretation_results[:, num_features:]

        # Normalize feature importances using the reusable normalize function
        normalized_features = self.normalize(feature_importances)

        # Normalize timestep importances using the reusable normalize function
        normalized_timesteps = self.normalize(timestep_importances)

        # Concatenate normalized features and timesteps back together
        return np.hstack((normalized_features, normalized_timesteps))