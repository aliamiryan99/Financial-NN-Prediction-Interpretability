import numpy as np
import shap

from Interpretations.interpretation_base import InterpretationBase

class InterpretationModel(InterpretationBase):

    def interpret(self, X_test):
        """
        Calculates the importance of features and timesteps separately using SHAP.
        :param X_test: Input data to be interpreted, shape (num_samples, seq_length, num_features)
        :return: An ndarray where each row corresponds to a single prediction and each column represents the importance of a feature or timestep.
        """
        # Ensure that self.X_train is available
        if not hasattr(self, 'X_train'):
            raise Exception("X_train is not available. Make sure to set self.X_train in the run method.")

        # Get background data
        X_train = self.X_train

        # For background data, we can use a subset of X_train
        background = X_train[:500]  # Adjust the number as needed

        # Create the SHAP GradientExplainer
        print("Creating SHAP GradientExplainer...")
        explainer = shap.GradientExplainer(self.forecasting_model.model, background)

        # Compute SHAP values for X_test
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(X_test)

        # SHAP values have shape [num_samples, seq_length, num_features, 1]
        # Convert to [num_samples, seq_length, num_features] by removing the last dimension
        shap_values = np.squeeze(shap_values)  # Shape: [num_samples, seq_length, num_features]

        # Calculate feature importances by summing over timesteps
        # Shape after sum: [num_samples, num_features]
        feature_importances = np.sum(np.abs(shap_values), axis=1)

        # Calculate timestep importances by summing over features
        # Shape after sum: [num_samples, seq_length]
        timestep_importances = np.sum(np.abs(shap_values), axis=2)

        # Concatenate feature importances and timestep importances
        importance_results = np.concatenate([feature_importances, timestep_importances], axis=1)

        return importance_results