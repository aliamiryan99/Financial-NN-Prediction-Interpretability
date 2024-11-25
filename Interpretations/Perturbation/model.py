# Perturbation.py

import numpy as np

from Interpretations.interpretation_base import InterpretationBase

class InterpretationModel(InterpretationBase):

    def interpret(self, X_test):
        """
        Calculates the importance of features and timesteps separately using perturbation.
        :param X_test: Input data to be interpreted, shape (num_samples, seq_length, num_features)
        :return: An ndarray where each row corresponds to a single prediction and each column represents the importance of a feature or timestep.
        """
        # Get original predictions
        print("Predicting original outputs...")
        y_pred_original = self.forecasting_model.model.predict(X_test)
        num_samples, seq_length, num_features = X_test.shape
        
        # Initialize arrays to store importances
        feature_importances = np.zeros((num_samples, num_features))
        timestep_importances = np.zeros((num_samples, seq_length))
        
        # For feature importance
        print("Calculating feature importances...")
        for f in range(num_features):
            X_test_perturbed = X_test.copy()
            # Perturb feature f by setting it to zero across all samples and timesteps
            X_test_perturbed[:, :, f] = 0
            y_pred_perturbed = self.forecasting_model.model.predict(X_test_perturbed)
            # Calculate the difference
            differences = np.abs(y_pred_original - y_pred_perturbed).flatten()
            # Store differences for this feature
            feature_importances[:, f] = differences
        
        # For timestep importance
        print("Calculating timestep importances...")
        for t in range(seq_length):
            X_test_perturbed = X_test.copy()
            # Perturb timestep t by setting all features at that timestep to zero across all samples
            X_test_perturbed[:, t, :] = 0
            y_pred_perturbed = self.forecasting_model.model.predict(X_test_perturbed)
            # Calculate the difference
            differences = np.abs(y_pred_original - y_pred_perturbed).flatten()
            # Store differences for this timestep
            timestep_importances[:, t] = differences
        
        # Concatenate feature importances and timestep importances
        importance_results = np.concatenate([feature_importances, timestep_importances], axis=1)
        return importance_results