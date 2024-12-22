# RIME Interpretation Model

import numpy as np
from Interpretations.Time.interpretation_base import InterpretationBase

class InterpretationModel(InterpretationBase):
    def interpret(self, X_test):
        """
        Implements the RIME (Random Input Sampling) method to calculate feature and timestep importance.
        :param X_test: Input data to be interpreted, shape (num_samples, seq_length, num_features)
        :return: An ndarray where each row corresponds to a single prediction and each column represents the importance of a feature or timestep.
        """
        print("Running RIME interpretation...")

        # Get original predictions
        y_pred_original = self.forecasting_model.model.predict(X_test)
        num_samples, seq_length, num_features = X_test.shape

        # Number of random masks to generate
        num_masks = 100

        # Initialize an array to store importance weights
        importance_weights = np.zeros_like(X_test, dtype=np.float32)

        # Probability of occlusion (zeros in the mask)
        prob_zero = 0.03  # Adjust this value to control the probability of zeros

        # Store unique masks to ensure they are distinct
        unique_masks = set()

        while len(unique_masks) < num_masks:
            # Generate random mask with a lower probability of zeros
            mask = np.random.choice([0, 1], size=X_test.shape, p=[prob_zero, 1 - prob_zero])

            # Convert mask to a hashable tuple for uniqueness check
            mask_tuple = tuple(mask.flatten())
            if mask_tuple not in unique_masks:
                unique_masks.add(mask_tuple)

                print(f"Processing mask {len(unique_masks)} of {num_masks}...")

                # Apply the mask to the input (occlusion)
                X_test_masked = X_test * mask

                # Get predictions for masked input
                y_pred_masked = self.forecasting_model.model.predict(X_test_masked)

                # Calculate prediction difference
                differences = np.abs(y_pred_original - y_pred_masked).reshape(num_samples, 1, 1)

                # Accumulate weighted differences
                importance_weights += differences * (-(mask-1))

        # Normalize importance weights by the number of masks
        importance_weights /= num_masks

        # Aggregate importance weights to get feature and timestep importance
        print("Aggregating feature and timestep importance...")
        feature_importances = importance_weights.mean(axis=1)  # Average over timesteps
        timestep_importances = importance_weights.mean(axis=2)  # Average over features

        # Concatenate feature and timestep importances
        importance_results = np.concatenate([feature_importances, timestep_importances], axis=1)

        return importance_results
