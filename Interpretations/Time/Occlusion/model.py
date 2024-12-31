# Interpretations/Time/Occlusion/model.py

import numpy as np
from Interpretations.Time.interpretation_base import InterpretationBase

class InterpretationModel(InterpretationBase):
    def interpret(self, X_test):
        """
        Occlusion-based importance:
        Return a 3D array (N, T, F), where each entry is how important
        that (timestep t, feature f) is for the i-th sample's prediction.
        """
        print("Predicting original outputs...")
        y_pred_original = self.forecasting_model.model.predict(X_test)

        num_samples, seq_length, num_features = X_test.shape

        # Prepare a 3D map for storing occlusion importances
        occlusion_map = np.zeros((num_samples, seq_length, num_features))

        print("Calculating occlusion importances...")
        # For each (t, f), we zero out that slice and measure the difference
        for t in range(seq_length):
            for f in range(num_features):
                X_perturbed = X_test.copy()
                X_perturbed[:, t, f] = 0.0

                y_pred_perturbed = self.forecasting_model.model.predict(X_perturbed)

                # The difference for each sample
                differences = np.abs(y_pred_original - y_pred_perturbed).flatten()

                # Assign to that position in the occlusion_map
                occlusion_map[:, t, f] = differences

        # Return shape (N, T, F)
        return occlusion_map
