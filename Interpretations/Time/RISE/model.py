import numpy as np
from Interpretations.Time.interpretation_base import InterpretationBase

class InterpretationModel(InterpretationBase):
    def interpret(self, X_test):
        """
        Return shape (N, T, F) with random sampling importance.
        """
        print("Running RISE interpretation...")

        # Original predictions
        y_pred_original = self.forecasting_model.model.predict(X_test)

        # We'll store the importance map directly in shape => (N, T, F)
        importance_map = np.zeros_like(X_test, dtype=np.float32)  # (N, T, F)

        # Number of random masks
        num_masks = 100
        prob_zero = 0.03  # Probability of occlusion

        # Keep track of unique masks
        unique_masks = set()

        while len(unique_masks) < num_masks:
            # Generate random mask
            mask = np.random.choice(
                [0, 1], 
                size=X_test.shape, 
                p=[prob_zero, 1 - prob_zero]
            )

            mask_tuple = tuple(mask.flatten())
            if mask_tuple not in unique_masks:
                unique_masks.add(mask_tuple)
                print(f"Processing mask {len(unique_masks)} of {num_masks}...")

                # Apply mask
                X_test_masked = X_test * mask

                # Predict
                y_pred_masked = self.forecasting_model.model.predict(X_test_masked)

                # Differences => shape (N,)
                differences = np.abs(y_pred_original - y_pred_masked).flatten()

                # Expand differences to shape (N, 1, 1) so we can broadcast
                differences_expanded = differences[:, None, None]

                # Accumulate
                # (mask == 0) indicates occluded dimension => 
                # we typically measure how dropping that dimension changed the output
                # You can decide how to accumulate. For example:
                importance_map += differences_expanded * (1 - mask)

        # Average across all masks
        importance_map /= num_masks

        return importance_map
