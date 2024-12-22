import numpy as np
import pandas as pd
from Interpretations.Spectral.interpretation_base import SpectralInterpretationBase

class InterpretationModel(SpectralInterpretationBase):
    def interpret(self, fft_data):
        """
        Implements the RIME (Random Input Sampling) method for frequency importance.

        :param fft_data: FFT-transformed input data, shape (num_samples, freq_length, num_features)
        :return: A pandas DataFrame where each row corresponds to a single prediction and each column represents the importance of a frequency.
        """
        print("Predicting original outputs with original FFT data...")
        # Reconstruct original inputs from FFT data
        X_test_original = self.inverse_fft(fft_data)
        y_pred_original = self.forecasting_model.model.predict(X_test_original)
        num_samples, freq_length, num_features = fft_data.shape

        # Initialize array to store frequency importances
        frequency_importances = np.zeros((num_samples, freq_length, num_features))

        # Number of random masks
        num_masks = 100

        # Probability of occlusion (zeros in the mask)
        prob_zero = 0.05  # Adjust this value to control the probability of zeros

        # Store unique masks to ensure they are distinct
        unique_masks = set()

        print("Generating random masks and calculating importance scores...")
        while len(unique_masks) < num_masks:
            
            # Generate random mask (binary 0 or 1 for each frequency and feature)
            random_mask = np.random.choice([0, 1], size=(num_samples, freq_length, num_features), p=[prob_zero, 1 - prob_zero])

            # Convert mask to a hashable tuple for uniqueness check
            mask_tuple = tuple(random_mask.flatten())
            if mask_tuple not in unique_masks:
                unique_masks.add(mask_tuple)

                print(f"Processing mask {len(unique_masks)} of {num_masks}...")

                # Apply mask to FFT data
                fft_masked = fft_data * random_mask

                # Reconstruct inputs from masked FFT data
                X_test_masked = self.inverse_fft(fft_masked)

                # Predict with masked inputs
                y_pred_masked = self.forecasting_model.model.predict(X_test_masked)

                # Calculate the differences between original and masked predictions
                differences = np.abs(y_pred_original - y_pred_masked).reshape(num_samples, 1, 1)

                # Aggregate importance scores weighted by the mask values
                frequency_importances += differences * (-(random_mask-1))

        # Normalize by the number of masks to get average importance scores
        frequency_importances = np.mean(frequency_importances, axis=2) / num_masks

        # Create a DataFrame with frequency importances
        freq_columns = [f'freq_{f:.5f}' for f in self.frequencies]
        df_importances = pd.DataFrame(frequency_importances, columns=freq_columns)
        return df_importances
