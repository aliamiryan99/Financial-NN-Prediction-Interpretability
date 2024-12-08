import numpy as np
import pandas as pd
from Interpretations.Spectral.interpretation_base import SpectralInterpretationBase

class InterpretationModel(SpectralInterpretationBase):
    def interpret(self, fft_data):
        """
        Calculates the importance of each frequency by perturbing its magnitude and measuring prediction changes.
        
        :param fft_data: FFT-transformed input data, shape (num_samples, freq_length, num_features)
        :return: A pandas DataFrame where each row corresponds to a single prediction and each column represents the importance of a frequency.
        """
        print("Predicting original outputs with original FFT data...")
        # Reconstruct original inputs from FFT data
        X_test_original = self.inverse_fft(fft_data)
        y_pred_original = self.forecasting_model.model.predict(X_test_original)
        num_samples, freq_length, num_features = fft_data.shape

        # Initialize array to store frequency importances
        frequency_importances = np.zeros((num_samples, freq_length))

        # Iterate over each frequency to perturb
        print("Calculating frequency importances by perturbing each frequency...")
        for f in range(freq_length):
            print(f"Perturbing frequency {f+1}/{freq_length}")
            fft_perturbed = fft_data.copy()
            # Perturb frequency f by setting its magnitude to zero across all samples and features
            fft_perturbed[:, f, :] = 0
            # Reconstruct perturbed inputs
            X_test_perturbed = self.inverse_fft(fft_perturbed)
            # Predict with perturbed inputs
            y_pred_perturbed = self.forecasting_model.model.predict(X_test_perturbed)
            # Calculate the absolute difference
            differences = np.abs(y_pred_original - y_pred_perturbed).flatten()
            # Store the differences as importance scores for frequency f
            frequency_importances[:, f] = differences

        # Create a DataFrame with frequency importances
        freq_columns = [f'freq_{f:.3f}' for f in self.frequencies]
        df_importances = pd.DataFrame(frequency_importances, columns=freq_columns)
        return df_importances

    def inverse_fft(self, fft_data):
        """
        Reconstruct the time-domain signal from perturbed FFT data.
        
        :param fft_data: FFT-transformed data with perturbations, shape (num_samples, freq_length, num_features)
        :return: Reconstructed time-domain data, shape (num_samples, seq_length, num_features)
        """
        num_samples, freq_length, num_features = fft_data.shape
        seq_lenght = self.config.model_parameters.seq_length
        # Reconstruct the full FFT by mirroring the positive frequencies
        full_fft = np.zeros((num_samples, seq_lenght, num_features), dtype=np.complex_)
        full_fft[:, :freq_length, :] = fft_data
        # Mirror the FFT data for negative frequencies
        full_fft[:, freq_length:, :] = np.conj(fft_data[:, 1:seq_lenght - freq_length + 1, :][:, ::-1, :])
        # Perform inverse FFT
        X_reconstructed = np.fft.ifft(full_fft, axis=1).real
        return X_reconstructed
