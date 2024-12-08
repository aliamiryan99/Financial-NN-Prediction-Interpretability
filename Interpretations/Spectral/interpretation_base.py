# SpectralInterpretationBase.py

import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from Models.model_base import ModelBase
from Configs.config_schema import Config
from Controllers.ModelModules.modules import preprocess_data, scale_data, split_data
from Utils.io import load_data, save_interpretability_results

class SpectralInterpretationBase(ABC):
    def __init__(self, config: Config, forecasting_model: ModelBase):
        self.config = config
        self.forecasting_model = forecasting_model
        self.frequencies = None

    @abstractmethod
    def interpret(self, fft_data):
        """
        Abstract method that should be implemented by derived classes for interpreting FFT data.
        :param fft_data: FFT-transformed input data to be interpreted.
        :return: A pandas DataFrame where each row corresponds to a single prediction and each column represents the importance of a frequency.
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
        scaled_data, scaler = scale_data(data, model_parameters.feature_columns)

        # Step 5: Split data
        train, test = split_data(scaled_data, model_parameters.train_ratio)
        self.X_train, self.y_train, self.X_test, self.y_test = self.forecasting_model.prepare_data(train, test)

        # Step 6: Compute FFT of the test data
        print("Step 2: Computing FFT of the Test Data")
        fft_data, frequencies = self.compute_fft(self.X_test)
        self.frequencies = frequencies

        # Save the original FFT data
        print("Step 3: Saving Original FFT Data")
        spec_path = self.config.data.spec_interpret_path
        spec_path_dir = os.path.dirname(spec_path)
        os.makedirs(spec_path_dir, exist_ok=True)
        self.save_fft_data(np.abs(fft_data), frequencies, spec_path_dir + "/Frequencies.csv")

        # Step 7: Interpret FFT data
        print("Step 4: Interpreting the FFT Data")
        interpretation_results = self.interpret(fft_data)

        # Step 8: Save interpretation results
        print("Step 5: Saving Interpretation Results")
        # Save to CSV
        interpretation_results.to_csv(spec_path, index=False)
        print(f"FFT data saved to {spec_path}")

    def compute_fft(self, X):
        """
        Compute the Fast Fourier Transform (FFT) of the input data.
        
        :param X: Input data, shape (num_samples, seq_length, num_features)
        :return: Tuple of (fft_data, frequencies)
            - fft_data: FFT-transformed data, shape (num_samples, freq_length, num_features)
            - frequencies: Array of frequency components
        """
        # Compute FFT along the sequence length axis
        fft_data = np.fft.fft(X, axis=1)
        # Only keep the positive frequencies
        fft_data = fft_data[:, :fft_data.shape[1] // 2 + 1, :]
        # Get the corresponding frequency components
        freq_length = fft_data.shape[1]
        sampling_rate = 3600  # Assuming H1 time frame;
        frequencies = np.abs(np.fft.fftfreq(X.shape[1], d=1/sampling_rate)[:freq_length])
        return fft_data, frequencies

    def save_fft_data(self, fft_data, frequencies, path):
        """
        Save the FFT data to a CSV file with reordered and renamed columns.
        
        :param fft_data: FFT-transformed data, shape (num_samples, freq_length, num_features)
        :param frequencies: Array of frequency components
        :param path: Path to save the CSV file
        """
        num_samples, freq_length, num_features = fft_data.shape
        
        # Create column names and reorder them
        feature_columns = self.config.model_parameters.feature_columns
        assert len(feature_columns) == num_features, "Mismatch between feature columns and num_features"
        
        # Reorganize columns so each feature's frequencies are grouped together
        columns = []
        ordered_data = []
        for feat_idx, feat_name in enumerate(feature_columns):
            for freq_idx, freq in enumerate(frequencies):
                columns.append(f"{feat_name}_freq_{abs(freq):.3f}")
                # Extract corresponding data for the feature-frequency pair
                ordered_data.append(fft_data[:, freq_idx, feat_idx])
        
        # Stack ordered data and transpose for DataFrame creation
        ordered_data = np.stack(ordered_data, axis=-1)
        ordered_data = ordered_data.reshape(num_samples, -1)
        
        # Create DataFrame
        df_fft = pd.DataFrame(ordered_data, columns=columns)
        
        # Save to CSV
        df_fft.to_csv(path, index=False)
        print(f"FFT data saved to {path}")
