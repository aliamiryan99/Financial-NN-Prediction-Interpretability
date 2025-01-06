# Interpretations/InternalLSTM/lstm_core_interpretation.py

import numpy as np
import pandas as pd
import copy
from typing import List
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanSquaredError
import os

class LSTMCoreInterpreter:
    def __init__(self, model: Model, X_test: np.ndarray, config):
        """
        Initializes the LSTMCoreInterpreter.

        Parameters:
        - model: Trained Keras LSTM model.
        - X_test: Test input data, shape (N, T, F).
        - config: Configuration object containing interpret_internal_path.
        """
        self.model = model
        self.X_test = X_test
        self.config = config
        self.metric = MeanSquaredError()
        self.original_preds = None
        self.original_mse = None

    def run(self):
        """
        Executes the occlusion analysis to determine the importance of each LSTM unit.
        Saves the results to a CSV file specified in config.data.interpret_internal_path.
        """
        print("Starting LSTM Core Interpretation using Occlusion Method...")

        # Step 1: Compute original predictions
        print("Computing original predictions on test data...")
        self.original_preds = self.model.predict(self.X_test, verbose=0).flatten()

        # Step 2: Identify LSTM layers
        lstm_layers = self._identify_lstm_layers()
        if not lstm_layers:
            print("No LSTM layers found in the model.")
            return

        print(f"Found {len(lstm_layers)} LSTM layer(s) in the model.")

        # Step 3: Prepare to store importance scores
        importance_results = []

        # Step 4: Iterate over each LSTM layer
        for layer_num, (layer_idx, layer) in enumerate(lstm_layers, start=1):
            print(f"\nAnalyzing Layer {layer_num} ({layer.name})...")

            # LSTM layers have weights in the order: kernel, recurrent_kernel, bias
            # Retrieve the weights
            kernel_weights, recurrent_kernel_weights, bias_weights = self.model.layers[layer_idx].get_weights()

            input_dim, total_units = kernel_weights.shape  # For LSTM, kernel shape is (input_dim, 4 * units)
            units = total_units // 4

            print(f"Layer {layer_num} has {units} units.")

            # Iterate over each unit in the LSTM layer
            for unit in range(units):
                print(f"  Occluding Unit {unit + 1}/{units}...", end='')

                # Calculate the gate indices for this unit
                # Gates are in the order: input, forget, cell, output
                # Each gate occupies a quarter of the weights
                gate_size = units  # Number of units per gate
                gate_order = ['input', 'forget', 'cell', 'output']
                gates = {}

                for i, gate in enumerate(gate_order):
                    gate_start = unit + i * gate_size
                    gate_end = gate_start + 1  # Single unit

                    # Extract the column indices for the current gate and unit
                    # Since we're dealing with single units, it's just the specific column
                    gates[gate] = {
                        'kernel': gate_start,
                        'recurrent_kernel': gate_start,
                        'bias': gate_start
                    }

                # Create a deep copy of the model's weights to modify
                original_weights = copy.deepcopy(self.model.get_weights())
                modified_weights = copy.deepcopy(self.model.get_weights())

                # Indexing in get_weights():
                # For each LSTM layer, weights are ordered as kernel, recurrent_kernel, bias
                kernel_idx = layer_idx * 3
                recurrent_kernel_idx = kernel_idx + 1
                bias_idx = kernel_idx + 2

                # Occlude the unit's weights by setting them to zero
                for gate in gate_order:
                    # Set kernel weights to zero
                    modified_weights[kernel_idx][:, gates[gate]['kernel']] = 0
                    # Set recurrent kernel weights to zero
                    modified_weights[recurrent_kernel_idx][:, gates[gate]['recurrent_kernel']] = 0
                    # Set bias weights to zero
                    modified_weights[bias_idx][gates[gate]['bias']] = 0

                # Apply the modified weights to the model
                self.model.set_weights(modified_weights)

                # Compute predictions with occluded weights
                occluded_preds = self.model.predict(self.X_test, verbose=0).flatten()

                # Compute MSE difference between occluded and original predictions
                mse_diff = np.mean((occluded_preds - self.original_preds) ** 2)

                # Restore the original weights to the model
                self.model.set_weights(copy.deepcopy(original_weights))  # Revert to original

                # Store the result
                importance_results.append({
                    'Layer': layer_num,
                    'Unit': unit + 1,
                    'MSE_Difference': mse_diff
                })

                print(f" MSE Difference: {mse_diff:.6f}")

        # Convert results to DataFrame
        results_df = pd.DataFrame(importance_results)

        # Sort the results by MSE_Difference in descending order
        results_df.sort_values(by='MSE_Difference', ascending=False, inplace=True)

        # Save the results to CSV
        output_path = self.config.data.interpret_internal_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nInterpretation results saved to {output_path}")

        print("LSTM Core Interpretation Completed Successfully.")

    def _identify_lstm_layers(self):
        """
        Identifies all LSTM layers within the model.

        Returns:
        - List of tuples containing (layer_index, layer_object).
        """
        lstm_layers = []
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, (self.model.layers[0].__class__, )) and 'lstm' in layer.name.lower():
                lstm_layers.append((idx, layer))
        return lstm_layers
