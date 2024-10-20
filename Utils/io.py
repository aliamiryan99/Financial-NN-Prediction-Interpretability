import os

import pandas as pd


def load_data(input_data_path):
    """Load the data from the CSV file."""
   
    data = pd.read_csv(input_data_path, parse_dates=['Time'], index_col='Time')
    return data


def save_results(data, predictions, out_path):
    """Save the forecasted results to a CSV file."""
    
    # Ensure the directory exists before saving the file
    output_dir = os.path.dirname(out_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Adjust the index of the test set to align with predictions
    results = data[-len(predictions):].copy()
    results['VolumeForecast'] = predictions
    
    # Save the results to CSV
    print(f"Results Saved to {out_path}")
    results.to_csv(out_path)

