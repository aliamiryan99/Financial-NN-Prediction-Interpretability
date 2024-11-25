import os

import pandas as pd


def load_data(input_data_path):
    """Load the data from the CSV file."""
   
    data = pd.read_csv(input_data_path, parse_dates=['Time'], index_col='Time')
    return data


def save_forecasting_results(data, predictions, out_path, target):
    """Save the forecasted results to a CSV file."""
    
    # Ensure the directory exists before saving the file
    output_dir = os.path.dirname(out_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Adjust the index of the test set to align with predictions
    results = data[-len(predictions):].copy()
    results[f'{target}Forecast'] = predictions
    
    # Save the results to CSV
    print(f"Results Saved to {out_path}")
    results.to_csv(out_path)

def save_interpretability_results(data, interpretations, out_path, feature_columns):
    """Save the interpretability results to a CSV file."""
    
    # Ensure the directory exists before saving the file
    output_dir = os.path.dirname(out_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Determine the number of features and timesteps from interpretations
    num_features = len(feature_columns)
    total_columns = interpretations.shape[1]
    num_timesteps = total_columns - num_features
    
    # Create column names for feature importances
    feature_importance_columns = [f'{feature}_imp' for feature in feature_columns]
    
    # Create column names for timestep importances
    timestep_importance_columns = [f'step_{i+1}_imp' for i in range(num_timesteps)]
    
    # Combine all column names
    interpretation_columns = feature_importance_columns + timestep_importance_columns
    
    # Create a DataFrame from the interpretations
    interpretations_df = pd.DataFrame(interpretations, columns=interpretation_columns)
    
    # Adjust the index of the data to align with interpretations
    results = data[-len(interpretations):].copy().reset_index(drop=True)
    
    # Concatenate the data and interpretations
    results = pd.concat([results, interpretations_df], axis=1)
    
    # Save the results to CSV
    print(f"Results Saved to {out_path}")
    interpretations_df.to_csv(out_path, index=False)

