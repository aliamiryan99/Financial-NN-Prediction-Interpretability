import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess_data(data, feature_columns, filter_holidays=False):
    """Preprocess the data by sorting, setting frequency, and handling missing values."""
    
    data.sort_index(inplace=True)  # Ensure data is sorted by Time
    data = data.asfreq('H')        # Set the frequency to hourly
    # Handle missing values by forward-filling
    data[feature_columns].fillna(method='ffill', inplace=True)

    if filter_holidays:
        # Remove rows where volume is zero
        if 'Volume' in feature_columns:
            data = data[data['Volume'] > 0]  # Keep only rows where volume is greater than 0
    
    return data


def scale_data(data, feature_columns, scaling_method='standard'):
    """
    Scale the data using Z-score scaling (StandardScaler) or MinMax scaling.
    
    Parameters:
    - data (pd.DataFrame): The input data.
    - feature_columns (list): List of columns to scale.
    - scaling_method (str): The scaling method ('standard' or 'minmax').
    
    Returns:
    - scaled_data (pd.DataFrame): Scaled data.
    - scalers (dict): Dictionary of scalers for each feature column.
    """
    
    scalers = {}
    scaled_data = pd.DataFrame(index=data.index)
    
    for column in feature_columns:
        if scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
            
        scaled_data[column] = scaler.fit_transform(data[[column]])
        scalers[column] = scaler  # Save the scaler for inverse transformation
    
    return scaled_data, scalers

def split_data(scaled_data, train_ratio):
    """Split the data into training and testing sets."""
    train_size = int(train_ratio * len(scaled_data))
    train = scaled_data.iloc[:train_size]
    test = scaled_data.iloc[train_size:]
    return train, test

