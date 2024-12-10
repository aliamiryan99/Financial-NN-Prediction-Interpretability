import os
import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource

from Configs.config_schema import Config

class ConfigLoader:
    """Load configuration settings."""
    def __init__(self, config: Config):
        self.config = config
        self.load_config()

    def load_config(self):
        self.show_aggregator = self.config.stream_visualization.show_aggregator
        if self.show_aggregator:
            self.config.data.out_path = f"Results/{self.config.data.name}/EnsembleAggregator.csv"
        self.CSV_FILE_PATH = self.config.data.out_path
        self.BATCH_SIZE = self.config.stream_visualization.batch_size
        self.UPDATE_INTERVAL = self.config.stream_visualization.update_interval
        self.MAX_POINTS = self.config.stream_visualization.max_points
        self.CANDLE_WIDTH = pd.Timedelta(f'0.7{self.config.stream_visualization.time_frame}')
        self.INTERPRETABILITY_PATH = self.config.data.interpret_path
        # Sequence length for dynamic frequency count
        self.seq_length = self.config.model_parameters.seq_length

class DataLoader:
    """Load and prepare data from CSV file."""
    def __init__(self, csv_file_path, show_aggregator):
        self.csv_file_path = csv_file_path
        self.show_aggregator = show_aggregator
        self.df = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.csv_file_path)

        if self.show_aggregator:
            required_columns = {
                'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VolumeForecast',
                'VolumeForecast_Min', 'VolumeForecast_Max', 'VolumeForecast_Var'
            }
        else:
            required_columns = {'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VolumeForecast'}

        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain the following columns: {required_columns}")

        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        df['Status'] = np.where(df['Close'] >= df['Open'], 'bullish', 'bearish')
        df['Color'] = np.where(df['Status'] == 'bullish', 'green', 'red')

        return df

class InterpretabilityDataLoader:
    """Load and prepare interpretability data from a single CSV file."""
    def __init__(self, interpretability_path, feature_columns):
        self.interpretability_path = interpretability_path
        self.feature_columns = feature_columns
        self.num_features = len(feature_columns)
        self.feature_importance_df, self.timestep_importance_df, self.timestep_columns = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.interpretability_path)
        # First columns are feature importance
        feature_importance_cols = df.columns[:self.num_features]
        # Remaining columns are timestep importance
        timestep_importance_cols = df.columns[self.num_features:]
        # Generate timestep column names if not present
        if len(timestep_importance_cols) == 0:
            timestep_importance_cols = [f"Timestep_{i+1}" for i in range(df.shape[1] - self.num_features)]
            df.columns = list(feature_importance_cols) + timestep_importance_cols

        feature_importance_df = df[list(feature_importance_cols)]
        timestep_importance_df = df[list(timestep_importance_cols)]
        return feature_importance_df, timestep_importance_df, timestep_importance_cols.tolist()

class FrequencyDataLoader:
    """Load and prepare frequency and frequency importance data."""
    def __init__(self, spec_interpret_path, feature_columns, config_loader: ConfigLoader):
        self.spec_interpret_path = spec_interpret_path
        self.feature_columns = feature_columns
        self.config_loader = config_loader
        # Frequencies.csv path
        self.frequencies_path = os.path.join(os.path.dirname(self.spec_interpret_path), "Frequencies.csv")
        self.num_frequencies = self.compute_num_frequencies()
        self.frequencies_df, self.frequency_importance_df = self.load_data()
        self.frequency_labels = self.extract_frequency_labels()

    def compute_num_frequencies(self):
        return self.config_loader.seq_length // 2 + 1  # Dynamic number of frequencies

    def load_data(self):
        frequencies_df = pd.read_csv(self.frequencies_path)
        frequency_importance_df = pd.read_csv(self.spec_interpret_path)
        
        # Validate number of frequencies
        for feature in self.feature_columns:
            feature_freq_cols = [col for col in frequencies_df.columns if col.startswith(f"{feature}_freq_")]
            if len(feature_freq_cols) != self.num_frequencies:
                raise ValueError(f"Expected {self.num_frequencies} frequency columns for feature '{feature}', but found {len(feature_freq_cols)}.")
        
        # Ensure frequency importance has correct number of columns
        if frequency_importance_df.shape[1] != self.num_frequencies:
            raise ValueError(f"Frequency importance data must have exactly {self.num_frequencies} columns.")
        
        return frequencies_df, frequency_importance_df

    def extract_frequency_labels(self):
        # Extract unique frequency labels from frequency importance columns
        return list(self.frequency_importance_df.columns)

class DataSourceManager:
    """Initialize ColumnDataSources for price, volume, interpretability, frequency data, and whisker plots."""
    def __init__(self, feature_columns, timestep_columns, frequency_labels):
        self.source_price = self.initialize_price_source()
        self.source_volume = self.initialize_volume_source()
        self.source_feature_importance = self.initialize_feature_importance_source(feature_columns)
        self.source_timestep_importance = self.initialize_timestep_importance_source(timestep_columns)
        self.source_frequencies = self.initialize_frequencies_source(frequency_labels, feature_columns)
        self.source_frequency_importance = self.initialize_frequency_importance_source(frequency_labels)
        
        # Whisker Data Sources
        self.source_timestep_whisker = self.initialize_whisker_source(timestep_columns)
        self.source_feature_whisker = self.initialize_whisker_source(feature_columns)
        self.source_frequency_whisker = self.initialize_whisker_source(frequency_labels)

        # Scatter Data Sources
        self.source_timestep_scatter = self.initialize_scatter_source()
        self.source_feature_scatter = self.initialize_scatter_source()
        self.source_frequency_scatter = self.initialize_scatter_source()

    @staticmethod
    def initialize_price_source():
        return ColumnDataSource(data=dict(
            Time=[],  # datetime objects
            Open=[],
            High=[],
            Low=[],
            Close=[],
            Color=[]
        ))

    @staticmethod
    def initialize_volume_source():
        return ColumnDataSource(data=dict(
            x1=[],  # datetime objects
            x2=[],  # datetime objects
            Time=[],  # datetime objects
            Volume=[],
            PredictedVolume=[],
            PredictedVolume_Min=[],
            PredictedVolume_Max=[]
        ))

    @staticmethod
    def initialize_feature_importance_source(feature_columns):
        return ColumnDataSource(data=dict(
            Feature=feature_columns,
            Importance=[0]*len(feature_columns)
        ))

    @staticmethod
    def initialize_timestep_importance_source(timestep_columns):
        return ColumnDataSource(data=dict(
            Timestep=timestep_columns,
            Importance=[0]*len(timestep_columns)
        ))

    @staticmethod
    def initialize_frequencies_source(frequency_labels, feature_columns):
        # 'Frequency' holds the x-axis labels
        # Each feature has a list of frequency values
        data = {'Frequency': frequency_labels}
        for feature in feature_columns:
            data[feature] = [0]*len(frequency_labels)  # Initialize with zeros
        return ColumnDataSource(data=data)

    @staticmethod
    def initialize_frequency_importance_source(frequency_labels):
        return ColumnDataSource(data=dict(
            Frequency=frequency_labels,
            Importance=[0]*len(frequency_labels)
        ))

    @staticmethod
    def initialize_whisker_source(feature_columns):
        return ColumnDataSource(data=dict(
            base=feature_columns,
            upper=[0]*len(feature_columns),
            lower=[0]*len(feature_columns)
        ))

    @staticmethod
    def initialize_scatter_source():
        return ColumnDataSource(data=dict(
            Feature=[],   # Feature name
            value=[]      # Corresponding importance value
        ))
