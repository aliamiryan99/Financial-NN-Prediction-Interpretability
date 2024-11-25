# Import Necessary Libraries
import warnings
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout

from Configs.config_schema import Config
from Controllers.ModelModules.modules import create_sequences
from Models.model_base import ModelBase  # Assuming ModelBase is in Controllers.ModelBase

warnings.filterwarnings("ignore")  # Suppress warnings

class ForecastingModel(ModelBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.seq_length = config.model_parameters.seq_length
        self.num_features = len(config.model_parameters.feature_columns)
        # Hyperparameters
        self.filters = 64
        self.kernel_size = 3
        self.pool_size = 2
        self.lstm_units = 50
        self.dense_units = 50
        self.dropout_rate = 0.2

    def prepare_data(self, train, test):
        model_parameters = self.config.model_parameters
        # Prepare sequences
        X_train, y_train = create_sequences(
            train[model_parameters.feature_columns].values,
            train[model_parameters.target_column].values,
            model_parameters.seq_length,
            reshape=True
        )
        X_test, y_test = create_sequences(
            test[model_parameters.feature_columns].values,
            test[model_parameters.target_column].values,
            model_parameters.seq_length,
            reshape=True
        )
        return X_train, y_train, X_test, y_test

    def build(self):
        """Build the LSTM-CNN model."""
        self.model = Sequential()
        # Add 1D Convolutional layers
        self.model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=(self.seq_length, self.num_features)))
        self.model.add(MaxPooling1D(pool_size=self.pool_size))
        # Add LSTM layers
        self.model.add(LSTM(units=self.lstm_units, return_sequences=False))
        self.model.add(Dropout(self.dropout_rate))
        # Fully connected layers
        self.model.add(Dense(units=self.dense_units, activation='relu'))
        self.model.add(Dense(1))  # Output layer for regression
        # Compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model

    def train(self, X_train, y_train):
        if self.model is None:
            raise Exception("Model needs to be built before training.")
        # Fit the model with training data
        self.model.fit(
            X_train, y_train, epochs=self.config.model_parameters.epochs, 
            batch_size=self.config.model_parameters.batch_size, verbose=self.config.model_parameters.verbose)

    def forecast(self, X_test):
        if self.model is None:
            raise Exception("Model needs to be built before forecasting.")
        y_pred = self.model.predict(X_test)
        return y_pred.reshape(-1, 1)

def run(config: Config):
    model = Model(config)
    model.run()
