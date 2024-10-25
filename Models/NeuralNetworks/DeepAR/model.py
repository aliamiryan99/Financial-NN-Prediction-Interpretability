# Import Necessary Libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np

from Configs.config_schema import Config
from Controllers.ModelModules.modules import create_sequences
from Models.model_base import ModelBase  # Assuming ModelBase is in Controllers.ModelBase

class DeepARModel(ModelBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.seq_length = config.model_parameters.seq_length
        self.num_features = len(config.model_parameters.feature_columns)
        self.model = None

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
        """Build the DeepAR model."""
        if self.seq_length is None or self.num_features is None:
            raise Exception("Sequence length and number of features must be set before building the model.")

        # Input layer
        inputs = Input(shape=(self.seq_length, self.num_features))

        # LSTM layers
        x = LSTM(50, return_sequences=True)(inputs)
        x = LSTM(50)(x)

        # Output layer for parameters of the probabilistic distribution
        params = Dense(2)(x)  # Output both mu and sigma

        # Define the model to output params directly
        self.model = Model(inputs=inputs, outputs=params)

        # Negative log likelihood loss function
        def nll(y_true, y_pred):
            mu = y_pred[:, 0]
            sigma = y_pred[:, 1]
            sigma = tf.keras.activations.softplus(sigma) + 1e-6  # Ensure positivity
            # Compute negative log-likelihood
            loss = 0.5 * tf.math.log(2 * np.pi * tf.square(sigma)) + tf.square(y_true[:, 0] - mu) / (2 * tf.square(sigma))
            return tf.reduce_mean(loss)

        # Compile the model
        self.model.compile(optimizer=self.config.model_parameters.optimizer, loss=nll)
        return self.model

    def train(self, X_train, y_train):
        if self.model is None:
            raise Exception("Model needs to be built before training.")

        # Reshape y_train to match the output shape
        y_train = y_train.reshape(-1, 1)

        self.model.fit(
            X_train,
            y_train,
            epochs=self.config.model_parameters.epochs,
            batch_size=self.config.model_parameters.batch_size,
            verbose=self.config.model_parameters.verbose
        )

    def forecast(self, X_test):
        if self.model is None:
            raise Exception("Model needs to be built before forecasting.")

        # Predict mu and sigma
        y_pred = self.model.predict(X_test)
        mu = y_pred[:, 0]
        sigma = tf.keras.activations.softplus(y_pred[:, 1]) + 1e-6  # Ensure positivity
        return mu

def run(config: Config):
    model = DeepARModel(config)
    model.run()
