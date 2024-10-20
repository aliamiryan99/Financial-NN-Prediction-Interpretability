# Import Necessary Libraries
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Lambda, TimeDistributed
from tensorflow.keras.optimizers import Adam

from Configs.config_schema import Config
from Controllers.ModelModules.modules import (
    preprocess_data,
    scale_data,
    split_data
)
from Utils.io import load_data, save_results

warnings.filterwarnings("ignore")  # Suppress warnings

# ============================
# Global VAE Parameters
# ============================

LATENT_DIM = 16
LEARNING_RATE = 0.001

# ============================

def create_sequences(data, seq_length):
    """Prepare sequences for the VAE model."""
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:(i + seq_length)])
    return np.array(sequences)

class Sampling(tf.keras.layers.Layer):
    """Sampling layer to sample latent variables."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(
            shape=tf.shape(z_mean), mean=0.0, stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    def __init__(self, seq_length, num_features, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_features = num_features

        # Encoder
        self.encoder_inputs = Input(shape=(seq_length, num_features))
        self.encoder_lstm = LSTM(64, activation='relu')
        self.z_mean_dense = Dense(latent_dim, name='z_mean')
        self.z_log_var_dense = Dense(latent_dim, name='z_log_var')
        self.sampling = Sampling()

        # Decoder
        self.decoder_repeat = RepeatVector(seq_length)
        self.decoder_lstm = LSTM(64, activation='relu', return_sequences=True)
        self.decoder_outputs = TimeDistributed(Dense(num_features))

    def encode(self, x):
        x = self.encoder_lstm(x)
        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        return z_mean, z_log_var

    def decode(self, z):
        x = self.decoder_repeat(z)
        x = self.decoder_lstm(x)
        x = self.decoder_outputs(x)
        return x

    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.sampling([z_mean, z_log_var])
        outputs = self.decode(z)

        # Compute the reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(inputs - outputs), axis=[1, 2]))

        # Compute the KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))

        # Add the losses as model losses
        self.add_loss(reconstruction_loss + kl_loss)
        return outputs

def train_model(model, X_train, epochs, batch_size):
    """Train the VAE model."""
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE))
    model.fit(X_train, epochs=epochs, batch_size=batch_size, shuffle=True)
    return model

def forecast(model, X_test):
    """Make predictions using the trained VAE model."""
    y_pred = model.predict(X_test)
    return y_pred

def run(config: Config):
    model_parameters = config.model_parameters

    # Load data
    print("Step 1: Loading the Data")
    data = load_data(config.data.in_path)

    # Preprocess data
    print("Step 2: Preprocessing the Data")
    data = preprocess_data(
        data,
        model_parameters.feature_columns,
        filter_holidays=config.preprocess_parameters
    )

    # Scale data
    print("Step 3: Scaling the Data")
    scaled_data, scalers = scale_data(
        data,
        model_parameters.feature_columns
    )

    # Split data
    print("Step 4: Splitting the Data")
    train, test = split_data(
        scaled_data,
        model_parameters.train_ratio
    )

    # Prepare sequences
    print("Step 5: Preparing the Data for VAE")
    X_train = create_sequences(
        train[model_parameters.feature_columns].values,
        model_parameters.seq_length
    )
    X_test = create_sequences(
        test[model_parameters.feature_columns].values,
        model_parameters.seq_length
    )

    # Build model
    print("Step 6: Building the VAE Model")
    num_features = len(model_parameters.feature_columns)
    model = VAE(seq_length=model_parameters.seq_length, num_features=num_features)

    # Train model
    print("Step 7: Training the Model")
    model = train_model(model, X_train, model_parameters.epochs, model_parameters.batch_size)

    # Forecast
    print("Step 8: Forecasting the Test Data")
    y_pred = forecast(model, X_test)

    # Inverse transform the predictions and actual values
    volume_index = model_parameters.feature_columns.index('Volume')
    volume_scaler = scalers['Volume']

    # Extract the last time step predictions
    y_pred_volume = y_pred[:, -1, volume_index]
    y_test_volume = X_test[:, -1, volume_index]

    # Reshape for inverse transform
    y_pred_inv = volume_scaler.inverse_transform(y_pred_volume.reshape(-1, 1))
    y_test_inv = volume_scaler.inverse_transform(y_test_volume.reshape(-1, 1))

    # Save results
    train_size = int(model_parameters.train_ratio * len(scaled_data))
    seq_length = model_parameters.seq_length
    total_length = len(data)

    # Prepare a DataFrame to hold predictions
    results = data.copy()
    results['VolumeForecast'] = np.nan

    # The predictions correspond to indices from (train_size + seq_length) to total_length - 1
    start_index = train_size + seq_length
    end_index = start_index + len(y_pred_inv)

    results.iloc[start_index:end_index, results.columns.get_loc('VolumeForecast')] = y_pred_inv.flatten()

    # Save the results
    results.to_csv(config.data.out_path)
