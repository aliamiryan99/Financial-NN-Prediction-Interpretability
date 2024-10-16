# Import Necessary Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LayerNormalization, Dropout,
                                     MultiHeadAttention, Embedding, GlobalAveragePooling1D)
from tensorflow.keras.models import Model

from Configs.ConfigSchema import Config
from Controllers.ModelModules.modules import (preprocess_data, scale_data,
                                              split_data)
from Utils.io import load_data, save_results

# Global variables for Transformer model parameters
NUM_TRANSFORMER_BLOCKS = 2
NUM_HEADS = 4
KEY_DIM = 32
FF_DIM = 64
DROPOUT_RATE = 0.1

def create_sequences(features, target, seq_length):
    """Prepare sequences for the Transformer model."""
    X = []
    y = []
    for i in range(len(features) - seq_length):
        X.append(features[i:(i + seq_length)])
        y.append(target[i + seq_length])
    X = np.array(X)
    y = np.array(y)
    return X, y


def build_model(seq_length, num_features):
    """Build the Transformer model."""
    inputs = Input(shape=(seq_length, num_features))

    def get_positional_encoding(seq_length, d_model):
        positions = np.arange(seq_length)[:, np.newaxis]
        dimensions = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    # Positional Encoding
    positional_encoding = get_positional_encoding(seq_length, num_features)
    x = inputs + positional_encoding

    # Transformer Blocks
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        # Multi-Head Attention
        attn_output = MultiHeadAttention(num_heads=NUM_HEADS,
                                         key_dim=KEY_DIM)(x, x)
        attn_output = Dropout(DROPOUT_RATE)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed Forward Network
        ffn_output = Dense(FF_DIM, activation='relu')(out1)
        ffn_output = Dense(num_features)(ffn_output)
        ffn_output = Dropout(DROPOUT_RATE)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs, batch_size):
    """Train the Transformer model."""
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def forecast(model, X_test):
    """Make predictions using the trained model."""
    y_pred = model.predict(X_test)
    return y_pred

def run(config: Config):
    model_parameters = config.model_parameters
    # Load data
    print("Step 1: Loading the Data")
    data = load_data(config.data.in_path)

    # Preprocess data
    print("Step 2: Preprocessing the Data")
    data = preprocess_data(data, model_parameters.feature_columns, filter_holidays=config.preprocess_parameters)

    # Scale data
    print("Step 3: Scaling the Data")
    scaled_data, scalers = scale_data(data, model_parameters.feature_columns)

    # Split data
    print("Step 4: Splitting the Data")
    train, test = split_data(scaled_data, model_parameters.train_ratio)

    # Prepare sequences
    print("Step 5: Preparing the Data for the Transformer Model")
    X_train, y_train = create_sequences(train[model_parameters.feature_columns].values,
                                        train[model_parameters.target_column].values,
                                        model_parameters.seq_length)
    X_test, y_test = create_sequences(test[model_parameters.feature_columns].values,
                                      test[model_parameters.target_column].values,
                                      model_parameters.seq_length)

    # Build model
    print("Step 6: Building the Transformer Model")
    model = build_model(model_parameters.seq_length, len(model_parameters.feature_columns))

    # Train model
    print("Step 7: Training the Model")
    model = train_model(model, X_train, y_train, model_parameters.epochs, model_parameters.batch_size)

    # Forecast
    print("Step 8: Forecasting the Test Data")
    y_pred = forecast(model, X_test)

    # Inverse transform the predictions and actual values
    volume_scaler = scalers[model_parameters.target_column]
    y_pred_inv = volume_scaler.inverse_transform(y_pred)
    y_test_inv = volume_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Save results
    train_size = int(model_parameters.train_ratio * len(scaled_data))
    save_results(data, y_pred_inv.flatten(), train_size, model_parameters.seq_length, config.data.out_path)
