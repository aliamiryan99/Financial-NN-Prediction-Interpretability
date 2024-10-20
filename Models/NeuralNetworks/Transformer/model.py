# Import Necessary Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LayerNormalization, Dropout,
                                     MultiHeadAttention, GlobalAveragePooling1D)
from tensorflow.keras.models import Model

from Configs.config_schema import Config
from Controllers.ModelModules.modules import create_sequences
from Models.model_base import ModelBase  # Assuming ModelBase is in Controllers.ModelBase

# Global variables for Transformer model parameters
NUM_TRANSFORMER_BLOCKS = 2
NUM_HEADS = 4
KEY_DIM = 32
FF_DIM = 64
DROPOUT_RATE = 0.1

class TransformerModel(ModelBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.seq_length = config.model_parameters.seq_length
        self.num_features = len(config.model_parameters.feature_columns)

    def prepare_data(self, train, test):
        model_parameters = self.config.model_parameters
        # Prepare sequences
        X_train, y_train = create_sequences(
            train[model_parameters.feature_columns].values,
            train[model_parameters.target_column].values,
            model_parameters.seq_length
        )
        X_test, y_test = create_sequences(
            test[model_parameters.feature_columns].values,
            test[model_parameters.target_column].values,
            model_parameters.seq_length
        )
        return X_train, y_train, X_test, y_test

    def get_positional_encoding(self, seq_length, d_model):
        positions = np.arange(seq_length)[:, np.newaxis]
        dimensions = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
        angle_rads = positions * angle_rates

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def build(self):
        """Build the Transformer model."""
        if self.seq_length is None or self.num_features is None:
            raise Exception("Sequence length and number of features must be set before building the model.")

        inputs = Input(shape=(self.seq_length, self.num_features))
        positional_encoding = self.get_positional_encoding(self.seq_length, self.num_features)
        x = inputs + positional_encoding

        for _ in range(NUM_TRANSFORMER_BLOCKS):
            # Multi-Head Attention
            attn_output = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(x, x)
            attn_output = Dropout(DROPOUT_RATE)(attn_output)
            out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

            # Feed Forward Network
            ffn_output = Dense(FF_DIM, activation='relu')(out1)
            ffn_output = Dense(self.num_features)(ffn_output)
            ffn_output = Dropout(DROPOUT_RATE)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.config.model_parameters.optimizer, loss=self.config.model_parameters.loss)
        return self.model

    def train(self, X_train, y_train):
        if self.model is None:
            raise Exception("Model needs to be built before training.")

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

        y_pred = self.model.predict(X_test)
        return y_pred


def run(config: Config):
    model = TransformerModel(config)
    model.run()
