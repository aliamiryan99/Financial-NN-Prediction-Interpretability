# Import Necessary Libraries
import warnings
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.models import Sequential

from Configs.config_schema import Config
from Modules.ModelModules.modules import create_sequences
from Models.model_base import ModelBase  # Assuming ModelBase is in Controllers.ModelBase

warnings.filterwarnings("ignore")  # Suppress warnings

class ForecastingModel(ModelBase):
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
        """Build the BiLSTM model."""
        if self.seq_length is None or self.num_features is None:
            raise Exception("Sequence length and number of features must be set before building the model.")
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(self.seq_length, self.num_features)))
        self.model.add(Bidirectional(LSTM(50)))
        self.model.add(Dense(1))
        self.model.compile(optimizer=self.config.model_parameters.optimizer, loss=self.config.model_parameters.loss)
        return self.model

    def train(self, X_train, y_train):
        if self.model is None:
            raise Exception("Model needs to be built before training.")

        # Get validation split from config or set default to 0.1
        validation_split = getattr(self.config.model_parameters, 'validation_split', 0.1)

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=self.config.model_parameters.epochs,
            batch_size=self.config.model_parameters.batch_size,
            verbose=self.config.model_parameters.verbose,
            validation_split=validation_split
        )

    def forecast(self, X_test):
        if self.model is None:
            raise Exception("Model needs to be built before forecasting.")

        y_pred = self.model.predict(X_test)
        return y_pred

def run(config: Config):
    model = Model(config)
    model.run()
