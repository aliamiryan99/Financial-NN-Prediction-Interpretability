# Import Necessary Libraries
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from Configs.config_schema import Config
from Controllers.ModelModules.modules import create_sequences
from Models.model_base import ModelBase  # Assuming ModelBase is in Controllers.ModelBase

class LSTMModel(ModelBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.seq_length = None
        self.num_features = None

    def prepare_data(self, train, test):
        model_parameters = self.config.model_parameters
        # Prepare sequences
        print("Preparing the Data for LSTM")
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
        self.seq_length = model_parameters.seq_length
        self.num_features = len(model_parameters.feature_columns)
        return X_train, y_train, X_test, y_test

    def build(self):
        """Build the LSTM model."""
        if self.seq_length is None or self.num_features is None:
            raise Exception("Sequence length and number of features must be set before building the model.")
        print("Building the LSTM Model")
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.seq_length, self.num_features)))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model

    def train(self, X_train, y_train):
        if self.model is None:
            raise Exception("Model needs to be built before training.")

        model_parameters = self.config.model_parameters
        print("Training the LSTM Model")
        self.model.fit(
            X_train,
            y_train,
            epochs=model_parameters.epochs,
            batch_size=model_parameters.batch_size,
            verbose=1
        )

    def forecast(self, X_test):
        if self.model is None:
            raise Exception("Model needs to be built before forecasting.")

        print("Forecasting the Test Data")
        y_pred = self.model.predict(X_test)
        return y_pred

def run(config: Config):
    model = LSTMModel(config)
    model.run()

    