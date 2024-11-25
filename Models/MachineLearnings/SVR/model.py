# Import Necessary Libraries
import warnings
import numpy as np
from sklearn.svm import SVR  # Import SVR from scikit-learn

from Configs.config_schema import Config
from Controllers.ModelModules.modules import create_sequences
from Models.model_base import ModelBase  # Assuming ModelBase is in Controllers.ModelBase

warnings.filterwarnings("ignore")  # Suppress warnings

# ============================
# Global SVR Parameters
# ============================
SVR_KERNEL = 'rbf'
SVR_C = 1.0
SVR_EPSILON = 0.1
SVR_GAMMA = 'scale'  # Can be 'scale', 'auto', or a float value
# ============================

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
            flatten=True
        )
        X_test, y_test = create_sequences(
            test[model_parameters.feature_columns].values,
            test[model_parameters.target_column].values,
            model_parameters.seq_length,
            flatten=True
        )
        return X_train, y_train, X_test, y_test

    def build(self):
        """Build the SVR model."""
        self.model = SVR(
            kernel=SVR_KERNEL,
            C=SVR_C,
            epsilon=SVR_EPSILON,
            gamma=SVR_GAMMA
        )
        return self.model

    def train(self, X_train, y_train):
        if self.model is None:
            raise Exception("Model needs to be built before training.")
        self.model.fit(X_train, y_train)

    def forecast(self, X_test):
        if self.model is None:
            raise Exception("Model needs to be built before forecasting.")
        y_pred = self.model.predict(X_test)
        return y_pred.reshape(-1, 1)


def run(config: Config):
    model = ForecastingModel(config)
    model.run()
