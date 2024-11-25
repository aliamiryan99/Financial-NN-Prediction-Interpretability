# VAR_Model.py
import numpy as np
from statsmodels.tsa.api import VAR
from Models.model_base import ModelBase
import warnings

from Configs.config_schema import Config

warnings.filterwarnings("ignore")  # Suppress warnings

class ForecastingModel(ModelBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.lags = 2  # Default lag order for VAR

    def build(self):
        """
        Set up the VAR model configuration.
        """
        super().build()
        self.model = "VAR"

    def prepare_data(self, train, test):
        """
        Prepare the data for VAR modeling.

        Parameters:
        - train (pd.DataFrame): The training dataset.
        - test (pd.DataFrame): The testing dataset.

        Returns:
        - X_train, y_train, X_test, y_test: Prepared data.
        """
        super().prepare_data(train, test)
        feature_columns = self.config.model_parameters.feature_columns

        # For VAR, we need all the feature variables
        X_train = np.array(train[feature_columns])
        y_train = np.array(train[feature_columns])
        X_test = np.array(test[feature_columns])
        y_test = np.array(test[feature_columns])

        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train):
        """
        Fit the VAR model to the training data.

        Parameters:
        - X_train (pd.DataFrame): The input variables for training.
        - y_train (pd.DataFrame): The target variables for training.
        """
        super().train(X_train, y_train)

        # Fit the VAR model
        model = VAR(endog=X_train)
        self.model = model.fit(maxlags=self.lags)
        print(self.model.summary())

    def forecast(self, X_test):
        """
        Make predictions using the trained VAR model.

        Parameters:
        - X_test (pd.DataFrame): The test data.

        Returns:
        - y_pred (np.ndarray): Forecasted values.
        """
        # Number of steps to forecast
        steps = len(X_test)

        # Get the last lag observations from the training data
        lag_order = self.model.k_ar
        last_obs = self.model.endog[-lag_order:]

        # Forecast future values
        forecast = self.model.forecast(y=last_obs, steps=steps)

        return forecast

def run(config: Config):
    """
    Orchestrate the VAR modeling process using the base class's run method.
    """
    model = ForecastingModel(config)
    model.run()
