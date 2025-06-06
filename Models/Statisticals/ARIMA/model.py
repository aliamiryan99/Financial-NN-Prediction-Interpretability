# ARIMA_Model.py
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from Models.model_base import ModelBase
import warnings

from Configs.config_schema import Config

warnings.filterwarnings("ignore")  # Suppress warnings

class ForecastingModel(ModelBase):
    def __init__(self, config:Config):
        super().__init__(config)
        self.order = (2, 1, 2)  # Default ARIMA order

    def build(self):
        """
        Set the ARIMA order from configuration.
        """
        super().build()
        self.model = "ARIMA"

    def prepare_data(self, train, test):
        """
        Prepare the data for ARIMA modeling.

        Parameters:
        - train (pd.DataFrame): The training dataset.
        - test (pd.DataFrame): The testing dataset.

        Returns:
        - X_train, y_train, X_test, y_test: Prepared data.
        """
        super().prepare_data(train, test)
        target_column = self.config.model_parameters.target_column

        # For ARIMA, we only need the target variable
        X_train = np.array(train[target_column])  # Is not using for this model
        y_train = np.array(train[target_column])
        X_test = np.array(test[target_column])
        y_test = np.array(test[target_column])

        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train):
        """
        Fit the ARIMA model to the training data.

        Parameters:
        - X_train: Not used in ARIMA.
        - y_train (np.ndarray): The target variable for training.
        """
        super().train(X_train, y_train)

        # Fit the ARIMA model
        self.model = ARIMA(y_train, order=self.order)
        self.model = self.model.fit()
        print(self.model.summary())

    def forecast(self, X_test):
        """
        Make predictions using the trained ARIMA model.

        Parameters:
        - X_test (np.ndarray): The test data (actual values).

        Returns:
        - y_pred (np.ndarray): Forecasted values.
        """
        # Prepare a numpy array to store predictions
        y_pred = np.empty(len(X_test))

        # Real-time prediction loop
        for i in range(len(X_test)):
            # Forecast the next time step
            forecast = self.model.forecast(steps=1)
            y_pred[i] = forecast[0]

            # Get the new observed value
            new_value = X_test[i]

            # Update the model with the new data point
            self.model = self.model.apply(endog=[new_value], refit=False)

        return y_pred

def run(config:Config):
    """
    Orchestrate the ARIMA modeling process using the base class's run method.
    """
    model = ARIMAModel(config)
    model.run()