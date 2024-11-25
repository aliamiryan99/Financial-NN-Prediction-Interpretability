# Import necessary libraries
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.models import Model
from Configs.config_schema import Config
from Controllers.ModelModules.modules import create_sequences
from Models.model_base import ModelBase

class ForecastingModel(ModelBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.num_features = len(config.model_parameters.feature_columns)
        self.seq_length = config.model_parameters.seq_length  # Sequence length for creating sequences
        
        # Static configurations for the model
        self.hidden_units = [8, 4]  # Hidden layers for univariate functions
        self.output_units = [16, 8]  # Additional layers after summation (if needed)
        self.activation = 'relu'      # Activation function for hidden layers

    def prepare_data(self, train, test):
        model_parameters = self.config.model_parameters
        # Use create_sequences to prepare data
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
        
        # No reshaping needed; data is returned as-is
        return X_train, y_train, X_test, y_test

    def build(self):
        """Build the KAN-like model manually."""
        if self.num_features is None or self.seq_length is None:
            raise Exception("Number of features and sequence length must be set before building the model.")

        # Adjusted input shape for the sequence input
        input_shape = self.num_features * self.seq_length
        inputs = Input(shape=(input_shape,))

        # Create univariate networks for each feature segment
        univariate_outputs = []
        for i in range(input_shape):
            # Extract each feature
            feature_input = Lambda(lambda x: x[:, i:i+1])(inputs)
            # Apply univariate transformation
            x = feature_input
            for units in self.hidden_units:
                x = Dense(units, activation=self.activation)(x)
            univariate_outputs.append(x)

        # Sum outputs of univariate transformations
        summed_output = Add()(univariate_outputs)

        # Additional layers after summation
        x = summed_output
        for units in self.output_units:
            x = Dense(units, activation=self.activation)(x)
        
        # Final output layer
        outputs = Dense(1)(x)

        # Compile model
        self.model = ForecastingModel(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
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
    model = ForecastingModel(config)
    model.run()
