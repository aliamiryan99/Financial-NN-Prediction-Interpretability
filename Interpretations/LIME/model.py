import numpy as np
from tqdm import tqdm
from lime.lime_tabular import LimeTabularExplainer
from Interpretations.interpretation_base import InterpretationBase

class InterpretationModel(InterpretationBase):

    def interpret(self, X_test):
        """
        Calculates the importance of features and timesteps separately using LIME.
        :param X_test: Input data to be interpreted, shape (num_samples, seq_length, num_features)
        :return: An ndarray where each row corresponds to a single prediction and each column represents the importance of a feature or timestep.
        """
        self.num_samples, self.seq_length, self.num_features = X_test.shape
        
        # Flatten the data for models like XGBoost, RandomForest, etc.
        if len(X_test.shape) == 3:
            # For LSTM, reshape it to (num_samples, seq_length * num_features)
            X_test_flat = X_test.reshape(self.num_samples, -1)
        else:
            # For models like XGBoost or RandomForest, ensure the input is already flat
            X_test_flat = X_test

        # Create the feature names by combining the time step and feature name
        feature_names = [f"Feature_{i+1}_Timestep_{t+1}" for t in range(self.seq_length) for i in range(self.num_features)]

        # Initialize the LIME Explainer
        self.lime_explainer = LimeTabularExplainer(
            training_data=X_test_flat,
            mode="regression",  # Or "classification" depending on your model
            feature_names=feature_names,
            class_names=["Prediction"]  # Can be customized depending on the model's output
        )
        
        feature_importances = np.zeros((self.num_samples, self.num_features))
        timestep_importances = np.zeros((self.num_samples, self.seq_length))

        print("Interpreting sample predictions with LIME...")

        for i in tqdm(range(self.num_samples)):
            # Get the instance to explain (flattened or original shape)
            instance = X_test[i].reshape(1, -1)

            # Alter predict method for models with sequence
            predict_method = self.forecasting_model.model.predict
            if len(X_test.shape) == 3:
                predict_method = self._predict_with_sequence
            
            # Use LIME's explain_instance method
            explanation = self.lime_explainer.explain_instance(
                instance.flatten(), 
                predict_method, 
                num_features=self.seq_length*self.num_features,
                num_samples=100
            )

            # Extract feature importances from LIME's explanation (flattened feature importance)
            explanation_local_exp = explanation.local_exp[0]  # Explanation for class 0 (for regression, there's only one class)
            feature_importance_values = np.array([exp[1] for exp in explanation_local_exp])  # Get the importance values for all features

            # Sum importances across timesteps for each feature
            for feature_idx in range(self.num_features):
                feature_importances[i, feature_idx] = np.sum(feature_importance_values[feature_idx::self.num_features])

            # Sum importances across features for each timestep
            for timestep_idx in range(self.seq_length):
                timestep_importances[i, timestep_idx] = np.sum(feature_importance_values[timestep_idx*self.num_features:(timestep_idx+1)*self.num_features])

        # Combine the results for features and timesteps
        importance_results = np.concatenate([feature_importances, timestep_importances], axis=1)
        return importance_results

    def _predict_with_sequence(self, X):
        """
        Reshapes the input X into the required shape for LSTM models (num_samples, seq_length, num_features).
        This function is used only for models that require sequential input.
        :param X: Input data, shape (num_samples, seq_length * num_features)
        :param num_sequence: Number of time steps (seq_length)
        :param num_features: Number of features per time step
        :return: reshaped input X of shape (num_samples, seq_length, num_features)
        """
        # Reshape the input data to (num_samples, seq_length, num_features)
        X_reshaped = X.reshape(-1, self.seq_length, self.num_features)
        return self.forecasting_model.model.predict(X_reshaped)