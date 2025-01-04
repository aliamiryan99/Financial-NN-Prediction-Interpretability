# Modules/Evaluations/interpretability_evaluations.py

import numpy as np
from typing import Callable, List
from tqdm import tqdm

def compute_iAUC(
    model,
    interpretability_method,
    X_test: np.ndarray,
    reference_input: np.ndarray,
    num_features: int = None
) -> List[float]:
    """
    Computes the Insertion Area Under the Curve (iAUC) for faithfulness evaluation.

    Parameters:
    - model: Trained regression model with a predict method.
    - interpretability_method: Method providing feature importance scores (e.g., SHAP, LIME).
                               Should have an interpret method that takes X and returns (N, T, F) explanations.
    - X_test: np.ndarray of shape (N, T, F), test input data.
    - reference_input: np.ndarray of shape (N, T, F), reference input data (e.g., all zeros or blurred).
    - num_features: int, number of top features to insert. If None, use F (number of features).

    Returns:
    - iAUC_scores: List of iAUC scores, one per input instance.
    """
    N, T, F = X_test.shape
    if num_features is None:
        num_features = T*F

    # Obtain feature importance scores from the interpretability method
    print("Obtaining feature importance scores from interpretability method...")
    explanations = interpretability_method.interpret(X_test)  # Shape: (N, T, F)

    # Aggregate explanations to obtain feature importance per input by summing over timesteps
    feature_importance = np.reshape(np.abs(explanations), (N, T*F))  # Shape: (N, T*F)

    # Sort features by importance in descending order for each input
    sorted_feature_indexes = np.argsort(-feature_importance, axis=1)  # Shape: (N, T*F)

    # Compute model predictions on reference inputs
    print("Computing model predictions on reference inputs...")
    reference_preds = model.predict(reference_input)  # Shape: (N, 1)
    reference_class_preds = reference_preds.flatten()  # Shape: (N,)

    # Initialize list to store iAUC scores
    iAUC_scores = []

    print("Computing iAUC for all input instances...")
    # Initialize modified_input as reference_input
    modified_input = reference_input.copy()  # Shape: (N, T, F)

    # Initialize list to store predictions with reference
    preds_list = [model.predict(modified_input).flatten()]  # List of arrays, each of shape (N,)

    # Iterate over the number of features to insert
    for k in range(num_features):
        # Get the feature to insert for each input
        feat_to_insert = sorted_feature_indexes[:, k]  # Shape: (N,)

        # Insert the k-th feature into the modified_input for all inputs
        for i in range(N):
            f_idx = feat_to_insert[i]
            modified_input[i, f_idx//F, f_idx%F] = X_test[i, f_idx//F, f_idx%F]
        
        # Predict on the modified_input
        current_preds = model.predict(modified_input).flatten()  # Shape: (N,)
        preds_list.append(current_preds.copy())

    # Convert preds_list to a numpy array: Shape (N, num_features + 1)
    preds_array = np.array(preds_list).T  # Shape: (N, num_features + 1)

    # Compute differences from reference predictions
    diffs = preds_array[:, 1:] - reference_class_preds[:, np.newaxis]  # Shape: (N, num_features)

    # Compute iAUC as the average difference over the number of features
    iAUC_scores = np.mean(diffs, axis=1).tolist()  # Shape: (N,)

    return iAUC_scores

def compute_max_sensitivity(
    model,
    interpretability_method,
    X_test: np.ndarray,
    perturb_function: Callable[[np.ndarray, float], np.ndarray],
    r: float,
    num_perturbations: int
) -> List[float]:
    """
    Computes the Max-Sensitivity for stability evaluation.

    Parameters:
    - model: Trained regression model with a predict method.
    - interpretability_method: Method providing feature importance scores (e.g., SHAP, LIME).
                               Should have an interpret method that takes X and returns (N, T, F) explanations.
    - X_test: np.ndarray of shape (N, T, F), test input data.
    - perturb_function: Function that takes an input array and a perturbation range r,
                         and returns a perturbed input array.
    - r: float, perturbation range parameter.
    - num_perturbations: int, number of perturbations per input.

    Returns:
    - max_sensitivity_scores: List of Max-Sensitivity scores, one per input instance.
    """
    N, T, F = X_test.shape

    # Obtain original explanations for all test inputs
    print("Obtaining original feature importance scores from interpretability method...")
    original_explanations = interpretability_method.interpret(X_test)  # Shape: (N, T, F)

    # Initialize array to store maximum sensitivity per input
    max_sensitivity_scores = np.zeros(N)

    print("Generating perturbed inputs and obtaining explanations...")
    for _ in (range(num_perturbations)):
        # Generate perturbed inputs
        perturbed_X = perturb_function(X_test, r)  # Shape: (N, T, F)

        # Obtain explanations for perturbed inputs
        perturbed_explanations = interpretability_method.interpret(perturbed_X)  # Shape: (N, T, F)

        # Compute difference norms for each input
        differences = np.linalg.norm(perturbed_explanations - original_explanations, axis=(1, 2))  # Shape: (N,)

        # Update max_sensitivity_scores with the maximum differences
        max_sensitivity_scores = np.maximum(max_sensitivity_scores, differences)

    return max_sensitivity_scores.tolist()
