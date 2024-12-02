from sklearn.metrics import classification_report, f1_score
import numpy as np

def evaluate_autoencoder(autoencoder, test_features, y_test, thresholds=None):
    """
    Evaluate the trained autoencoder by calculating the reconstruction error,
    normalizing it, and classifying based on a range of thresholds.
    Returns predictions, the best threshold, and the F1-score.

    Args:
        autoencoder (Model): The trained autoencoder model.
        test_features (numpy.ndarray): The features to be tested.
        y_test (numpy.ndarray): The true labels (0 for 'no novelty', 1 for 'novelty').
        thresholds (list or None): List of thresholds to evaluate. If None, default thresholds are used.

    Returns:
        dict: A dictionary with the following keys:
              - "best_threshold": The threshold with the highest F1-score.
              - "best_f1_score": The highest F1-score achieved.
              - "y_pred": Predictions using the best threshold.
              - "classification_report": The classification report for the best threshold.
    """
    # Predict the reconstruction (get the reconstructed features)
    reconstructed = autoencoder.predict(test_features)

    # Calculate the reconstruction error (Mean Squared Error for each sample)
    reconstruction_error = np.mean(np.square(test_features - reconstructed), axis=1)

    # # Normalize the reconstruction error to be between 0 and 1
    min_error = np.min(reconstruction_error)
    max_error = np.max(reconstruction_error)
    normalized_error = (reconstruction_error - min_error) / (max_error - min_error)

    # Default thresholds if none are provided
    if thresholds is None:
        thresholds = []
        current_threshold = 0.01
        while current_threshold <= 1.0:
            thresholds.append(current_threshold)
            current_threshold += 0.01

    # Evaluate F1-scores for all thresholds
    macro_f1_scores = [] 
    for threshold in thresholds:
        # Classify as novelty (1) or non-novelty (0) based on the threshold
        y_pred = (normalized_error > threshold).astype(int)
        
        # (el promedio simple del F1-score para ambas clases) en lugar de usar solo el F1-score de una clase.y
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate macro F1-score (average F1 for both classes)
        macro_f1 = report['macro avg']['f1-score']
        macro_f1_scores.append(macro_f1)

    # Find the best threshold
    best_index = np.argmax(macro_f1_scores)
    best_threshold = thresholds[best_index]
    best_f1_score = macro_f1_scores[best_index]

    # Generate predictions and classification report using the best threshold
    y_pred = (normalized_error > best_threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=False)

    return {
        "best_threshold": best_threshold,
        "best_f1_score": best_f1_score,
        "y_pred": y_pred,
        "classification_report": report,
    }


# Evaluar el autoencoder
results = evaluate_autoencoder(trained_autoencoder, test_features_combined, y_test)

# Imprimir los resultados
print("Best Threshold:", results["best_threshold"])
print("Best F1-Score:", results["best_f1_score"])
print("Classification Report:")
print(results["classification_report"])