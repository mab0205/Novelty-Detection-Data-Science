from sklearn.metrics import classification_report, f1_score
import numpy as np
import pandas as pd
# ---- Helper Functions ----

def evaluate_novelty(y_true, predictions):
    """
    Evaluate novelty detection results.
    """
    print(classification_report(y_true, predictions))

def prepare_features_without_embeddings(df,selected_columns):
    """
    Prepare features excluding embeddings for novelty detection.
    """
    selected_data = df[selected_columns]
    return selected_data

def combine_features(df, embeddings, selected_columns):
    """
    Combine embeddings with all LDA topic distributions and PCA features.
    """
    selected_data = df[selected_columns]
    
    combined_df = pd.concat([selected_data.reset_index(drop=True), embeddings.reset_index(drop=True)], axis=1)
    return combined_df


def generate_many_similarity_intervals(diagonal_similarities, num_points=20):
    """
    Genera una lista de múltiples intervalos de similitud coseno basados en la distribución.

    Args:
        diagonal_similarities (np.ndarray): Similitudes coseno (diagonal).
        num_points (int): Número de puntos de referencia (divisiones) en el rango de similitudes.

    Returns:
        list of tuples: Lista de intervalos [(min1, max1), (min2, max2), ...].
    """
    # Obtener el rango de diagonal_similarities
    min_value = np.min(diagonal_similarities)
    max_value = np.max(diagonal_similarities)

    # Generar puntos equidistantes en el rango
    points = np.linspace(min_value, max_value, num_points)

    # Crear intervalos combinando los puntos
    intervals = []
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            lower_bound = round(points[i], 3)
            upper_bound = round(points[j], 3)
            intervals.append((lower_bound, upper_bound))

    return intervals
