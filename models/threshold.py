from sklearn.metrics import classification_report, f1_score
import numpy as np

def evaluate_combined_thresholds(diagonal_similarities, reconstruction_errors, y_test, similarity_intervals, error_thresholds=None):
    """
    Evalúa combinaciones de intervalos de similitud coseno y thresholds de error de reconstrucción.

    Args:
        diagonal_similarities (np.ndarray): Similitudes coseno.
        reconstruction_errors (np.ndarray): Errores de reconstrucción.
        y_test (np.ndarray): Etiquetas reales.
        similarity_intervals (list of tuples): Lista de intervalos de similitud coseno (e.g., [(0.75, 0.95)]).
        error_thresholds (list or None): Lista de thresholds para el error de reconstrucción. Si es None, se generan automáticamente.

    Returns:
        dict: Diccionario con los mejores parámetros y métricas.
    """
    best_f1_score = 0
    best_similarity_interval = None
    best_error_threshold = None
    best_report = None

    # Generar thresholds automáticamente si no se proporcionan
    if error_thresholds is None:
        error_thresholds = []
        current_threshold = 0.01
        while current_threshold <= 1.0:
            error_thresholds.append(current_threshold)
            current_threshold += 0.01

    # Iterar sobre thresholds de error
    for error_threshold in error_thresholds:
        # Por cada threshold de error, recorrer los intervalos de similitud
        for sim_interval in similarity_intervals:
            # Clasificación combinada
            novelty_predictions = ((diagonal_similarities < sim_interval[0]) | 
                                    (diagonal_similarities > sim_interval[1]) | 
                                    (reconstruction_errors > error_threshold)).astype(int)

            # Calcular F1-score y reporte de clasificación
            f1 = f1_score(y_test, novelty_predictions)
            report = classification_report(y_test, novelty_predictions, output_dict=True)

            # Condición: Ambos F1-scores >= 0.4
            if report["0"]["f1-score"] >= 0.5 and report["1"]["f1-score"] >= 0.5:
                # Si mejora el F1-score global, actualizar los mejores resultados
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_similarity_interval = sim_interval
                    best_error_threshold = error_threshold
                    best_report = report

    return {
        "best_similarity_interval": best_similarity_interval,
        "best_error_threshold": best_error_threshold,
        "best_f1_score": best_f1_score,
        "classification_report": best_report
    }
