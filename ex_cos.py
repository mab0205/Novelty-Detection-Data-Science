from sklearn.metrics import f1_score, classification_report
import numpy as np

def evaluate_cosine_similarity(diagonal_similarities, y_test, thresholds=None):
    """
    Evalúa el rendimiento de la similitud coseno usando F1-score para diferentes umbrales.

    Args:
        diagonal_similarities (np.ndarray): Similitudes coseno en la diagonal (original vs reconstruido).
        y_test (np.ndarray): Etiquetas reales (0 = no novelty, 1 = novelty).
        thresholds (list or None): Lista de umbrales para probar. Si es None, se generan automáticamente.

    Returns:
        dict: Diccionario con los siguientes valores:
              - "best_threshold": Mejor umbral que maximiza el F1-score.
              - "best_f1_score": El F1-score máximo alcanzado.
              - "classification_report": Reporte de clasificación con el mejor umbral.
              - "f1_scores": Lista de F1-scores para cada umbral probado.
              - "thresholds": Lista de umbrales probados.
    """
    # Generar umbrales automáticamente si no se especifican
    if thresholds is None:
        thresholds = np.arange(0.74, 0.99, 0.001)  # Ajusta el rango según la distribución

    best_f1_score = 0
    best_threshold = 0
    f1_scores = []

    for threshold in thresholds:
        # Clasificar como novedad (1) si la similitud está por debajo del umbral
        novelty_predictions = (diagonal_similarities < threshold).astype(int)
        
        # Calcular F1-score
        f1 = f1_score(y_test, novelty_predictions)
        f1_scores.append(f1)

        # Actualizar el mejor F1-score y su umbral
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold

    # Generar reporte de clasificación para el mejor umbral
    best_predictions = (diagonal_similarities < best_threshold).astype(int)
    report = classification_report(y_test, best_predictions, output_dict=False)

    return {
        "best_threshold": best_threshold,
        "best_f1_score": best_f1_score,
        "classification_report": report,
        "f1_scores": f1_scores,
        "thresholds": thresholds
    }


from sklearn.metrics.pairwise import cosine_similarity

# Obtén los embeddings reconstruidos
reconstructed = trained_autoencoder.predict(test_features_combined)

# Calcula la similitud coseno para cada muestra
cosine_similarities = cosine_similarity(test_features_combined, reconstructed)
diagonal_similarities = cosine_similarities.diagonal()
# Imprime la matriz de similitud
print(cosine_similarities)

# Similitud coseno en la diagonal
diagonal_similarities = cosine_similarities.diagonal()

# Histograma de la diagonal
plt.hist(diagonal_similarities, bins=50, alpha=0.75)
plt.xlabel('Cosine Similarity (Diagonal)')
plt.ylabel('Frequency')
plt.title('Distribution of Cosine Similarities (Diagonal)')
plt.show()

# Opcional: Analiza la similitud promedio
average_similarity = cosine_similarities.diagonal().mean()  # Similitud promedio en la diagonal
print(f"Average Cosine Similarity: {average_similarity}")


results = evaluate_cosine_similarity(diagonal_similarities, y_test)

# Imprimir el mejor umbral y el F1-score
print(f"Best Threshold: {results['best_threshold']}")
print(f"Best F1-Score: {results['best_f1_score']}")
print("Classification Report:")
print(results["classification_report"])

# Visualizar los F1-scores para cada umbral probado
import matplotlib.pyplot as plt
plt.plot(results["thresholds"], results["f1_scores"], marker='o')
plt.xlabel('Threshold')
plt.ylabel('F1-Score')
plt.title('F1-Score vs Threshold')
plt.show()