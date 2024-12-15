import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def classify_news_with_kmeans(test_data, base_data, n_clusters=10, threshold=0.75):
    """
    Classify news as 'novelty' or 'no novelty' using KMeans clustering and cosine similarity.

    Args:
        test_data (np.ndarray): Array containing the test news data.
        base_data (np.ndarray): Array containing the base news data.
        n_clusters (int): Number of clusters for KMeans. Default is 10.
        threshold (float): Cosine similarity threshold for classification. Default is 0.75.

    Returns:
        list: List of classifications ('novelty' or 'no novelty') for each test instance.
    """
    # Apply KMeans clustering to the base data
    kmeans = KMeans(n_clusters=n_clusters, random_state=12)
    kmeans.fit(base_data)
    cluster_centers = kmeans.cluster_centers_

    classifications = []
    for test_row in test_data:
        # Compute cosine similarities between the test row and cluster centers
        similarities = cosine_similarity(test_row.reshape(1, -1), cluster_centers)
        max_similarity = similarities.max()

        # Classify based on the threshold
        if max_similarity < threshold:
            classifications.append("novelty")
        else:
            classifications.append("no novelty")
    
    return classifications

def kmeans_model_and_evaluate_thresholds(test_df, base_df, y_test, thresholds=None, n_clusters=20, n_components=20):
    """
    Evaluate metrics for different thresholds and identify the best one based on F1-Score.

    Args:
        test_df (pd.DataFrame or np.ndarray): Test dataset.
        base_df (pd.DataFrame or np.ndarray): Base dataset.
        y_test (list or np.ndarray): True labels of the test dataset.
        thresholds (np.ndarray, optional): Threshold range for evaluation. Defaults to np.linspace(0.5, 1.0, 51).
        n_clusters (int, optional): Number of clusters for KMeans. Defaults to 20.
        n_components (int, optional): Number of PCA components for dimensionality reduction. Defaults to 20.

    Returns:
        float: Best threshold based on F1-Score.
        dict: Metrics associated with the best threshold.
    """
    # Set default thresholds if not provided
    if thresholds is None:
        thresholds = np.arange(0.5, 0.99, 0.01)

    precision_list, recall_list, f1_list, accuracy_list = [], [], [], [] # metrics lists

    # Apply PCA for dimensionality reduction of embeddings 
    pca = PCA(n_components=n_components)
    base_pca = pca.fit_transform(base_df)
    test_pca = pca.transform(test_df)

    for threshold in thresholds:
        # Classify test data using KMeans and cosine similarity
        predictions = classify_news_with_kmeans(test_pca, base_pca, n_clusters=n_clusters, threshold=threshold)
        predictions_binary = [1 if label == "novelty" else 0 for label in predictions]

        # Compute metrics
        precision_list.append(precision_score(y_test, predictions_binary, zero_division=0))
        recall_list.append(recall_score(y_test, predictions_binary, zero_division=0))
        f1_list.append(f1_score(y_test, predictions_binary, zero_division=0))
        accuracy_list.append(accuracy_score(y_test, predictions_binary))

    # Identify the best threshold based on F1-Score
    best_threshold_index = np.argmax(f1_list)
    best_threshold = thresholds[best_threshold_index]

    # Metrics for the best threshold
    best_metrics = {
        "threshold": best_threshold,
        "precision": precision_list[best_threshold_index],
        "recall": recall_list[best_threshold_index],
        "f1_score": f1_list[best_threshold_index],
        "accuracy": accuracy_list[best_threshold_index],
    }

    # Plot metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision_list, label="Precision", marker='o')
    plt.plot(thresholds, recall_list, label="Recall", marker='o')
    plt.plot(thresholds, f1_list, label="F1-Score", marker='o')
    plt.plot(thresholds, accuracy_list, label="Accuracy", marker='o')
    plt.title("Metrics as a Function of Similarity Threshold")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Metrics at Best Threshold: {best_metrics}")

    return best_threshold, best_metrics
