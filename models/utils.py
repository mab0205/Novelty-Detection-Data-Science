from sklearn.metrics import classification_report, f1_score
import numpy as np
import pandas as pd
# ---- Helper Functions ----

def evaluate_novelty(y_true, predictions):
    """
    Evaluate novelty detection results.
    """
    print(classification_report(y_true, predictions))

def prepare_features_without_embeddings(df):
    """
    Prepare features excluding embeddings for novelty detection.
    """
    pca_features = df[['PC1']].to_numpy()
    lda_features = df[[col for col in df.columns if col.startswith('topic_')]].to_numpy()
    combined_features = np.hstack([pca_features, lda_features])
    return combined_features

def combine_features(df, embeddings):
    """
    Combine embeddings with all LDA topic distributions and PCA features.
    """
    embeddings = embeddings.cpu().numpy()
    
    pca_features = df[['PC1']].to_numpy()
    lda_features = df[[col for col in df.columns if col.startswith('topic_')]].to_numpy()
    combined_features = np.hstack([embeddings, pca_features, lda_features])
    return combined_features



