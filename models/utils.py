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





