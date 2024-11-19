import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  f1_score

# ---- Models ----

# 1. Cosine Similarity
def detect_novelty_cosine(train_embeddings, test_embeddings, threshold=0.5):
    """
    Detect novelty using cosine similarity.
    """
    try:
        train_embeddings = train_embeddings.cpu().numpy()
        test_embeddings = test_embeddings.cpu().numpy()
    except:
        print("ok")

    similarities = cosine_similarity(test_embeddings, train_embeddings)
    novelty_scores = 1 - np.max(similarities, axis=1)
    predictions = (novelty_scores > threshold).astype(int)

    return predictions, novelty_scores

# 2. Random Forest with GridSearchCV
def optimize_random_forest(train_features, train_labels):
    """
    Optimize Random Forest parameters using GridSearchCV.
    """
    # features = train_features.cpu().numpy()
    # labels = train_labels.cpu().numpy()

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1_macro', verbose=2, n_jobs=-1)
    grid_search.fit(train_features, train_labels)
    print("Best Random Forest Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_

# 3. Local Outlier Factor (LOF)
def optimize_lof(train_features, test_features, test_labels):
    """
    Optimize LOF parameters manually using a parameter grid.
    """
    # try:
    #     tfeatures = train_features.cpu().numpy()
    #     tfeatures = test_features.cpu().numpy()
    #     test = test_labels.cpu().numpy()
    # except:
    #     print('ok')

    param_grid = {
        'n_neighbors': [5, 10, 20, 30],
        'metric': ['minkowski', 'euclidean', 'manhattan']
    }
    best_params = None
    best_f1 = 0
    best_predictions = None
    for params in ParameterGrid(param_grid):
        lof = LocalOutlierFactor(n_neighbors=params['n_neighbors'], metric=params['metric'], novelty=True)
        lof.fit(train_features)
        predictions = lof.predict(test_features)
        predictions = (predictions == -1).astype(int)
        f1 = f1_score(test_labels, predictions, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_predictions = predictions
    print("Best LOF Parameters:", best_params)
    return best_predictions, best_params