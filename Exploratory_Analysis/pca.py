from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


def lot_components(explained_variance):
 # plot the explained variance
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(explained_variance)+1), explained_variance, color='blue', alpha=0.7)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance of Principal Components')
    plt.show()


def pca_analises(X_normalized,features):

    pca = PCA()
    X_pca = pca.fit_transform(X_normalized)

    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

    print(df_pca)

    explained_variance = pca.explained_variance_ratio_

    lot_components(explained_variance)

    loadings = pca.components_[:2].T

    # create a dataframe with the loadings
    df_loadings = pd.DataFrame(loadings, index=features, columns=['PC1', 'PC2'])
    print("")
    print(df_loadings)

    return df_pca