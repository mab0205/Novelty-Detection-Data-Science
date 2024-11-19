from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def build_autoencoder(input_dim):
    """
    Build a simple autoencoder model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def detect_novelty_autoencoder(train_embeddings, test_embeddings, epochs=50, batch_size=32, threshold=None):
    """
    Detect novelty using an autoencoder.
    """
    
    # train_embeddings = train_embeddings.cpu().numpy()
    # test_embeddings = test_embeddings.cpu().numpy()
    
    # Normalize data
    scaler = MinMaxScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)

    # Build and train the autoencoder
    input_dim = train_embeddings.shape[1]
    autoencoder = build_autoencoder(input_dim)
    autoencoder.fit(train_embeddings, train_embeddings, epochs=epochs, batch_size=batch_size, verbose=0)

    # Reconstruction error
    reconstructions = autoencoder.predict(test_embeddings)
    reconstruction_errors = np.mean(np.square(test_embeddings - reconstructions), axis=1)

    # Define novelty threshold if not provided
    if threshold is None:
        threshold = np.percentile(reconstruction_errors, 95)  # Top 5% as novelty

    predictions = (reconstruction_errors > threshold).astype(int)
    return predictions, reconstruction_errors