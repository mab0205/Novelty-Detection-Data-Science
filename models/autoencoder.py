import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def cosine_similarity_loss(y_true, y_pred):
    """
    Calcula la pérdida basada en la similitud coseno entre los vectores de salida y entrada.
    Args:
        y_true: Valores reales (entrada al autoencoder).
        y_pred: Valores predichos por el autoencoder.

    Returns:
        Cosine similarity loss.
    """
    y_true_normalized = tf.nn.l2_normalize(y_true, axis=1)
    y_pred_normalized = tf.nn.l2_normalize(y_pred, axis=1)
    cosine_similarity = tf.reduce_sum(y_true_normalized * y_pred_normalized, axis=1)
    return 1 - tf.reduce_mean(cosine_similarity)  # 1 - similitud promedio


# def train_autoencoder(train_features, epochs=50, batch_size=256):
#     """
#     Train an autoencoder on the given dataset.

#     Args:
#         train_features (numpy.ndarray): The features to be used for training.
#         epochs (int): The number of epochs to train the autoencoder. Default is 50.
#         batch_size (int): The batch size used for training. Default is 256.

#     Returns:
#         Model: The trained autoencoder model.
#     """
#     input_dim = train_features.shape[1]

#     # Input layer
#     input_layer = Input(shape=(input_dim,))

#     # Encoder
#     encoded = Dense(128, activation='relu')(input_layer)
#     encoded = Dense(64, activation='relu')(encoded)
#     encoded = Dense(32, activation='relu')(encoded)
#     encoded = Dense(16, activation='relu')(encoded)

#     # Bottleneck
#     bottleneck = Dense(8, activation='relu')(encoded)

#     # Decoder
#     decoded = Dense(16, activation='relu')(bottleneck)
#     decoded = Dense(32, activation='relu')(decoded)
#     decoded = Dense(64, activation='relu')(decoded)
#     decoded = Dense(128, activation='relu')(decoded)


#     # Output layer
#     output_layer = Dense(input_dim, activation='sigmoid')(decoded)

#     # Autoencoder model
#     autoencoder = Model(input_layer, output_layer)

#     # Compile the model
#     autoencoder.compile(optimizer='adam', loss='mse')

#     early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

#     autoencoder.fit(
#         train_features, train_features,
#         epochs=epochs,
#         batch_size=batch_size,
#         shuffle=True,
#         validation_split=0.2,
#         callbacks=[early_stopping]
#     )

#     return autoencoder


# trained_autoencoder = train_autoencoder(train_features_combined, epochs=500, batch_size=512)