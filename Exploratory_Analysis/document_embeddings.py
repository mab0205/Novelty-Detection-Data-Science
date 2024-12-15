import os
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
# from sklearn.preprocessing import MinMaxScaler

def generate_doc_vectors(df, tokens_column, base_dir, name, vector_size=100, min_count=6, epochs=300):
    """
    Generates document embeddings using Doc2Vec.

    Args:
        df (pd.DataFrame): DataFrame containing the documents.
        tokens_column (str): Name of the column in the DataFrame that contains tokenized words (lists of tokens).
        base_dir (str): Base directory to save the results.
        name (str): Name of the output file.
        vector_size (int, optional): Size of the document vectors. Defaults to 100.
        min_count (int, optional): Minimum frequency for a word to be included in the vocabulary. Defaults to 6.
        epochs (int, optional): Number of training epochs. Defaults to 300.

    Returns:
        np.ndarray: Array of normalized document vectors.
    """
    file_path = os.path.join(base_dir, "results", f"{name}.csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        print(f"File '{file_path}' exists. Loading normalized vectors...")
        return pd.read_csv(file_path).values


    # Create tagged data for training
    tagged_data = [
        TaggedDocument(words=row[tokens_column], tags=[str(index)]) 
        for index, row in df.iterrows()
    ]

    # Doc2Vec model
    model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Generate document vectors
    document_vectors = [
        model.infer_vector(row[tokens_column]) 
        for _, row in df.iterrows()
    ]

    # Normalize vectors to L2 norm for cosine similarity
    normalized_vectors = normalize(document_vectors, norm='l2')

    # Save 
    pd.DataFrame(normalized_vectors).to_csv(file_path, index=False)
    print(f"Normalized vectors saved to '{file_path}'.")

    return normalized_vectors

def save_vectors(data, base_dir, name):
    """
    Save embeddings in a CSV file.

    Args:
        data (pd.DataFrame or np.ndarray): Data containing the embeddings to save.
        base_dir (str): Base directory to save the file.
        name (str): File name (without extension).

    Returns:
        None
    """
    output_file = os.path.join(base_dir, "results", f"{name}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(data, columns=[f"embedding_{i}" for i in range(len(data[0]))])

    data.to_csv(output_file, index=False)
    print(f"Vectors saved to: {output_file}")

