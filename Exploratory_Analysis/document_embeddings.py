from sentence_transformers import SentenceTransformer

def generate_embeddings(df, text_column, model_name='all-MiniLM-L6-v2'):
    """
    Generate document embeddings using Sentence-BERT.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    text_column (str): Column containing the text to generate embeddings from.
    model_name (str): Pre-trained Sentence-BERT model to use.

    Returns:
    np.ndarray: Array of embeddings for the input text.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[text_column].tolist(), convert_to_tensor=True)
    return embeddings