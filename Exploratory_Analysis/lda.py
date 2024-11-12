import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import CoherenceModel

nltk.download('punkt')

def tokenize_text(df, text_column):
    """Tokenizes the text in the specified column."""
    return df[text_column].apply(word_tokenize)

def create_dictionary(tokenized_texts):
    """Creates a dictionary from tokenized texts and filters extremes."""
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    return dictionary

def create_corpus(dictionary, tokenized_texts):
    """Converts tokenized texts to a bag-of-words corpus using the dictionary."""
    return [dictionary.doc2bow(text) for text in tokenized_texts]

def train_lda_model(corpus, dictionary, num_topics=11):
    """Trains an LDA model on the corpus with a fixed number of topics (default=11)."""
    return LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)

def get_topic_distributions(lda_model, corpus, num_topics=11):
    """Gets topic distribution for each document in the corpus, ensuring `num_topics` topics."""
    topic_distributions = []
    for doc in corpus:
        topic_dist = {f"topic_{i}": 0 for i in range(num_topics)}  # Initialize with 0s
        for topic_id, prob in lda_model.get_document_topics(doc, minimum_probability=0):
            topic_dist[f"topic_{topic_id}"] = prob
        topic_distributions.append(topic_dist)
    return topic_distributions

def evaluate_coherence(lda_model, corpus, dictionary, texts, coherence_type='c_v'):
    """
    Evaluates the coherence score of an LDA model.

    Parameters:
    lda_model (LdaModel): The trained LDA model.
    corpus (list of list of tuples): The bag-of-words representation of the corpus.
    dictionary (Dictionary): The dictionary used to build the LDA model.
    texts (list of list of str): The original texts tokenized (required for c_v coherence).
    coherence_type (str): The type of coherence measure to use. Options are 'c_v', 'u_mass', 'c_npmi', 'c_uci'.

    Returns:
    float: The coherence score of the model.
    """
    coherence_model = CoherenceModel(model=lda_model, texts=texts, corpus=corpus, dictionary=dictionary, coherence=coherence_type)
    coherence_score = coherence_model.get_coherence()
    return coherence_score


def lda_on_all_documents(df, text_column='content_no_sw', num_topics=11):
    """
    Applies LDA on the entire dataset with a specified number of topics and returns a DataFrame with topic distributions.

    Parameters:
    df (pd.DataFrame): DataFrame containing text data.
    text_column (str): The column name with text data without stop words.
    num_topics (int): The number of topics for the LDA model.

    Returns:
    pd.DataFrame: DataFrame with topic distributions for each document.
    """
    # Tokenize all documents
    df['tokens'] = tokenize_text(df, text_column)
    
    # Create dictionary and corpus from all tokens
    dictionary = create_dictionary(df['tokens'])
    corpus = create_corpus(dictionary, df['tokens'])

    # Train LDA model on all documents
    lda_model = train_lda_model(corpus, dictionary, num_topics=num_topics)

    # Get topic distributions for each document
    topic_distributions = get_topic_distributions(lda_model, corpus, num_topics=num_topics)
    
    # Create DataFrame from topic distributions
    df_topics = pd.DataFrame(topic_distributions)
    
    # Concatenate original DataFrame with topic distributions
    final_df = pd.concat([df.reset_index(drop=True), df_topics], axis=1)

        # Evaluar la coherencia de un modelo LDA entrenado
    coherence_score = evaluate_coherence(lda_model, corpus, dictionary, df['tokens'], coherence_type='c_v')
    print(f'Coherence Score: {coherence_score}')

    
    return final_df