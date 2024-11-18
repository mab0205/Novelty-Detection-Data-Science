import os
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pyLDAvis.gensim
import pickle
import pyLDAvis

nltk.download('punkt')
nltk.download('stopwords')

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
    """
    coherence_model = CoherenceModel(
        model=lda_model, texts=texts, corpus=corpus, dictionary=dictionary, coherence=coherence_type
    )
    return coherence_model.get_coherence()


def lda_on_all_documents(df, text_column='content_no_sw', num_topics=11):
    """
    Applies LDA on the entire dataset and returns a DataFrame with topic distributions.
    """
    df['tokens'] = tokenize_text(df, text_column)
    dictionary = create_dictionary(df['tokens'])
    corpus = create_corpus(dictionary, df['tokens'])
    lda_model = train_lda_model(corpus, dictionary, num_topics=num_topics)
    topic_distributions = get_topic_distributions(lda_model, corpus, num_topics=num_topics)
    df_topics = pd.DataFrame(topic_distributions)
    final_df = pd.concat([df.reset_index(drop=True), df_topics], axis=1)

    # Evaluate coherence
    coherence_score = evaluate_coherence(lda_model, corpus, dictionary, df['tokens'], coherence_type='c_v')
    print(f'Coherence Score: {coherence_score}')

    return final_df, lda_model, corpus, dictionary


def preprocess_text_dual(df, text_column, custom_stopwords):
    """
    Preprocess text: Keep two versions of the text - one with stopwords removed and one full text.
    """
    stopwords_set = set(stopwords.words('english')) | set(custom_stopwords)
    df['cleaned_text'] = df[text_column].apply(
        lambda text: ' '.join(word for word in text.split() if word not in stopwords_set)
    )
    df['full_text'] = df[text_column]
    return df


def visualize_lda(lda_model, corpus, id2word, num_topics, output_dir='./results'):
    """
    Visualize LDA topics and save the visualization to an HTML file.

    Parameters:
    lda_model (gensim.models.LdaModel): Trained LDA model.
    corpus (list): Bag-of-words representation of the corpus.
    id2word (gensim.corpora.Dictionary): Gensim dictionary for the corpus.
    num_topics (int): Number of topics in the LDA model.
    output_dir (str): Directory to save the visualization.

    Returns:
    pyLDAvis.PreparedData: Prepared visualization data.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Filepaths for saving the visualization
    ldavis_filepath = os.path.join(output_dir, f'ldavis_prepared_{num_topics}.pkl')
    html_filepath = os.path.join(output_dir, f'ldavis_prepared_{num_topics}.html')

    # Prepare the LDA visualization
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

    # Save the visualization data for reuse
    with open(ldavis_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

    # Save the visualization as an HTML file
    pyLDAvis.save_html(LDAvis_prepared, html_filepath)

    return LDAvis_prepared
