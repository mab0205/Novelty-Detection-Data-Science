from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

def plot_word_clouds_by_topic(df, token_column='tokens', topic_column='topic', max_words=100):
    """
    Plots word clouds by topic based on token frequencies.

    Parameters:
    df (pd.DataFrame): DataFrame containing tokens and topics.
    token_column (str): The column with tokens lists.
    topic_column (str): The column with topic labels.
    max_words (int): The maximum number of words in the word cloud.
    """
    if token_column not in df.columns:
        raise ValueError(f"Column '{token_column}' not found in DataFrame. Ensure it exists after tokenization.")
    
    topics = df[topic_column].unique()
    
    for topic in topics:
        # Filter DataFrame for the current topic
        df_topic = df[df[topic_column] == topic]
        
        # Flatten the list of tokens for the current topic
        all_tokens = [token for tokens in df_topic[token_column] for token in tokens]
        
        # Count the frequency of each token
        token_counts = Counter(all_tokens)
        
        # Generate and plot the word cloud
        wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white')
        wordcloud.generate_from_frequencies(token_counts)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for Topic: {topic}")
        plt.show()
