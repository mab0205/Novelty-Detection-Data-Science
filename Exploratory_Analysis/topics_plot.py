import matplotlib.pyplot as plt

def plot_topic_counts(topic_counts):
    """
    Plots a bar chart for the number of documents per topic.

    Parameters:
    topic_counts (dict): A dictionary where keys are topics and values are the number of documents per topic.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(topic_counts.keys(), topic_counts.values())
    plt.xlabel('Topics')
    plt.ylabel('Number of Documents')
    plt.title('Number of Documents per Topic')
    plt.xticks(rotation=45)
    plt.show()

