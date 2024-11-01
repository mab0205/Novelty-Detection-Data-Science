import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def len_words_distribution(df_source, df_target ):
    plt.figure(figsize=(12, 6))
    sns.histplot(df_source['words'], color='blue', label='Source', kde=True, bins=20)
    sns.histplot(df_target['words'], color='orange', label='Target', kde=True, bins=20)
    plt.title('Distribución de la longitud de palabras en Source y Target')
    plt.xlabel('Cantidad de palabras')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()

def count_sentences_distribution(df_source, df_target ):
    plt.figure(figsize=(12, 6))
    sns.histplot(df_source['sentence'], color='blue', label='Source', kde=True, bins=20)
    sns.histplot(df_target['sentence'], color='orange', label='Target', kde=True, bins=20)
    plt.title('Distribución de la cantidad de oraciones en Source y Target')
    plt.xlabel('Cantidad de oraciones')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()

def publisher_distribution(df_news):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_news, x='publisher', hue='is_source', palette='Set1')
    plt.title('Distribución de Publisher en Source y Target')
    plt.xlabel('Publisher')
    plt.ylabel('Frecuencia')
    plt.legend(title='Source', labels=['Source', 'Target'])
    plt.xticks(rotation=45)
    plt.show()
