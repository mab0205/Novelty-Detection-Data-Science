import os
import nltk
from nltk.tokenize import word_tokenize

# Descargar stopwords y configurar path de datos de nltk
nltk.download('stopwords')
nltk.download('punkt')
base_dir = os.getcwd()
corpus_dir = os.path.join(base_dir, 'nltk_data')

# Lista de stopwords en inglés
stopwords = nltk.corpus.stopwords.words('english')
print("First 10 stopwords:", stopwords[:10])

def tokenize_and_remove_punctuation(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    
    return tokens

def remove_stopwords(txt_tokenized):
    return [word for word in txt_tokenized if word.lower() not in stopwords]
