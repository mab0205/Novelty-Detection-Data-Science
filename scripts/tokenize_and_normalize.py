import nltk
from nltk.tokenize import word_tokenize

# Descargar stopwords y configurar path de datos de nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.data.path.append('/home/mab0205/respaldos_linux_10_2024/UTFPR/semestre 2024.2/Ciencias de dados 2/nltk_data')

# Lista de stopwords en inglés
stopwords = nltk.corpus.stopwords.words('english')
print("First 10 stopwords:", stopwords[:10])

def tokenize_and_remove_punctuation(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    
    return tokens

def remove_stopwords(txt_tokenized):
    return [word for word in txt_tokenized if word.lower() not in stopwords]
