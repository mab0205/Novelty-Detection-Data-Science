import spacy
import re

nlp = spacy.load("en_core_web_sm")

def lemmatize_text(df_news):
    # Asegurarse de que todas las entradas sean cadenas
    df_news['content_no_sw'] = df_news['content_no_sw'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

    # Función de lematización
    def aux(text):
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_stop])
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Quita espacios extra
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Quita caracteres especiales
        return text

    # Aplicar la lematización
    df_news['content_no_sw'] = df_news['content_no_sw'].apply(aux)
    df_news['content_no_sw'] = df_news['content_no_sw'].apply(clean_text)

    return df_news