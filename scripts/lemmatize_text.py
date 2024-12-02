import spacy
import re
import os
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def lemmatize_text(df_news, base_dir):
    output_dir = os.path.join(base_dir, "results") # results exist? 
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"lemmatize_text.csv")

    if os.path.exists(file_path):
        print(f"File '{file_path}' exists...")
        df_news = pd.read_csv(file_path)

        return df_news
    
    else:    
        df_news['content_no_sw'] = df_news['content_no_sw'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        
        def aux(text):
            doc = nlp(text)
            return ' '.join([token.lemma_ for token in doc if not token.is_stop])
        
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'\s+', ' ', text)  
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
            return text

        df_news['content_no_sw'] = df_news['content_no_sw'].apply(aux)
        df_news['content_no_sw'] = df_news['content_no_sw'].apply(clean_text)

        df_news.to_csv(file_path, index=False)
        return df_news