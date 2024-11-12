import os
import pandas as pd
import xml.etree.ElementTree as ET

class CorpusParser:
    """
    A class to parse and load `.txt` documents and their associated `.xml` metadata from a corpus.
    
    Steps:
        1. Parse and load the `.txt` documents and their associated `.xml` metadata.
        2. Organize the information into a DataFrame, with columns representing both the content and metadata of the news articles.
        3. Save the data in a structured format (CSV) to facilitate analysis and model training for novelty detection.
    """

    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.data = []

    def parse(self):
        """
        Parses the news corpus, loading both text and metadata into a DataFrame,
        and saves the DataFrame as a CSV file.
        
        Returns:
            pd.DataFrame: A DataFrame containing all news articles and metadata.
        """
        # Itera sobre todas as categorias (pastas) no nível mais alto do diretório do corpus
        for category in os.listdir(self.corpus_dir):
            category_path = os.path.join(self.corpus_dir, category)
            
            # Verifica se é uma pasta
            if os.path.isdir(category_path):
                for event_id in os.listdir(category_path):
                    event_path = os.path.join(category_path, event_id)

                    for folder in ['source', 'target']:
                        folder_path = os.path.join(event_path, folder)
                        if os.path.exists(folder_path):
                            is_source = folder == 'source'
                            for file_name in os.listdir(folder_path):
                                if file_name.endswith('.txt'):
                                    news_id = file_name.split('.')[0]
                                    txt_path = os.path.join(folder_path, file_name)
                                    
                                    # Carrega o conteúdo do arquivo .txt
                                    content = self._load_content(txt_path)
                                    
                                    # Valores padrão de metadados
                                    metadata = {
                                        'DOP': None,
                                        'publisher': None,
                                        'title': None,
                                        'eventid': None,
                                        'eventname': None,
                                        'topic': None,
                                        'sentence': None,
                                        'words': None,
                                        'sourceid': None,
                                        'DLA': None,
                                        'SLNS': None
                                    }
                                    
                                    # Procura o arquivo .xml correspondente
                                    xml_file = os.path.join(folder_path, f"{news_id}.xml")
                                    if os.path.exists(xml_file):
                                        metadata = self._load_metadata(xml_file, metadata)
                                    
                                    # Adiciona os dados à lista de dados
                                    self.data.append({
                                        'category': category,
                                        'event_id': event_id,
                                        'news_id': news_id,
                                        'content': content,
                                        'is_source': is_source,
                                        **metadata
                                    })

        # Converte a lista de dados para um DataFrame
        df = pd.DataFrame(self.data)
        
        # Define os tipos de dados desejados para cada coluna
        dtype_dict = {
            'category': 'string',
            'event_id': 'string',
            'news_id': 'string',
            'content': 'string',
            'is_source': 'bool',
            'DOP': 'string',  # Ou 'datetime64[ns]' se quiserem como data
            'publisher': 'string',
            'title': 'string',
            'eventid': 'string',
            'eventname': 'string',
            'topic': 'string',
            'sentence': 'Int64',  # Inteiro anulável
            'words': 'Int64',     # Inteiro anulável
            'sourceid': 'string',
            'DLA': 'string',
            'SLNS': 'string'
        }
        
        # Aplica os tipos de dados ao DataFrame
        df = df.astype(dtype_dict)

        output_path = os.path.join(os.getcwd(), 'docs', 'corpus_all_categories.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return df

    def _load_content(self, txt_path):
        """Load the content of a .txt file with error handling for encoding."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(txt_path, 'r', encoding='ISO-8859-1') as f:
                content = f.read()
        return content

    def _load_metadata(self, xml_file, metadata):
        """Parse an XML file to extract metadata and update the metadata dictionary."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for feature in root.findall('feature'):
            for key, value in feature.attrib.items():
                metadata[key] = value
        return metadata
