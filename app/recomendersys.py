from typing import Dict, List
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import nltk


nltk.download('punkt_tab')
nltk.download('stopwords')

stopwords_list = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by:
    - Lowercasing
    - Removing special characters and numbers
    - Tokenizing
    - Removing stopwords
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords_list]
    return ' '.join(filtered_tokens)