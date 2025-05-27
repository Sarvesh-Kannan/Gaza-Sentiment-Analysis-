import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
def download_nltk_resources():
    """Download required NLTK resources"""
    print("Downloading NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Preprocessing function
def preprocess_text(text):
    """
    Preprocess text by performing the following operations:
    - Convert to lowercase
    - Remove URLs
    - Remove punctuation
    - Remove numbers
    - Tokenize
    - Remove stopwords
    - Lemmatize
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Combine text columns into a single text field
def combine_text_columns(df, text_columns):
    """
    Combine multiple text columns into a single text field
    
    Args:
        df (pandas.DataFrame): DataFrame containing text columns
        text_columns (list): List of column names to combine
        
    Returns:
        pandas.Series: Combined text
    """
    return df[text_columns].apply(
        lambda row: ' '.join(str(cell) for cell in row if isinstance(cell, (str, int, float)) and pd.notna(cell) and str(cell).strip() != ''), 
        axis=1
    )

# Calculate text statistics
def calculate_text_statistics(df, text_column='combined_text'):
    """
    Calculate basic text statistics (length and word count)
    
    Args:
        df (pandas.DataFrame): DataFrame containing text
        text_column (str): Column name containing text
        
    Returns:
        pandas.DataFrame: DataFrame with added statistics columns
    """
    df['text_length'] = df[text_column].apply(len)
    df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
    return df 