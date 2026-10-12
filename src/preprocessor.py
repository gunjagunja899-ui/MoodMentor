"""
Text Preprocessing Module for MoodMentor
Handles text cleaning, tokenization, stop-word removal, and lemmatization.
"""

import re
import string
from typing import Dict, List, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_resources():
    """Ensure required NLTK corpora are downloaded."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)


_ensure_nltk_resources()


class TextPreprocessor:
    """
    Cleans, normalizes, tokenizes, and lemmatizes input text.
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

        # Retain certain emotion-carrying words in sentiment analysis context if needed
        # but for pure stopword removal we keep standard set
        self.punctuation_table = str.maketrans('', '', string.punctuation)

    def clean_text(self, text: str) -> str:
        """
        Removes URLs, HTML tags, special symbols, and collapses whitespace.
        Preserves core punctuation meaning for sentences where needed.
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove HTML tags
        cleaned = re.sub(r'<.*?>', ' ', text)
        # Remove URLs
        cleaned = re.sub(r'http\S+|www\.\S+', ' ', cleaned)
        # Remove emails
        cleaned = re.sub(r'\S+@\S+', ' ', cleaned)
        # Remove excessive repeated symbols (e.g. !!! -> !)
        cleaned = re.sub(r'([!?.])\1+', r'\1', cleaned)
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def process(self, text: str) -> Dict[str, Any]:
        """
        Performs full preprocessing pipeline on input text:
        1. Validation
        2. Cleaning
        3. Tokenization (sentences & words)
        4. Stop-word removal
        5. Lemmatization
        6. Statistical summary

        Returns:
            Dict containing cleaned text, tokens, lemmatized tokens, and text statistics.
        """
        if not text or not text.strip():
            return {
                "is_empty": True,
                "raw_text": "",
                "clean_text": "",
                "sentences": [],
                "tokens": [],
                "filtered_tokens": [],
                "lemmatized_tokens": [],
                "processed_string": "",
                "stats": {
                    "word_count": 0,
                    "char_count": 0,
                    "sentence_count": 0,
                    "unique_words": 0
                }
            }

        cleaned = self.clean_text(text)

        # Sentence tokenization
        try:
            sentences = sent_tokenize(cleaned)
        except Exception:
            sentences = [s.strip() for s in cleaned.split('.') if s.strip()]

        # Word tokenization
        try:
            raw_tokens = word_tokenize(cleaned)
        except Exception:
            raw_tokens = cleaned.split()

        # Lowercase tokens and filter
        tokens = [t.lower() for t in raw_tokens]

        # Filter stop words and punctuation for lexical analysis
        filtered_tokens = [
            t for t in tokens
            if t.isalpha() and t not in self.stop_words
        ]

        # Lemmatization
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(t)
            for t in filtered_tokens
        ]

        processed_string = " ".join(lemmatized_tokens)

        unique_words = len(set(filtered_tokens))
        stats = {
            "word_count": len(tokens),
            "char_count": len(text),
            "sentence_count": len(sentences),
            "unique_words": unique_words
        }

        return {
            "is_empty": False,
            "raw_text": text,
            "clean_text": cleaned,
            "sentences": sentences,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "processed_string": processed_string,
            "stats": stats
        }
