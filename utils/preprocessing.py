# from nltk.sentiment import SentimentIntensityAnalyzer
# import nltk
# nltk.download('vader_lexicon')

# def analyze_sentiment(text):
#     """Enhanced financial sentiment analysis"""
#     sia = SentimentIntensityAnalyzer()
    
#     # Financial-specific word weighting
#     financial_lexicon = {
#         'profit': 2.0,
#         'growth': 1.5,
#         'loss': -2.0,
#         'decline': -1.5,
#         'revenue': 1.2,
#         'earnings': 1.3
#     }
    
#     sia.lexicon.update(financial_lexicon)
#     sentiment = sia.polarity_scores(text)
    
#     # Custom thresholds for financial documents
#     if sentiment['compound'] >= 0.1:
#         sentiment_label = "Positive"
#     elif sentiment['compound'] <= -0.1:
#         sentiment_label = "Negative"
#     else:
#         sentiment_label = "Neutral"
    
#     return {
#         "sentiment": sentiment_label,
#         "score": sentiment['compound'],
#         "positive": sentiment['pos'],
#         "negative": sentiment['neg'],
#         "neutral": sentiment['neu']
#     }

# try1
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from typing import Dict
import re

nltk.download('vader_lexicon')

def analyze_sentiment(text: str) -> Dict:
    """Enhanced financial sentiment analysis with section-specific weighting"""
    sia = SentimentIntensityAnalyzer()
    
    # Extended financial lexicon with weights
    financial_lexicon = {
        'profit': 2.0, 'growth': 1.5, 'increase': 1.3, 'strong': 1.2,
        'loss': -2.0, 'decline': -1.5, 'decrease': -1.3, 'weak': -1.2,
        'revenue': 1.2, 'earnings': 1.3, 'margin': 1.1, 
        'debt': -1.2, 'risk': -1.5, 'uncertainty': -1.7,
        'dividend': 1.1, 'buyback': 1.1, 'guidance': 1.4
    }
    
    # Update lexicon
    sia.lexicon.update(financial_lexicon)
    
    # Preprocess text - remove numbers and special chars that might confuse sentiment
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Get sentiment scores
    sentiment = sia.polarity_scores(clean_text)
    
    # Financial-specific thresholds
    if sentiment['compound'] >= 0.15:  # Higher threshold for positive
        sentiment_label = "Positive"
    elif sentiment['compound'] <= -0.15:  # Lower threshold for negative
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    return {
        "sentiment": sentiment_label,
        "score": round(sentiment['compound'], 3),
        "positive": round(sentiment['pos'], 3),
        "negative": round(sentiment['neg'], 3),
        "neutral": round(sentiment['neu'], 3),
        "keywords": [word for word in financial_lexicon if word in text.lower()]
    }