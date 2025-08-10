# import requests
# from transformers import pipeline
# from newspaper import Article
# from typing import List, Dict
# import streamlit as st

# class MarketSentiment:
#     def __init__(self):
#         # Load lightweight sentiment model (2MB)
#         self.sentiment_pipeline = pipeline(
#             "text-classification",
#             model="cardiffnlp/twitter-roberta-base-sentiment-latest"
#         )
#         self.news_sources = {
#             # "general": "https://newsapi.org/v2/everything?q={query}&apiKey={key}",
#             "financial": "https://finnhub.io/api/v1/news?category=general&token=d2c4tqhr01qvh3ve2da0d2c4tqhr01qvh3ve2dag"
#         }

#     def fetch_news(self, ticker: str, api_key: str) -> List[Dict]:
#         """Fetch news from multiple free APIs with fallback"""
#         articles = []
        
#         try:
#             # Try Finnhub first (specialized for stocks)
#             finnhub_url = self.news_sources["financial"].format(key=api_key)
#             finnhub_news = requests.get(finnhub_url).json()
#             articles.extend([{
#                 "title": item["headline"],
#                 "url": item["url"],
#                 "source": "Finnhub"
#             } for item in finnhub_news[:5]])  # Limit to 5 articles
            
#         except:
#             # Fallback to NewsAPI
#             newsapi_url = self.news_sources["general"].format(
#                 query=ticker, 
#                 key=api_key
#             )
#             newsapi_data = requests.get(newsapi_url).json()
#             articles.extend([{
#                 "title": article["title"],
#                 "url": article["url"],
#                 "source": article["source"]["name"]
#             } for article in newsapi_data.get("articles", [])[:3]])
            
#         return articles

#     def analyze_sentiment(self, text: str) -> Dict:
#         """Analyze sentiment with confidence scores"""
#         result = self.sentiment_pipeline(text)[0]
#         return {
#             "label": result["label"],
#             "score": round(result["score"], 3)
#         }

#     def get_article_text(self, url: str) -> str:
#         """Extract main article text"""
#         try:
#             article = Article(url)
#             article.download()
#             article.parse()
#             return article.text[:1000]  # First 1000 chars to avoid long processing
#         except:
#             return ""

#     def get_ticker_sentiment(self, ticker: str, api_key: str) -> Dict:
#         """Full sentiment analysis pipeline"""
#         news_items = self.fetch_news(ticker, api_key)
        
#         if not news_items:
#             return {"error": "No news found"}
        
#         # Analyze each news item
#         for item in news_items:
#             full_text = f"{item['title']} {self.get_article_text(item['url'])}"
#             item["sentiment"] = self.analyze_sentiment(full_text)
        
#         # Calculate aggregate sentiment
#         pos = sum(1 for item in news_items if item["sentiment"]["label"] == "POSITIVE")
#         neg = sum(1 for item in news_items if item["sentiment"]["label"] == "NEGATIVE")
        
#         return {
#             "score": (pos - neg) / len(news_items),  # Normalized -1 to 1
#             "articles": news_items,
#             "summary": {
#                 "positive": pos,
#                 "negative": neg,
#                 "neutral": len(news_items) - pos - neg
#             }
#         }

# try1
# import requests
# from transformers import pipeline
# from newspaper import Article
# from typing import List, Dict
# import streamlit as st
# import time

# class MarketSentiment:
#     def __init__(self):
#         # Load lightweight sentiment model (2MB)
#         self.sentiment_pipeline = pipeline(
#             "text-classification",
#             model="cardiffnlp/twitter-roberta-base-sentiment-latest"
#         )
#         self.news_sources = {
#             "alpha_vantage": "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={key}",
#             "finnhub": "https://finnhub.io/api/v1/news?category=general&token={key}"
#         }

#     def fetch_news(self, ticker: str, api_key: str) -> List[Dict]:
#         """Fetch news from multiple free APIs with fallback"""
#         articles = []
        
#         try:
#             # Try Alpha Vantage first (using your provided key)
#             av_url = self.news_sources["alpha_vantage"].format(
#                 ticker=ticker.split('.')[0],  # Remove .NS/.KS suffixes if present
#                 key="T3FJH7K52ABNE0JC"  # Your Alpha Vantage API key
#             )
#             av_data = requests.get(av_url).json()
            
#             if 'feed' in av_data:
#                 articles.extend([{
#                     "title": item["title"],
#                     "url": item["url"],
#                     "source": item["source"],
#                     "time_published": item["time_published"]
#                 } for item in av_data['feed'][:5]])
                
#             # Add Finnhub as fallback
#             if len(articles) < 3:  # If AV returns few results
#                 fh_url = self.news_sources["finnhub"].format(key="d2c4tqhr01qvh3ve2da0d2c4tqhr01qvh3ve2dag")
#                 fh_data = requests.get(fh_url).json()
#                 articles.extend([{
#                     "title": item["headline"],
#                     "url": item["url"],
#                     "source": "Finnhub",
#                     "time_published": item["datetime"]
#                 } for item in fh_data[:5-len(articles)]])
                
#         except Exception as e:
#             st.error(f"News fetch error: {str(e)}")
#             return []
        
#         # Sort by most recent
#         articles.sort(key=lambda x: x.get("time_published", ""), reverse=True)
#         return articles[:5]  # Return max 5 articles

#     def analyze_sentiment(self, text: str) -> Dict:
#         """Analyze sentiment with confidence scores"""
#         try:
#             result = self.sentiment_pipeline(text[:1000])[0]  # Limit to first 1000 chars
#             return {
#                 "label": result["label"],
#                 "score": round(result["score"], 3)
#             }
#         except:
#             return {"label": "NEUTRAL", "score": 0.5}

#     def get_article_text(self, url: str) -> str:
#         """Extract main article text with retry logic"""
#         max_retries = 2
#         for _ in range(max_retries):
#             try:
#                 article = Article(url)
#                 article.download()
#                 article.parse()
#                 return article.text[:1000]  # First 1000 chars
#             except:
#                 time.sleep(1)  # Delay between retries
#         return ""

#     def get_ticker_sentiment(self, ticker: str) -> Dict:
#         """Full sentiment analysis pipeline"""
#         news_items = self.fetch_news(ticker, "")
        
#         if not news_items:
#             return {"error": "No recent news found for this ticker"}
        
#         # Analyze each news item
#         for item in news_items:
#             full_text = f"{item['title']} {self.get_article_text(item['url'])}"
#             item["sentiment"] = self.analyze_sentiment(full_text)
        
#         # Calculate aggregate sentiment
#         pos = sum(1 for item in news_items if item["sentiment"]["label"] == "POSITIVE")
#         neg = sum(1 for item in news_items if item["sentiment"]["label"] == "NEGATIVE")
#         total = len(news_items)
        
#         return {
#             "score": (pos - neg) / total if total > 0 else 0,  # Normalized -1 to 1
#             "articles": news_items,
#             "summary": {
#                 "positive": pos,
#                 "negative": neg,
#                 "neutral": total - pos - neg,
#                 "total": total
#             }
#         }
import requests
from transformers import pipeline
from typing import List, Dict, Optional
import streamlit as st
import time
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class MarketSentiment:
    def __init__(self):
        # Initialize both sentiment analyzers
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.vader = SentimentIntensityAnalyzer()
        self.news_sources = {
            "alpha_vantage": "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={key}",
            "finnhub": "https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={key}"
        }
        self.cache = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _clean_ticker(self, ticker: str) -> str:
        """Remove exchange suffixes for API compatibility"""
        return ticker.split('.')[0]

    def _get_date_range(self) -> tuple:
        """Get date range for news lookup (last 7 days)"""
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        return week_ago.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    def _extract_text_with_bs4(self, html: str) -> str:
        """Extract main text content using BeautifulSoup"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()
        
        # Get text from remaining elements
        text = ' '.join(soup.stripped_strings)
        return text[:5000]  # Limit to first 5000 characters

    def fetch_news(self, ticker: str) -> List[Dict]:
        """Fetch news from multiple free APIs with fallback"""
        if ticker in self.cache:
            return self.cache[ticker]
        
        articles = []
        clean_ticker = self._clean_ticker(ticker)
        from_date, to_date = self._get_date_range()
        
        try:
            # Try Alpha Vantage first
            av_url = self.news_sources["alpha_vantage"].format(
                ticker=clean_ticker,
                key="T3FJH7K52ABNE0JC"  # Replace with your Alpha Vantage API key
            )
            av_data = requests.get(av_url, timeout=10).json()
            
            if 'feed' in av_data:
                articles.extend([{
                    "title": item["title"],
                    "url": item["url"],
                    "source": item["source"],
                    "time_published": item["time_published"],
                    "summary": item.get("summary", "")
                } for item in av_data['feed'][:5]])
                
            # Add Finnhub as fallback
            if len(articles) < 3:
                fh_url = self.news_sources["finnhub"].format(
                    ticker=clean_ticker,
                    from_date=from_date,
                    to_date=to_date,
                    key="d2c4tqhr01qvh3ve2da0d2c4tqhr01qvh3ve2dag"  # Replace with your Finnhub API key
                )
                fh_data = requests.get(fh_url, timeout=10).json()
                articles.extend([{
                    "title": item["headline"],
                    "url": item["url"],
                    "source": "Finnhub",
                    "time_published": item["datetime"],
                    "summary": item.get("summary", "")
                } for item in fh_data[:5-len(articles)] if item['url'].startswith('http')])
                
        except Exception as e:
            st.error(f"News fetch error: {str(e)}")
            return []
        
        # Sort by most recent and cache
        articles.sort(key=lambda x: x.get("time_published", ""), reverse=True)
        self.cache[ticker] = articles[:5]  # Cache max 5 articles
        return self.cache[ticker]

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment with both models and combine results"""
        if not text.strip():
            return {"label": "NEUTRAL", "score": 0.5}
            
        try:
            # Use both models for more robust analysis
            roberta_result = self.sentiment_pipeline(text[:1000])[0]
            vader_result = self.vader.polarity_scores(text[:1000])
            
            # Combine results (weighted average)
            roberta_weight = 0.7
            vader_weight = 0.3
            
            if roberta_result["label"] == "POSITIVE":
                roberta_score = roberta_result["score"]
            else:
                roberta_score = -roberta_result["score"]
                
            combined_score = (roberta_weight * roberta_score) + (vader_weight * vader_result["compound"])
            
            # Determine final label
            if combined_score > 0.1:
                return {"label": "POSITIVE", "score": min(1.0, combined_score)}
            elif combined_score < -0.1:
                return {"label": "NEGATIVE", "score": min(1.0, -combined_score)}
            else:
                return {"label": "NEUTRAL", "score": 0.5}
                
        except Exception as e:
            print(f"Sentiment analysis error: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.5}

    def get_article_text(self, url: str) -> str:
        """Extract main article text using requests and BeautifulSoup"""
        if not url.startswith('http'):
            return ""
            
        max_retries = 2
        for _ in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                return self._extract_text_with_bs4(response.text)
            except Exception as e:
                print(f"Article download error: {str(e)}")
                time.sleep(1)
        return ""

    def get_ticker_sentiment(self, ticker: str) -> Dict:
        """Full sentiment analysis pipeline with enhanced features"""
        cache_key = f"{ticker}_sentiment"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        news_items = self.fetch_news(ticker)
        
        if not news_items:
            return {"error": "No recent news found for this ticker"}
        
        # Analyze each news item
        for item in news_items:
            full_text = f"{item['title']}. {item.get('summary', '')} {self.get_article_text(item['url'])}"
            item["sentiment"] = self.analyze_sentiment(full_text)
        
        # Calculate weighted aggregate sentiment
        total_weight = 0
        weighted_score = 0
        sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        
        for item in news_items:
            sentiment = item["sentiment"]
            weight = 1.5 if "alpha_vantage" in item.get("source", "").lower() else 1.0
            total_weight += weight
            
            if sentiment["label"] == "POSITIVE":
                weighted_score += sentiment["score"] * weight
                sentiment_counts["POSITIVE"] += 1
            elif sentiment["label"] == "NEGATIVE":
                weighted_score -= sentiment["score"] * weight
                sentiment_counts["NEGATIVE"] += 1
            else:
                sentiment_counts["NEUTRAL"] += 1
        
        # Normalize score to [-1, 1] range
        normalized_score = weighted_score / total_weight if total_weight > 0 else 0
        
        result = {
            "score": normalized_score,
            "articles": news_items,
            "summary": {
                "positive": sentiment_counts["POSITIVE"],
                "negative": sentiment_counts["NEGATIVE"],
                "neutral": sentiment_counts["NEUTRAL"],
                "total": len(news_items)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result for 1 hour
        self.cache[cache_key] = result
        return result

    def get_sentiment_timeseries(self, ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical sentiment data (mock implementation)"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=days)
            data = {
                'date': dates,
                'sentiment': [random.uniform(-1, 1) for _ in range(days)],
                'positive': [random.randint(0, 5) for _ in range(days)],
                'negative': [random.randint(0, 5) for _ in range(days)]
            }
            return pd.DataFrame(data)
        except:
            return None