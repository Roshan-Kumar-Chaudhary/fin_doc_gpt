# try1
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class MarketDataFetcher:
    def __init__(self):
        self.ticker_cache = {}  

    def get_historical_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch historical market data with timezone handling"""
        if ticker in self.ticker_cache:
            return self.ticker_cache[ticker]
        
        data = yf.Ticker(ticker).history(period=period)
        
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        self.ticker_cache[ticker] = data
        return data

    def get_key_metrics(self, ticker: str) -> dict:
        """Get fundamental analysis metrics"""
        stock = yf.Ticker(ticker)
        try:
            hist = stock.history(period='1d')
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
        except:
            current_price = None
            
        return {
            'current_price': current_price,
            'pe_ratio': stock.info.get('trailingPE', None),
            'market_cap': stock.info.get('marketCap', None),
            '52_week_high': stock.info.get('fiftyTwoWeekHigh', None),
            '52_week_low': stock.info.get('fiftyTwoWeekLow', None)
        }