
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from prophet import Prophet
from pandas.api.types import is_datetime64_any_dtype
import plotly.graph_objects as go
import yfinance as yf

class FinancialForecaster:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}

    def fetch_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Fetch historical market data from Yahoo Finance"""
        df = yf.download(ticker, period=period)
        df.dropna(inplace=True)
        return df

    def forecast_prophet(self, data: pd.DataFrame, periods: int = 30, sentiment_weight: float = 0) -> pd.DataFrame:
        """Time series forecasting using Prophet with sentiment adjustment"""
        df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Ensure tz-naive datetimes
        if is_datetime64_any_dtype(df['ds']):
            try:
                df['ds'] = df['ds'].dt.tz_localize(None)
            except Exception:
                # If already tz-naive this will raise; ignore safely
                pass
        
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Add sentiment as an extra regressor if significant
        if abs(sentiment_weight) > 0.2:
            df['sentiment'] = np.linspace(0, sentiment_weight, len(df))
            model.add_regressor('sentiment')
        
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods)
        
        # Include sentiment in future predictions
        if abs(sentiment_weight) > 0.2:
            future['sentiment'] = np.linspace(sentiment_weight, sentiment_weight * 0.9, len(future))
        
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def forecast_movement(self, data: pd.DataFrame, sentiment_weight: float = 0) -> tuple:
        """Predict next day movement with explainability and sentiment adjustment"""
        df = data.copy()
        
        # Create technical indicators
        df['5d_ma'] = df['Close'].rolling(5).mean()
        df['20d_ma'] = df['Close'].rolling(20).mean()
        df['Trend'] = (df['Close'] > df['20d_ma']).astype(int)
        df['Volume_Spike'] = (df['Volume'] > 1.5 * df['Volume'].rolling(20).mean()).astype(int)
        df['Near_High'] = ((df['High'] - df['Close']) < (0.002 * df['Close'])).astype(int)
        df['3d_momentum'] = df['Close'].pct_change(3)
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        
        # Add sentiment as a feature if significant
        if abs(sentiment_weight) > 0.2:
            df['Sentiment'] = np.linspace(0, sentiment_weight, len(df))
        
        # Prepare target
        df['Tomorrow'] = df['Close'].shift(-1)
        df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
        df = df.dropna()
        
        # Define predictors
        predictors = [
            'Close', 'Volume', 'Open', 'High', 'Low',
            '5d_ma', '20d_ma', 'Trend', 'Volume_Spike', 
            'Near_High', '3d_momentum', 'RSI'
        ]
        
        # Add sentiment to predictors if used
        if abs(sentiment_weight) > 0.2:
            predictors.append('Sentiment')
        
        X = df[predictors]
        y = df['Target']
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_split=50,
            random_state=42,
            max_depth=5
        )
        model.fit(X, y)
        
        # Get prediction and feature importance
        last_row = X.tail(1)
        prob = float(model.predict_proba(last_row)[0][1])
        
        # Adjust probability based on sentiment if significant
        if abs(sentiment_weight) > 0.2:
            prob = min(0.99, max(0.01, prob + (sentiment_weight * 0.1)))
        
        importance = dict(zip(predictors, model.feature_importances_))
        
        # Generate explanation
        explanation = self._generate_explanation(last_row.iloc[0], importance, prob, sentiment_weight)
        
        return prob, explanation, importance

    def recommend_trade(self, history: pd.DataFrame, forecast: pd.DataFrame,
                       move_prob: float, sentiment_score: float,
                       forecast_horizon_days: int) -> dict:
        """Generate BUY/SELL/HOLD recommendation with rationale"""
        if history.empty or forecast.empty:
            return {
                "action": "HOLD",
                "combined_score": 0,
                "strength": 0,
                "forecast_change_pct": 0,
                "last_price": np.nan,
                "forecast_price": np.nan,
                "reasons": ["Insufficient data"],
                "rationale": "No data to generate recommendation."
            }

        last_price = float(history['Close'].iloc[-1])
        forecast_price = float(forecast['yhat'].iloc[-1])
        abs_change = forecast_price - last_price
        change_pct = (abs_change / last_price) * 100.0

        forecast_signal = np.tanh(change_pct / 10.0)  # Normalize to [-1, 1]
        move_signal = (move_prob - 0.5) * 2.0  # Convert probability to [-1, 1] range
        sentiment_signal = float(np.clip(sentiment_score, -1.0, 1.0))

        # Weighted combination of signals
        w_forecast, w_move, w_sent = 0.5, 0.3, 0.2
        combined_score = (w_forecast * forecast_signal) + (w_move * move_signal) + (w_sent * sentiment_signal)

        # Determine recommendation based on combined score
        if combined_score >= 0.35:
            action = "STRONG BUY"
        elif combined_score >= 0.10:
            action = "BUY"
        elif combined_score <= -0.35:
            action = "STRONG SELL"
        elif combined_score <= -0.10:
            action = "SELL"
        else:
            action = "HOLD"

        # <<< FIXED: added missing closing parenthesis here >>>
        strength = int(round(min(100, abs(combined_score) * 100)))

        reasons = [
            f"Forecast horizon ({forecast_horizon_days}d) change: {change_pct:.2f}%",
            f"Next-day model probability: {move_prob:.2f}",
            f"News sentiment score: {sentiment_score:.2f}"
        ]

        if action in ("STRONG BUY", "BUY"):
            rationale = "Positive forecast and momentum signals suggest upside potential."
            if sentiment_signal > 0.2:
                rationale += " Positive market sentiment further supports buying."
        elif action in ("STRONG SELL", "SELL"):
            rationale = "Negative forecast and momentum signals suggest downside risk."
            if sentiment_signal < -0.2:
                rationale += " Negative market sentiment further supports selling."
        else:
            rationale = "Signals are mixed or weak - maintaining current position is recommended."

        return {
            "action": action,
            "combined_score": combined_score,
            "strength": strength,
            "forecast_change_pct": change_pct,
            "last_price": last_price,
            "forecast_price": forecast_price,
            "reasons": reasons,
            "rationale": rationale
        }

    def _generate_explanation(self, features, importance, prob, sentiment_weight=0) -> str:
        """Generate human-readable explanation with sentiment context"""
        reasons = []
        
        # Trend analysis
        if features.get('Trend', 0) == 1:
            reasons.append((
                "Price is above 20-day moving average (bullish trend)",
                importance.get('Trend', 0)
            ))
        
        # Volume spike
        if features.get('Volume_Spike', 0) == 1:
            reasons.append((
                "Significant volume spike (150%+ of 20-day average)",
                importance.get('Volume_Spike', 0)
            ))
        
        # Price position
        if features.get('Near_High', 0) == 1:
            reasons.append((
                "Closed near session high (strong finish)",
                importance.get('Near_High', 0)
            ))
        
        # Momentum
        if features.get('3d_momentum', 0) > 0:
            reasons.append((
                f"Positive 3-day momentum ({features.get('3d_momentum', 0):.2%})",
                importance.get('3d_momentum', 0)
            ))
        
        # RSI
        rsi = features.get('RSI', 50)
        if rsi < 30:
            reasons.append((
                f"Oversold (RSI: {rsi:.1f})",
                importance.get('RSI', 0)
            ))
        elif rsi > 70:
            reasons.append((
                f"Overbought (RSI: {rsi:.1f})",
                importance.get('RSI', 0)
            ))
        
        # Add sentiment reason if significant
        if abs(sentiment_weight) > 0.2:
            sentiment_label = "Bullish" if sentiment_weight > 0 else "Bearish"
            reasons.append((
                f"Market sentiment is {sentiment_label} ({sentiment_weight:.2f})",
                abs(sentiment_weight) * 0.5  # Moderate importance
            ))
        
        # Sort by most important factors
        reasons.sort(key=lambda x: x[1], reverse=True)
        
        # Build explanation text
        explanation = [f"Prediction Confidence: {prob:.0%} chance of increase tomorrow"]
        explanation.append("\nTop Influencing Factors:")
        
        for reason, weight in reasons[:3]:  # Top 3 factors
            explanation.append(f"- {reason} (impact: {weight:.1%})")
        
        if len(reasons) > 3:
            explanation.append(f"\nOther considerations: {len(reasons)-3} additional factors")
        
        return "\n".join(explanation)

    def _calculate_rsi(self, series, period=14):
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
