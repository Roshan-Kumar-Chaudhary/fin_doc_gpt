
import streamlit as st
from transformers import pipeline
from utils.document_loader import load_document, extract_financial_sections
from utils.preprocessing import analyze_sentiment
from utils.anomaly_detection import detect_anomalies, detect_tabular_anomalies
from utils.market_data import MarketDataFetcher
from utils.forecasting import FinancialForecaster
from utils.sentiment import MarketSentiment
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="FinDocGPT",
    page_icon="📊",
    layout="wide"
)

load_dotenv()

@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        token=os.getenv("HF_API_KEY")
    )

@st.cache_resource
def load_summarization_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        token=os.getenv("HF_API_KEY")
    )

def display_sentiment(sentiment):
    """Helper method to display sentiment results"""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if sentiment['sentiment'] == "Positive":
            st.success(f"**{sentiment['sentiment']}**")
        elif sentiment['sentiment'] == "Negative":
            st.error(f"**{sentiment['sentiment']}**")
        else:
            st.info(f"**{sentiment['sentiment']}**")
        
        st.metric("Score", f"{sentiment['score']:.2f}")
    
    with col2:
        st.progress(sentiment['positive'], text=f"Positive: {sentiment['positive']:.1%}")
        st.progress(sentiment['negative'], text=f"Negative: {sentiment['negative']:.1%}")
        st.progress(sentiment['neutral'], text=f"Neutral: {sentiment['neutral']:.1%}")
        
        if sentiment.get('keywords'):
            st.caption(f"Key terms: {', '.join(sentiment['keywords'][:5])}{'...' if len(sentiment['keywords']) > 5 else ''}")

market_data = MarketDataFetcher()
forecaster = FinancialForecaster()
sentiment_analyzer = MarketSentiment()

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["📄 Document Analysis", "📈 Market Analysis"]
)

if app_mode == "📄 Document Analysis":
    st.title("📄 FinDocGPT - Document Analysis")
    st.markdown("Upload financial documents to extract insights, analyze sentiment, and detect anomalies")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Financial Document (PDF/Excel/Image/TXT)",
        type=["pdf", "xlsx", "xls", "csv", "txt", "png", "jpg", "jpeg"]
    )

    if uploaded_file:
        try:
            # Document processing
            with st.spinner("Processing document..."):
                text = load_document(uploaded_file)
                sections = extract_financial_sections(text)
                
                # Generate executive summary for large documents
                if len(text) > 10000:
                    with st.spinner("Generating executive summary..."):
                        summarizer = load_summarization_model()
                        summary = summarizer(text[:10000], max_length=300, min_length=100, do_sample=False)
                        text_summary = summary[0]['summary_text']
                else:
                    text_summary = "Document is concise enough for direct analysis"
            
            st.success("✅ Document processed successfully!")
            
            # Display summary and sections
            with st.expander("📝 Executive Summary"):
                st.write(text_summary)
                
            with st.expander("📄 Full Document Sections"):
                for section, content in sections.items():
                    if content:
                        st.subheader(section.replace("_", " ").title())
                        st.text(content[:2000] + ("..." if len(content) > 2000 else ""))
            
            # Initialize models
            qa_model = load_qa_model()
            
            # Enhanced Q&A Section
            st.subheader("📊 Financial Document Q&A")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                question = st.text_input("Ask a financial question about the document:", 
                                        placeholder="E.g., What was the Q3 revenue growth?")
            
            with col2:
                focus_section = st.selectbox("Focus on:", 
                                           ["Entire Document", "Income Statement", 
                                            "Balance Sheet", "Cash Flow", 
                                            "Management Discussion", "Risk Factors"])
            
            if question:
                # Select context based on focus section
                if focus_section == "Income Statement" and sections["income_statement"]:
                    context = sections["income_statement"]
                elif focus_section == "Balance Sheet" and sections["balance_sheet"]:
                    context = sections["balance_sheet"]
                elif focus_section == "Cash Flow" and sections["cash_flow"]:
                    context = sections["cash_flow"]
                elif focus_section == "Management Discussion" and sections["management_discussion"]:
                    context = sections["management_discussion"]
                elif focus_section == "Risk Factors" and sections["risks"]:
                    context = sections["risks"]
                else:
                    context = text  # Fall back to full text
                
                with st.spinner("Analyzing document..."):
                    try:
                        answer = qa_model(question=question, context=context[:50000])  # Limit context size
                        st.markdown(f"**Answer:** {answer['answer']}")
                        st.progress(float(answer['score']))
                        st.caption(f"Confidence: {answer['score']:.2%}")
                        
                        # Show context snippet
                        with st.expander("View relevant context"):
                            start = max(0, answer['start']-100)
                            end = min(len(context), answer['end']+100)
                            st.text(context[start:end])
                    except Exception as e:
                        st.error(f"Could not answer question: {str(e)}")
            
            # Enhanced Sentiment Analysis
            st.subheader("📈 Sentiment Analysis by Section")
            if sections["management_discussion"] or sections["risks"]:
                tab1, tab2 = st.tabs(["Management Discussion", "Risk Factors"])
                
                with tab1:
                    if sections["management_discussion"]:
                        sentiment = analyze_sentiment(sections["management_discussion"])
                        display_sentiment(sentiment)
                    else:
                        st.info("No management discussion section found")
                
                with tab2:
                    if sections["risks"]:
                        sentiment = analyze_sentiment(sections["risks"])
                        display_sentiment(sentiment)
                    else:
                        st.info("No risk factors section found")
            else:
                sentiment = analyze_sentiment(text)
                display_sentiment(sentiment)
            
            # Enhanced Anomaly Detection
            if uploaded_file.name.endswith(('.xlsx', '.xls', '.csv')):
                st.subheader("🔍 Financial Anomaly Detection")
                try:
                    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
                    
                    if not df.empty:
                        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
                        
                        if numerical_cols:
                            selected_cols = st.multiselect(
                                "Select columns for anomaly analysis:",
                                numerical_cols,
                                default=numerical_cols[:2]
                            )
                            
                            if selected_cols:
                                with st.spinner("Detecting anomalies..."):
                                    anomalies = detect_tabular_anomalies(df[selected_cols])
                                
                                if anomalies:
                                    for col, result in anomalies.items():
                                        st.warning(f"🚨 Found {len(result['anomaly_indices'])} anomalies in {col} ({result['percentage']:.1f}%)")
                                        st.dataframe(df.iloc[result['anomaly_indices']][selected_cols])
                                else:
                                    st.success("No anomalies detected in selected columns")
                        else:
                            st.info("No numeric columns found for anomaly detection")
                    else:
                        st.warning("The uploaded file is empty")
                except Exception as e:
                    st.error(f"Error processing tabular data: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

else:  # Market Analysis mode
    st.title("📈 FinDocGPT - Market Analysis")
    st.markdown("Analyze market trends and forecast future performance")
    
    # API-powered stock selection with auto-complete
    @st.cache_data(ttl=3600)
    def get_popular_stocks():
        try:
            indices = {
                "S&P 500 (^GSPC)": "^GSPC",
                "Dow Jones (^DJI)": "^DJI",
                "NASDAQ (^IXIC)": "^IXIC",
                "Nifty 50 (^NSEI)": "^NSEI"
            }
            
            stocks = {
                # US Stocks
                **{f"{yf.Ticker(t).info['shortName']} ({t})": t 
                   for t in ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL', 'NVDA', 'META']},
                
                # Indian Stocks
                **{f"{yf.Ticker(t).info['shortName']} ({t})": t 
                   for t in ['RELIANCE.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS', 'INFY.NS']},
                
                # International
                **{f"{yf.Ticker(t).info['shortName']} ({t})": t 
                   for t in ['NESN.SW', '005930.KS', 'BABA', 'TSM']}
            }
            
            return {**indices, **stocks}
        except:
            return {
                "Apple (AAPL)": "AAPL",
                "Microsoft (MSFT)": "MSFT",
                "Tesla (TSLA)": "TSLA",
                "Amazon (AMZN)": "AMZN",
                "Reliance (RELIANCE.NS)": "RELIANCE.NS",
                "Nestlé (NESN.SW)": "NESN.SW"
            }
    
    popular_stocks = get_popular_stocks()
    
    # Enhanced stock search
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("Search stock:", value="", 
                                  placeholder="Type to search (AAPL, MSFT, RELIANCE.NS...)")
        
        filtered_stocks = {
            k: v for k, v in popular_stocks.items() 
            if search_term.upper() in k.upper() or search_term.upper() in v.upper()
        }
        
        if not filtered_stocks:
            st.info("No matches found. Try a different search or enter a ticker manually")
            manual_ticker = st.text_input("Or enter ticker manually:", value="")
            ticker = manual_ticker.upper() if manual_ticker else None
        else:
            selected_stock = st.selectbox(
                "Select stock:",
                options=list(filtered_stocks.keys()),
                index=0
            )
            ticker = filtered_stocks[selected_stock]
    
    with col2:
        period = st.selectbox(
            "History Period", 
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=3
        )
    
    forecast_days = st.slider(
        "Forecast Period (days)", 
        min_value=7, max_value=90, value=30,
        key="forecast_days"
    )
    
    if ticker:
        try:
            # Fetch and display key metrics
            with st.spinner("Fetching market data..."):
                metrics = market_data.get_key_metrics(ticker)
                history = market_data.get_historical_data(ticker, period)
                sentiment_data = sentiment_analyzer.get_ticker_sentiment(ticker)
            
            if history.empty:
                st.warning("No historical data available for this ticker/period")
                st.stop()
            
            # Key metrics display
            st.subheader(f"{ticker} Key Metrics")
            cols = st.columns(5)
            cols[0].metric("Current Price", f"${metrics['current_price']:.2f}" if metrics['current_price'] else "N/A")
            cols[1].metric("P/E Ratio", f"{metrics['pe_ratio']:.2f}" if metrics['pe_ratio'] else "N/A")
            cols[2].metric("Market Cap", f"${metrics['market_cap']/1e9:.2f}B" if metrics['market_cap'] else "N/A")
            cols[3].metric("52W Range", 
                         f"{metrics['52_week_low']:.2f}-{metrics['52_week_high']:.2f}" 
                         if metrics['52_week_low'] and metrics['52_week_high'] else "N/A")
            
            # Sentiment metric
            sentiment_score = sentiment_data.get('score', 0) if 'error' not in sentiment_data else 0
            sentiment_label = "Bullish" if sentiment_score > 0.2 else "Bearish" if sentiment_score < -0.2 else "Neutral"
            
            if sentiment_score > 0.2:
                delta_color = "normal"
            elif sentiment_score < -0.2:
                delta_color = "inverse"
            else:
                delta_color = "off"
                
            cols[4].metric(
                "Market Sentiment",
                f"{sentiment_label} ({sentiment_score:.2f})",
                delta=f"{sentiment_data['summary']['positive']}👍 / {sentiment_data['summary']['negative']}👎" if 'error' not in sentiment_data else "N/A",
                delta_color=delta_color
            )
            
            # Price history chart
            st.subheader(f"{ticker} Price History")
            st.line_chart(history['Close'])
            
            # Forecasting and Recommendation Section
            if st.button("Generate Forecast & Recommendation"):
                with st.spinner("Running analysis..."):
                    try:
                        # Generate forecast
                        forecast = forecaster.forecast_prophet(history, forecast_days, sentiment_score)
                        
                        # Generate movement prediction
                        move_prob, explanation, importance = forecaster.forecast_movement(history, sentiment_score)
                        
                        # Generate recommendation
                        recommendation = forecaster.recommend_trade(
                            history=history,
                            forecast=forecast,
                            move_prob=move_prob,
                            sentiment_score=sentiment_score,
                            forecast_horizon_days=forecast_days
                        )
                        
                        # Display forecast chart
                        fig = px.line(
                            forecast, 
                            x='ds', 
                            y='yhat',
                            title=f"{ticker} {forecast_days}-Day Price Forecast"
                        )
                        fig.add_scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_upper'],
                            fill='tonexty',
                            mode='lines',
                            line=dict(color='rgba(0,100,80,0.2)'),
                            name='Upper Bound'
                        )
                        fig.add_scatter(
                            x=forecast['ds'],
                            y=forecast['yhat_lower'],
                            fill='tonexty',
                            mode='lines',
                            line=dict(color='rgba(0,100,80,0.2)'),
                            name='Lower Bound'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display recommendation
                        st.subheader("📊 Trading Recommendation")
                        
                        rec_action = recommendation['action']
                        rec_strength = recommendation['strength']
                        
                        if rec_action.startswith("BUY"):
                            st.success(f"Recommendation: {rec_action} — Confidence: {rec_strength}%")
                        elif rec_action.startswith("SELL"):
                            st.error(f"Recommendation: {rec_action} — Confidence: {rec_strength}%")
                        else:
                            st.info(f"Recommendation: {rec_action} — Confidence: {rec_strength}%")
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Price", f"${recommendation['last_price']:.2f}")
                        col2.metric(f"{forecast_days}-Day Forecast", f"${recommendation['forecast_price']:.2f}",
                                    f"{recommendation['forecast_change_pct']:.2f}%")
                        col3.metric("Composite Score", f"{recommendation['combined_score']:.2f}")
                        
                        # Display rationale
                        with st.expander("Recommendation Details"):
                            st.write(recommendation['rationale'])
                            st.write("**Key Factors:**")
                            for reason in recommendation['reasons']:
                                st.write(f"- {reason}")
                            
                            st.write("\n**Next-Day Movement Prediction:**")
                            st.write(explanation)
                            
                            # Feature importance
                            importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Importance'])
                            importance_df = importance_df.sort_values('Importance', ascending=False)
                            
                            fig = go.Figure(go.Bar(
                                x=importance_df.index,
                                y=importance_df['Importance'],
                                marker_color='#636EFA'
                            ))
                            fig.update_layout(
                                xaxis_title='Features',
                                yaxis_title='Relative Importance',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
            
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
