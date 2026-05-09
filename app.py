import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import plotly.graph_objects as go
import nltk

from data import load_data, prepare_data
from model import train_model, evaluate_model, predict_latest, feature_importance
from utils import get_news_sentiment, generate_signal

nltk.download('vader_lexicon', quiet=True)

st.set_page_config(layout="wide")
st.title("Stock Price Direction Predictor")

refresh_ms = st.sidebar.slider("Auto Refresh (ms)", 0, 300000, 60000)
st_autorefresh(interval=refresh_ms, key="refresh")

stocks = st.sidebar.text_input(
    "Enter Stocks (comma separated)",
    "RELIANCE.NS,TCS.NS,INFY.NS"
)

stock_list = [s.strip() for s in stocks.split(",")]
portfolio_results = []

for stock in stock_list:
    st.subheader(f"📊 {stock}")

    try:
        df = prepare_data(load_data(stock))

        features = ['MA10','MA50','RSI','MACD','Return','Lag1','Lag2','Lag3']
        X = df[features]
        y = df['Target']

        split = int(len(df)*0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = train_model(X_train, y_train)

        acc, baseline, y_pred = evaluate_model(model, X_test, y_test)
        # Backtesting
        returns = df['Return'].iloc[split:]
        strategy = returns * y_pred

        cumulative_strategy = (1 + strategy).cumprod()
        cumulative_market = (1 + returns).cumprod()

        
        pred, proba, confidence = predict_latest(model, X)

        sentiment = get_news_sentiment(stock)
        signal = generate_signal(proba, sentiment)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2f}")
        col2.metric("Prediction", "UP" if pred==1 else "DOWN")
        col3.metric("Confidence", f"{confidence:.2f}")
        col4.metric("Baseline", f"{baseline:.2f}")
        st.info("📊 Note: Stock movement is highly noisy. Even small improvements over baseline are meaningful.")

        st.write(f"### 🔔 Signal: {signal}")
        st.caption("Signal = 70% model probability + 30% sentiment score")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Price"))
        st.plotly_chart(fig, use_container_width=True)
        st.write("### 📈 Strategy vs Market Performance")

        backtest_df = {
            "Strategy": cumulative_strategy,
            "Market": cumulative_market
        }

        st.line_chart(backtest_df)

        feat_df = feature_importance(model, features)
        st.bar_chart(feat_df.set_index('Feature'))
        st.caption("Feature importance shows which indicators most influence the model predictions.")

        portfolio_results.append({
            "Stock": stock,
            "Signal": signal,
            "Accuracy": round(acc, 2)
        })

    except Exception as e:
        st.error(f"Error loading {stock}: {e}")

st.header("📊 Portfolio Summary")
st.dataframe(pd.DataFrame(portfolio_results))
