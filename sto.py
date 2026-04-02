# =========================================
# 📚 Imports
# =========================================
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import ta
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download sentiment model once
nltk.download('vader_lexicon')

# =========================================
# ⚙️ Page Config
# =========================================
st.set_page_config(layout="wide")
st.title("Stock Price Direction Predictor")

# =========================================
# 🔄 AUTO REFRESH (SAFE WAY)
# =========================================
refresh_ms = st.sidebar.slider("Auto Refresh (ms)", 0, 300000, 60000)

st_autorefresh(interval=refresh_ms, key="refresh")

# =========================================
# 📌 MULTI STOCK INPUT
# =========================================
stocks = st.sidebar.text_input(
    "Enter Stocks (comma separated)",
    "RELIANCE.NS,TCS.NS,INFY.NS"
)

stock_list = [s.strip() for s in stocks.split(",")]

# =========================================
# 📰 NEWS SENTIMENT
# =========================================
@st.cache_data
def get_news_sentiment(stock):
    try:
        import requests
        from bs4 import BeautifulSoup
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        clean_stock = stock.replace(".NS", "")
        url = f"https://news.google.com/search?q={clean_stock}%20stock&hl=en-IN&gl=IN&ceid=IN:en"

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        soup = BeautifulSoup(response.text, "html.parser")

        headlines = [h.text for h in soup.select("a.DY5T1d")[:10]]

        if not headlines:
            return 0

        sid = SentimentIntensityAnalyzer()
        scores = [sid.polarity_scores(h)['compound'] for h in headlines]

        return float(np.mean(scores))

    except Exception as e:
        return 0

# =========================================
# 📥 LOAD DATA
# =========================================
@st.cache_data
def load_data(stock):
    df = yf.download(stock, start="2018-01-01", interval="1d")
    df.columns = df.columns.get_level_values(0)
    return df

# =========================================
# 🧠 FEATURE ENGINEERING
# =========================================
def prepare_data(df):
    close = df['Close']
    
    df['Target'] = (close.shift(-1) > close).astype(int)
    df['MA10'] = close.rolling(10).mean()
    df['MA50'] = close.rolling(50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(close).rsi()
    df['MACD'] = ta.trend.MACD(close).macd()
    df['Return'] = close.pct_change()
    
    df['Lag1'] = close.shift(1)
    df['Lag2'] = close.shift(2)
    df['Lag3'] = close.shift(3)
    
    df.dropna(inplace=True)
    return df

# =========================================
# 🤖 MODEL
# =========================================
def train_model(X_train, y_train):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    return model

# =========================================
# 📊 MAIN DASHBOARD
# =========================================
portfolio_results = []

for stock in stock_list:
    st.subheader(f"📊 {stock}")
    
    try:
        df = load_data(stock)
        df = prepare_data(df)
        
        features = ['MA10','MA50','RSI','MACD','Return','Lag1','Lag2','Lag3']
        
        X = df[features]
        y = df['Target']
        
        split = int(len(df)*0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = train_model(X_train, y_train)
        
        # Accuracy
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Prediction
        latest = X.iloc[-1:]
        pred = model.predict(latest)[0]
        proba = model.predict_proba(latest)[0]
        
        # Sentiment
        sentiment = get_news_sentiment(stock)
        
        # Signal Logic
        sentiment = max(min(sentiment, 0.2), -0.2)

        score = (proba[1] * 0.7) + ((sentiment + 1)/2 * 0.3)

        if score > 0.6:
            signal = "BUY 📈"
        elif score < 0.4:
            signal = "SELL 📉"
        else:
            signal = "HOLD ⏳"
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.2f}")
        col2.metric("Prediction", "UP" if pred==1 else "DOWN")
        col3.metric("Confidence", f"{max(proba):.2f}")
       
        
        st.write(f"### 🔔 Signal: {signal}")
        
        # Plot price
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index,   # ✅ THIS IS THE FIX
            y=df['Close'],
            name="Price"
        ))

        fig.update_layout(
            title=stock,
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(fig, use_container_width=True)
        
        portfolio_results.append({
            "Stock": stock,
            "Signal": signal,
            "Accuracy": round(acc,2)
        })
    
    except Exception as e:
        st.error(f"Error loading {stock}: {e}")

# =========================================
# 📊 PORTFOLIO SUMMARY
# =========================================
st.header("📊 Portfolio Summary")
portfolio_df = pd.DataFrame(portfolio_results)
st.dataframe(portfolio_df)

