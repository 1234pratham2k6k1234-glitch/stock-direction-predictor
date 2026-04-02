# 📈 Stock Price Direction Predictor

A machine learning-powered Streamlit dashboard that predicts stock price direction (UP/DOWN) and generates BUY/SELL signals using technical indicators and news sentiment.

---

## 🚀 Features

- 📊 Real-time stock data using Yahoo Finance
- 🤖 XGBoost machine learning model
- 📰 News sentiment analysis using NLP (VADER)
- 📈 Interactive charts with Plotly
- 🔄 Auto-refresh dashboard
- 📉 Portfolio summary across multiple stocks

---

## 🧠 How It Works

1. Fetch historical stock data  
2. Generate technical indicators:
   - Moving Averages (MA10, MA50)
   - RSI (Relative Strength Index)
   - MACD
   - Daily Returns  
3. Train an XGBoost classifier  
4. Predict next-day price direction (UP/DOWN)  
5. Fetch latest news and compute sentiment  
6. Combine ML prediction + sentiment into final signal  

Signals:
- BUY 📈  
- SELL 📉  
- HOLD ⏳  

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/stock-direction-predictor.git
cd stock-direction-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📌 Example Tickers

- RELIANCE.NS  
- TCS.NS  
- INFY.NS  
- TSLA  
- AAPL  

---

## 📊 Output Includes

- 📈 Prediction (UP / DOWN)  
- 📊 Confidence score  
- 🔔 Trading signal  
- 📉 Interactive price chart  
- 📊 Portfolio summary  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- XGBoost  
- Pandas / NumPy  
- Plotly  
- Yahoo Finance  
- NLTK (VADER)  
- BeautifulSoup  

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
It is **not financial advice**.

---

## 📸 Screenshot

<img width="800" height="500" alt="Screenshot 2026-04-02 170647" src="https://github.com/user-attachments/assets/4aa6b230-f0b6-4476-baf6-e50f7b5d7bb1" />
<img width="800" height="500" alt="Screenshot 2026-04-02 170700" src="https://github.com/user-attachments/assets/05ade80d-d495-45a2-9c5f-b1512b5f7bf9" />
<img width="800" height="500" alt="Screenshot 2026-04-02 170710" src="https://github.com/user-attachments/assets/27774400-ec13-4232-8440-b6aaa76c58d2" />


## 👨‍💻 Author

**Pratham Agarwal**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
