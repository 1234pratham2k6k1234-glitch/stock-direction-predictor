import yfinance as yf
import pandas as pd
import ta

def load_data(stock):
    df = yf.download(stock, start="2023-01-01", interval="1d")
    df.columns = df.columns.get_level_values(0)
    return df

def prepare_data(df):
    close = df['Close']

    df['Future_Return'] = close.shift(-1) / close - 1
    df['Target'] = (df['Future_Return'] > 0.005).astype(int)

    df['MA10'] = close.rolling(10).mean()
    df['MA50'] = close.rolling(50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(close).rsi()
    df['MACD'] = ta.trend.MACD(close).macd()

    df['Return'] = close.pct_change()
    df['Lag1'] = df['Return'].shift(1)
    df['Lag2'] = df['Return'].shift(2)
    df['Lag3'] = df['Return'].shift(3)

    df['Volatility'] = close.pct_change().rolling(10).std()
    df['Momentum'] = close - close.shift(10)

    df.dropna(inplace=True)
    return df
