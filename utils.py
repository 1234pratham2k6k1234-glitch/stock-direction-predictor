import numpy as np

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

    except:
        return 0


def generate_signal(proba, sentiment):
    sentiment = max(min(sentiment, 0.2), -0.2)

    score = (proba[1] * 0.7) + ((sentiment + 1)/2 * 0.3)

    if score > 0.6:
        return "BUY 📈"
    elif score < 0.4:
        return "SELL 📉"
    else:
        return "HOLD ⏳"
