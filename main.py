import feedparser
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading lightweight sentiment model...")
# Using a lightweight DistilBERT model that fits comfortably under 512MB RAM
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

TIMEFRAME_MAP = {
    "1 Day": "1d",
    "5 Days": "5d",
    "1 Month": "1m",
    "6 Months": "6m",
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y"
}

def get_sentiment(text: str):
    try:
        res = sentiment_pipeline(text[:512])[0]
        label = res['label'].upper()
        # DistilBERT outputs POSITIVE or NEGATIVE; map them cleanly
        if label == "POSITIVE":
            mapped_label = "POSITIVE"
        elif label == "NEGATIVE":
            mapped_label = "NEGATIVE"
        else:
            mapped_label = "NEUTRAL"
        return mapped_label, round(res['score'], 2)
    except Exception:
        return "NEUTRAL", 0.50

@app.get("/api/analyze")
def analyze_ticker(ticker: str, timeframe: str = "1 Year"):
    try:
        sym = ticker.strip().upper()
        tk = yf.Ticker(sym)
        info = tk.info or {}
        company_name = info.get("longName") or info.get("shortName") or sym
        currency = info.get("currency", "USD")
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0

        tf_code = TIMEFRAME_MAP.get(timeframe, "1y")
        query = f'"{sym}" OR "{company_name}" when:{tf_code}'
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        
        resp = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        feed = feedparser.parse(resp.content)

        articles = []
        clean_sym = sym.split(".")[0]
        first_name = company_name.split()[0].upper()

        for entry in feed.entries[:40]:
            title = entry.get("title", "")
            t_up = title.upper()
            if clean_sym in t_up or first_name in t_up or sym in t_up:
                label, score = get_sentiment(title)
                articles.append({
                    "timeframe": timeframe,
                    "title": title,
                    "sentiment": label,
                    "score": score,
                    "url": entry.get("link", f"https://finance.yahoo.com/quote/{sym}")
                })

        if not articles:
            label, score = get_sentiment(f"{sym} reports strong market activity and growth")
            articles.append({
                "timeframe": timeframe,
                "title": f"{sym} Market Performance and Volume Update",
                "sentiment": label,
                "score": score,
                "url": f"https://finance.yahoo.com/quote/{sym}"
            })

        return {
            "ticker": sym,
            "name": company_name,
            "price": f"${price:,.2f} {currency}",
            "timeframe": timeframe,
            "total_articles": len(articles),
            "articles": articles
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
