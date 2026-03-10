import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("STOCKNEWSAPI_TOKEN")
BASE_URL = "https://stocknewsapi.com/api/v1"
OUTPUT_FILE = "data_stocknewsapi\\historical_news.csv"


def generate_month_ranges(start_date, end_date, month_interval):
    current = start_date
    while current <= end_date:
        next_month = current + relativedelta(months=month_interval)
        yield current, min(next_month - timedelta(days=1), end_date)
        current = next_month


def download_news_for_range(ticker, start_date, end_date):
    page = 1
    all_data = []

    while True:
        params = {
            "tickers": ticker,
            "date": f"{start_date.strftime('%m%d%Y')}-{end_date.strftime('%m%d%Y')}",
            "items": 3,
            "page": page,
            "token": API_TOKEN
        }

        response = requests.get(BASE_URL, params=params, timeout=15)
        print(response.json())
        response.raise_for_status()

        data = response.json()
        news_items = data.get("data", [])

        if not news_items:
            break

        for item in news_items:
            item["ticker"] = ticker  # important for ML later

        all_data.extend(news_items)

        print(f"{ticker} | {start_date.date()} to {end_date.date()} | Page {page} | {len(news_items)} items")

        page += 1
        time.sleep(0.4)

    return all_data


def save_to_csv(data, output_file):
    if not data:
        return

    df = pd.DataFrame(data)

    # Clean newline characters
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace("\n", " ").str.replace("\r", " ")

    # Remove duplicates if file exists
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        df = pd.concat([existing, df]).drop_duplicates(subset=["news_url"], keep="first")

    df.to_csv(output_file, index=False, encoding="utf-8")


def download_historical_news(ticker):
    start_date = datetime(2019, 4, 1)   # ✅ March 2019
    end_date = datetime(2026, 12, 31)

    for start, end in generate_month_ranges(start_date, end_date, 1):
        news = download_news_for_range(ticker, start, end)
        save_to_csv(news, OUTPUT_FILE)

def download_historical_news_count():
    start_date = datetime(2019, 3, 1)   # ✅ March 2019
    end_date = datetime(2026, 2, 24)
    count_requests = 0

    for start, end in generate_month_ranges(start_date, end_date, 1):
        count_requests += 1

    print(count_requests)
    
if __name__ == "__main__":
    tickers = ["NVDA"]  # add more here

    for ticker in tickers:
        download_historical_news(ticker)
        # download_historical_news_count()