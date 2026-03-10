from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('ALPACA_KEY', None)
SECRET_KEY = os.getenv('ALPACA_SECRET', None)

OUTPUT_FILE = "data_alpacanews\\historical_news_poland_and_bigcorp-01012015-25022026.csv"

def get_news():
    client = NewsClient(api_key = API_KEY, secret_key = SECRET_KEY)

    request_params = NewsRequest(
                            symbols="NVDA, TSLA",
                            start=datetime.strptime("2020-01-01", '%Y-%m-%d')
                            )

    news = client.get_news(request_params)

    # convert to dataframe
    return news.df

def get_all_news(tickers : list[str], start, end):
    client = NewsClient(api_key=API_KEY, secret_key=SECRET_KEY)

    if isinstance(tickers, list):
        tickers = ",".join(tickers)

    request_params = NewsRequest(
        symbols=tickers,
        start=start,
        end=end
    )

    news = client.get_news(request_params)

    if not news.data:
        return pd.DataFrame()

    return news.df


def preprocess_news(df: pd.DataFrame) -> pd.DataFrame:
    print(df.head())
    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    df["full_text"] = (
        df["headline"].fillna("") + ". " +
        df["summary"].fillna("")
    )

    for col in ["headline", "summary", "full_text"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace("\r", " ", regex=False)
        )

    # Keep symbols exactly as returned (no explode)
    df["symbols"] = df["symbols"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else str(x)
    )

    df = df[
        [
            "created_at",
            "symbols",
            "headline",
            "summary",
            "full_text",
            "source",
            "url"
        ]
    ]

    return df


def save_to_csv(df: pd.DataFrame, filename: str):
    if df.empty:
        print("No data to save.")
        return

    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {filename}")

def test():
    news_df : pd.DataFrame = get_news()
    print(news_df.columns)
    print(news_df.head())
    print("______________")
    print("symbols: ", news_df['symbols'].iloc[0])
    print("Summary: ", news_df['summary'].iloc[0])
    print("Content: ", news_df['content'].iloc[0])
    print("Headline: ", news_df['headline'].iloc[0])
    print("Url: ", news_df['url'].iloc[0])
    print("______________")
    print(news_df.iloc[0])
    print(len(news_df))

def create_dataset():
    # NVDA, MSFT, TSLA, AMZN
    # PKN.WA, KGH.WA, PKO.WA, CDR.WA
    # KGHM -> KGHPF
    # ORLEN -> PSKOF
    # PKOBP -> PSZKF
    # CDR (CD Projekt) -> OTGLF
    # Dino -> DNOPY
    tickers = ["NVDA", "TSLA", "MSFT", "AMZN", "KGHPF", "PSKOF", "PSZKF", "OTGLF", "GOOGL", "META"]
    print(str(tickers))

    start_date = datetime(2015, 1, 1)
    end_date = None

    print("Downloading news...")
    raw_news = get_all_news(tickers, start_date, end_date)

    print("Preprocessing news...")
    processed_news = preprocess_news(raw_news)

    print("Saving to CSV...")
    save_to_csv(processed_news, OUTPUT_FILE)

    print("Done.")

if __name__ == "__main__":
    # test()
    create_dataset()