import requests
from dotenv import load_dotenv
import os
import pandas as pd
import json

load_dotenv()

def save_news_to_csv():
    news_api_params = {
        'api_token': os.getenv('NEWS_TOKEN'),
        'language': 'en,pl'    
    }
    
    get_all_news = requests.get(
        url = 'https://api.thenewsapi.com/v1/news/all',
        params = news_api_params
    )

    news_data : dict = get_all_news.json()
    print(json.dumps(news_data, indent=4))

    articles : dict = news_data.get("data", [])

    news_df : pd.DataFrame = pd.DataFrame(articles)
    news_df["categories"] = news_df["categories"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else ""
    )


    news_df.to_csv("data_news_api\\news_data.csv", encoding="utf-8")

def main():
    print("NEWS API")
    save_news_to_csv()

if __name__ == "__main__":
    main()