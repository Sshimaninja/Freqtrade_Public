import requests
import json
import pandas as pd
from datetime import datetime


def fetch_fear_and_greed_index(limit=3000):
    response = requests.get(f"https://api.alternative.me/fng/?limit={limit}")
    data = json.loads(response.text)
    df = pd.DataFrame(data["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df.set_index("timestamp", inplace=True)
    return df


fgi_data = fetch_fear_and_greed_index()

# This saves the data.
fgi_data.to_csv("user_data/data/FGI_data/fgi_data.csv")
