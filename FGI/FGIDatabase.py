import requests
import json
import pandas as pd
from datetime import datetime


# That's a great idea! You can indeed fetch a year's worth of Fear and Greed Index (FGI) data, save it to a file, and then load it during backtesting. Here's how you can do it:

# First, write a script to fetch the FGI data and save it to a CSV file. You can run this script once a day to update the file with the latest FGI data.


def fetch_fear_and_greed_index(limit=3000):
    response = requests.get(f"https://api.alternative.me/fng/?limit={limit}")
    data = json.loads(response.text)
    df = pd.DataFrame(data["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df.set_index("timestamp", inplace=True)
    return df


# Fetch the FGI data
fgi_data = fetch_fear_and_greed_index()

# Save the FGI data to a CSV file
fgi_data.to_csv("user_data/data/FGI_data/fgi_data.csv")


# This will use the FGI data from the CSV file in backtesting.
# Please note that this assumes that the FGI data is available for all dates in the backtesting period.
# If the FGI data is not available for a certain date, the self.fgi_data.loc[current_date]['value'] line will raise a KeyError. You might want to add some error handling code to deal with this situation.
