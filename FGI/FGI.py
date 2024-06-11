import requests
import json

from FreqCtrlBTC import FreqCtrlBTC

# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

# --------------------------------
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    stoploss_from_open,
    merge_informative_pair,
    DecimalParameter,
    IntParameter,
    CategoricalParameter,
)
import technical.indicators as ftt
import math
import logging

logger = logging.getLogger(__name__)


# This is a simple daily check for the Fear and Greed Index from alternative.me (the originator of the API)

# However, if you ware happy with FGITest.py backtests, and you want a more performant bot, you can run this on the live bot instead of hosting the full backtest database

# This is an appropriate way to run for example an Oracle free tier bot with low resources.


class FGI(FreqCtrlBTC):

    def fetch_fear_and_greed_index(self):
        response = requests.get(
            "https://api.alternative.me/fng/?limit=1"
        )  # Fetch the latest data for today
        data = json.loads(response.text)
        return int(data["data"][0]["value"])

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)

        fear_and_greed_index = self.fetch_fear_and_greed_index()

        greed = []
        # fear = []
        # Control buy trend based on Fear and Greed Index
        # if fear_and_greed_index < 30:  # Fear
        # 	# Modify this to your desired condition when the market is in fear

        # 	fear.append(condition)
        if fear_and_greed_index > 70:  # Greed
            dataframe.loc[:, "buy"] = 0
        return dataframe
