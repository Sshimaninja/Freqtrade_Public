import json
import requests
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce

import json
import requests
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from pandas import read_csv
from pandas import to_datetime

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
from YOURSTRATEGY import YOURSTRATEGY
import technical.indicators as ftt
import math
import logging

logger = logging.getLogger(__name__)

# A control class for the YOURSTRATEGY strategy written by Justin Michael Boon (https://github.com/sshimaninja)


class FGIHMABTCCtrl(YOURSTRATEGY):
    INTERFACE_VERSION = 3
    buy_params = {
        "hma": 20,
        "hma_offset": 10,
        "hma_on": False,
        "hma_offset_switch": False,
        "FGIGREED_on": True,
        "FGIFEAR_on": True,
        "GREED": 70,
        "FEAR": 30,
        # "sma": 2,
    }

    sell_params = {}
    timeframe = "5m"
    inf_1d = "1d"

    # Hyperoptable parameters
    # Buy hyperspace params:
    hma_1d_tf = IntParameter(
        5, 500, default=buy_params["hma"], space="buy", optimize=True
    )
    hma_1d_IO = CategoricalParameter(
        [True, False], default=buy_params["hma_on"], space="buy", optimize=True
    )
    # sma_1d_tf = IntParameter(5, 75, default=buy_params["sma"], space="buy", optimize=True)
    hma_1d_offset_tf = IntParameter(
        1, 200, default=buy_params["hma_offset_switch"], space="buy", optimize=True
    )
    hma_1d_offset_IO = CategoricalParameter(
        [True, False], default=True, space="buy", optimize=True
    )
    # hma_on = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    # hma_offset_switch = CategoricalParameter([True, False], default=True, space="buy", optimize=True)

    # FGI data comes from a CSV file in order to allow for backtesting
    # We pull the most recent data in the file later.
    fgi_data = read_csv(
        "user_data/data/FGI_data/fgi_data.csv", index_col="timestamp", parse_dates=True
    )
    fgi_fear_on = CategoricalParameter(
        [True, False], default=buy_params["FGIFEAR_on"], space="buy", optimize=True
    )
    fgi_greed_on = CategoricalParameter(
        [True, False], default=buy_params["FGIGREED_on"], space="buy", optimize=True
    )
    extreme_greed = CategoricalParameter(
        [70, 75, 80, 85, 90, 95],
        default=buy_params["GREED"],
        space="buy",
        optimize=True,
    )
    extreme_fear = CategoricalParameter(
        [5, 10, 15, 20, 25, 30], default=buy_params["FEAR"], space="buy", optimize=True
    )
    # print(fgi_data)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "1d") for pair in pairs]

        if self.config["stake_currency"] in [
            "USDT",
            "BUSD",
            "USDC",
            "DAI",
            "TUSD",
            "PAX",
            "USD",
            "EUR",
            "GBP",
        ]:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        informative_pairs.append((btc_info_pair, self.timeframe))
        informative_pairs.append((btc_info_pair, self.inf_1d))

        return informative_pairs

    def fetch_FGI(self, dataframe: DataFrame) -> DataFrame:
        # Convert the index to datetime and remove timezone information
        dataframe.index = to_datetime(dataframe.index).tz_localize(None)
        self.fgi_data.index = to_datetime(self.fgi_data.index).tz_localize(None)

        # Resample both dataframes to daily data
        dataframe = dataframe.resample("D").last()
        self.fgi_data = self.fgi_data.resample("D").last()

        # Merge the dataframes
        dataframe = dataframe.merge(
            self.fgi_data[["value"]], left_index=True, right_index=True, how="left"
        )
        dataframe.rename(columns={"value": "FGI"}, inplace=True)

        return dataframe

    def info_tf1d_btc_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:
        # Indicators
        # -----------------------------------------------------------------------------------------
        # dataframe["sma"] = ta.SMA(dataframe["close"], timeperiod=self.buy_params["sma"])
        dataframe["hma"] = qtpylib.hull_moving_average(
            dataframe["close"], window=self.buy_params["hma"]
        )
        dataframe["1d_close"] = dataframe["close"]
        dataframe["hma_offset"] = dataframe["hma"].shift(self.buy_params["hma_offset"])
        # Add prefix
        # -----------------------------------------------------------------------------------------
        # ignore_columns = ["date", "open", "high", "low", "close", "volume"]
        # dataframe.rename(
        # 	columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True
        # )
        # FGI
        # -----------------------------------------------------------------------------------------
        if self.buy_params["FGIFEAR_on"] or self.buy_params["FGIGREED_on"]:
            dataframe = self.fetch_FGI(dataframe)
            if "FGI" not in dataframe.columns:
                logger.error("Failed to create 'FGI' column in dataframe")
                raise ValueError("Failed to create 'FGI' column in dataframe")
                # print('FGI' in dataframe.columns)
            # else:
            # 	logger.info("Successfully created 'FGI' column in dataframe")
            # print('FGI' in dataframe.columns)
        # print(dataframe.index)
        # print(self.fgi_data.index)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)

        if self.config["stake_currency"] in ["USDT", "BUSD"]:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        btc_info_tf_1d = self.dp.get_pair_dataframe(btc_info_pair, "1d")
        btc_info_tf_1d = self.info_tf1d_btc_indicators(
            btc_info_tf_1d, metadata
        )  # Call the method here
        dataframe = merge_informative_pair(
            dataframe, btc_info_tf_1d, self.timeframe, "1d", ffill=True
        )

        drop_columns = [
            f"{s}_{self.inf_1d}"
            for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        dataframe.drop(
            columns=dataframe.columns.intersection(drop_columns), inplace=True
        )

        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        available_stake_amount = self.wallets.get_available_stake_amount()
        max_open_trades = self.config["max_open_trades"]

        if proposed_stake < 5000:
            available_stake_amount / (max_open_trades - available_stake_amount)
        else:
            proposed_stake = 5000
        return proposed_stake

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)
        guard = []

        print(dataframe["FGI_1d"])

        if self.buy_params["hma_on"]:
            guard.append(dataframe["1d_close_1d"] > dataframe["hma_offset_1d"])
            print(dataframe["1d_close_1d"])
        if self.buy_params["FGIFEAR_on"]:
            guard.append(dataframe["FGI_1d"] < self.buy_params["FEAR"])
            print(dataframe["FGI_1d"])
        if self.buy_params["FGIGREED_on"]:
            guard.append(dataframe["FGI_1d"] > self.buy_params["GREED"])
            print(dataframe["FGI_1d"])

        # Set the 'buy' column to 0 where any of the conditions in 'guard' are True
        for condition in guard:
            dataframe.loc[condition, "buy"] = 0

        # Print the 'FGI_ExtremeGreed' and 'buy' columns

        return dataframe
