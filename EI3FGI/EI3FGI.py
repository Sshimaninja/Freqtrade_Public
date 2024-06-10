# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
from pandas import to_datetime
from pandas import read_csv


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

# @Rallipanos # changes by IcHiAT


def EWO(dataframe, ema_length=5, ema2_length=3):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["close"] * 100
    return emadif


class EI3FGI(IStrategy):
    INTERFACE_VERSION = 2
    """
	# ROI table:
	minimal_roi = {
		"0": 0.08,
		"20": 0.04,
		"40": 0.032,
		"87": 0.016,
		"201": 0,
		"202": -1
	}
	"""
    # Buy hyperspace params:
    buy_params = {
        "hma": 20,
        "hma_offset": 10,
        "hma_on": False,
        "hma_offset_switch": False,
        "FGIGREED_switch": True,
        "FGIFEAR_switch": True,
        "GREED": 80,
        "FEAR": 40,
        "fgi_switch": True,
        "base_nb_candles_buy": 12,
        "rsi_buy": 58,
        "ewo_high": 3.001,
        "ewo_low": -10.289,
        "low_offset": 0.987,
        "lambo2_ema_14_factor": 0.981,
        "lambo2_enabled": True,
        "lambo2_rsi_14_limit": 39,
        "lambo2_rsi_4_limit": 44,
        "buy_adx": 20,
        "buy_fastd": 20,
        "buy_fastk": 22,
        "buy_ema_cofi": 0.98,
        "buy_ewo_high": 4.179,
    }

    protection_params = {
        # protections
        # Cool down period
        "cooldown_stop_duration_candles": 5,
        # MaxDrawdown
        "max_drawdown_lookback_period_candles": 48,
        "max_drawdown_trade_limit": 20,
        "max_drawdown_stop_duration_candles": 4,
        "max_drawdown_max_allowed_drawdown": 0.2,
        # StoplossGuard
        "stoplossguard_lookback_period_candles": 24,
        "stoplossguard_trade_limit": 4,
        "stoplossguard_stop_duration_candles": 2,
        # lowprofitpairs
        "lowprofitpairs0_lookback_period_candles": 6,
        "lowprofitpairs0_trade_limit": 2,
        "lowprofitpairs0_stop_duration_candles": 60,
        "lowprofitpairs0_required_profit": 0.02,
        # lowprofitpairs
        "lowprofitpairs1_lookback_period_candles": 24,
        "lowprofitpairs1_trade_limit": 4,
        "lowprofitpairs1_stop_duration_candles": 2,
        "lowprofitpairs1_required_profit": 0.01,
    }

    # Sell hyperspace params:
    sell_params = {
        "FGIGREED_cut": True,
        "FGIFEAR_cut": True,
        "GREED": 95,
        "FEAR": 40,
        "unclog": False,
        "unclog_days": 4,
        "base_nb_candles_sell": 22,
        "high_offset": 1.014,
        "high_offset_2": 1.01,
    }

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.protection_params[
                    "cooldown_stop_duration_candles"
                ],
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": self.protection_params[
                    "max_drawdown_lookback_period_candles"
                ],
                "trade_limit": self.protection_params["max_drawdown_trade_limit"],
                "stop_duration_candles": self.protection_params[
                    "max_drawdown_stop_duration_candles"
                ],
                "max_allowed_drawdown": self.protection_params[
                    "max_drawdown_max_allowed_drawdown"
                ],
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": self.protection_params[
                    "stoplossguard_lookback_period_candles"
                ],
                "trade_limit": self.protection_params["stoplossguard_trade_limit"],
                "stop_duration_candles": self.protection_params[
                    "stoplossguard_stop_duration_candles"
                ],
                "only_per_pair": False,
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": self.protection_params[
                    "lowprofitpairs0_lookback_period_candles"
                ],
                "trade_limit": self.protection_params["lowprofitpairs0_trade_limit"],
                "stop_duration_candles": self.protection_params[
                    "lowprofitpairs0_stop_duration_candles"
                ],
                "required_profit": self.protection_params[
                    "lowprofitpairs0_required_profit"
                ],
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": self.protection_params[
                    "lowprofitpairs1_lookback_period_candles"
                ],
                "trade_limit": self.protection_params["lowprofitpairs1_trade_limit"],
                "stop_duration_candles": self.protection_params[
                    "lowprofitpairs1_stop_duration_candles"
                ],
                "required_profit": self.protection_params[
                    "lowprofitpairs1_required_profit"
                ],
            },
        ]

    # ROI table:
    minimal_roi = {
        "0": 0.99,
    }

    # Stoploss:
    stoploss = -0.99

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        8, 20, default=buy_params["base_nb_candles_buy"], space="buy", optimize=False
    )
    base_nb_candles_sell = IntParameter(
        8, 20, default=sell_params["base_nb_candles_sell"], space="sell", optimize=False
    )
    low_offset = DecimalParameter(
        0.985, 0.995, default=buy_params["low_offset"], space="buy", optimize=False
    )
    high_offset = DecimalParameter(
        1.005, 1.015, default=sell_params["high_offset"], space="sell", optimize=False
    )
    high_offset_2 = DecimalParameter(
        1.010, 1.020, default=sell_params["high_offset_2"], space="sell", optimize=False
    )

    # lambo2
    lambo2_ema_14_factor = DecimalParameter(
        0.8,
        1.2,
        decimals=3,
        default=buy_params["lambo2_ema_14_factor"],
        space="buy",
        optimize=False,
    )
    lambo2_rsi_4_limit = IntParameter(
        5, 60, default=buy_params["lambo2_rsi_4_limit"], space="buy", optimize=False
    )
    lambo2_rsi_14_limit = IntParameter(
        5, 60, default=buy_params["lambo2_rsi_14_limit"], space="buy", optimize=False
    )

    # Protection
    fast_ewo = 50
    slow_ewo = 200

    ewo_low = DecimalParameter(
        -20.0, -8.0, default=buy_params["ewo_low"], space="buy", optimize=False
    )
    ewo_high = DecimalParameter(
        3.0, 3.4, default=buy_params["ewo_high"], space="buy", optimize=False
    )
    rsi_buy = IntParameter(
        30, 70, default=buy_params["rsi_buy"], space="buy", optimize=False
    )

    # Cool down period
    cooldown_stop_duration_candles = IntParameter(
        0,
        30,
        default=protection_params["cooldown_stop_duration_candles"],
        space="protection",
        optimize=False,
    )

    # MaxDrawdown
    max_drawdown_lookback_period_candles = IntParameter(
        24,
        48,
        default=protection_params["max_drawdown_lookback_period_candles"],
        space="protection",
        optimize=False,
    )
    max_drawdown_trade_limit = IntParameter(
        4, 20, default=protection_params["max_drawdown_trade_limit"], space="protection"
    )
    max_drawdown_stop_duration_candles = IntParameter(
        2,
        4,
        default=protection_params["max_drawdown_stop_duration_candles"],
        space="protection",
        optimize=False,
    )
    max_drawdown_max_allowed_drawdown = DecimalParameter(
        0.1,
        0.3,
        default=protection_params["max_drawdown_max_allowed_drawdown"],
        space="protection",
        optimize=False,
    )

    # StoplossGuard
    stoplossguard_lookback_period_candles = IntParameter(
        24,
        48,
        default=protection_params["stoplossguard_lookback_period_candles"],
        space="protection",
        optimize=False,
    )
    stoplossguard_trade_limit = IntParameter(
        4,
        20,
        default=protection_params["stoplossguard_trade_limit"],
        space="protection",
        optimize=False,
    )
    stoplossguard_stop_duration_candles = IntParameter(
        2,
        4,
        default=protection_params["stoplossguard_stop_duration_candles"],
        space="protection",
        optimize=False,
    )

    # lowprofitpairs 0
    lowprofitpairs0_lookback_period_candles = IntParameter(
        6,
        24,
        default=protection_params["lowprofitpairs0_lookback_period_candles"],
        space="protection",
        optimize=False,
    )
    lowprofitpairs0_trade_limit = IntParameter(
        2,
        10,
        default=protection_params["lowprofitpairs0_trade_limit"],
        space="protection",
        optimize=False,
    )
    lowprofitpairs0stop_duration_candles = IntParameter(
        2,
        60,
        default=protection_params["lowprofitpairs0_stop_duration_candles"],
        space="protection",
        optimize=False,
    )
    lowprofitpairs0_required_profit = DecimalParameter(
        0.01,
        0.03,
        default=protection_params["lowprofitpairs0_required_profit"],
        space="protection",
        optimize=False,
    )

    # lowprofitpairs 1
    lowprofitpairs1_lookback_period_candles = IntParameter(
        24,
        48,
        default=protection_params["lowprofitpairs1_lookback_period_candles"],
        space="protection",
        optimize=False,
    )
    lowprofitpairs1_trade_limit = IntParameter(
        4,
        20,
        default=protection_params["lowprofitpairs1_trade_limit"],
        space="protection",
        optimize=False,
    )
    lowprofitpairs1stop_duration_candles = IntParameter(
        2,
        4,
        default=protection_params["lowprofitpairs1_stop_duration_candles"],
        space="protection",
        optimize=False,
    )
    lowprofitpairs1_required_profit = DecimalParameter(
        0.01,
        0.03,
        default=protection_params["lowprofitpairs1_required_profit"],
        space="protection",
        optimize=False,
    )

    # Unclog:
    unclog = CategoricalParameter(
        [True, True], default=True, space="sell", optimize=False
    )

    unclog_days = IntParameter(
        1, 35, default=sell_params["unclog_days"], space="sell", optimize=False
    )

    # Hyperoptable parameters
    # Buy hyperspace params:
    hma_1d_tf = IntParameter(
        5, 200, default=buy_params["hma"], space="buy", optimize=False
    )
    hma_1d_IO = CategoricalParameter(
        [True, False], default=buy_params["hma_on"], space="buy", optimize=False
    )
    # sma_1d_tf = IntParameter(5, 75, default=buy_params["sma"], space="buy", optimize=False)
    hma_1d_offset_tf = IntParameter(
        1, 100, default=buy_params["hma_offset"], space="buy", optimize=False
    )
    hma_1d_offset_IO = CategoricalParameter(
        [True, False],
        default=buy_params["hma_offset_switch"],
        space="buy",
        optimize=False,
    )
    # hma_on = CategoricalParameter([True, False], default=True, space="buy", optimize=False)
    # hma_offset_switch = CategoricalParameter([True, False], default=True, space="buy", optimize=False)

    # FGI data comes from a CSV file in order to allow for backtesting
    # We pull the most recent data in the file later.
    fgi_switch: CategoricalParameter = CategoricalParameter(
        [True, False], default=buy_params["fgi_switch"], space="buy", optimize=False
    )
    fgi_data = read_csv(
        "user_data/data/FGI_data/fgi_data.csv", index_col=0, parse_dates=True
    )

    # FGI
    fgi_fear_switch = CategoricalParameter(
        [True, False], default=buy_params["FGIFEAR_switch"], space="buy", optimize=False
    )
    fgi_greed_switch = CategoricalParameter(
        [True, False],
        default=buy_params["FGIGREED_switch"],
        space="buy",
        optimize=False,
    )
    extreme_greed_guard = CategoricalParameter(
        [60, 65, 70, 75, 80, 85, 90, 92, 95],
        default=buy_params["GREED"],
        space="buy",
        optimize=False,
    )
    extreme_fear_guard = CategoricalParameter(
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        default=buy_params["FEAR"],
        space="buy",
        optimize=False,
    )

    fgi_fear_cut = CategoricalParameter(
        [True, False], default=sell_params["FGIFEAR_cut"], space="sell", optimize=False
    )
    fgi_greed_cut = CategoricalParameter(
        [True, False], default=sell_params["FGIGREED_cut"], space="sell", optimize=False
    )
    extreme_greed_cut = CategoricalParameter(
        [60, 65, 70, 72, 75, 78, 80, 82, 85, 90, 95],
        default=sell_params["GREED"],
        space="sell",
        optimize=False,
    )
    extreme_fear_cut = CategoricalParameter(
        [5, 10, 12, 15, 18, 20, 25, 30, 35, 40, 42, 45, 48, 50, 55],
        default=sell_params["FEAR"],
        space="sell",
        optimize=False,
    )

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    # cofi
    is_optimize_cofi = False
    buy_ema_cofi = DecimalParameter(0.96, 0.98, default=0.97, optimize=is_optimize_cofi)
    buy_fastk = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_fastd = IntParameter(20, 30, default=20, optimize=is_optimize_cofi)
    buy_adx = IntParameter(20, 30, default=30, optimize=is_optimize_cofi)
    buy_ewo_high = DecimalParameter(2, 12, default=3.553, optimize=is_optimize_cofi)

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    # Optional order time in force.
    order_time_in_force = {"buy": "gtc", "sell": "gtc"}

    # Optimal timeframe for the strategy
    timeframe = "5m"
    inf_1h = "1h"
    inf_1d = "1d"

    process_only_new_candles = True
    startup_candle_count = 400

    plot_config = {
        "main_plot": {
            "hma_50": {"color": "blue"},
            "sma_9": {"color": "green"},
            "ema_14": {"color": "red"},
            "ema_8": {"color": "purple"},
            "dema_30": {"color": "orange"},
            "dema_200": {"color": "brown"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "blue"},
                "rsi_fast": {"color": "green"},
                "rsi_slow": {"color": "red"},
                "rsi_4": {"color": "purple"},
                "rsi_14": {"color": "orange"},
            },
            "EWO": {
                "EWO": {"color": "blue"},
            },
            "Stochastic Fast": {
                "fastd": {"color": "green"},
                "fastk": {"color": "red"},
            },
            "ADX": {
                "adx": {"color": "blue"},
            },
            "Pump Strength": {
                "pump_strength": {"color": "green"},
            },
        },
    }

    def custom_sell(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        # Sell any positions at a loss if they are held for more than 7 days.
        if self.unclog.value == True:
            if (
                current_profit < -0.04
                and (current_time - trade.open_date_utc).days
                >= self.unclog_days.value  # 2
            ):
                return "unclog"

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, "1h") for pair in pairs]
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
        informative_pairs.append((btc_info_pair, self.inf_1h))
        informative_pairs.append((btc_info_pair, self.inf_1d))

        return informative_pairs

    def fetch_FGI(self, dataframe: DataFrame) -> DataFrame:
        if self.fgi_data.index.tz is None:
            self.fgi_data.index = self.fgi_data.index.tz_localize("UTC")
        # Convert the 'date' column to datetime
        dataframe["date"] = to_datetime(dataframe["date"])
        self.fgi_data.index = to_datetime(self.fgi_data.index)

        # Print unique dates in dataframe and fgi_data
        # print("Unique dates in dataframe:", dataframe['date'].unique())
        # print("Unique dates in fgi_data:", self.fgi_data.index.unique())
        # print("Dataframe columns: ", dataframe.columns)
        # print("FGI data columns: ", self.fgi_data.columns)
        # print("Dataframe index: ", dataframe.index)

        # Initialize a new 'FGI' column in dataframe
        dataframe["FGI"] = np.nan

        # Iterate over each row in dataframe
        for i, row in dataframe.iterrows():
            # If the date exists in fgi_data, add the 'value' to the 'FGI' column in dataframe
            if row["date"] in self.fgi_data.index:
                dataframe.loc[i, "FGI"] = self.fgi_data.loc[row["date"], "value"]

        return dataframe

    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # print("df pump_dump_protection:", dataframe)
        df36h = dataframe.copy().shift(432)  # TODO FIXME: This assumes 5m timeframe
        df24h = dataframe.copy().shift(288)  # TODO FIXME: This assumes 5m timeframe

        dataframe["volume_mean_short"] = dataframe["volume"].rolling(4).mean()
        dataframe["volume_mean_long"] = df24h["volume"].rolling(48).mean()
        dataframe["volume_mean_base"] = df36h["volume"].rolling(288).mean()

        dataframe["volume_change_percentage"] = (
            dataframe["volume_mean_long"] / dataframe["volume_mean_base"]
        )

        dataframe["rsi_mean"] = dataframe["rsi"].rolling(48).mean()

        dataframe["pnd_volume_warn"] = np.where(
            (dataframe["volume_mean_short"] / dataframe["volume_mean_long"] > 5.0),
            -1,
            0,
        )

        return dataframe

    def base_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # print("df base_tf_btc_ind:", dataframe)
        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe["price_trend_long"] = (
            dataframe["close"].rolling(8).mean()
            / dataframe["close"].shift(8).rolling(144).mean()
        )

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ["date", "open", "high", "low", "close", "volume"]
        dataframe.rename(
            columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True
        )

        return dataframe

    def info_tf_btc_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe["rsi_8"] = ta.RSI(dataframe, timeperiod=8)

        # Add prefix
        # -----------------------------------------------------------------------------------------
        ignore_columns = ["date", "open", "high", "low", "close", "volume"]
        dataframe.rename(
            columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True
        )

        return dataframe

    def info_tf1d_btc_indicators(
        self, dataframe: DataFrame, metadata: dict
    ) -> DataFrame:

        # Indicators
        # -----------------------------------------------------------------------------------------
        dataframe["1d_close"] = dataframe["close"]
        hma = self.hma_1d_tf.value
        dataframe["hma"] = qtpylib.hull_moving_average(dataframe["close"], window=hma)
        dataframe["hma_offset"] = dataframe["hma"].shift(self.hma_1d_offset_tf.value)

        # FGI
        # -----------------------------------------------------------------------------------------
        # if self.fgi_switch.value == True:
        dataframe = self.fetch_FGI(dataframe)
        if "FGI" not in dataframe.columns:
            logger.error("Failed to create 'FGI' column in dataframe")
            raise ValueError("Failed to create 'FGI' column in dataframe")

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            print("DataProvider not available")
            # Don't do anything if DataProvider is not available.
            return dataframe

        if self.config["stake_currency"] in ["USDT", "BUSD"]:
            btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            btc_info_pair = "BTC/USDT"

        btc_info_tf_1d = self.dp.get_pair_dataframe(btc_info_pair, "1d")

        # Check if 'close' column exists in the dataframe
        if "close" not in btc_info_tf_1d.columns:
            logger.error("'close' column not in btc_info_tf_1d dataframe")
            return dataframe

        btc_info_tf_1d = self.info_tf1d_btc_indicators(
            btc_info_tf_1d, metadata
        )  # Call the method here
        dataframe = merge_informative_pair(
            dataframe, btc_info_tf_1d, self.timeframe, "1d", ffill=True
        )

        drop_columns = [
            f"{s}_{self.inf_1d}" for s in ["date", "open", "high", "low", "volume"]
        ]
        dataframe.drop(
            columns=dataframe.columns.intersection(drop_columns), inplace=True
        )

        # print("EI3 populate_indicators:", dataframe.columns)
        btc_info_tf_1h = self.dp.get_pair_dataframe(btc_info_pair, self.inf_1h)
        # print("btc_inf_tf:", btc_info_tf_1h)
        btc_info_tf_1h = self.info_tf_btc_indicators(btc_info_tf_1h, metadata)
        dataframe = merge_informative_pair(
            dataframe, btc_info_tf_1h, self.timeframe, self.inf_1h, ffill=True
        )
        drop_columns = [
            f"{s}_{self.inf_1h}"
            for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        dataframe.drop(
            columns=dataframe.columns.intersection(drop_columns), inplace=True
        )

        btc_base_tf = self.dp.get_pair_dataframe(btc_info_pair, self.timeframe)
        btc_base_tf = self.base_tf_btc_indicators(btc_base_tf, metadata)
        dataframe = merge_informative_pair(
            dataframe, btc_base_tf, self.timeframe, self.timeframe, ffill=True
        )
        drop_columns = [
            f"{s}_{self.timeframe}"
            for s in ["date", "open", "high", "low", "close", "volume"]
        ]
        dataframe.drop(
            columns=dataframe.columns.intersection(drop_columns), inplace=True
        )

        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f"ma_buy_{val}"] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f"ma_sell_{val}"] = ta.EMA(dataframe, timeperiod=val)

        dataframe["hma_50"] = qtpylib.hull_moving_average(dataframe["close"], window=50)

        dataframe["sma_9"] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe["EWO"] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)

        # lambo2
        dataframe["ema_14"] = ta.EMA(dataframe, timeperiod=14)
        dataframe["rsi_4"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)

        # Pump strength
        dataframe["dema_30"] = ftt.dema(dataframe, period=30)
        dataframe["dema_200"] = ftt.dema(dataframe, period=200)
        dataframe["pump_strength"] = (
            dataframe["dema_30"] - dataframe["dema_200"]
        ) / dataframe["dema_30"]

        # Cofi
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]
        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["ema_8"] = ta.EMA(dataframe, timeperiod=8)

        dataframe = self.pump_dump_protection(dataframe, metadata)

        return dataframe

    def FGI_cut(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe = super().populate_sell_trend(dataframe, metadata)
        cutLoss = []

        if self.fgi_fear_cut.value == True:
            cutLoss.append(dataframe["FGI_1d"] < self.extreme_fear_cut.value)

        if self.fgi_greed_cut.value == True:
            cutLoss.append(dataframe["FGI_1d"] > self.extreme_greed_cut.value)

        for condition in cutLoss:
            dataframe.loc[condition, "sell"] = 1

        return dataframe

    def FGI_guard(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        guard = []

        if self.fgi_fear_switch.value == True:
            guard.append(dataframe["FGI_1d"] < self.extreme_fear_guard.value)

        if self.fgi_greed_switch.value == True:
            guard.append(dataframe["FGI_1d"] > self.extreme_greed_guard.value)

        for condition in guard:
            dataframe.loc[condition, "sell"] = 0

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
        conditions = []
        dataframe.loc[:, "buy_tag"] = ""

        lambo2 = (
            # bool(self.lambo2_enabled.value) &
            # (dataframe['pump_warning'] == 0) &
            (
                dataframe["close"]
                < (dataframe["ema_14"] * self.lambo2_ema_14_factor.value)
            )
            & (dataframe["rsi_4"] < int(self.lambo2_rsi_4_limit.value))
            & (dataframe["rsi_14"] < int(self.lambo2_rsi_14_limit.value))
        )
        dataframe.loc[lambo2, "buy_tag"] += "lambo2_"
        conditions.append(lambo2)

        buy1ewo = (
            (dataframe["rsi_fast"] < 35)
            & (
                dataframe["close"]
                < (
                    dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"]
                    * self.low_offset.value
                )
            )
            & (dataframe["EWO"] > self.ewo_high.value)
            & (dataframe["rsi"] < self.rsi_buy.value)
            & (dataframe["volume"] > 0)
            & (
                dataframe["close"]
                < (
                    dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                    * self.high_offset.value
                )
            )
        )
        dataframe.loc[buy1ewo, "buy_tag"] += "buy1eworsi_"
        conditions.append(buy1ewo)

        buy2ewo = (
            (dataframe["rsi_fast"] < 35)
            & (
                dataframe["close"]
                < (
                    dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"]
                    * self.low_offset.value
                )
            )
            & (dataframe["EWO"] < self.ewo_low.value)
            & (dataframe["volume"] > 0)
            & (
                dataframe["close"]
                < (
                    dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                    * self.high_offset.value
                )
            )
        )
        dataframe.loc[buy2ewo, "buy_tag"] += "buy2ewo_"
        conditions.append(buy2ewo)

        is_cofi = (
            (dataframe["open"] < dataframe["ema_8"] * self.buy_ema_cofi.value)
            & (qtpylib.crossed_above(dataframe["fastk"], dataframe["fastd"]))
            & (dataframe["fastk"] < self.buy_fastk.value)
            & (dataframe["fastd"] < self.buy_fastd.value)
            & (dataframe["adx"] > self.buy_adx.value)
            & (dataframe["EWO"] > self.buy_ewo_high.value)
        )
        dataframe.loc[is_cofi, "buy_tag"] += "cofi_"
        conditions.append(is_cofi)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "buy"] = 1

        dont_buy_conditions = []
        guard = []

        # don't buy if there seems to be a Pump and Dump event.
        dont_buy_conditions.append((dataframe["pnd_volume_warn"] < 0.0))

        # BTC price protection
        dont_buy_conditions.append((dataframe["btc_rsi_8_1h"] < 35.0))

        if dont_buy_conditions:
            for condition in dont_buy_conditions:
                dataframe.loc[condition, "buy"] = 0

        # print(dataframe.columns)

        if self.hma_1d_offset_IO.value == True:
            guard.append(dataframe["1d_close_1d"] > dataframe["hma_offset_1d"])
            # print(dataframe["1d_close_1d"])

        # Set the 'buy' column to 0 where any of the conditions in 'guard' are True
        for condition in guard:
            dataframe.loc[condition, "buy"] = 0

        dataframe = self.FGI_guard(dataframe, metadata)
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe["close"] > dataframe["hma_50"])
                & (
                    dataframe["close"]
                    > (
                        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                        * self.high_offset_2.value
                    )
                )
                & (dataframe["rsi"] > 50)
                & (dataframe["volume"] > 0)
                & (dataframe["rsi_fast"] > dataframe["rsi_slow"])
            )
            | (
                (dataframe["close"] < dataframe["hma_50"])
                & (
                    dataframe["close"]
                    > (
                        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                        * self.high_offset.value
                    )
                )
                & (dataframe["volume"] > 0)
                & (dataframe["rsi_fast"] > dataframe["rsi_slow"])
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "sell"] = 1

        # Call FGI_cut function here
        dataframe = self.FGI_cut(dataframe, metadata)

        return dataframe

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        sell_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:

        # trade.sell_reason = sell_reason + "_" + trade.buy_tag

        return True


def pct_change(a, b):
    return (b - a) / a
