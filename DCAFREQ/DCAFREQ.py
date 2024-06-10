# --- Do not remove these libs ---
from importlib import metadata
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from typing import Optional
from pandas import DataFrame, Series
from freqtrade.persistence import Trade

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

# from  import (
#     buy_params,
#     sell_params
#     )


import technical.indicators as ftt
import math
import logging

logger = logging.getLogger(__name__)

# @Rallipanos # changes by IcHiAT
# Position Adjustment adjusted and cheks implemented by sshimaninja https://github.com/Sshimaninja/

# Quick hyperopt command with tee results:
#  freqtrade hyperopt -c user_data/configDCATEST.json -s YOURSTRATEGY -j 1 --timerange 20230101-20240212 --hyperopt-loss SharpeHyperOptLoss -e 2000 | tee user_data/strategies/hyperopt_results/YOURSTRATEGY_20240218.txt


class DCAFREQ(YOURSTRATEGY):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_trade_timestamp = None

    position_adjustment_enable = True
    plot_config = {
        "main_plot": {
            "position_adjustment": {
                "color": "orange",
            },
        }
    }

    buy_params = {
        "dca_buy": True,
        "initial_position_adjustment_trigger": -7.945,
        "max_entry_position_adjustment": 8,
        "position_adjustment_step_scale": 0.893,
        "position_adjustment_volume_scale": 1.008,
        "dca_min_rsi": 35,
    }
    sell_params = {
        "dca_sell": False,
        # "dca_stoploss": False,
        "initial_position_reduction_trigger": 1.8,  # We want to use sell signals from parent class.
        "max_exit_position_reduction": 3,
        "position_reduction_step_scale": 1.1,
        "position_reduction_volume_scale": 1.2,
    }

    # Hyperoptable parameters#
    dca_buy = CategoricalParameter(
        [True, False], default=buy_params["dca_buy"], space="buy", optimize=False
    )
    max_entry_adjustment = IntParameter(
        3,
        7,
        default=buy_params["max_entry_position_adjustment"],
        space="buy",
        optimize=True,
    )
    initial_position_adjustment_trigger = DecimalParameter(
        -7.0,
        -1.0,
        default=buy_params["initial_position_adjustment_trigger"],
        space="buy",
        optimize=True,
    )
    position_adjustment_step_scale = DecimalParameter(
        0.8,
        1.23,
        default=buy_params["position_adjustment_step_scale"],
        space="buy",
        optimize=True,
    )
    position_adjustment_volume_scale = DecimalParameter(
        1.0,
        2.0,
        default=buy_params["position_adjustment_volume_scale"],
        space="buy",
        optimize=True,
    )

    # append buy_params of parent class
    buy_params.update(YOURSTRATEGY.buy_params)
    sell_params.update(YOURSTRATEGY.sell_params)

    dca_min_rsi = IntParameter(
        15, 75, default=buy_params["dca_min_rsi"], space="buy", optimize=False
    )

    # Set up logging
    logger = logging.getLogger("trade_sizes")
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler("tradeSizes.txt", mode="w")
    handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        available_stake_amount = self.wallets.get_available_stake_amount()
        max_open_trades = self.config["max_open_trades"]
        max_entry_adjustments = self.max_entry_adjustment.value
        position_adjustment_volume_scale = self.position_adjustment_volume_scale.value

        if self.dca_buy.value == True:

            total_order_multiple_single_trade = (
                1
                * (
                    1
                    - math.pow(
                        position_adjustment_volume_scale, max_entry_adjustments + 1
                    )
                )
                / (1 - position_adjustment_volume_scale)
            )

            open_slots = max_open_trades - len(self.wallets.get_all_positions())

            # The maximum stake available for each trade
            if open_slots == 0:
                current_max_stake_available_per_trade = 0
                return False
            else:
                current_max_stake_available_per_trade = available_stake_amount / (
                    open_slots
                )

                # Calculate the initial stake amount
                initial_stake_amount = current_max_stake_available_per_trade / (
                    total_order_multiple_single_trade * (max_entry_adjustments + 1)
                )

            return initial_stake_amount

        if self.dca_buy.value == False:
            # Limit the proposed stake to the available balance
            return min(proposed_stake, self.wallets.get_available_stake_amount())

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float]:

        ### This is an attempt to mimic the behaviour of backtesting on a live bot. Ref: https://www.freqtrade.io/en/latest/backtesting/#assumptions-made-by-backtesting
        # Unfortunately, this seems to reduce profit far below the backtesting results, whereas, if it worked, it should mimic the backtest logic exactly. More testing is needed.

        # self.last_trade_timestamp = current_time
        # if (
        #    self.last_trade_timestamp is not None
        #    and current_time <= self.last_trade_timestamp + timedelta(minutes=5)
        # ):
        #    print(
        #        f"Cannot order more than once per candle. Last order time: {self.last_trade_timestamp}"
        #    )
        #    return None
        # print(f"New candle: orders made: {trade.get_open_trade_count()}")

        available_stake_amount = self.wallets.get_available_stake_amount()
        max_open_trades = self.config["max_open_trades"]
        max_entry_adjustments = self.max_entry_adjustment.value
        position_adjustment_volume_scale = self.position_adjustment_volume_scale.value
        open_slots = max_open_trades - trade.get_open_trade_count()

        def calculate_position_adjustment_size(
            stake_amount: float, count_of_entries: int
        ) -> float:

            return (
                stake_amount
                * (1 + (count_of_entries * self.position_adjustment_volume_scale.value))
                * 0.99  # Error tolerance
            )

        def simulate_full_order_book(self, trade: Trade) -> bool:
            # THIS IS A SIMULATED CALCULATION OF THE DCA SETTINGS TO ENSURE THEY WORK IN GENERAL

            # I ended up calculating this in real-time in the position_adjustment class, thus, this is not necessary, but remains here a monolith to my attempt to understand the logic.
            # You can remove it, or if you want to use the code for another purpose, feel free to do so. Credit me (Sshimaninja) if possible.

            # Available stake amount is the amount of capital that is not currently in active trades
            current_free_capital = available_stake_amount  # 3000  # Since active trades cause a negative balance, we need to set a dummy capital for the simulation
            open_slots = max_open_trades - trade.get_open_trade_count()
            # The maximum stake required for each trade
            total_order_multiple_single_trade = (
                1
                * (
                    1
                    - math.pow(
                        position_adjustment_volume_scale, max_entry_adjustments + 1
                    )
                )
                / (1 - position_adjustment_volume_scale)
            )
            # The maximum stake available for each trade
            if open_slots == 0:
                return False
            else:
                # The current stake amount taking available capital and open trades into account
                current_stake_amount = current_free_capital / (
                    total_order_multiple_single_trade * open_slots  # max_open_trades
                )
                current_max_stake_available_per_trade = current_free_capital / (
                    open_slots
                )
            order_sizes_per_new_trade = []
            # Calculate the order size for each new trade
            for _ in range(open_slots):
                # Create a list to store the order sizes for this trade
                new_order_sizes = [current_stake_amount]
                # Calculate the order size for each additional simulated_entry_adjustments_each
                for _ in range(max_entry_adjustments):
                    # The next order size is the previous order size multiplied by the position_adjustment_volume_scale
                    next_order_size = (
                        new_order_sizes[-1] * position_adjustment_volume_scale
                    ) * 0.99  # value to round down to the nearest 0.01
                    new_order_sizes.append(next_order_size)
                # Add the order sizes for this trade to the main list
                order_sizes_per_new_trade.append(new_order_sizes)
            # Calculate committed capital
            fully_committed_available_capital = sum(
                sum(new_order_sizes) for new_order_sizes in order_sizes_per_new_trade
            )
            current_max_stake_required_per_trade = max(
                sum(order_sizes) for order_sizes in order_sizes_per_new_trade
            )
            # Subtract from operating capital for each trade
            operating_capital = current_free_capital - fully_committed_available_capital

            if (
                current_max_stake_required_per_trade
                > current_max_stake_available_per_trade
            ):
                print(
                    ">>>>>>>>>>>>>> max_stake_required > max_stake_available per trade <<<<<<<<<<<<<<<"
                )
                return False
            if operating_capital < 0:
                print(">>>>>>>>>>>>>> DCA SETTINGS EXCEED CAPITAL <<<<<<<<<<<<<<<")
                return False

        def calculate_position_adjustment_trigger(
            initial_trigger: float, count_of_entries: int
        ) -> float:
            if self.position_adjustment_step_scale.value == 1:
                return abs(initial_trigger)
            else:
                return abs(initial_trigger) + (
                    abs(initial_trigger)
                    * self.position_adjustment_step_scale.value
                    * (
                        math.pow(
                            self.position_adjustment_step_scale.value,
                            (count_of_entries - 1),
                        )
                        - 1
                    )
                    / (self.position_adjustment_step_scale.value - 1)
                )

        if self.dca_buy.value == True:

            # I wrote this into the logic of each trade below, so this is deprecated.
            # if simulate_full_order_book(self, trade) == False:
            #    return None

            # Refers this trade specifically.
            filled_entries = trade.select_filled_orders(trade.entry_side)
            count_of_entries = trade.nr_of_successful_entries

            if 1 <= count_of_entries < self.max_entry_adjustment.value:

                position_adjustment_trigger = calculate_position_adjustment_trigger(
                    self.initial_position_adjustment_trigger.value, count_of_entries
                )

                if current_profit <= position_adjustment_trigger:
                    try:

                        # Uses the first order in the already opened trade to caclculate the following trades in a loop.
                        stake_amount = filled_entries[0].cost
                        # stake_amount = filled_entries[count_of_entries - 1].cost
                        stake_amount = calculate_position_adjustment_size(
                            stake_amount, count_of_entries
                        )
                        total_stake_amount_required = (
                            stake_amount
                            * (
                                1
                                - position_adjustment_volume_scale
                                ** (max_entry_adjustments)
                            )
                            / (1 - position_adjustment_volume_scale)
                        )
                        remaining_required_stake_amount = (
                            stake_amount
                            * (
                                1
                                - position_adjustment_volume_scale
                                ** ((max_entry_adjustments + 1) - count_of_entries)
                            )
                            / (1 - position_adjustment_volume_scale)
                        )
                        simulated_fully_committed_available_capital = (
                            total_stake_amount_required * open_slots
                        )
                        # print( # Debugging
                        #    f"_______________________{asset}_____________________________"
                        # )

                        simulate_total_position_adjustment_trigger = abs(
                            self.initial_position_adjustment_trigger.value
                        ) + (
                            abs(self.initial_position_adjustment_trigger.value)
                            * self.position_adjustment_step_scale.value
                            * (
                                math.pow(
                                    self.position_adjustment_step_scale.value,
                                    (max_entry_adjustments - 1),
                                )
                                - 1
                            )
                            / (self.position_adjustment_step_scale.value - 1)
                        )

                        # Check the values don't exceed price range.
                        if simulate_total_position_adjustment_trigger > 99.0:
                            print(
                                f"Total adjustments exceeds 100% of price: Total adjustments: {simulate_total_position_adjustment_trigger}"
                            )
                            return None
                        if available_stake_amount < remaining_required_stake_amount:
                            print(
                                f"Available stake amount: {available_stake_amount} < Required stake amount: {remaining_required_stake_amount}"
                            )
                            return None
                        if (
                            simulated_fully_committed_available_capital
                            > available_stake_amount
                        ):
                            print(
                                f"Required stake amount for full order book: {simulated_fully_committed_available_capital} > Available stake amount: {available_stake_amount}"
                            )
                            return None
                        return stake_amount

                    except Exception as exception:
                        print(f"EXCEPTION: {exception}")
                        return None
