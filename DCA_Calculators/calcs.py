import math
import json


with open("user_data/strategies/YOURSTRATEGY.json") as f:
    data = json.load(f)
params = data["params"]["buy"]

# print(params)

# params = {
#    "max_entry_adjustment": 8,
#    "initial_position_adjustment_trigger": -7.945,
#    "position_adjustment_step_scale": 0.893,
#    "position_adjustment_volume_scale": 1.008,
# }
# max_open_trades = 5

# replace these with your actual values

initial_position_adjustment_trigger = params["initial_position_adjustment_trigger"]
max_entry_adjustments = params["max_entry_adjustment"]
position_adjustment_step_scale = params["position_adjustment_step_scale"]
position_adjustment_volume_scale = params["position_adjustment_volume_scale"]
max_open_trades = 5
starting_capital = 3000
operating_capital = starting_capital
commited_capital = 0
simulated_open_trades = 5
simulated_entry_adjustments_each = max_entry_adjustments

if simulated_open_trades > max_open_trades:
    print(">>>>>>>>>>>>>> MAX OPEN TRADES EXCEEDED <<<<<<<<<<<<<<<")
    exit()
if simulated_entry_adjustments_each > max_entry_adjustments:
    print(">>>>>>>>>>>>>> MAX ENTRY ADJUSTMENTS EXCEEDED <<<<<<<<<<<<<<<")
    exit()

# Calculate the total volume of orders for a single trade
total_order_volume_single_trade = (
    1
    * (1 - math.pow(position_adjustment_volume_scale, max_entry_adjustments + 1))
    / (1 - position_adjustment_volume_scale)
)

total_position_adjustment_trigger = (
    abs(initial_position_adjustment_trigger)
    * (1 - math.pow(position_adjustment_step_scale, max_entry_adjustments))
    / (1 - position_adjustment_step_scale)
)

total_position_adjustment_volume = (
    1
    * (1 - math.pow(position_adjustment_volume_scale, max_entry_adjustments))
    / (1 - position_adjustment_volume_scale)
)

total_order_volume = (
    1
    * (1 - math.pow(position_adjustment_volume_scale, max_entry_adjustments))
    / (1 - position_adjustment_volume_scale)
)


total_orders = max_open_trades * (max_entry_adjustments + 1)
total_entry_adjustments = total_orders - max_open_trades

# Calculate the initial stake amount
# initial_stake_amount = (
#    S
#    * (1 - position_adjustment_volume_scale)
#    / (1 - position_adjustment_volume_scale**max_entry_adjustments)
# )
initial_stake_amount = starting_capital / (
    total_order_volume_single_trade * max_open_trades
)


operating_capital = starting_capital - initial_stake_amount * simulated_open_trades

order_sizes_per_trade = []

# Calculate the order size for each open trade
for _ in range(simulated_open_trades):
    # Create a list to store the order sizes for this trade
    order_sizes = [initial_stake_amount]

    # Calculate the order size for each additional simulated_entry_adjustments_each
    for _ in range(simulated_entry_adjustments_each):
        # The next order size is the previous order size multiplied by the position_adjustment_volume_scale
        next_order_size = (order_sizes[-1] * position_adjustment_volume_scale) * 0.99
        order_sizes.append(next_order_size)

    # Add the order sizes for this trade to the main list
    order_sizes_per_trade.append(order_sizes)

# Calculate committed capital
committed_capital = sum(sum(order_sizes) for order_sizes in order_sizes_per_trade)

# Subtract from operating capital for each trade
operating_capital = starting_capital - committed_capital


if operating_capital < 0:
    print(">>>>>>>>>>>>>> DCA SETTINGS EXCEED CAPITAL <<<<<<<<<<<<<<<")

print(f"Starting capital: {starting_capital}")

print(f"Initial stake amount: {initial_stake_amount}")

# Print the order sizes
for i, order_sizes in enumerate(order_sizes_per_trade):
    print(f"Order sizes for trade {i+1}:")
    for j, order_size in enumerate(order_sizes):
        print(f"  Order size for entry adjustment {j}: {order_size}")

print(f"Open Orders: {simulated_open_trades}")

print(f"Total_Entry Adjustments: {total_entry_adjustments}")

print(f"Operating capital: {operating_capital}")

print(f"Commited capital: {committed_capital}")

print(f"Max Entry Position Adjustment: {max_entry_adjustments}")

print(f"Position Adjustment Step Scale: {position_adjustment_step_scale}")

print(f"Position Adjustment Volume Scale: {position_adjustment_volume_scale}")

print(f"Total position adjustment volume: {total_position_adjustment_volume}")

print(f"Total position adjustment trigger: {total_position_adjustment_trigger}")

if operating_capital < 0:
    print(">>>>>>>>>>>>>> REQUIRED CAPITAL EXCEEDS AVAILABLE CAPITAL <<<<<<<<<<<<<<<")
if total_position_adjustment_trigger > 99.9:
    print(
        ">>>>>>>>>>>>>> POSITION ADJUSTMENT TRIGGER EXCEEDS 99 of price% <<<<<<<<<<<<<<<"
    )
