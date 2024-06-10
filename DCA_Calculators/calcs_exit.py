import math
import json


with open("user_data/strategies/YOURSTRATEGY.json") as f:
    data = json.load(f)
params = data["params"]["sell"]

# print(params)

# params = {
#    "max_exit_reduction": 8,
#    "initial_position_reduction_trigger": -7.945,
#    "position_reduction_step_scale": 0.893,
#    "position_reduction_volume_scale": 1.008,
# }
# max_open_trades = 5

# replace these with your actual values
# __________________________POSITION REDUCTION CALCULATIONS__________________________
initial_position_reduction_trigger = params["initial_position_reduction_trigger"]
max_exit_reductions = params["max_exit_reduction"]
position_reduction_step_scale = params["position_reduction_step_scale"]
position_reduction_volume_scale = params["position_reduction_volume_scale"]
trade_size = 1000
operating_tokens = trade_size
commited_tokens = 1000
simulated_exit_reductions_each = max_exit_reductions


# Calculate the total volume of orders for a single trade
total_trade_volume = (
    1
    * (1 - math.pow(position_reduction_volume_scale, max_exit_reductions + 1))
    / (1 - position_reduction_volume_scale)
)

total_position_reduction_trigger = (
    abs(initial_position_reduction_trigger)
    * (1 - math.pow(position_reduction_step_scale, max_exit_reductions))
    / (1 - position_reduction_step_scale)
)

total_position_reduction_volume = (
    1
    * (1 - math.pow(position_reduction_volume_scale, max_exit_reductions))
    / (1 - position_reduction_volume_scale)
)

total_order_volume = (
    1
    * (1 - math.pow(position_reduction_volume_scale, max_exit_reductions))
    / (1 - position_reduction_volume_scale)
)


total_orders = max_exit_reductions + 1


initial_token_amount = trade_size / total_trade_volume

trade_remainder = trade_size - initial_token_amount

order_sizes_per_trade = []

# Calculate the order size for each open trade

# Create a list to store the order sizes for this trade
order_sizes = [initial_token_amount]

# Calculate the order size for each additional simulated_exit_reductions_each
for _ in range(simulated_exit_reductions_each):
    # The next order size is the previous order size multiplied by the position_reduction_volume_scale
    next_order_size = (order_sizes[-1] * position_reduction_volume_scale) * 0.99
    order_sizes.append(next_order_size)


# Calculate committed tokens
remaining_tokens = sum(order_sizes)


if operating_tokens < 0:
    print(">>>>>>>>>>>>>> DCA SETTINGS EXCEED tokens <<<<<<<<<<<<<<<")

print(f"Starting tokens: {trade_size}")

print(f"Initial token amount: {initial_token_amount}")

# Print the order sizes
for i, order_sizes in enumerate(order_sizes_per_trade):
    print(f"Order sizes for trade {i+1}:")
    for j, order_size in enumerate(order_sizes):
        print(f"  Order size for exit reduction {j}: {order_size}")


print(f"Commited tokens: {remaining_tokens}")

print(f"Max exit Position reduction: {max_exit_reductions}")

print(f"Position reduction Step Scale: {position_reduction_step_scale}")

print(f"Position reduction Volume Scale: {position_reduction_volume_scale}")

print(f"Total position reduction volume: {total_position_reduction_volume}")

print(f"Total position reduction trigger: {total_position_reduction_trigger}")

if operating_tokens < 0:
    print(">>>>>>>>>>>>>> REQUIRED tokens EXCEEDS AVAILABLE tokens <<<<<<<<<<<<<<<")
if total_position_reduction_trigger > 99.9:
    print(
        ">>>>>>>>>>>>>> POSITION reduction TRIGGER EXCEEDS 99 of price% <<<<<<<<<<<<<<<"
    )
# __________________________POSITION REDUCTION CALCULATIONS__________________________
