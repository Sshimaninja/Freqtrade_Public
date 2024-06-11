# Indicators, Freqtrade strategies, improvements on existing strategies, helper scripts:

I ususally have one main file for each strategy that works, and several copies that I'm experimenting with. 

If this were a publicly maintained repo, I wouldn't allow that kidn of mess, but this is only up here to showcase my ability to work with Python and develop finance trading strategies based on technical indicators, and to create parent classes that control those strategies with higher timeframe/external data sources. 


# Fear and Greed Index (FGI):

### Example:

Step 1: If you want to backtest anything, you need a database. FGIDatabase.py calls that FGI data for you and puts it in a .csv file. The data is called from alternative.me, who are the original authors of the Bitcoin Fear and Greed Index, which calls data from inflows, outflows, and among other things user surveys.


Step 2: In the example FGIHMABTCCtrl.py the FGI indicator and database call is integrated into the sample HMA offset strategy, controlling the bot using BTC Fear and Greed Index on the 1D timeframe by adding an infomrative timeframe column with the indicator, adding the FGI dailiy data to it, and then controlling the populate_buy/sell_trend functions.

## Notes: 

- FGI.py will give you a simple daily indicator that takes data from the maintainers of the actual FGI. You can't backtest this.

- FGITest.py will give you the backtestable FGI indicator which uses all the stored data for testing. 

In order to use this, you must manually download historical FGI data for the db (see example above). 

I've written bots for trading for several years, and I think the biggest problem is lack of context with the world outside of the strategy itself. Data gathered from only the technical indicators operate in a vacuum. The Fear and Greed Indicator, though it is flawed in several ways as a reliable indicator, still has some information that I think is valuable, specifically a 'sentiment' indication, partially weighted by *user surveys* which I think is extremely unique and difficult to find without directed effort or funding.

So, the most interesting parent indicator I have on here is the Fear and Greed Index daily controller. I've done some hyperopting with this and found that actually, it is useful as a shitcoin loss-protection device. 

The only other indicator I'd like to try is a Twitter (presently 'X') sentiment indicator, which shouldn't be too difficult but will take some time to implement. 

I find that simply using the Bitcoin Daily FGI > 85 or < 20 works great as a power on/off switch for your alt-coin trading bots, because most poeple will dump when Bitcoin looks weak.

Conversely, it's always a good idea to get out early by turning off the bots when Bitcoin greed gets super insane. Most strategies have difficulty pinpointing a cliff-edge, while it's generally clear to human observers, so the greed index works quite well when extreme greed hits, and you turn off your bots.


## FGI Guard:

Turns the bot on/off at FGI specified level.

## FGI Cut:

There's also a 'stoploss' style instant sell for any in-process trades at somewhere near 90. It makes sense and tests well to minimuse losses.

# Position Adjustment Backtest Improvements:

## DCAFREQ: 

I wrote a position adjustment strategy (sometimes incorrectly referred to as 'DCA') that inherits a given parent strategy, which simulates a full order book before allowing trades to open. 

The reason I did this is because backtest results were wildly optimistic because the hyperopt could seriously over-commit the order book, or make it so that a single trade opened with the rest of the position_adjustment trades far below 100% of the price range.

## DCA_Calculators

Plug in your preferred settings and your simulated trade numbers, and see what your spread looks like. Very simple but I didn't find a shake and bake solution so I made my own. Let me know if you like it!

### Donations:
Donations:

Bitcoin: 
bc1q2s096sa6h63uqqlqu68g62xa8hqkasj9jrthqw 

ERC20: 
0xd4268778C7a462FDa525DaB2A113CcFe46fabdE2
