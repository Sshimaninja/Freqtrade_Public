from pandas import DataFrame
from FreqCtrlBTC import YOURSTRATEGY


class FGITest(YOURSTRATEGY):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load the FGI data
        self.fgi_data = pd.read_csv(
            "user_data/data/FGI_data/fgi_data.csv",
            index_col="timestamp",
            parse_dates=True,
        )

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_buy_trend(dataframe, metadata)

        # Get the FGI value for the current date
        current_date = dataframe.iloc[-1].name.date()
        fear_and_greed_state = self.fgi_data.loc[current_date]["state"]

        # Add a condition to only buy when the FGI is not in the "Extreme Greed" state
        if fear_and_greed_state != "Extreme Greed":
            dataframe.loc[
                (
                    # ... your existing conditions here ...
                ),
                "buy",
            ] = 1

        return dataframe
