import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest import Strategy, Portfolio

class SNPForecastingStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol."""

    def __init__(self, symbol, bars, prelim_signals, w):
        self.symbol = symbol
        self.bars = bars
        self.prelim_signals = prelim_signals
        self.w = w


    def generate_signals(self):
        """Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0)."""
        signals = self.prelim_signals
        # volatility = self.bars['Close'].rolling(self.w).std(ddof=0)
        # signals['Volatiity'] = volatility
        # vals = np.array(self.bars['Close'])
        # vals = (vals - np.mean(vals))/np.std(vals)
        # signals['z_score'] = vals
        # dup = signals.pivot_table(index=['signal'], aggfunc='size')
        # signals[abs(signals['z_score']) > 2] = 0
        # dup = signals.pivot_table(index=['signal'], aggfunc='size')
        return signals

class MarketIntradayPortfolio(Portfolio):
    """Buys or sells 500 shares of an asset at the opening price of
    every bar, depending upon the direction of the forecast, closing
    out the trade at the close of the bar.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, symbol, bars, signals, initial_capital=100000.0):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()

    def generate_positions(self):
        """Generate the positions DataFrame, based on the signals
        provided by the 'signals' DataFrame."""
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)

        # Long or short 500 shares of SPY based on
        # directional signal every day
        positions[self.symbol] = 500 * self.signals['signal']
        return positions

    def backtest_portfolio(self):
        """Backtest the portfolio and return a DataFrame containing
        the equity curve and the percentage returns."""

        # Set the portfolio object to have the same time period
        # as the positions DataFrame
        portfolio = pd.DataFrame(index=self.positions.index)
        pos_diff = self.positions.diff()

        # Work out the intraday profit of the difference
        # in open and closing prices and then determine
        # the daily profit by longing if an up day is predicted
        # and shorting if a down day is predicted
        portfolio['price_diff'] = self.bars['Close'] - self.bars['Open']
        portfolio['price_diff'][0:5] = 0.0
        portfolio['profit'] = self.positions[self.symbol] * (portfolio['price_diff'])
        portfolio['fees'] = 500*(bars['Close']*0.002 + bars['Open']*0.002)* abs(portfolio['price_diff'])
        # Generate the equity curve and percentage returns
        portfolio['total'] = self.initial_capital + portfolio['profit'].cumsum() - portfolio['fees'].cumsum()
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio


def DataReader(symbol, start_date, end_date, symbol_class="forex" , model="SVC", pred_results=False, is_ret=False):
    if pred_results:
        if model == "SVC" or model == "RidgeClassifier":
            csv = symbol+"_"+model+"_sliding_30_results.csv"
            data = pd.read_csv(csv)
            data = data.set_index('Date')
            print(data)
        else:
            if is_ret:
                csv = symbol + "_Close_Ret_" + model + ".csv"
            else:
                csv = symbol+"_Close_" + model + ".csv"
            data = pd.read_csv(csv)
            data = data[['Date', 'y_pred_class']]
            data = data.set_index('Date')
    else:
        csv = "prep_" + symbol_class + ".csv"
        if symbol_class == "forex":
            data = pd.read_csv(csv, header=[0, 1], index_col=0)
            forex_features_bt = ["Open", "Close", "High", "Low", "Volume"]
            forex_cols_bt = [x for x in data.columns if x[1] == symbol]
            data = data[[col for col in forex_cols_bt if col[0] in forex_features_bt]][:-1]
            data.columns = [x[0] for x in list(data.columns)]
            data = data.dropna(how='any')
        else:
            data = pd.read_csv(csv, header=[0, 1, 2], index_col=0)

    data.index = pd.to_datetime(data.index)
    mask = (data.index > start_date) & (data.index <= end_date)
    data = data.loc[mask]

    return data

if __name__ == "__main__":
    start_test = datetime.datetime(2018, 1, 1)
    end_period = datetime.datetime(2019, 12, 30)

    symbol = "BDT" #input()
    symbol_class = "forex"
    model = "RidgeClassifier" #input()
    is_ret = False
    w = 5
    # Obtain the bars for SPY ETF which tracks the S&P500 index
    bars = DataReader(symbol, start_test, end_period, symbol_class)
    # Create the S&P500 forecasting strategy
    signals = DataReader(symbol, start_test, end_period, model=model, pred_results=True, is_ret=is_ret)
    signals.columns = ["signal"]
    strategy = SNPForecastingStrategy(symbol, bars, signals, w)

    signals = strategy.generate_signals()

    # Create the portfolio based on the forecaster
    portfolio = MarketIntradayPortfolio(symbol, bars, signals,
                                        initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    print(returns)
    # Plot results
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    # Plot the price of the SPY ETF
    ax1 = fig.add_subplot(211, ylabel=symbol+'ETF price in $')
    bars['Close'].plot(ax=ax1, color='r', lw=2.)

    # Plot the equity curve
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    returns['total'].plot(ax=ax2, lw=2.)

    # if is_ret:
    #     fig.savefig("./images/" + symbol + "_" + model + "_ret_res.png")
    # else:
    #     fig.savefig("./images/"+symbol + "_" + model + "_with_brok_res.png")
