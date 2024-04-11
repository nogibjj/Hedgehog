def long_feature_df(df):
    ''' Used to turn stocks.csv into a feature ready format for ML'''
    df = df.copy()
    # Extract sentiment features
    sentiment = df.groupby('DATE')[["tweet_sentiment", "news_sentiment"]].mean().dropna()

    # Drop sentiment
    df.drop(columns=["tweet_sentiment", "news_sentiment"], inplace=True)

    # Convert to long format
    df_long = df.melt(id_vars=['DATE', 'Ticker'], var_name='Feature', value_name='Value')

    # Pivot to wide format with aggregation
    df_pivot = df_long.pivot_table(index='DATE', columns=['Ticker', 'Feature'], values='Value', aggfunc='first')

    # Flatten the MultiIndex in the columns
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

    # Reset the index
    df_pivot.reset_index(inplace=True)

    # Merge sentiment features
    df_pivot = df_pivot.merge(sentiment, left_on='DATE', right_on='DATE', how='left')

    return df_pivot

def fill_short_interest_ratio(df):
    ''' Fill missing short interest ratio values with forward fill'''
    df = df.copy()
    # find columns with short interest ratio
    selected_columns = list(filter(lambda x: "SHORT INTEREST RATIO" in x, df.columns))

    # Forward fill missing values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].ffill()

    return df

def drop_ticker(df, ticker):
    ''' Drop a ticker from the dataframe'''
    df = df.copy()
    selected_columns = list(filter(lambda x: ticker not in x, df.columns))
    return df[selected_columns]

############################################################################################################
# technical indicators
############################################################################################################
import numpy as np
import pandas as pd


def calculate_macd(close_prices, short_range=12, long_range=26, signal_range=9):
    """calculate macd"""
    short_EMA = close_prices.ewm(span=short_range, adjust=False).mean()
    long_EMA = close_prices.ewm(span=long_range, adjust=False).mean()
    macd = short_EMA - long_EMA
    signal = macd.ewm(span=signal_range, adjust=False).mean()
    return macd, signal


def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window=period).mean()
    avg_loss = abs(down.rolling(window=period).mean())
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(df):
    """add technical indicators to dataframe"""
    df = df.copy()
    for instrument in df.Ticker.unique():
        # log ret in price 1d, 5d, 8d, 13d
        for i in [1, 3, 5, 8, 13, 21]:
            df.loc[df.Ticker == instrument, f"log_ret_{i}d"] = np.log(
                df.loc[df.Ticker == instrument, "LAST"]
                / df.loc[df.Ticker == instrument, "LAST"].shift(i)
            )
        # sma of log ret in price 1d, 5d, 8d, 13d
        # for i in [1, 5, 8, 13]:
        #     df.loc[df.Instrument == instrument, f"std_log_ret_{i}d"] = (
        #         df.loc[df.Instrument == instrument, f"log_ret_{i}d"].rolling(i).mean()
        #     )
        for i in [1, 3, 5, 8, 13, 21]:
            df.loc[df.Ticker == instrument, f"ewm_log_ret_{i}d"] = (
                df.loc[df.Ticker == instrument, f"log_ret_{i}d"].ewm(span=i).mean()
            )
        # macd
        macd, macd_sig = calculate_macd(df.loc[df.Ticker == instrument, "LAST"])
        df.loc[df.Ticker == instrument, "macd"] = macd
        df.loc[df.Ticker == instrument, "macd_sig"] = macd_sig
        # macd slopes
        for i in [1, 3, 5, 8, 13, 21]:
            df.loc[
                df.Ticker == instrument, f"macd_{i}d_slope"
                ] = df.loc[
                    df.Ticker == instrument, "macd"
                    ] - df.loc[df.Ticker == instrument, "macd"].shift(i)
        # df.loc[df.Ticker == instrument, "macd_slope"] = df.loc[
        #     df.Ticker == instrument, "macd"
        # ] - df.loc[df.Ticker == instrument, "macd"].shift(9)

        # rsi
        df.loc[df.Ticker == instrument, "rsi"] = calculate_rsi(
            df.loc[df.Ticker == instrument, "LAST"], 14
        )
        # rsi slopes
        for i in [1, 3, 5, 8, 13, 21]:
            df.loc[
                df.Ticker == instrument, f"rsi_{i}d_slope"
                ] = df.loc[
                    df.Ticker == instrument, "rsi"
                    ] - df.loc[df.Ticker == instrument, "rsi"].shift(i)
        # df.loc[df.Ticker == instrument, "rsi_slope"] = df.loc[
        #     df.Ticker == instrument, "rsi"
        # ] - df.loc[df.Ticker == instrument, "rsi"].shift(9)

        # clean ohl
        df.loc[df.Ticker == instrument, "HIGH"] = np.log(df.HIGH)-np.log(df.LAST.shift(1))
        df.loc[df.Ticker == instrument, "LOW"] = np.log(df.LOW)-np.log(df.LAST.shift(1))
        df.loc[df.Ticker == instrument, "OPEN"] = np.log(df.OPEN)-np.log(df.LAST.shift(1))

        # ivol slopes
        for i in [1, 3, 5, 8, 13, 21]:
            df.loc[
                df.Ticker == instrument, f"ivol_{i}d_slope"
                ] = df.loc[
                    df.Ticker == instrument, "3M IMPLIED VOL"
                    ] - df.loc[df.Ticker == instrument, "3M IMPLIED VOL"].shift(i)



    return df


############################################################################################################
# x, y prep and train test split
############################################################################################################
from sklearn.model_selection import train_test_split

def ml_prep(flattened_df, days=13):
    assert days in [1, 5, 8, 13, 21], "days must be 1, 5, 8, 13, or 21"
    # flattened_df.dropna(inplace=True)
    # define target variable and create y_true
    flattened_df["y_true"] = flattened_df["SPY_ewm_log_ret_1d"].rolling(window = days).sum().shift(-days)
    flattened_df.dropna(inplace=True)
    # define features
    y_true = flattened_df["y_true"]
    X = flattened_df.drop(columns=["DATE", "y_true"])
    X = X.reset_index()
    assert len(X) == len(y_true), "X and y_true must have the same length"

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.2, random_state=257
    )

    return X_train, X_test, y_train, y_test



