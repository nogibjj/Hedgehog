import talib
import pandas as pd



##### Momentum
def compute_momentum_indicators(df):
    # Prepare data
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open1 = df["open"].values

    # Initialize dictionary to hold results
    results = {}

    # Compute momentum indicators
    results["ADX"] = talib.ADX(high, low, close, timeperiod=14)
    results["APO"] = talib.APO(close, fastperiod=12, slowperiod=21, matype=0)
    results["AROONOSC"] = talib.AROONOSC(high, low, timeperiod=14)
    results["BOP"] = talib.BOP(open1, high, low, close)
    results["CCI"] = talib.CCI(high, low, close, timeperiod=14)
    results["CMO"] = talib.CMO(close, timeperiod=14)
    results["MACD"], results["MACDSIGNAL"], results["MACDHIST"] = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    results["MOM"] = talib.MOM(close, timeperiod=10)
    results["RSI"] = talib.RSI(close, timeperiod=14)
    results["STOCH_FASTK"], results["STOCH_FASTD"] = talib.STOCHF(
        high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    results["STOCHRSI_K"], results["STOCHRSI_D"] = talib.STOCHRSI(
        close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    results["TRIX"] = talib.TRIX(close, timeperiod=13)
    results["ULTOSC"] = talib.ULTOSC(
        high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28
    )
    results["WILLR"] = talib.WILLR(high, low, close, timeperiod=14)

    # Convert dictionary to DataFrame
    results_df = pd.DataFrame(results)
    results_df.index = df.index
    return results_df


#### Trend Indicators
def compute_trend_indicators(df):
    # Ensure all necessary columns are floats
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values

    # Initialize dictionary to hold results
    results = {}

    # Moving Averages
    results["SMA"] = talib.SMA(close, timeperiod=14)
    results["EMA"] = talib.EMA(close, timeperiod=14)
    results["WMA"] = talib.WMA(close, timeperiod=14)
    results["DEMA"] = talib.DEMA(close, timeperiod=14)
    results["TRIMA"] = talib.TRIMA(close, timeperiod=14)
    results["KAMA"] = talib.KAMA(close, timeperiod=14)
    mama, fama = talib.MAMA(
        close, fastlimit=0.5, slowlimit=0.05
    )  # MAMA returns MAMA and FAMA
    results["MAMA"] = mama
    results["FAMA"] = fama

    # Directional Movement Indicators
    results["ADX"] = talib.ADX(high, low, close, timeperiod=14)
    results["PLUS_DI"] = talib.PLUS_DI(high, low, close, timeperiod=14)
    results["MINUS_DI"] = talib.MINUS_DI(high, low, close, timeperiod=14)

    # Others
    results["SAR"] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    results["AROON_DOWN"], results["AROON_UP"] = talib.AROON(high, low, timeperiod=14)
    results["AROONOSC"] = talib.AROONOSC(high, low, timeperiod=14)
    # results['VI_PLUS'], results['VI_MINUS'] = talib.VI(high, low, close, timeperiod=14)

    # Convert dictionary to DataFrame and ensure it aligns with the original DataFrame's index
    results_df = pd.DataFrame(results, index=df.index)

    return results_df


#### Volatility Indicators
def compute_volatility_indicators(df):
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values

    # Initialize dictionary to hold results
    results = {}

    # Bollinger Bands
    results["BBANDS_UPPER"], results["BBANDS_MIDDLE"], results["BBANDS_LOWER"] = (
        talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    )

    # Average True Range
    results["ATR"] = talib.ATR(high, low, close, timeperiod=14)

    # Normalized Average True Range
    results["NATR"] = talib.NATR(high, low, close, timeperiod=14)

    # True Range
    results["TRANGE"] = talib.TRANGE(high, low, close)

    # Chandelier Exit (custom calculation, not a direct TA-Lib function)
    # Typically uses a 22-day period and a multiplier of 3 times the ATR
    atr_22 = talib.ATR(high, low, close, timeperiod=22)
    highest_high_22 = talib.MAX(high, timeperiod=22)
    lowest_low_22 = talib.MIN(low, timeperiod=22)
    results["CHANDELIER_EXIT_LONG"] = highest_high_22 - (atr_22 * 3)
    results["CHANDELIER_EXIT_SHORT"] = lowest_low_22 + (atr_22 * 3)

    # Convert dictionary to DataFrame and ensure it aligns with the original DataFrame's index
    results_df = pd.DataFrame(results, index=df.index)

    return results_df


#### Price Transformation
def compute_price_transform_indicators(df):
    # Ensure all necessary columns are floats
    open_ = df["open"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values

    # Initialize dictionary to hold results
    results = {}

    # Weighted Close Price
    results["WCLPRICE"] = talib.WCLPRICE(high, low, close)

    # Typical Price
    results["TYPPRICE"] = talib.TYPPRICE(high, low, close)

    # Median Price
    results["MEDPRICE"] = talib.MEDPRICE(high, low)

    # Price Rate of Change
    results["ROC"] = talib.ROC(close, timeperiod=10)

    # Average Price
    results["AVGPRICE"] = talib.AVGPRICE(open_, high, low, close)

    # Convert dictionary to DataFrame and ensure it aligns with the original DataFrame's index
    results_df = pd.DataFrame(results, index=df.index)

    return results_df


def compute_cycle_indicators(df):
    # Ensure 'close' column is a float
    close = df["close"].astype(float).values

    # Initialize dictionary to hold results
    results = {}

    # Hilbert Transform - Dominant Cycle Period
    results["HT_DCPERIOD"] = talib.HT_DCPERIOD(close)

    # Hilbert Transform - Phasor Components
    results["HT_PHASOR_inphase"], results["HT_PHASOR_quadrature"] = talib.HT_PHASOR(
        close
    )

    # Hilbert Transform - Trend vs Cycle Mode
    results["HT_TRENDMODE"] = talib.HT_TRENDMODE(close)

    # Convert dictionary to DataFrame and ensure it aligns with the original DataFrame's index
    results_df = pd.DataFrame(results, index=df.index)

    return results_df



def technical_indicators(df):
    all_results = [compute_momentum_indicators(df),
         compute_trend_indicators(df),
         compute_price_transform_indicators(df),
         compute_volatility_indicators(df),
         compute_cycle_indicators(df)
         ]

    return all_results