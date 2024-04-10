def long_feature_df(df):
    ''' Used to turn stocks.csv into a feature ready format for ML'''
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
    df_pivot = df_pivot.merge(sentiment, left_on='DATE', right_on_='DATE', how='left')

    return df_pivot

def fill_short_interest_ratio(df):
    ''' Fill missing short interest ratio values with forward fill'''
    # find columns with short interest ratio
    selected_columns = list(filter(lambda x: "SHORT INTEREST RATIO" in x, df.columns))

    # Forward fill missing values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].ffill()

    return df

def drop_ticker(df, ticker):
    ''' Drop a ticker from the dataframe'''
    selected_columns = list(filter(lambda x: ticker not in x, df.columns))
    return df[selected_columns]

