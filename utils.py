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

