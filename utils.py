def long_feature_df(df):
    ''' Used to turn stocks.csv into a feature ready format for ML'''
    # Convert to long format
    df_long = df.melt(id_vars=['DATE', 'Ticker'], var_name='Feature', value_name='Value')

    # Pivot to wide format with aggregation
    df_pivot = df_long.pivot_table(index='DATE', columns=['Ticker', 'Feature'], values='Value', aggfunc='first')

    # Flatten the MultiIndex in the columns
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

    # Reset the index
    df_pivot.reset_index(inplace=True)

    return df_pivot

