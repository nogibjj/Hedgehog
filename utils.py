def long_feature_df(df):
    """Used to turn stocks.csv into a feature ready format for ML"""
    df = df.copy()
    # Extract sentiment features
    sentiment = (
        df.groupby("DATE")[["tweet_sentiment", "news_sentiment"]].mean().dropna()
    )

    # Drop sentiment
    df.drop(columns=["tweet_sentiment", "news_sentiment"], inplace=True)

    # Convert to long format
    df_long = df.melt(
        id_vars=["DATE", "Ticker"], var_name="Feature", value_name="Value"
    )

    # Pivot to wide format with aggregation
    df_pivot = df_long.pivot_table(
        index="DATE", columns=["Ticker", "Feature"], values="Value", aggfunc="first"
    )

    # Flatten the MultiIndex in the columns
    df_pivot.columns = ["_".join(col).strip() for col in df_pivot.columns.values]

    # Reset the index
    df_pivot.reset_index(inplace=True)

    # Merge sentiment features
    df_pivot = df_pivot.merge(sentiment, left_on="DATE", right_on="DATE", how="left")

    return df_pivot


def fill_short_interest_ratio(df):
    """Fill missing short interest ratio values with forward fill"""
    df = df.copy()
    # find columns with short interest ratio
    selected_columns = list(filter(lambda x: "SHORT INTEREST RATIO" in x, df.columns))

    # Forward fill missing values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].ffill()

    return df


def drop_ticker(df, ticker):
    """Drop a ticker from the dataframe"""
    df = df.copy()
    selected_columns = list(filter(lambda x: ticker not in x, df.columns))
    return df[selected_columns]


def feature_type_map(df):
    """Map feature names to their types"""
    df = df.copy()
    # feature_map = {
    #     "sentiment": ["SPY_ewm_log_ret_1d"],
    #     "ohlc": ["SPY_ewm_log_ret_1d"],
    #     "spy_ohlc": ["SPY_ewm_log_ret_1d"],
    #     "returns": ["SPY_ewm_log_ret_1d"],
    #     "spy_returns": ["SPY_ewm_log_ret_1d"],
    #     "technical": ["SPY_ewm_log_ret_1d"],
    # }
    feature_map = {
        "sentiment": ["SPY_ewm_log_ret_1d"],
        "ohlc": ["SPY_ewm_log_ret_1d"],
        "spy_ohlc": ["SPY_ewm_log_ret_1d"],
        "returns": ["SPY_ewm_log_ret_1d"],
        "spy_returns": ["SPY_ewm_log_ret_1d"],
        "technical": ["SPY_ewm_log_ret_1d"],
    }
    for col in df.columns:
        if "sentiment" in col:
            (
                feature_map["sentiment"].append(col)
                if col not in feature_map["sentiment"]
                else None
            )
        elif "HIGH" in col or "LOW" in col or "OPEN" in col:
            feature_map["ohlc"].append(col) if col not in feature_map["ohlc"] else None
            if "SPY" in col:
                (
                    feature_map["spy_ohlc"].append(col)
                    if col not in feature_map["spy_ohlc"]
                    else None
                )
        elif "ret" in col:
            if "ewm" in col:
                (
                    feature_map["technical"].append(col)
                    if col not in feature_map["technical"]
                    else None
                )
            else:
                (
                    feature_map["returns"].append(col)
                    if col not in feature_map["returns"]
                    else None
                )
                if "SPY" in col:
                    (
                        feature_map["spy_returns"].append(col)
                        if col not in feature_map["spy_returns"]
                        else None
                    )
        else:
            (
                feature_map["technical"].append(col)
                if col not in feature_map["technical"]
                else None
            )

    return feature_map


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
            df.loc[df.Ticker == instrument, f"macd_{i}d_slope"] = df.loc[
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
            df.loc[df.Ticker == instrument, f"rsi_{i}d_slope"] = df.loc[
                df.Ticker == instrument, "rsi"
            ] - df.loc[df.Ticker == instrument, "rsi"].shift(i)
        # df.loc[df.Ticker == instrument, "rsi_slope"] = df.loc[
        #     df.Ticker == instrument, "rsi"
        # ] - df.loc[df.Ticker == instrument, "rsi"].shift(9)

        # clean ohl
        df.loc[df.Ticker == instrument, "HIGH"] = np.log(df.HIGH) - np.log(
            df.LAST.shift(1)
        )
        df.loc[df.Ticker == instrument, "LOW"] = np.log(df.LOW) - np.log(
            df.LAST.shift(1)
        )
        df.loc[df.Ticker == instrument, "OPEN"] = np.log(df.OPEN) - np.log(
            df.LAST.shift(1)
        )

        # ivol slopes
        for i in [1, 3, 5, 8, 13, 21]:
            df.loc[df.Ticker == instrument, f"ivol_{i}d_slope"] = df.loc[
                df.Ticker == instrument, "3M IMPLIED VOL"
            ] - df.loc[df.Ticker == instrument, "3M IMPLIED VOL"].shift(i)

    return df


############################################################################################################
# x, y prep and train test split
############################################################################################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ml_prep(flattened_df, days=13, random_state=257):
    """Prepare data for regression"""
    assert days in [1, 5, 8, 13, 21], "days must be 1, 5, 8, 13, or 21"
    # flattened_df.dropna(inplace=True)
    # define target variable and create y_true
    flattened_df["y_true"] = (
        flattened_df["SPY_ewm_log_ret_1d"].rolling(window=days).sum().shift(-days)
    )
    flattened_df = flattened_df.dropna()
    # define features
    y_true = flattened_df["y_true"].values
    if "DATE" in flattened_df.columns:
        X = flattened_df.drop(columns=["DATE", "y_true"]).values
    else:
        X = flattened_df.drop(columns=["y_true"]).values
    assert len(X) == len(y_true), "X and y_true must have the same length"

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.2, random_state=random_state, shuffle=True
    )

    return X_train, X_test, y_train, y_test


def prep_classifier_data(df, days=13, random_state=257):
    """Prepare data for classifier"""
    # Create target variable
    df["y_true"] = df["SPY_ewm_log_ret_1d"].rolling(window=days).sum().shift(-days)
    df["y_true"] = df["y_true"].apply(lambda x: 1 if x > 0 else 0)
    df = df.dropna()
    # Define features
    y_true = df["y_true"].values
    if "DATE" in df.columns:
        X = df.drop(columns=["DATE", "y_true"]).values
    else:
        X = df.drop(columns=["y_true"]).values
    assert len(X) == len(y_true), "X and y_true must have the same length"
    # # train test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y_true, test_size=0.2, random_state=random_state, shuffle=True
    # )

    test_size = 0.2
    # Calculate split index
    split_idx = int(len(X) * (1 - test_size))

    # Split into training and testing sets based on the calculated index
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y_true[:split_idx]
    y_test = y_true[split_idx:]

    # # Standardization
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def prep_regression_data(df, days=13, random_state=257):
    """Prepare data for regressor"""
    # Create target variable
    df["y_true"] = df["SPY_ewm_log_ret_1d"].rolling(window=days).sum().shift(-days)
    df = df.dropna()
    # Define features
    y_true = df["y_true"].values
    if "DATE" in df.columns:
        X = df.drop(columns=["DATE", "y_true"]).values
    else:
        X = df.drop(columns=["y_true"]).values
    assert len(X) == len(y_true), "X and y_true must have the same length"
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=0.2, random_state=random_state, shuffle=True
    )

    return X_train, X_test, y_train, y_test


############################################################################################################
# models
############################################################################################################
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB


def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    """Naive Bayes Classifier"""
    # Train model
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Evaluate model
    print(f"Accuracy: {accuracy_score(y_test, y_pred) :.5f}")
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, cm, clf, (y_test, clf.predict_proba(X_test)[:, 1])


def logistic_regression(X_train, X_test, y_train, y_test, random_state=257):
    """Logistic Regression"""
    # train model
    clf = LogisticRegression(random_state=random_state)
    clf.fit(X_train, y_train)

    # predict on test data
    y_pred = clf.predict(X_test)

    # evaluate model
    print(f"Accuracy: {accuracy_score(y_test, y_pred) :.5f}")
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return y_pred, cm, clf, (y_test, clf.predict_proba(X_test)[:, 1])


def Linear_regressor(X_train, X_test, y_train, y_test, random_state=257):
    clf = LinearRegression(random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"MSE:{test_mse}, R2:{test_r2}")

    return clf, y_pred


# def random_forest_classifier(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=50, random_state=257):
#     """Random Forest Classifier"""

#     # train model
#     clf = RandomForestClassifier(
#         n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
#     )
#     clf.fit(X_train, y_train)

#     # predict on test data
#     y_pred = clf.predict(X_test)

#     # evaluate model
#     print(f"Accuracy: {accuracy_score(y_test, y_pred) :.5f}" )
#     # Generate the confusion matrix
#     cm = confusion_matrix(y_test, y_pred)

#     return y_pred, cm, clf


def random_forest_classifier(X_train, X_test, y_train, y_test):
    """Random Forest Classifier with Random Search CV"""

    # Define the model
    clf = RandomForestClassifier(random_state=257)

    # Set up the parameter grid to sample from during fitting
    param_distributions = {
        "n_estimators": np.arange(100, 1100, 100),
        "max_depth": [None] + list(np.arange(10, 110, 10)),
        "min_samples_split": np.arange(2, 21),
        "min_samples_leaf": np.arange(1, 21),
        # "max_features": ["auto", "sqrt", "log2"],
    }

    # Create the random search with CV object
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=100,  # Number of parameter settings sampled
        scoring="accuracy",  # Can be changed to other metrics
        cv=3,  # 3-fold cross-validation
        verbose=1,  # Higher the number, more the verbosity
        random_state=257,
        n_jobs=-1,  # Use all available cores
    )

    # Fit the random search model
    random_search.fit(X_train, y_train)

    # Best model found by random search
    best_clf = random_search.best_estimator_

    # Predict on test data using the best model
    y_pred = best_clf.predict(X_test)

    # Evaluate the best model
    print(f"Accuracy: {accuracy_score(y_test, y_pred) :.5f}")
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # # Print best model parameters
    # print("Best model parameters:", random_search.best_params_)

    return y_pred, cm, best_clf, (y_test, best_clf.predict_proba(X_test)[:, 1])


def random_forest_regressor(X_train, X_test, y_train, y_test):
    """Random Forest Regressor with Random Search CV"""
    from sklearn.ensemble import RandomForestRegressor

    # Define the model
    clf = RandomForestRegressor(random_state=257)

    # Set up the parameter grid to sample from during fitting
    param_distributions = {
        "n_estimators": np.arange(100, 1100, 100),
        "max_depth": [None] + list(np.arange(10, 110, 10)),
        "min_samples_split": np.arange(2, 21),
        "min_samples_leaf": np.arange(1, 21),
    }

    # Create the random search with CV object
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=100,  # Number of parameter settings sampled
        scoring="neg_mean_squared_error",  # Minimize MSE
        cv=3,  # 3-fold cross-validation
        verbose=1,  # Higher the number, more the verbosity
        random_state=257,
        n_jobs=-1,  # Use all available cores
    )

    # Fit the random search model
    random_search.fit(X_train, y_train)

    # Best model found by random search
    best_clf = random_search.best_estimator_

    # Predict on test data using the best model
    y_pred = best_clf.predict(X_test)

    # Evaluate the best model
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    print(f"MSE:{test_mse}, R2:{test_r2}")

    return best_clf, y_pred


def xgboost_regressor(X_train, X_test, y_train, y_test):
    """XGBoost Regressor with Random Search CV"""
    from xgboost import XGBRegressor

    # Define the model
    clf = XGBRegressor(random_state=257)

    # Set up the parameter grid to sample from during fitting
    param_distributions = {
        "n_estimators": np.arange(100, 1100, 100),
        "max_depth": np.arange(3, 15),
        "learning_rate": np.linspace(0.01, 0.3, num=30),
        "subsample": np.linspace(0.5, 1.0, num=6),
        "colsample_bytree": np.linspace(0.5, 1.0, num=6),
    }

    # Create the random search with CV object
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=100,  # Number of parameter settings sampled
        scoring="neg_mean_squared_error",  # Minimize MSE
        cv=3,  # 3-fold cross-validation
        verbose=1,  # Higher the number, more the verbosity
        random_state=257,
        n_jobs=-1,  # Use all available cores
    )

    # Fit the random search model
    random_search.fit(X_train, y_train)

    # Best model found by random search
    best_clf = random_search.best_estimator_

    # Predict on test data using the best model
    y_pred = best_clf.predict(X_test)

    # Evaluate the best model
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    print(f"MSE:{test_mse}, R2:{test_r2}")

    return best_clf, y_pred


# def xgboost_classifier(X_train, X_test, y_train, y_test, random_state=257):
#     ''' XGBoost Classifier'''
#     from xgboost import XGBClassifier
#     # train model
#     clf = XGBClassifier(random_state=random_state)
#     clf.fit(X_train, y_train)

#     # predict on test data
#     y_pred = clf.predict(X_test)

#     # evaluate model
#     print(f"Accuracy: {accuracy_score(y_test, y_pred) :.5f}" )
#     # Generate the confusion matrix
#     cm = confusion_matrix(y_test, y_pred)

#     return y_pred, cm, clf


def xgboost_classifier(X_train, X_test, y_train, y_test, random_state=257):
    """XGBoost Classifier with Random Search CV"""

    # Define the classifier
    clf = XGBClassifier(random_state=random_state)

    # Set up the parameter grid to sample from during fitting
    param_distributions = {
        "n_estimators": np.arange(50, 400, 50),
        "max_depth": np.arange(3, 15, 1),
        "learning_rate": np.linspace(0.01, 0.3, 30),
        "subsample": np.linspace(0.7, 1, 30),
        "colsample_bytree": np.linspace(0.5, 1, 30),
        "min_child_weight": np.arange(1, 10, 1),
    }

    # Create the random search with CV object
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=100,  # Number of parameter settings sampled
        scoring="accuracy",  # Can be changed to other metrics
        cv=3,  # 5-fold cross-validation
        verbose=1,  # Higher the number, more the verbosity
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
    )

    # Fit the random search model
    random_search.fit(X_train, y_train)

    # Best model found by random search
    best_clf = random_search.best_estimator_

    # Predict on test data using the best model
    y_pred = best_clf.predict(X_test)

    # Evaluate the best model
    print(f"Accuracy: {accuracy_score(y_test, y_pred) :.5f}")
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # # Print best model parameters
    # print("Best model parameters:", random_search.best_params_)

    return y_pred, cm, best_clf, (y_test, best_clf.predict_proba(X_test)[:, 1])


############################################################################################################
# plots
############################################################################################################


def plot_confusion_matrix(y_test, y_pred, days, title):
    """Plot confusion matrix"""
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.title(f"{title} for {days} day return")
    plt.show()


def plot_feature_importance(clf, X_train, days):
    """Plot feature importance"""
    feature_importance = clf.feature_importances_
    feature_names = X_train.columns
    sorted_idx = feature_importance.argsort()
    plt.figure(figsize=(10, 7))
    plt.barh(feature_names[sorted_idx], feature_importance[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance for {days} day return")
    plt.show()


def plot_roc(clf, X_test, y_test, days, model_type):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, roc_auc_score

    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f"AUC: {auc:.2f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for {days} day return")
    plt.legend()
    plt.show()
