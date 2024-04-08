from datasets import load_dataset
from transformers import pipeline


# models
models = {
    "twitter": {
        "regular": ["cardiffnlp/twitter-roberta-base-sentiment"], 
        "finance_tweets": ["nickmuchi/finbert-tone-finetuned-fintwitter-classification",
                           "StephanAkkerman/FinTwitBERT",
                           "StephanAkkerman/FinTwitBERT-sentiment"],
    "financial_news": ["mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", "ahmedrachid/FinancialBERT-Sentiment-Analysis"],
    }
}

# datasets
datasets = {
    "twitter": {
        "finance_tweets": ["StephanAkkerman/financial-tweets",
                           "StephanAkkerman/stock-market-tweets-data",
                           "mjw/stock_market_tweets",
                           "zeroshot/twitter-financial-news-sentiment",
                           "emad12/stock_tweets_sentiment"],
        },
    "financial_news": "ashraq/financial-news",
}

kaggle_crypto_dir = "kaggle datasets download -d rezasemyari/crypto-sentiment-tweets"

def create_pipeline(model_name, device = 0):
    try:
        return pipeline("sentiment-analysis", model = model_name, device=device)
    except:
        return pipeline("sentiment-analysis", model = model_name)

# dataset = load_dataset(datasets['twitter']['finance_tweets'][4])

def test_speed(pipeline, dataset, column_name = 'tweet_text', num_obs = 100):
    import time
    import random
    sample_idx = random.sample(range(len(dataset["train"][column_name])), num_obs)
    sentences = [dataset["train"][column_name][i] for i in sample_idx]
    sentences = [sentence for sentence in sentences if sentence is not None]
    start = time.time()
    _ = pipeline(sentences)
    end = time.time()
    len_document = len(dataset["train"][column_name])
    print(f"Time taken: {end-start} seconds")
    print(f"Estimated time for document: {len_document/num_obs*(end-start)/60:.2f} min")

def get_labels(pipeline, dataset, column_name = 'tweet_text'):
    # remove na and keep track of indices
    sentences = [(i, sentence) for i, sentence in enumerate(dataset["train"][column_name]) if sentence is not None]

    # Generate labels
    labels = pipeline([sentence for i, sentence in sentences], truncation=True, max_length=512)

    # Map indices to labels
    index_label_map = {sentences[i][0]: label for i, label in enumerate(labels)}

    return index_label_map

# def get_labels(label_generator, dataset, column_name = 'tweet_text'):
#     # remove na and keep track of indices
#     sentences = [(i, sentence) for i, sentence in enumerate(dataset["train"][column_name]) if sentence is not None]

#     # Truncate sentences
#     sentences = [(i, label_generator.tokenizer(sentence, truncation='longest_first', max_length=512)['input_ids']) for i, sentence in sentences]

#     # Generate labels
#     labels = label_generator([sentence for i, sentence in sentences])

#     # Map indices to labels
#     index_label_map = {sentences[i][0]: label for i, label in enumerate(labels)}

#     return index_label_map

def store_labels(index_label_map, dataset_name):
    import pickle
    with open(f"data/{dataset_name}_labels.pkl", "wb") as f:
        pickle.dump(index_label_map, f)

def load_labels(dataset_name):
    import pickle
    with open(f"data/{dataset_name}_labels.pkl", "rb") as f:
        return pickle.load(f)
    

import pandas as pd
import dask.dataframe as dd

def labels_dates_df(index_label_map, dataset, date_column):
    dates_series = pd.Series(dataset["train"][date_column])
    dates_df = dates_series.to_frame()  # Convert Series to DataFrame
    dates = dd.from_pandas(dates_df, npartitions=1800).loc[list(index_label_map.keys())].compute()
    
    labels = [index_label_map[key]['label'] for key in index_label_map]
    scores = [index_label_map[key]['score'] for key in index_label_map]
    
    return pd.DataFrame({"date": dates.squeeze(), "label": labels, "score": scores})  # Use squeeze to convert DataFrame back to Series