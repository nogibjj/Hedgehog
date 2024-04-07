# models
models = {
    "twitter": {
        "regular": ["cardiffnlp/twitter-roberta-base-sentiment"], 
        "finance_tweets": ["nickmuchi/finbert-tone-finetuned-fintwitter-classification",
                           "StephanAkkerman/FinTwitBERT"],
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

# import financial-news dataset from transformers
from datasets import load_dataset
from transformers import pipeline

try:
    label_generator = pipeline("text-classification", model = models['twitter']['finance_tweets'][1], device=0)
except:
    label_generator = pipeline("text-classification", model = models['twitter']['finance_tweets'][1])

dataset = load_dataset(datasets['twitter']['finance_tweets'][2])

def test_speed(label_generator, dataset, column_name = 'tweet_text', num_obs = 100):
    import time
    import random
    sample_idx = random.sample(range(len(dataset["train"][column_name])), num_obs)
    sentences = [dataset["train"][column_name][i] for i in sample_idx]
    sentences = [sentence for sentence in sentences if sentence is not None]
    start = time.time()
    labels = label_generator(sentences)
    end = time.time()
    len_document = len(dataset["train"][column_name])
    print(f"Time taken: {end-start} seconds")
    print(f"Estimated time for document: {len_document/num_obs*(end-start)/60:.2f} min")

def get_labels(label_generator, dataset, column_name = 'tweet_text'):
    # remove na and keep track of indices
    sentences = [(i, sentence) for i, sentence in enumerate(dataset["train"][column_name]) if sentence is not None]

    # Generate labels
    labels = label_generator([sentence for i, sentence in sentences])

    # Map indices to labels
    index_label_map = {sentences[i][0]: label for i, label in enumerate(labels)}

    return index_label_map

def store_labels(index_label_map, dataset_name):
    import pickle
    with open(f"data/{dataset_name}_labels.pkl", "wb") as f:
        pickle.dump(index_label_map, f)

if __name__ == "__main__":
    index_label_map = get_labels(label_generator, dataset, 'body')
    store_labels(index_label_map, 'stock_market_tweets')
    print("Done!")