# models
models = {
    "twitter": {
        "regular": ["cardiffnlp/twitter-roberta-base-sentiment"], 
        "finance_tweets": ["nickmuchi/finbert-tone-finetuned-fintwitter-classification"],
    "financial_news": ["mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", "ahmedrachid/FinancialBERT-Sentiment-Analysis"],
    }
}

# datasets
datasets = {
    "twitter": {
        "finance_tweets": ["StephanAkkerman/financial-tweets-crypto",
                           "StephanAkkerman/financial-tweets-stocks",
                           "StephanAkkerman/financial-tweets-other",
                           "mjw/stock_market_tweets",
                           "zeroshot/twitter-financial-news-sentiment"],
        },
    "financial_news": "ashraq/financial-news",
}


# import financial-news dataset from transformers
from datasets import load_dataset

dataset = load_dataset("ashraq/financial-news")

# import Finance-Bert
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# Use a pipeline as a high-level helper
# from transformers import pipeline

pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")


# sample 50 observations from dataset
import random
random.seed(42)
sample = random.sample(range(len(dataset["train"])), 50)
sentences = [dataset["train"]['headline'][i] for i in sample]
results = nlp(sentences)
print(results)


import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the dataset
dataset = load_dataset("ashraq/financial-news")

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis", num_labels=3).to(device)
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

# Create the sentiment analysis pipeline
if device == "cuda":
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)  # device=0 refers to the GPU
else:
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Sample 50 observations from the dataset
import random
random.seed(42)
sample = random.sample(range(len(dataset["train"])), 50)
sentences = [dataset["train"]['headline'][i] for i in sample]

# Generate labels for the dataset
results = nlp(sentences)

# Print the results
print(results)

############################################################################################################
# to generate labels for the entire dataset
############################################################################################################

labels = nlp(dataset["train"]['headline'])