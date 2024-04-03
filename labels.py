import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, BertForSequenceClassification, pipeline

# # check if CuDA is available and set the device accordingly
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# load the dataset
dataset = load_dataset("ashraq/financial-news")

# load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", num_labels=3)
tokenizer = RobertaTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# create the sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# sample 50 observations from the dataset
import random
random.seed(42)
sample = random.sample(range(len(dataset["train"])), 50)
sentences = [dataset["train"]["headline"][i] for i in sample]

# generate labels for the dataset
results = nlp(sentences)

# print the results
print(results)

