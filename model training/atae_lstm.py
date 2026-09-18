import os
import re
import random
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# 1. CONFIGURATION

SEED = 42

DATA_PATH = "./dataset/data/combined_train.csv"
GLOVE_PATH = "./dataset/embeddings/glove.6B.100d.txt"
MODEL_SAVE_PATH = "./checkpoints/atae_lstm_best.pt"

EMBEDDING_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
DROPOUT = 0.30
MAX_SENTENCE_LEN = 100
MAX_ASPECT_LEN = 10
MIN_WORD_FREQUENCY = 2

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    "conflict": 3
}

ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


# 2. REPRODUCIBILITY AND DEVICE

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)


# 3. TEXT PREPROCESSING

def tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


# 4. LOAD AND CLEAN DATA

data = pd.read_csv(DATA_PATH)

print("Original shape:", data.shape)
print("Original labels:")
print(data["polarity"].value_counts(dropna=False))

data["polarity"] = (
    data["polarity"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({
        "postive": "positive",
        "po": "positive"
    })
)

data = data.dropna(subset=["text", "aspect_term", "polarity"]).copy()

data = data[
    data["text"].astype(str).str.strip().ne("") &
    data["aspect_term"].astype(str).str.strip().ne("")
].copy()

data = data[data["polarity"].isin(LABEL_TO_ID)].copy()

data["text_tokens"] = data["text"].apply(tokenize)
data["aspect_tokens"] = data["aspect_term"].apply(tokenize)
data["label"] = data["polarity"].map(LABEL_TO_ID)

data = data[
    data["text_tokens"].map(len).gt(0) &
    data["aspect_tokens"].map(len).gt(0)
].reset_index(drop=True)

print("\nCleaned shape:", data.shape)
print("\nCleaned label distribution:")
print(data["polarity"].value_counts())

print("\nExamples:")
print(data[["text", "aspect_term", "polarity"]].head())


# 5. TRAIN / VALIDATION / TEST SPLIT

train_df, temp_df = train_test_split(
    data,
    test_size=0.20,
    random_state=SEED,
    stratify=data["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTraining-label distribution:")
print(train_df["polarity"].value_counts())


# 6. BUILD VOCABULARY FROM TRAINING TEXT ONLY

word_counter = Counter()

for tokens in train_df["text_tokens"]:
    word_counter.update(tokens)

for tokens in train_df["aspect_tokens"]:
    word_counter.update(tokens)

vocab = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID
}

for word, count in word_counter.items():
    if count >= MIN_WORD_FREQUENCY:
        vocab[word] = len(vocab)

id_to_word = {idx: word for word, idx in vocab.items()}

print("\nVocabulary size:", len(vocab))


# 7. LOAD PRETRAINED GLOVE EMBEDDINGS

def load_glove_embeddings(glove_path, vocabulary, embedding_dim):
    embedding_matrix = np.random.normal(
        loc=0.0,
        scale=0.05,
        size=(len(vocabulary), embedding_dim)
    ).astype(np.float32)

    embedding_matrix[PAD_ID] = np.zeros(embedding_dim, dtype=np.float32)

    found_words = 0

    with open(glove_path, "r", encoding="utf-8") as glove_file:
        for line in glove_file:
            values = line.rstrip().split(" ")

            word = values[0]

            if word not in vocabulary:
                continue

            vector = np.asarray(values[1:], dtype=np.float32)

            if vector.shape[0] != embedding_dim:
                continue

            embedding_matrix[vocabulary[word]] = vector
            found_words += 1

    coverage = (found_words / max(len(vocabulary) - 2, 1)) * 100

    print(f"GloVe vectors found: {found_words:,}")
    print(f"GloVe vocabulary coverage: {coverage:.2f}%")

    return torch.tensor(embedding_matrix, dtype=torch.float32)


embedding_matrix = load_glove_embeddings(
    GLOVE_PATH,
    vocab,
    EMBEDDING_DIM
)
