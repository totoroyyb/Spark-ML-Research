#!/usr/bin/python

# -*- coding: utf-8 -*-
import pandas as pd
from enum import Enum
import string

# This enum is never used currently
class NEWS_TYPES(Enum):
    REAL_TITLE = 0
    REAL_BODY = 1
    FAKE_TITLE = 2
    FAKE_BODY = 3
    
# count the word frequency given the certain string
def word_count(input_str: str, word_dict: dict):
    wc = input_str.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in wc]
    for word in stripped:
        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1

# count the word frequency given the certain pandas Series
def count_series(word_series, word_dict):
    for index, value in word_series.items():
        word_count(value, word_dict)


# 0 stands for fake, 1 stands for true
data1_fake = pd.read_csv("data/Fake.csv", usecols=["title", "text"])
data1_fake["label"] = 0

data1_true = pd.read_csv("data/True.csv", usecols=["title", "text"])
data1_true["label"] = 1

data2 = pd.read_csv("data/data.csv", usecols=["Headline", "Body", "Label"])
data2.columns = ["title", "text", "label"]

data3 = pd.read_csv("data/train.csv", usecols=["title", "text", "label"])
# Reverse the label to match up with the current dataset settings
data3["label"] = data3["label"].map(lambda x: 0 if x == 1 else 1)

# Appending all trimmed dataset together
data = data1_true.append(data1_fake.append(data2, ignore_index=True), ignore_index=True)
data = data.append(data3, ignore_index=True)
assert len(data.index) == len(data1_fake.index) + len(data1_true.index) + len(data2.index) + len(data3.index), "Appending dataset results a length error"

# Remove extra data
del data1_true, data1_fake, data2, data3

# Remove all duplicated data and NaN values
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

data.to_csv("data/dataset.csv")

# wc_real_title = {}
# wc_fake_title = {}
# wc_real_body = {}
# wc_fake_body = {}

# # All real data
# real = data[data["label"] == 1]
# count_series(real["title"], wc_real_title)
# count_series(real["text"], wc_real_body)

# # All fake data
# fake = data[data["label"] == 0]
# count_series(fake["title"], wc_fake_title)
# count_series(fake["text"], wc_fake_body)
