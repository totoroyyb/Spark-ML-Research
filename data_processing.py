#!/usr/bin/python

# -*- coding: utf-8 -*-
import pandas as pd
from enum import Enum
import string

class NEWS_TYPES(Enum):
    REAL_TITLE = 0
    REAL_BODY = 1
    FAKE_TITLE = 2
    FAKE_BODY = 3
    
def word_count(input_str: str, word_dict: dict):
    wc = input_str.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in wc]
    for word in stripped:
        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1
            
def count_series(word_series, word_dict):
    for index, value in word_series.items():
        word_count(value, word_dict)



# 0 stands for fake, 1 stands for true
data1_fake = pd.read_csv("Fake.csv", usecols=["title", "text"])
data1_fake["label"] = 0

data1_true = pd.read_csv("True.csv", usecols=["title", "text"])
data1_true["label"] = 1

data2 = pd.read_csv("data.csv", usecols=["Headline", "Body", "Label"])
data2.columns = ["title", "text", "label"]

# Appending all trimmed dataset together
data = data1_true.append(data1_fake.append(data2, ignore_index=True), ignore_index=True)
assert len(data.index) == len(data1_fake.index) + len(data1_true.index) + len(data2.index), "Appending dataset results a length error"

# Remove extra data
del data1_true, data1_fake, data2

# Remove all duplicated data and NaN values
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

wc_real_title = {}
wc_fake_title = {}
wc_real_body = {}
wc_fake_body = {}

# All real data
real = data[data["label"] == 1]
count_series(real["title"], wc_real_title)
count_series(real["text"], wc_real_body)

# All fake data
fake = data[data["label"] == 0]
count_series(fake["title"], wc_fake_title)
count_series(fake["text"], wc_fake_body)
