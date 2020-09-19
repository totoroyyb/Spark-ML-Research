# Machine Learning for detection of fake news

This is a simple project which trained a model for fake news detection

## Dataset Source

**All dataset is obtained from online source, kaggle, and they are placed in the `data` folder**

- [train.csv](https://www.kaggle.com/c/fake-news/data?select=train.csv)
- [Fake.csv, True.csv](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv)
- [data.csv](https://www.kaggle.com/jruvika/fake-news-detection)

## Build up dataset

The dataset used for training process is combined from all three sources listed above, and file `data_processing.py` is used to build the standalone dataset named `dataset.csv` which is used in training process.

To generate `dataset.csv`, please run `python data_processing.py` first.

## Training the model

Run `fake_news_lstm.py`

