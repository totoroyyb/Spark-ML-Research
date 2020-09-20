# Machine Learning for detection of fake news

This is a simple project which trained a model for fake news detection



## Dataset Source

**All datasets are obtained from online source, kaggle, and they are placed in the `data` folder**

- [train.csv](https://www.kaggle.com/c/fake-news/data?select=train.csv)
- [Fake.csv, True.csv](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv)
- [data.csv](https://www.kaggle.com/jruvika/fake-news-detection)



## Build up dataset

The dataset used for training process is combined from all three sources listed above, and file `data_processing.py` is used to build the standalone dataset named `dataset.csv` which is used in training process.

To generate `dataset.csv`, please run `python data_processing.py` first.



## Train the model

Run `fake_news_lstm.py`



## Test out parameters

 `fake_news_lstm.py` supports several parameters and `script.py` can be used for generate a batch of output based on certain parameter.

### Usage

`python script.py --variable [parameter_name] --init [initial_value] --stride [interval] --rep [number_of_repetition]`

For example, if you want to get different results of setting `vocab_size` starting at 2000, 3000, 4000, 5000 and so on 5 times. You can run `python script.py --variable vocab_size --init 2000 --stride 1000, --rep 5`.

- `--variable` is the name of parameter that you want to tested out. It can be

  - `vocab_size`
  - `sentence_length`
  - `hidden_size`
  - `embedding_size`
  - `dropout_rate`
  - `learning_rate`
  - `batch_size`

- `--init` is the initial value for a series of operations. It accepts a `float`, but only `dropout_rate` and `learning_rate` will actually use it as float, otherwise, it will be converted to an `int`.

- `--stride` has different meaning based on different contexts.

  - In `vocab_size`, `sentence_length`, `dropout_rate`

    It stands for the increment. For example, if run `python script.py --vocab_size --init 1000 --stride 1000 --rep 5`, it will automatically run `fake_news_lstm.py`, changing `vocab_size` to 1000, 2000, 3000, 4000 and 5000, 5 times in total.

  - In `hidden_size`, `embedding_size`, `learning_rate`, `batch_size`

    It stands for the exponent. For example, if set `--init 8 --stride 2 --rep 4`, it will run with 8, $8\times2^1$, $8\times2^2$ and $8\times2^3$, 4 times in total.

- `--rep` represents how many times you want to run. It accepts only integer.

### Output

By default, the output of `script.py` will be placed in `./output` relative to the root path of this project. The naming convention will follow `[variable]-[parameter]`. For example, if you run `python script.py --vocab_size --init 1000 --stride 1000 --rep 5`, then 5 files will be generated in `./output`:

* `vocab_size-1000`
* `vocab_size-2000`
* `vocab_size-3000`
* `vocab_size-4000`
* `vocab_size-5000`



## Generate plots

`gene_plot.py` is another script used to generate plots based on the result produced by `script.py`

### Usage

`python gene_plot.py --variable [variable_name] --type [type]`

For example, if you want to generate the graph of all `vocab_size` results, you may run `python gene_plot.py --variable vocab_size`. `--variable` accepts a `str`.

`--type` can be `line` or `bar`, to generate different types of charts.

### Output

It will automatically place the output image at the same level of data files as `[variable]_[type].png`.

For example, if you run `python gene_plot.py --variable vocab_size --type line`, it will generate `vocab_size_line.png`.

### Note

By default, `gene_plot.py` will search out data files located in the folder `./output/[variable_name]`. So if you run `python gene_plot.py --variable vocab_size`, it will search out files at `./output/vocab_size/vocab_size-1000`, etc. Hence, please make sure to organize the data files into their corresponding folders, considering `script.py` will not organize it for you.