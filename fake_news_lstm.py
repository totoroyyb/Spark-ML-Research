import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu 50% of MEM
# config.gpu_options.allow_growth = True #allocate dynamically
# sess = tf.Session(config = config)

def read_csv(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()
    labels = df["label"]
    texts = []
    for i, row in df.iterrows():
        # removing all special characters
        review = re.sub('[^a-zA-Z]', ' ', row['title'])
        review = review.lower()
        review = review.split()
        review = " ".join(review)
        texts.append(review)
    return texts, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='data', type=str,
                        help='The input data dir.')
    parser.add_argument('--output_dir', default='output', type=str,
                        help='The output directory where the model will be saved.')
    parser.add_argument('--vocab_size', default=5000, type=int,
                        help='The maximum number of words to keep, based on word frequency.')
    parser.add_argument('--sentence_length', default=20, type=int,
                        help='Maximum length of all sequences.')
    parser.add_argument('--hidden_size', default=16, type=int,
                        help='Dimensionality of the output space.')
    parser.add_argument('--embedding_size', default=16, type=int,
                        help='Dimension of the dense embedding.')
    parser.add_argument('--dropout_rate', default=0.2, type=float,
                        help='Fraction of the input units to drop.')
    parser.add_argument('--activation', default='relu', type=str,
                        help='The activation function of the lstm.')
    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        help='The initial learning rate')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='The optimizer name.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training.')
    parser.add_argument('--name', default='log.txt', type=str, 
                        help="The trailing text of output file.")
    parser.add_argument('--write', default=True, type=bool,
                        help="Whether write out output files.")
    args = parser.parse_args()


    dataset_dir = args.dataset_dir
    texts, labels = read_csv(os.path.join(dataset_dir, 'dataset.csv'))

    vocab_size = args.vocab_size
    sentence_length = args.sentence_length
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)

    padded_sequences = pad_sequences(sequences, padding='pre', maxlen=sentence_length)

    X = np.array(padded_sequences)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=1234)

    ## Creating model
    embedding_size= args.embedding_size
    hidden_size = args.hidden_size
    dropout_rate = args.dropout_rate
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sentence_length))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(hidden_size, activation=args.activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(32, activation="sigmoid"))
    model.add(Dense(1, activation='sigmoid'))

    if args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = Adam(learning_rate=args.learning_rate)
    elif args.optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate=args.learning_rate)
    else:
        optimizer = Adadelta(learning_rate=args.learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epoch, batch_size=args.batch_size)

    y_pred = model.predict_classes(X_test)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    if args.write:
        with open(os.path.join(args.output_dir, args.name), 'w', encoding='utf-8') as file:
            acc = accuracy_score(y_test, y_pred)
            # print('Acc in test: {}'.format(acc))
            file.write('Acc in test:{}\n'.format(acc))
            report = classification_report(y_test, y_pred, digits=5)
            # print(report)
            file.write('report:\n{}\n'.format(report))
            matrix = confusion_matrix(y_test, y_pred)
            # print(matrix)
            file.write('confusion_matrix:{}\n'.format(str(matrix)))
            df_cm = pd.DataFrame(matrix, columns=['fake', 'truth'], index=['fake', 'truth'])
            sns.heatmap(df_cm, annot=True, fmt='g')
            # plt.show()