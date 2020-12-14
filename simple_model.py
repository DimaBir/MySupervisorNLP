import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils import plot_history


def load_data():
    filename = r"D:\MySupervisorNLP\dataset2.csv"
    df = pd.read_csv(filename, encoding='latin_1')
    wordy = df.loc[df['Class'] == 1]
    clear = df.loc[df['Class'] == 0].sample(n=len(wordy))

    df = pd.concat([wordy, clear])
    return df


if __name__ == '__main__':
    print('Loading data...')
    df = load_data()
    df = df.sample(frac=1).reset_index(drop=True)
    sentences = df["Sentence"]

    # integer encode the documents
    vocab_size = 5000
    data = [one_hot(sentence, vocab_size) for sentence in sentences]
    labels = np.asarray(df["Class"])
    print(data)

    # pad documents to a max length of 4 words
    max_length = 80
    padded_docs = pad_sequences(data, maxlen=max_length, padding='post')
    print(padded_docs)

    num_validation_samples = int(0.2 * len(data))

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    print(len(x_train), 'train sequences')
    print(len(x_val), 'val sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)

    print('Build model...')
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_length))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    # summarize the model
    print(model.summary())

    # fit the model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=16)
    plot_history(history)

    # we use the first 100 training examples as our background dataset to integrate over
    explainer = shap.DeepExplainer(model, x_train[:100])

    # explain the first 10 predictions
    # explaining each prediction requires 2 * background dataset size runs
    shap_values = explainer.shap_values(x_val[:10])
