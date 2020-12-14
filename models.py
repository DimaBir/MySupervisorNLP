import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import RegexpTokenizer
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers.merge import concatenate
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model

from sklearn.utils import class_weight
import sklearn
import shap
from sklearn.model_selection import train_test_split


from nltk.tokenize import sent_tokenize, word_tokenize

import gensim
from gensim.models import Word2Vec


# def word2vec(dataframe):
#     data = []
#     temp = []
#
#     for sentence in dataframe["Sentence"]:
#         # tokenize the sentence into words
#         for j in word_tokenize(sentence):
#             temp.append(j.lower())
#
#         data.append(temp)
#
#     # Create CBOW model
#     model1 = gensim.models.Word2Vec(data, min_count=1,
#                                     size=100, window=5)
#
#     print("Done")


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False, extra_conv=True):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    # Add a 1D convnet with global maxpooling
    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    if extra_conv == True:
        x = Dropout(0.25)(l_merge)
    else:
        # Original Yoon Kim model
        x = Dropout(0.5)(pool)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)

    preds = Dense(labels_index, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    return model


if __name__ == '__main__':
    word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    tokenizer = RegexpTokenizer(r'\w+')

    filename = r"D:\MySupervisorNLP\dataset2.csv"
    df = pd.read_csv(filename, encoding='latin_1')
    wordy = df.loc[df['Class'] == 1]
    clear = df.loc[df['Class'] == 0].sample(n=len(wordy))

    df = pd.concat([wordy, clear])

    df["Tokens"] = df["Sentence"].apply(tokenizer.tokenize)

    all_words = [word for tokens in df["Tokens"] for word in tokens]
    sentence_lengths = [len(tokens) for tokens in df["Tokens"]]

    VOCAB = sorted(list(set(all_words)))

    print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    print("Max sentence length is %s" % max(sentence_lengths))

    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 35
    VOCAB_SIZE = len(VOCAB)

    VALIDATION_SPLIT = .2
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(df["Sentence"].tolist())
    sequences = tokenizer.texts_to_sequences(df["Sentence"].tolist())

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(df["Class"]))

    indices = np.arange(cnn_data.shape[0])
    np.random.shuffle(indices)
    cnn_data = cnn_data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])

    embedding_weights = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, index in word_index.items():
        embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
    print(embedding_weights.shape)

    x_train = cnn_data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = cnn_data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    model = ConvNet(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index) + 1, EMBEDDING_DIM,
                    len(list(df["Class"].unique())), False)

    # earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=64,
              callbacks=[mcp_save])

    # # we use the first 100 training examples as our background dataset to integrate over
    # explainer = shap.DeepExplainer(model, x_train[:100])
    #
    # # explain the first 10 predictions
    # # explaining each prediction requires 2 * background dataset size runs
    # shap_values = explainer.shap_values(x_val[:10])
    #
    # # plot the explanation of the first prediction
    # # Note the model is "multi-output" because it is rank-2 but only has one column
    # shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_val[:10])

    print("Done")
