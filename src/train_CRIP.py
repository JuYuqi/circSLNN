import gzip
import os

import numpy as np
import random
import matplotlib.pyplot as plt

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Bidirectional, LSTM, Conv1D
from keras.optimizers import RMSprop
from keras_contrib.layers import CRF
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from fscore_callback import fscore_callback

sequence_len = 101 - 9
total_num = 6000

def load_CRIP_data():
    def read_seq(seq_file):
        seq_list = []
        seq = ''
        with gzip.open(seq_file, 'r') as fp:
            for line in fp:
                if line[0] == '>':
                    name = line[1:-1]
                    if len(seq):
                        seq_list.append(seq)
                    seq = ''
                else:
                    seq = seq + line[:-1]
            if len(seq):
                seq_list.append(seq)
        return seq_list

    protein = '6_CLIP-seq-eIF4AIII_1'
    path = 'iDeep/datasets/clip/{}/5000/training_sample_0'.format(protein)
    train_data = read_seq(os.path.join(path, 'sequences.fa.gz'))
    train_type = np.loadtxt(gzip.open(os.path.join(path, 'matrix_Response.tab.gz')), skiprows=1)
    train_label = list()
    for item in train_type:
        if item == 0:
            train_label.append('O' * sequence_len)
        else:
            train_label.append('I' * sequence_len)
    path = 'iDeep/datasets/clip/{}/5000/test_sample_0'.format(protein)
    test_data = read_seq(os.path.join(path, 'sequences.fa.gz'))
    test_type = np.loadtxt(gzip.open(os.path.join(path, 'matrix_Response.tab.gz')), skiprows=1)
    test_label = list()
    for item in test_type:
        if item == 0:
            test_label.append('O' * sequence_len)
        else:
            test_label.append('I' * sequence_len)

    return train_data + test_data, train_label + test_label

origin_data, origin_label = load_CRIP_data()


label_vocab = ['O', 'I']

texts = np.asarray([[line[index - 5 : index + 5] for index in range(5, sequence_len + 9 + 5)] for line in origin_data])
words = np.unique(texts)
max_words = len(words)
word_index = dict((w, i) for i, w in enumerate(words))
data = np.asarray([[word_index[text] for text in line] for line in texts])

embedding_index = dict()
with open('temp/10MerVector.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype=float)
        embedding_index[word] = coefs

embedding_dim = 30
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector



label = [[label_vocab.index(char) for char in line] for line in origin_label]
label = np.expand_dims(label, 2)

train_data = data[:total_num / 6 * 5]
val_data = data[total_num / 6 * 5:]
train_label = label[:total_num / 6 * 5]
val_label = label[total_num / 6 * 5:]


model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_shape=(101, )))  # Random embedding
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.add(Conv1D(256, 10))
model.add(Bidirectional(LSTM(256 // 2, return_sequences=True)))
crf = CRF(len(label_vocab), sparse_target=True)
model.add(crf)
model.compile(optimizer=RMSprop(), loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()

history = model.fit(train_data, train_label, batch_size=256, epochs=10, validation_data=(val_data, val_label),
                    callbacks=[fscore_callback(training_data=[train_data, train_label], validation_data=[val_data, val_label])])


print model.evaluate(val_data, val_label)
pred_val_label = model.predict(val_data)


origin_val_label = [''.join([label_vocab[index[0]] for index in line]) for line in val_label]
origin_pred_val_label = [''.join([label_vocab[np.argmax(index)] for index in line]) for line in pred_val_label]


acc = history.history['crf_viterbi_accuracy']
val_acc = history.history['val_crf_viterbi_accuracy']

plt.plot(range(len(acc)), acc, '')
plt.plot(range(len(val_acc)), val_acc, 'o')

plt.show()

