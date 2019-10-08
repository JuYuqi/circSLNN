import numpy as np
import random
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Conv1D
from keras.optimizers import RMSprop
from keras_contrib.layers import CRF
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score


sequence_len = 101
def load_data(generate=True):
    if generate:
        protein = "AGO1"

        bindingsites_data = np.loadtxt("data/bindingsites/" + protein + ".csv", dtype=str, skiprows=1)
        bindingsites_dict = dict()
        for line in bindingsites_data:
            if not line[0] in bindingsites_dict:
                bindingsites_dict[line[0]] = list()
            bindingsites_dict[line[0]].append([int(line[8]) - 1, int(line[9]) - 1])

        rna_dict = dict()
        with open("data/human_hg19_circRNAs.fa") as f:
            is_name_line = True
            for line in f:
                if is_name_line:
                    rna_name = line[1:17]
                    is_name_line = False
                else:
                    rna_content = line[:-1]
                    is_name_line = True
                    rna_dict[rna_name] = rna_content

        positive_num = 3000
        negtive_num = 3000

        positive_data = list()
        positive_label = list()
        for i in range(positive_num):
            while True:
                rna_name = random.sample(bindingsites_dict.keys(), 1)[0]
                if len(rna_dict[rna_name]) == 0:
                    continue
                binding_start_index, binding_end_index = random.sample(bindingsites_dict[rna_name], 1)[0]
                mid_index = (binding_start_index + binding_end_index) / 2
                start_index = mid_index - sequence_len / 2
                end_index = mid_index + sequence_len / 2
                if start_index - 9 < 0 or start_index - 9 > len(rna_dict[rna_name]) - 1 or end_index + 9 < 0 or end_index + 9 > len(rna_dict[rna_name]) - 1:
                    continue
                sequence = rna_dict[rna_name][start_index - 9 : end_index + 9 + 1]
                label = ""
                has_other_binding_sites = False
                for j in range(start_index, end_index + 1):
                    success = False
                    for bindingsite in bindingsites_dict[rna_name]:
                        if j == bindingsite[0]:
                            label += 'I'
                            success = True
                            break
                        if j == bindingsite[1]:
                            label += 'I'
                            success = True
                            break
                        if j > bindingsite[0] and j < bindingsite[1]:
                            label += 'I'
                            success = True
                            break
                    if not success:
                        label += 'O'
                    if success and (j < binding_start_index or j > binding_end_index):
                        has_other_binding_sites = True
                        break
                if has_other_binding_sites:
                    continue
                break
            positive_data.append(sequence)
            positive_label.append(label)


        negtive_data = list()
        negtive_label = list()
        for i in range(negtive_num):
            while True:
                rna_name = random.sample(rna_dict.keys(), 1)[0]
                if len(rna_dict[rna_name]) == 0:
                    continue
                mid_index = random.randrange(0, len(rna_dict[rna_name]))
                start_index = mid_index - sequence_len / 2
                end_index = mid_index + sequence_len / 2
                if start_index - 9 < 0 or start_index - 9 > len(rna_dict[rna_name]) - 1 or end_index + 9 < 0 or end_index + 9 > len(rna_dict[rna_name]) - 1:
                    continue
                if rna_name in bindingsites_dict:
                    valid = True
                    for j in range(start_index, end_index + 1):
                        for bindingsite in bindingsites_dict[rna_name]:
                            if j >= bindingsite[0] and j <= bindingsite[1]:
                                valid = False
                                break
                    if not valid:
                        continue
                sequence = rna_dict[rna_name][start_index - 9 : end_index + 9 + 1]
                label = 'O' * sequence_len
                break
            negtive_data.append(sequence)
            negtive_label.append(label)

        data = positive_data + negtive_data
        label = positive_label + negtive_label

        data_label = list(zip(data, label))
        random.shuffle(data_label)
        data, label = zip(*data_label)
        np.savetxt('temp/data.txt', data, '%s')
        np.savetxt('temp/label.txt', label, '%s')
        return np.asarray(data), np.asarray(label)
    else:
        data = np.loadtxt('temp/data.txt', dtype=str)
        label = np.loadtxt('temp/label.txt', dtype=str)
        return data, label

origin_data, origin_label = load_data(False)


label_vocab = ['O', 'I']

texts = np.asarray([[line[index] for index in range(5, sequence_len + 9 + 5)] for line in origin_data])
words = np.unique(texts)
max_words = len(words)
word_index = dict((w, i) for i, w in enumerate(words))
data = np.asarray([[word_index[text] for text in line] for line in texts])


embedding_dim = 30



label = [[label_vocab.index(char) for char in line] for line in origin_label]
label = np.expand_dims(label, 2)

train_data = data[:5000]
val_data = data[5000:]
train_label = label[:5000]
val_label = label[5000:]


model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_shape=(110, )))  # Random embedding
model.add(Conv1D(32, 10))
model.add(Bidirectional(LSTM(256 // 2, return_sequences=True)))
crf = CRF(len(label_vocab), sparse_target=True)
model.add(crf)
model.compile(optimizer=RMSprop(), loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()

history = model.fit(train_data, train_label, batch_size=512, epochs=10, validation_data=(val_data, val_label))


print model.evaluate(val_data, val_label)
pred_val_label = model.predict(val_data)
print precision_score(val_label.reshape(-1), pred_val_label.argmax(axis=-1).reshape(-1))
print recall_score(val_label.reshape(-1), pred_val_label.argmax(axis=-1).reshape(-1))
print f1_score(val_label.reshape(-1), pred_val_label.argmax(axis=-1).reshape(-1))


origin_val_label = [''.join([label_vocab[index[0]] for index in line]) for line in val_label]
origin_pred_val_label = [''.join([label_vocab[np.argmax(index)] for index in line]) for line in pred_val_label]


acc = history.history['crf_viterbi_accuracy']
val_acc = history.history['val_crf_viterbi_accuracy']

plt.plot(range(len(acc)), acc, '')
plt.plot(range(len(val_acc)), val_acc, 'o')

plt.show()

