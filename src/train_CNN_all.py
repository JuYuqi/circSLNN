import numpy as np
import random
import os

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, Bidirectional, LSTM, Conv1D
from keras.optimizers import RMSprop
from keras_contrib.layers import CRF
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from fscore_callback import fscore_callback

sequence_len = 101
def load_data(protein, generate=True):
    if generate:
        protein = 'ALKBH5'

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

        positive_num = 0
        positive_data = list()
        positive_label = list()
        for rna_name in bindingsites_dict.keys():
            if len(rna_dict[rna_name]) == 0:
                continue
            for binding_site in bindingsites_dict[rna_name]:
                binding_start_index, binding_end_index = binding_site
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
                positive_data.append(sequence)
                positive_label.append(label)
                positive_num += 1


        negtive_num = positive_num

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
        np.savetxt('temp/data-' + protein + '.txt', data, '%s')
        np.savetxt('temp/label-' + protein + '.txt', label, '%s')
        return np.asarray(data), np.asarray(label)
    else:
        data = np.loadtxt('temp/data-' + protein + '.txt', dtype=str)
        label = np.loadtxt('temp/label-' + protein + '.txt', dtype=str)
        return data, label


def model(protein):
    print protein
    origin_data, origin_label = load_data(protein, True)
    total_num = len(origin_data)

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
    model.add(Embedding(max_words, embedding_dim, input_shape=(110, )))  # Random embedding
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.add(Conv1D(256, 10))
    model.add(Bidirectional(LSTM(256 // 2, return_sequences=True)))
    crf = CRF(len(label_vocab), sparse_target=True)
    model.add(crf)
    model.compile(optimizer=RMSprop(), loss=crf.loss_function, metrics=[crf.accuracy])
    model.summary()

    callback = fscore_callback(training_data=[train_data, train_label], validation_data=[val_data, val_label])
    history = model.fit(train_data, train_label, batch_size=256, epochs=10, validation_data=(val_data, val_label),
                        callbacks=[callback])
    model.save('model/' + protein + '_bestmodel.h5')

    with open('output/best_score.txt', 'a') as o:
        o.write(protein + '\t' + str(callback.best_precision) + '\t' + str(callback.best_recall) + '\t' + str(callback.best_f1score) + '\t' + str(callback.best_auc) + '\n')


    # print model.evaluate(val_data, val_label)
    # pred_val_label = model.predict(val_data)
    #
    # origin_val_label = [''.join([label_vocab[index[0]] for index in line]) for line in val_label]
    # origin_pred_val_label = [''.join([label_vocab[np.argmax(index)] for index in line]) for line in pred_val_label]
    #
    #
    # acc = history.history['crf_viterbi_accuracy']
    # val_acc = history.history['val_crf_viterbi_accuracy']
    #
    # plt.plot(range(len(acc)), acc, '')
    # plt.plot(range(len(val_acc)), val_acc, 'o')
    #
    # plt.show()

for file in os.listdir('data/bindingsites'):
    model(file.split('.')[0])
    exit()