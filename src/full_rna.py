import numpy as np
import pandas as pd
from model_predict import dealwithdata
from keras.models import load_model
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import os
from scipy.stats import pearsonr
import random
from keras_contrib.layers import CRF, crf


def cor(protein):
    # protein = "AGO1"
    bindingsites_data = np.loadtxt("data/bindingsites/" + protein + ".csv", dtype=str, skiprows=1)
    bindingsites_dict = dict()
    for line in bindingsites_data:
        if not line[0] in bindingsites_dict:
            bindingsites_dict[line[0]] = list()
        bindingsites_dict[line[0]].append([int(line[8]) - 1, int(line[9]) - 1])

    model = load_model('web/static/models/' + protein.upper() + "_bestmodel.hdf5")
    model.summary()


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

    # origin_data, origin_label = load_data(protein, False)
    # sequence_len = 101
    # texts = np.asarray([[line[index - 5 : index + 5] for index in range(5, sequence_len + 9 + 5)] for line in origin_data])
    # words = np.unique(texts)
    # word_index = dict((w, i) for i, w in enumerate(words))


    x = list()
    y = list()
    for i in range(1000):
        rna_name = random.sample(rna_dict.keys(), 1)[0]
        rna_content = rna_dict[rna_name]
        if not rna_name in bindingsites_dict:
            continue
        try:
            test_X = dealwithdata(rna_content)

            predictions = model.predict_proba(np.asarray(test_X))[:, 1]
            gold = np.zeros((len(rna_content)))
            if rna_name in bindingsites_dict:
                for item in bindingsites_dict[rna_name]:
                    gold[item[0]:item[1] + 1] = 1
                bind_num = len(bindingsites_dict[rna_name])
                predict_label = np.zeros((len(rna_content)))
                for i in range(len(predict_label)):
                    if i / 101 < predictions.shape[0] and predictions[i / 101] > 0.5:
                        predict_label[i] = 1
                print rna_name + '\t' + str(gold.sum()) + '\t' + str(predictions.round().sum()) + '\t' + str(precision_score(gold, predict_label)) \
                      + '\t' + str(recall_score(gold, predict_label)) + '\t' + str(f1_score(gold, predict_label))
                i += 1
            x.append(bind_num)
            y.append(np.average(predictions) * predictions.shape[0])
        except Exception as e:
            pass


    print 'relevance:' + '\t' + str(pd.Series(x).corr(pd.Series(y)))
    r, p_value = pearsonr(x, y)
    print 'relevance:' + '\t' + str(r)
    print '\n'

for files in os.listdir("web/static/models"):
    file = files.split('_')[0]
    file = 'FMRP'
    cor(file)
