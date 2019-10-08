from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import keras
import numpy as np


class fscore_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_precision = 0
        self.best_recall = 0
        self.best_f1score = 0
        self.best_auc = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        val_label = self.y_val
        pred_val_label = self.model.predict(self.x_val)
        p1 =  precision_score(val_label.reshape(-1), pred_val_label.argmax(axis=-1).reshape(-1))
        r1 = recall_score(val_label.reshape(-1), pred_val_label.argmax(axis=-1).reshape(-1))
        f1 =  f1_score(val_label.reshape(-1), pred_val_label.argmax(axis=-1).reshape(-1))
        print p1, r1, f1

        pred_val_label_mid = pred_val_label.argmax(axis=-1).copy()
        for line in pred_val_label_mid:
            for i, item in enumerate(line):
                if i > 0 and i < len(line) - 1 and item == 1:
                    if line[i - 1] == 0 and line[i + 1] == 0:
                        line[i] = 0
        p2 = precision_score(val_label.reshape(-1), pred_val_label_mid.reshape(-1))
        r2 = recall_score(val_label.reshape(-1), pred_val_label_mid.reshape(-1))
        f2 = f1_score(val_label.reshape(-1), pred_val_label_mid.reshape(-1))
        print p2, r2, f2

        auc1 = roc_auc_score(np.where(val_label.squeeze().sum(axis=1) > 0, 1, 0), np.where(pred_val_label.argmax(axis=-1).sum(axis=1) > 0, 1, 0))
        auc2 = roc_auc_score(np.where(val_label.squeeze().sum(axis=1) > 0, 1, 0), np.where(pred_val_label_mid.sum(axis=1) > 0, 1, 0))
        print auc1, auc2
        if f1 > self.best_f1score:
            self.best_precision = p1
            self.best_recall = r1
            self.best_f1score = f1
            self.best_auc = auc1
        if f2 > self.best_f1score:
            self.best_precision = p2
            self.best_recall = r2
            self.best_f1score = f2
            self.best_auc = auc2
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return