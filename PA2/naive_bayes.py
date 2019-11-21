# -*- coding: utf-8 -*-

__author__ = "Zifeng Wang"
__email__  = "wangzf18@mails.tsinghua.edu.cn"
__date__   = "20191005"

import numpy as np
import pdb

np.random.seed(2019)

class BernoulliNaiveBayes:
    def __init__(self):
        return

    def train(self, x_train, y_train):
        """Learn the model.
        Inputs:
            x_train: np.array, shape (num_samples, num_features)
            y_train: np.array, shape (num_samples, )

        Outputs：
            None
        """

        # learn the p(y=1)
        self.p_y1 = y_train.sum() / len(y_train)
        self.p_y0 = 1 - self.p_y1

        # learn the p(x|y=0), p(x|y=1)
        # DO NOT forget the Laplace smoothing!
        self.p_x_y0 = (x_train[y_train == 0].sum(0) + 1) / (y_train[y_train == 0].shape[0] + 2)
        self.p_x_y1 = (x_train[y_train == 1].sum(0) + 1) / (y_train[y_train == 1].shape[0] + 2)


    def predict(self, x_test):
        """Do prediction via the learnt model.
        Inputs:
            x_test: np.array, shape (num_samples, num_features)

        Outputs:
            pred: np.array, shape (num_samples, )
        """

        x_test = x_test.astype(int)

        log_p0 = np.log(self.p_y0) + (x_test * np.log(self.p_x_y0)).sum(1) \
            + ((1 - x_test) * np.log(1 - self.p_x_y0)).sum(1)

        log_p1 = np.log(self.p_y1) + (x_test * np.log(self.p_x_y1)).sum(1) \
            + ((1 - x_test) * np.log(1 - self.p_x_y1)).sum(1)

        pred = np.zeros(x_test.shape[0])
        pred[log_p1 > log_p0] = 1

        return pred

def load_data(data_path="a1a.txt"):
    labels = []
    x = None
    with open(data_path, "r") as f:
        for i,line in enumerate(f.readlines()):
            if i % 200 == 0:
                print("Processing line No.{}.".format(i))
            data_list = line.split()
            label = (int(data_list[0]) + 1) / 2
            feature_idx = [int(l.split(":")[0]) for l in data_list[1:]]
            labels.append(label)
            features = np.zeros(120)
            features[feature_idx] = 1.0

            if x is None:
                x = features
            else:
                x = np.c_[x,features]

    x = x.T
    labels = np.array(labels).astype(int)
    all_idx = np.arange(x.shape[0])
    np.random.shuffle(all_idx)
    x_train = x[all_idx[:1000]]
    y_train = labels[all_idx[:1000]]
    x_test  = x[all_idx[1000:]]
    y_test  = labels[all_idx[1000:]]

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    print("# of Training data:",x_train.shape[0])
    print("# of Test data:",x_test.shape[0])

    acc_func = lambda x,y: 100 * (x == y).sum() / y.shape[0]

    clf = BernoulliNaiveBayes()

    clf.train(x_train, y_train)

    pred = clf.predict(x_test)

    test_acc = acc_func(pred, y_test)
    print("You model acquires Test Acc:{:.2f} %".format(test_acc))

    if test_acc > 75:
        print("Congratulations! Your Naive Bayes classifier WORKS!")
    else:
        print("Check your code! Sth went wrong.")
