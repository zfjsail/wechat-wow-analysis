import random
import numpy as np
import os
from os.path import join
import sklearn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def random_guess(data="weibo", train_ratio=50, valid_ratio=25, seed=42):
    file_dir = join(settings.DATA_DIR, data)
    labels = np.load(os.path.join(file_dir, "label.npy"))
    logger.info("labels loaded!")

    y_score_random = np.array([random.random() for _ in range(len(labels))]).reshape((len(labels), 1))

    labels, y_score_random = sklearn.utils.shuffle(labels, y_score_random, random_state=seed)

    N = len(labels)
    train_start, valid_start, test_start = \
        0, int(N * train_ratio / 100), int(N * (train_ratio + valid_ratio) / 100)
    y_score_random_test = y_score_random[test_start:]
    labels_test = labels[test_start:]

    y_score_random_test = np.concatenate((y_score_random_test, 1 - y_score_random_test), axis=1)
    y_pred_random = np.array([round(y[1]) for y in y_score_random_test])

    prec, rec, f1, _ = precision_recall_fscore_support(labels_test, y_pred_random, average="binary")
    auc = roc_auc_score(labels_test, y_score_random_test[:, 1])
    logger.info('random pred results %.4f %.4f %.4f', prec, rec, f1)
    logger.info('random auc score %.4f', auc)


if __name__ == "__main__":
    random_guess()
    logger.info("done")
