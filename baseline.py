import random
import numpy as np
import os
from os.path import join
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def train_baseline_weibo(method="lr", train_ratio=50, valid_ratio=25, seed=42):
    file_dir = join(settings.DATA_DIR, "weibo")

    vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
    logger.info("vertex ids loaded!")

    vertex_features = np.load(os.path.join(file_dir, "vertex_feature.npy"))
    # vertex_features = vertex_features[:, -1, :]

    ego_vids = vertices[:, -1]
    vertex_features = vertex_features[ego_vids]

    other_features = []
    with open(join(file_dir, "weibo_sample_other_features.txt")) as rf:
        for i, line in enumerate(rf):
            if i % 100000 == 0:
                logger.info("read other features line %d", i)
            cur_other_f = [float(x) for x in line.strip().split()]
            other_features.append(cur_other_f)

    other_features = np.array(other_features)

    x = np.concatenate((vertex_features, other_features), axis=1)

    labels = np.load(os.path.join(file_dir, "label.npy"))
    logger.info("labels loaded!")

    labels, x = sklearn.utils.shuffle(labels, x, random_state=seed)

    N = len(labels)
    train_start, valid_start, test_start = \
        0, int(N * train_ratio / 100), int(N * (train_ratio + valid_ratio) / 100)
    x_train = x[:test_start]
    x_test = x[test_start:]
    y_train = labels[:test_start]
    y_test = labels[test_start:]

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if method == "lr":
        clf = LogisticRegression(n_jobs=10, class_weight="balanced")
    elif method == "rf":
        clf = RandomForestClassifier(random_state=0, verbose=True, n_jobs=10,
                                     max_depth=4, class_weight="balanced")
    else:
        raise

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    # y_score_random = np.array([random.random() for _ in range(len(y_pred))]).reshape((len(y_pred), 1))
    # y_score_random = np.concatenate((y_score_random, 1 - y_score_random), axis=1)
    # y_pred_random = np.array([round(y[1]) for y in y_score_random])
    y_score = clf.predict_proba(x_test)
    try:
        print(clf.feature_importances_)
    except:
        pass
    logger.info('y_true len %d, %s', len(y_test), y_test)
    logger.info('y_pred len %d %s', len(y_pred), y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    auc = roc_auc_score(y_test, y_score[:, 1])
    # print(list(y_score[:, 1]))
    logger.info('pred results %.4f %.4f %.4f', prec, rec, f1)
    logger.info('auc score %.4f', auc)


if __name__ == "__main__":
    # random_guess()
    train_baseline_weibo(method="rf")
    logger.info("done")
