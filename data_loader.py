import os
import numpy as np
import sklearn
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from utils import load_w2v_feature
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired data points
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class InfluenceDataset(Dataset):
    def __init__(self, file_dir, seed, shuffle):
        self.vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
        logger.info("vertex ids loaded!")

        embedding_path = os.path.join(file_dir, "prone.emb2")
        max_vertex_idx = np.max(self.vertices)
        embedding = load_w2v_feature(embedding_path, max_vertex_idx)
        self.embedding = torch.FloatTensor(embedding)
        logger.info("global prone embedding loaded")

        vertex_features = np.load(os.path.join(file_dir, "vertex_feature.npy"))
        vertex_features = preprocessing.scale(vertex_features)
        self.vertex_features_dim = vertex_features.shape[1]
        vertex_features = np.concatenate((vertex_features, np.zeros(shape=(1, self.vertex_features_dim))), axis=0)
        self.vertex_features = torch.FloatTensor(vertex_features)
        del vertex_features
        logger.info("global vertex features loaded!")

        self.graphs = np.load(os.path.join(file_dir, "adjacency_matrix.npy")).astype(np.float32)

        # self-loop trick, the input graphs should have no self-loop
        identity = np.identity(self.graphs.shape[2], dtype=np.bool_)
        self.graphs += identity
        self.graphs[self.graphs != False] = True
        logger.info("graphs loaded!")

        # whether a user has been influenced
        # whether he/she is the ego user
        self.influence_features = np.load(
            os.path.join(file_dir, "influence_feature.npy")).astype(np.float32)
        logger.info("influence features loaded!")

        self.labels = np.load(os.path.join(file_dir, "label.npy"))
        logger.info("labels loaded!")

        if shuffle:
            self.graphs, self.influence_features, self.labels, self.vertices = \
                sklearn.utils.shuffle(
                    self.graphs, self.influence_features,
                    self.labels, self.vertices,
                    random_state=seed
                )

        self.N = len(self.graphs)
        if self.N > settings.TEST_SIZE:
            self.graphs = self.graphs[: settings.TEST_SIZE]
            self.influence_features = self.influence_features[: settings.TEST_SIZE]
            self.labels = self.labels[: settings.TEST_SIZE]
            self.vertices = self.vertices[: settings.TEST_SIZE]
            self.N = settings.TEST_SIZE

        logger.info("%d ego networks loaded, each with size %d" % (self.N, self.graphs.shape[1]))

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.labels))
        self.class_weight = torch.FloatTensor(class_weight)

    def get_embedding(self):
        return self.embedding

    def get_vertex_features(self):
        return self.vertex_features

    def get_feature_dimension(self):
        return self.influence_features.shape[-1]

    def get_num_class(self):
        return np.unique(self.labels).shape[0]

    def get_class_weight(self):
        return self.class_weight

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.graphs[idx], self.influence_features[idx], self.labels[idx], self.vertices[idx]
