from os.path import join
import numpy as np
import os
from collections import defaultdict as dd
import sklearn
import networkx as nx
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def gen_wl_features(dataset="weibo"):
    file_dir = join(settings.DATA_DIR, dataset)

    if dataset == "weibo":
        graphs = np.load(os.path.join(file_dir, "adjacency_matrix.npy")).astype(np.float32)
        N = len(graphs)
        vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
        logger.info("vertex ids loaded!")

        wf = open(join(file_dir, "{}_wl_features.txt".format(dataset)), "w")
        wf.write("x x x x\n")
        for i in range(N):
            if i % 10000 == 0:
                logger.info("process %d", i)
            cur_adj = graphs[i]
            cur_nodes = vertices[i]

            wf.write(str(i)+" ")
            wf.write(str(i)+" ")
            wf.write(str(i)+" ")

            edge_str = ""

            cur_g = nx.from_numpy_array(cur_adj)

            if dataset == "weibo":
                ego_node = len(cur_adj) - 1
            else:
                ego_node = 0

            for e in cur_g.edges():
                v1, v2 = e
                v1_map = cur_nodes[v1]
                v2_map = cur_nodes[v2]
                if v1 == ego_node:
                    edge_str += "{},-1|".format(v2_map)
                elif v2 == ego_node:
                    edge_str += "{},-1|".format(v1_map)
                else:
                    edge_str += "{},{}|".format(v1_map, v2_map)

            if edge_str[-1] == "|":
                edge_str = edge_str[:-1]

            wf.write(edge_str +"\n")

        wf.close()
    else:
        roles = ["train", "valid", "test"]
        for role in roles:
            graphs = np.load(os.path.join(file_dir, "{}_adjacency_matrix.npy".format(role))).astype(np.float32)
            N = len(graphs)

            vertices = np.load(os.path.join(file_dir, "{}_vertex_ids.npy".format(role)))
            logger.info("vertex ids loaded!")

            wf = open(join(file_dir, "{}_{}_wl_features.txt".format(dataset, role)), "w")
            wf.write("x x x x\n")
            for i in range(N):
                if i % 10000 == 0:
                    logger.info("process %d", i)
                cur_adj = graphs[i]
                cur_nodes = vertices[i]

                wf.write(str(i) + " ")
                wf.write(str(i) + " ")
                wf.write(str(i) + " ")

                edge_str = ""

                cur_g = nx.from_numpy_array(cur_adj)

                if dataset == "weibo":
                    ego_node = len(cur_adj) - 1
                else:
                    ego_node = 0

                for e in cur_g.edges():
                    v1, v2 = e
                    v1_map = cur_nodes[v1]
                    v2_map = cur_nodes[v2]
                    if v1 == ego_node:
                        edge_str += "{},-1|".format(v2_map)
                    elif v2 == ego_node:
                        edge_str += "{},-1|".format(v1_map)
                    else:
                        edge_str += "{},{}|".format(v1_map, v2_map)

                if len(edge_str) and edge_str[-1] == "|":
                    edge_str = edge_str[:-1]

                wf.write(edge_str + "\n")
            wf.close()


def gen_weibo_sample_features():
    file_dir = join(settings.DATA_DIR, "weibo")

    graphs = np.load(os.path.join(file_dir, "adjacency_matrix.npy")).astype(np.float32)
    N = len(graphs)
    vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
    logger.info("vertex ids loaded!")

    inf_features = np.load(
            os.path.join(file_dir, "influence_feature.npy")).astype(np.float32)

    friend_dict = dd(set)
    with open(join(file_dir, "all_edges.txt")) as rf:
        for i, line in enumerate(rf):
            if i % 100000 == 0:
                logger.info("read edges line %d", i)
            items = line.strip().split()
            u, v = int(items[0]), int(items[1])
            friend_dict[u].add(v)
            friend_dict[v].add(u)

    wf = open(join(file_dir, "weibo_sample_other_features.txt"), "w")

    for i in range(N):
        if i % 10000 == 0:
            logger.info("process %d", i)
            wf.flush()
        cur_adj = graphs[i]
        cur_nodes = vertices[i]

        cur_g = nx.from_numpy_array(cur_adj)
        cur_inf_features = inf_features[i]

        n_act_nbrs = sum(cur_inf_features[:, 0])

        n_cc = len(list(nx.connected_components(cur_g)))

        lcc = float(2 * cur_g.number_of_edges())/(cur_g.number_of_nodes()*(cur_g.number_of_nodes() - 1))

        ego_friends = friend_dict[int(cur_nodes[-1])]
        n_ego_f = len(ego_friends)
        f_friends = [friend_dict[int(f)] for f in cur_nodes[:-1] if int(f) in friend_dict]
        common_f_ego = [len(ego_friends.intersection(ff))/n_ego_f for ff in f_friends]
        common_f_f = [len(ego_friends.intersection(ff))/len(ff) for ff in f_friends]
        avg_ff_1 = np.mean(common_f_ego)
        avg_ff_2 = np.mean(common_f_f)
        sum_ff_1 = sum(common_f_ego)
        sum_ff_2 = sum(common_f_f)

        cur_features = [n_act_nbrs, n_cc, lcc, avg_ff_1, avg_ff_2, sum_ff_1, sum_ff_2]
        wf.write("\t".join([str(x) for x in cur_features])+"\n")

    wf.close()


def split_weibo_features(train_ratio=50, valid_ratio=25, seed=42):
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

    np.save(join(file_dir, "x_train_baseline_features.npy"), x_train)
    np.save(join(file_dir, "y_train_baseline.npy"), y_train)
    np.save(join(file_dir, "x_test_baseline_features.npy"), x_test)
    np.save(join(file_dir, "y_test_baseline.npy"), y_test)


if __name__ == "__main__":
    # gen_wl_features(dataset="wechat")
    # gen_weibo_sample_features()
    split_weibo_features()
    logger.info("done")
