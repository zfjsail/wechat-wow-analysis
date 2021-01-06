from os.path import join
import numpy as np
import os
import networkx as nx
import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def gen_wl_features(dataset="weibo"):
    file_dir = join(settings.DATA_DIR, dataset)

    vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
    logger.info("vertex ids loaded!")

    graphs = np.load(os.path.join(file_dir, "adjacency_matrix.npy")).astype(np.float32)
    N = len(graphs)

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


if __name__ == "__main__":
    gen_wl_features(dataset="weibo")
    logger.info("done")
    