from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from os.path import join

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from data_loader import ChunkSampler
from data_loader import InfluenceDatasetWeChat, InfluenceDatasetOthers
from model import BatchGAT, BatchWrapDiffGATPool

import settings

import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='diffpool_prone', help="models used")
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')  # wow: 0.01, click: 0.1
parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.2, help='adj Dropout rate.')  # little use
parser.add_argument('--use-vertex-feature', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use vertices' structural features")
parser.add_argument('--label-type', type=str, default="click", help="Label type")
parser.add_argument('--data', type=str, default="weibo", help="Dataset Type")
parser.add_argument('--mu', type=float, default=0.4, help='mu')
parser.add_argument('--theta', type=float, default=7, help='theta')
parser.add_argument('--num-pooling', type=int, default=1, help="Number of hierarchical pooling layers")
parser.add_argument('--use-pretrain', type=bool, default=True, help="whether pre-train as input")
parser.add_argument('--batch', type=int, default=1024, help="Batch size")


parser.add_argument('--tensorboard-log', type=str, default='', help="name of this run")
# parser.add_argument('--model', type=str, default='gat', help="models used")
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden-units', type=str, default="16,8",
                    help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8,1",
                    help="Heads in each layer, splitted with comma")  # adjust
parser.add_argument('--dim', type=int, default=64, help="Embedding dimension")
parser.add_argument('--check-point', type=int, default=10, help="Check point")
parser.add_argument('--instance-normalization', action='store_true', default=False,
                    help="Enable instance normalization")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--file-dir', type=str, default=join(settings.DATA_DIR, "wechat"),
                    help="Input file directory")
parser.add_argument('--train-ratio', type=float, default=50, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=25, help="Validation ratio (0, 100)")
parser.add_argument('--class-weight-balanced', action='store_true', default=True,
                    help="Adjust weights inversely proportional"
                         " to class frequencies in the input data")
parser.add_argument('--debug', type=bool, default=False, help="Debug or not")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("arg use vertex feature", args.use_vertex_feature)

# if args.label_type == "like":  # best paras wechat
#     args.mu = 1.0
#     args.theta = 5
# else:
#     args.mu = 0.2
#     args.theta = 7
print(args)


def mae_loss(output, target, class_weight):
    output = output[:, 1]
    w = class_weight[target]
    return torch.mean(torch.abs(output - target) * w)


def evaluate(epoch, loader, thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    for i_batch, batch in enumerate(loader):
        graph, features, labels, vertices = batch
        graph = graph.type(torch.ByteTensor)
        bs = graph.size(0)
        embed = emb(vertices)
        cur_vertex_features = vertex_features[vertices]

        if args.cuda:
            features = features.cuda()
            graph = graph.cuda()
            labels = labels.cuda()
            embed = embed.cuda()
            cur_vertex_features = cur_vertex_features.cuda()

        output, _ = model(features, graph, embed, cur_vertex_features)
        if args.model == "gat":
            output = output[:, 0, :]
        loss_batch = F.nll_loss(output, labels, class_weight)
        # loss_batch = mae_loss(output, labels, class_weight)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("model %s loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                args.model, loss / total, auc, prec, rec, f1)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return best_thr, [auc, prec, rec, f1]
    else:
        return None, [auc, prec, rec, f1]


def train(epoch, train_loader, valid_loader, test_loader, log_desc='train_'):
    global best_auc, best_valid, best_test, best_epoch, wf

    model.train()

    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(train_loader):
        graph, features, labels, vertices = batch
        graph = graph.type(torch.ByteTensor)
        bs = graph.size(0)
        embed = emb(vertices)
        cur_vertex_features = vertex_features[vertices]

        if args.cuda:
            features = features.cuda()
            graph = graph.cuda()
            labels = labels.cuda()
            embed = embed.cuda()
            cur_vertex_features = cur_vertex_features.cuda()

        optimizer.zero_grad()
        output, assign_mat = model(features, graph, embed, cur_vertex_features)
        if args.model == "gat":
            output = output[:, 0, :]
        loss_train = F.nll_loss(output, labels, class_weight)
        # loss_train = mae_loss(output, labels, class_weight)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()

        if i_batch == 0 and assign_mat is not None:
            out_dir = join(settings.OUT_DIR, "vis-clusters")
            os.makedirs(out_dir, exist_ok=True)
            np.save(join(out_dir, "adj-batch-{}-0.npy".format(args.label_type)), graph.cpu().numpy())
            np.save(join(out_dir, "inf-features-batch-{}-0.npy".format(args.label_type)), features.cpu().numpy())
            np.save(join(out_dir, "assign-mat-batch-{}-0.npy".format(args.label_type)), assign_mat.cpu().numpy())
            np.save(join(out_dir, "labels-batch-{}-0.npy".format(args.label_type)), labels.cpu().numpy())

    logger.info("train loss in this epoch %f", loss / total)
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thr, cur_valid_r = evaluate(epoch, valid_loader, return_best_thr=True, log_desc='valid_')
        _, cur_test_r = evaluate(epoch, test_loader, thr=best_thr, log_desc='test_')
        if best_auc < cur_valid_r[0]:
            best_valid = cur_valid_r
            best_test = cur_test_r
            best_auc = cur_valid_r[0]
            best_epoch = epoch
        logger.info("********************BEST UNTIL NOW IN EPOCH %d***********************", best_epoch)
        logger.info("model %s, u=%f, theta=%f, best validation until now: AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                    args.model, args.mu, args.theta, best_valid[0], best_valid[1], best_valid[2], best_valid[3])
        logger.info("model %s, u=%f, theta=%f Best test until now: AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                    args.model, args.mu, args.theta, best_test[0], best_test[1], best_test[2], best_test[3])

        wf.write("********************BEST UNTIL NOW IN EPOCH {}***********************\n".format(best_epoch))
        wf.write("model {}, u={}, theta={}, best validation until now: AUC: {:.04} Prec: {:.04} Rec: {:.04} "
                 "F1: {:.04}\n".format(args.model, args.mu, args.theta, best_valid[0], best_valid[1], best_valid[2],
                                       best_valid[3]))
        wf.write("model {}, u={}, theta={}, best validation until now: AUC: {:.04} Prec: {:.04} Rec: {:.04} "
                 "F1: {:.04}\n".format(args.model, args.mu, args.theta, best_test[0], best_test[1], best_test[2],
                                       best_test[3]))
        wf.flush()


seeds = [42]

wf_temp = "test_results_model_{}_epoch_{}_lr_{}_dropout_{}_attn_dp_{}_vfeature_{}_label_type_{}_data_{}_mu_{}_theta" \
          "_{}_num_pooling_{}_pretrain_{}.txt"

# for seed in seeds:
# for n_pool in range(0, 5):
# for mu in range(0, 6):
for theta in range(0, 5):
#     args.mu = 0.2 * mu
#     args.num_pooling = n_pool
    args.theta = 2 * theta + 1.0
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("seed", seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print("args current loop", args)

    wfname = wf_temp.format(args.model, args.epochs, args.lr, args.dropout, args.attn_dropout, args.use_vertex_feature,
                            args.label_type, args.data, args.mu, args.theta, args.num_pooling, args.use_pretrain)
    out_dir = join(settings.OUT_DIR, "results", args.data)
    os.makedirs(out_dir, exist_ok=True)
    wf = open(join(out_dir, wfname), "w")

    if args.data == "wechat":
        influence_dataset_train = InfluenceDatasetWeChat(args.file_dir, args.dim, args.seed, args.shuffle,
                                                         args.label_type,
                                                         args.debug, "train")
        influence_dataset_valid = InfluenceDatasetWeChat(args.file_dir, args.dim, args.seed, args.shuffle,
                                                         args.label_type,
                                                         args.debug, "valid")
        influence_dataset_test = InfluenceDatasetWeChat(args.file_dir, args.dim, args.seed, args.shuffle,
                                                        args.label_type,
                                                        args.debug, "test")

        N = len(influence_dataset_train)
        logger.info("Number of training samples: %d", N)
        Nvalid = len(influence_dataset_valid)
        Ntest = len(influence_dataset_test)
        n_classes = 2
        class_weight = influence_dataset_train.get_class_weight() \
            if args.class_weight_balanced else torch.ones(n_classes)
        logger.info("class_weight=%.2f:%.2f", class_weight[0], class_weight[1])
        feature_dim = influence_dataset_train.get_feature_dimension()

        train_loader = DataLoader(influence_dataset_train, batch_size=args.batch,
                                  sampler=ChunkSampler(N, 0))
        valid_loader = DataLoader(influence_dataset_valid, batch_size=args.batch,
                                  sampler=ChunkSampler(Nvalid, 0))
        test_loader = DataLoader(influence_dataset_test, batch_size=args.batch,
                                 sampler=ChunkSampler(Ntest, 0))
        vertex_feature_dim = influence_dataset_train.vertex_features_dim
        emb_origin = influence_dataset_train.get_embedding()
        vertex_features = influence_dataset_train.get_vertex_features()
    else:
        args.file_dir = join(settings.DATA_DIR, args.data)
        influence_dataset = InfluenceDatasetOthers(args.file_dir, args.seed, args.shuffle)
        N = len(influence_dataset)
        train_start, valid_start, test_start = \
            0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
        train_loader = DataLoader(influence_dataset, batch_size=args.batch,
                                  sampler=ChunkSampler(valid_start - train_start, 0))
        valid_loader = DataLoader(influence_dataset, batch_size=args.batch,
                                  sampler=ChunkSampler(test_start - valid_start, valid_start))
        test_loader = DataLoader(influence_dataset, batch_size=args.batch,
                                 sampler=ChunkSampler(N - test_start, test_start))
        feature_dim = influence_dataset.get_feature_dimension()
        n_classes = 2
        class_weight = influence_dataset.get_class_weight() \
            if args.class_weight_balanced else torch.ones(n_classes)
        vertex_feature_dim = influence_dataset.vertex_features_dim
        emb_origin = influence_dataset.get_embedding()
        vertex_features = influence_dataset.get_vertex_features()

    n_units = [feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")] + [n_classes]
    logger.info("feature dimension=%d", feature_dim)
    logger.info("number of classes=%d", n_classes)

    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = None
    if args.model == "gat":
        model = BatchGAT(pretrained_emb_dim=args.dim,
                         vertex_feature_dim=vertex_feature_dim,
                         use_vertex_feature=args.use_vertex_feature,
                         n_units=n_units, n_heads=n_heads,
                         dropout=args.dropout, instance_normalization=args.instance_normalization)
    elif args.model == "diffpool_prone":
        model = BatchWrapDiffGATPool(pretrained_emb_dim=args.dim,
                                     vertex_feature_dim=vertex_feature_dim,
                                     use_vertex_feature=args.use_vertex_feature,
                                     n_units=n_units, n_heads=n_heads,
                                     dropout=args.dropout, instance_normalization=args.instance_normalization,
                                     use_diffpool=True, use_deepinf=False, use_prone=True,
                                     mu=args.mu, theta=args.theta,
                                     attn_dropout=args.attn_dropout,
                                     num_pooling=args.num_pooling,
                                     # num_pooling=n_pool,
                                     use_pretrain=args.use_pretrain,
                                     args=args)
    elif args.model == "diffpool":
        model = BatchWrapDiffGATPool(pretrained_emb_dim=args.dim,
                                     vertex_feature_dim=vertex_feature_dim,
                                     use_vertex_feature=args.use_vertex_feature,
                                     n_units=n_units, n_heads=n_heads,
                                     dropout=args.dropout, instance_normalization=args.instance_normalization,
                                     use_diffpool=True, use_deepinf=False, use_prone=False,
                                     mu=args.mu, theta=args.theta,
                                     attn_dropout=args.attn_dropout,
                                     num_pooling=args.num_pooling,
                                     use_pretrain=args.use_pretrain,
                                     args=args)

    print(model)
    if args.cuda:
        model.cuda()
        class_weight = class_weight.cuda()

    emb = torch.nn.Embedding(emb_origin.size(0), emb_origin.size(1))
    emb.weight = torch.nn.Parameter(emb_origin)
    emb.weight.requires_grad = False

    params = [{'params': filter(lambda p: p.requires_grad, model.parameters())
    if args.model != "gat" else model.layer_stack.parameters()}]
    optimizer = optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)

    best_epoch = -1
    best_auc = 0
    best_valid = [0] * 4
    best_test = [0] * 4

    # eval first
    best_thr1, _ = evaluate(0, valid_loader, return_best_thr=True, log_desc='valid_')
    evaluate(0, test_loader, thr=best_thr1, log_desc='test_')

    # Train model
    t_total = time.time()
    logger.info("training...")
    for epoch in range(args.epochs):
        train(epoch, train_loader, valid_loader, test_loader)
    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    logger.info("--------------------------------------------")
    logger.info("best is in epoch %d", best_epoch)
    logger.info("model %s, best validation until now: AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                args.model, best_valid[0], best_valid[1], best_valid[2], best_valid[3])
    logger.info("model %s, Best test until now: AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                args.model, best_test[0], best_test[1], best_test[2], best_test[3])

    wf.write("best is in epoch {}\n".format(best_epoch))
    wf.write("model {}, u={}, theta={}, best validation until now: AUC: {:.04} Prec: {:.04} Rec: {:.04} "
             "F1: {:.04}\n".format(args.model, args.mu, args.theta, best_valid[0], best_valid[1], best_valid[2],
                                   best_valid[3]))
    wf.write("model {}, u={}, theta={}, best validation until now: AUC: {:.04} Prec: {:.04} Rec: {:.04} "
             "F1: {:.04}\n".format(args.model, args.mu, args.theta, best_test[0], best_test[1], best_test[2],
                                   best_test[3]))
    wf.close()
