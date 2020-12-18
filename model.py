from scipy.special import iv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from layers import GATEncoderGraph, GraphConv, BatchMultiHeadGraphAttention


class BatchGAT(nn.Module):
    def __init__(self, pretrained_emb_dim, vertex_feature_dim, use_vertex_feature,
                 n_units=[1433, 8, 7], n_heads=[8, 1],
                 dropout=0.1, attn_dropout=0.0,
                 instance_normalization=False):
        super(BatchGAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb_dim, momentum=0.0, affine=True)

        n_units[0] += pretrained_emb_dim

        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            n_units[0] += vertex_feature_dim

        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            # consider multi head from last layer
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(n_heads[i], f_in=f_in,
                                             f_out=n_units[i + 1], attn_dropout=attn_dropout)
            )

    def forward(self, x, adj, emb, vertex_features):
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        if self.use_vertex_feature:
            x = torch.cat((x, vertex_features), dim=2)
        bs, n = adj.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj)  # bs x n_head x n x f_out
            if i + 1 == self.n_layer:
                x = x.mean(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1), None


class SoftPoolingGATEncoder(GATEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, n_head, attn_dropout, use_diffpool, use_deepinf,
                 assign_ratio=0.5, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=False, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''
        super(SoftPoolingGATEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, n_head, attn_dropout,
                                                    pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                    args=None, bn=bn, dropout=dropout)

        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.args = args

        self.use_diffpool = use_diffpool
        self.use_deepinf = use_deepinf

        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):  # conv on clusters
            # use self to register the modules in self.modules()
            # conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
            #     num_layers, n_head, self.pred_input_dim, hidden_dim,
            #     embedding_dim, attn_dropout, attn_mask=False
            # )
            conv_first2 = BatchMultiHeadGraphAttention(n_head, f_in=self.pred_input_dim,
                                                       f_out=embedding_dim, attn_dropout=attn_dropout, attn_mask=False)
            conv_block2 = nn.ModuleList()
            conv_last2 = None
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)

        for i in range(num_pooling):
            if i == 0:
                cur_attn_mask = True
            else:
                cur_attn_mask = True  # old False
            assign_dims.append(assign_dim)
            # assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
            #     assign_num_layers, n_head, assign_input_dim, assign_hidden_dim, assign_dim, attn_dropout, cur_attn_mask)
            assign_conv_first = BatchMultiHeadGraphAttention(n_head, f_in=assign_input_dim,
                                                             f_out=assign_dim, attn_dropout=attn_dropout,
                                                             attn_mask=cur_attn_mask)
            assign_conv_block = nn.ModuleList()
            assign_conv_last = None
            # assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred_input_dim = assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        self.pred_model = self.build_pred_layers(embedding_dim * (num_pooling + 2), pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

        self.gat_sep_layer_1_1 = BatchMultiHeadGraphAttention(8, f_in=hidden_dim, f_out=hidden_dim,
                                                              attn_dropout=attn_dropout, attn_mask=True)
        self.gat_sep_layer_1_2 = BatchMultiHeadGraphAttention(1, f_in=hidden_dim, f_out=self.pred_input_dim,
                                                              attn_dropout=attn_dropout, attn_mask=True)

        self.merge_fc_2 = nn.Linear(label_dim + self.pred_input_dim, label_dim)
        init.xavier_normal_(self.merge_fc_2.weight.data)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        embedding_tensor, emb_first = self.gcn_forward(x, adj,
                                                       self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        ego_embs = []
        gat_add_tensor = F.elu(self.gat_sep_layer_1_1(emb_first, adj))
        gat_add_tensor = self.gat_sep_layer_1_2(gat_add_tensor, adj)
        gat_add_tensor = gat_add_tensor.mean(dim=1)

        ego_embs.append(gat_add_tensor[:, 0, :])

        # out, _ = torch.max(embedding_tensor, dim=1)
        # out = torch.sum(embedding_tensor, dim=1)
        if self.args.data == "wechat":
            out = embedding_tensor[:, 0, :]
        else:
            out = embedding_tensor[:, -1, :]
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        first_assignment_mat = None
        ego_assign = None

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor, _ = self.gcn_forward(x_a, adj,
                                                     self.assign_conv_first_modules[i],
                                                     self.assign_conv_block_modules[i],
                                                     self.assign_conv_last_modules[i],
                                                     embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            if i == 0:
                first_assignment_mat = self.assign_tensor.clone().detach()

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor, cluster_emb_first = self.gcn_forward(x, adj,
                                                                   self.conv_first_after_pool[i],
                                                                   self.conv_block_after_pool[i],
                                                                   self.conv_last_after_pool[i])

            # print("emb shape", embedding_tensor.shape)
            # out, _ = torch.max(embedding_tensor, dim=1)
            # out = torch.sum(embedding_tensor, dim=1)
            if ego_assign is None:
                if self.args.data == "wechat":
                    ego_assign = self.assign_tensor[:, 0, :].unsqueeze(1)
                else:
                    ego_assign = self.assign_tensor[:, -1, :].unsqueeze(1)
            else:
                ego_assign = torch.bmm(ego_assign, self.assign_tensor)
            out = torch.bmm(ego_assign, embedding_tensor)

            out_all.append(out)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        if self.use_diffpool and self.use_deepinf:
            # tmp
            ypred = self.pred_model(output)
            ypred = torch.cat([ypred] + ego_embs, dim=1)
            ypred = self.merge_fc_2(F.relu(ypred))
        elif self.use_diffpool and not self.use_deepinf:
            ypred = self.pred_model(output)
        elif self.use_deepinf and not self.use_diffpool:
            ypred = ego_embs[0]
        else:
            raise NotImplementedError

        return ypred, first_assignment_mat


class BatchWrapDiffGATPool(nn.Module):
    def __init__(self, pretrained_emb_dim, vertex_feature_dim, use_vertex_feature,
                 n_units=[1433, 8, 7], n_heads=[8, 1],
                 dropout=0.1, attn_dropout=0.0,
                 instance_normalization=False, use_diffpool=True, use_deepinf=True, use_prone=True,
                 mu=1, theta=3.5, num_pooling=1, args=None):
        super(BatchWrapDiffGATPool, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        self.use_prone = use_prone
        self.use_diffpool = use_diffpool
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb_dim, momentum=0.0, affine=True)

        n_units[0] += pretrained_emb_dim

        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            n_units[0] += vertex_feature_dim

        self.layer_stack = nn.ModuleList()
        n_units[-1] = 5
        label_dim = 32
        node_feature_input_dim = n_units[0]

        self.pool_layer = SoftPoolingGATEncoder(max_num_nodes=32, input_dim=node_feature_input_dim, hidden_dim=16,
                                                embedding_dim=16, label_dim=label_dim, num_layers=2,
                                                assign_hidden_dim=32, n_head=n_heads[0], attn_dropout=attn_dropout,
                                                use_diffpool=use_diffpool, use_deepinf=use_deepinf,
                                                num_pooling=num_pooling, bn=True, dropout=self.dropout, args=args)

        self.fc_after_pool = nn.Linear(label_dim, 2)
        self.fc_after_prone = nn.Linear(node_feature_input_dim, 2)

        self.mu = torch.nn.Parameter(torch.FloatTensor(1))
        self.theta = theta
        self.order = 3
        torch.nn.init.constant_(self.mu, mu)

    def added_forward(self, batch_adj, batch_feature):
        batchsize, nodenum, feature_dim = batch_feature.shape
        A = batch_adj
        rowsum = torch.sum(batch_adj, dim=2)
        d_inv = torch.pow(rowsum, -1.)
        d_inv[torch.isinf(d_inv)] = 0
        d_inv = d_inv.unsqueeze(2)
        d_inv = d_inv.expand_as(A)
        DA = d_inv * A

        identity = torch.eye(nodenum)
        if torch.cuda.is_available():
            identity = identity.cuda()
        L = identity - DA

        M = L - self.mu * identity
        M = torch.cuda.FloatTensor(M)

        Lx0 = torch.eye(nodenum)
        Lx0 = Lx0.unsqueeze(0).expand_as(M)
        if torch.cuda.is_available():
            Lx0 = Lx0.cuda()
        Lx1 = 0.5 * (torch.bmm(M, M) - Lx0)

        # ----------------------------------------------------

        conv = iv(0, self.theta) * Lx0
        conv -= 2 * iv(1, self.theta) * Lx1
        for i in range(2, self.order):
            Lx2 = torch.bmm(M, Lx1)
            Lx2 = (torch.bmm(M, Lx2) - Lx1) - Lx0
            if i % 2 == 0:
                conv += 2 * iv(i, self.theta) * Lx2
            else:
                conv -= 2 * iv(i, self.theta) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
        mm = torch.bmm(conv, batch_feature)

        return mm

    def forward(self, x, adj, emb, vertex_features):
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((x, emb), dim=2)
        if self.use_vertex_feature:
            x = torch.cat((x, vertex_features), dim=2)

        xx = x
        if self.use_prone:
            xx = self.added_forward(adj, xx)

        assign_mat = None
        if self.use_diffpool:
            xx, assign_mat = self.pool_layer(xx, adj.float(), None)
            x = self.fc_after_pool(F.relu(xx))
        else:
            xx = xx[:, 0, :]
            x = self.fc_after_prone(F.relu(xx))

        return F.log_softmax(x, dim=-1), assign_mat
