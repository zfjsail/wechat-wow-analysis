from scipy.special import iv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from layers import BatchMultiHeadGraphAttention, GraphConv, GATEncoderGraph


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


class BatchWrapDiffGATPool(nn.Module):
    def __init__(self, pretrained_emb_dim, vertex_feature_dim, use_vertex_feature,
                 n_units=[1433, 8, 7], n_heads=[8, 1],
                 dropout=0.1, attn_dropout=0.0,
                 instance_normalization=False, use_diffpool=True, use_deepinf=True, use_prone=True,
                 mu=1, theta=3.5, num_pooling=1, args=None, use_pretrain=True, attn_type="aa"):
        super(BatchWrapDiffGATPool, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        self.use_prone = use_prone
        self.use_diffpool = use_diffpool
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(pretrained_emb_dim, momentum=0.0, affine=True)

        self.use_pretrain = use_pretrain

        if self.use_pretrain:
            n_units[0] += pretrained_emb_dim

        self.use_vertex_feature = use_vertex_feature
        if self.use_vertex_feature:
            n_units[0] += vertex_feature_dim

        second_order_dim = 16

        if self.use_pretrain and self.use_vertex_feature:
            self.emb_second_order = nn.ModuleList([
                nn.Linear(2, second_order_dim),
                nn.Linear(pretrained_emb_dim, second_order_dim),
                nn.Linear(vertex_feature_dim, second_order_dim)
            ])

        if not self.use_pretrain and self.use_vertex_feature:
            self.emb_second_order_wo_emb = nn.ModuleList([
                nn.Linear(2, second_order_dim),
                # nn.Linear(pretrained_emb_dim, second_order_dim),
                nn.Linear(vertex_feature_dim, second_order_dim)
            ])

        if self.use_pretrain and not self.use_vertex_feature:
            self.emb_second_order_wo_vf = nn.ModuleList([
                nn.Linear(2, second_order_dim),
                nn.Linear(pretrained_emb_dim, second_order_dim),
                # nn.Linear(vertex_feature_dim, second_order_dim)
            ])

        self.layer_stack = nn.ModuleList()
        n_units[-1] = 5
        label_dim = 32
        # node_feature_input_dim = n_units[0]
        node_feature_input_dim = n_units[0] + second_order_dim

        self.pool_layer = SoftPoolingGATEncoder(max_num_nodes=32, input_dim=node_feature_input_dim, hidden_dim=16,
                                                embedding_dim=16, label_dim=label_dim, num_layers=2,
                                                assign_hidden_dim=32, n_head=n_heads[0], attn_dropout=attn_dropout,
                                                use_diffpool=use_diffpool, use_deepinf=use_deepinf,
                                                num_pooling=num_pooling, bn=True, dropout=self.dropout, args=args,
                                                attn_type=attn_type)

        if self.use_diffpool:
            self.fc_after_pool = nn.Linear(label_dim, 2)
        else:
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

    def add_fm(self, x, emb, vertex_features):

        if self.use_pretrain and self.use_vertex_feature:
            input_x_cat = [x, emb, vertex_features]
            fm_second_order_emb_arr = [w(input_x_cat[f_idx]) for f_idx, w in enumerate(self.emb_second_order)]
        elif self.use_vertex_feature and not self.use_pretrain:
            input_x_cat = [x, vertex_features]
            fm_second_order_emb_arr = [w(input_x_cat[f_idx]) for f_idx, w in enumerate(self.emb_second_order_wo_emb)]
        elif not self.use_vertex_feature and self.use_pretrain:
            input_x_cat = [x, emb]
            fm_second_order_emb_arr = [w(input_x_cat[f_idx]) for f_idx, w in enumerate(self.emb_second_order_wo_vf)]
        else:
            raise

        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5

        out_inter = fm_second_order
        return out_inter

    def forward(self, x, adj, emb, vertex_features):

        xx = self.add_fm(x, emb, vertex_features)

        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)
        if self.use_pretrain:
            x_2 = torch.cat((x, emb), dim=2)
        else:
            x_2 = x

        if self.use_vertex_feature:
            x_2 = torch.cat((x_2, vertex_features), dim=2)

        xx = torch.cat((x_2, xx), dim=2)
        # xx = x_2

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


class SoftPoolingGATEncoder(GATEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, n_head, attn_dropout, use_diffpool, use_deepinf,
                 assign_ratio=0.5, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], concat=True, bn=False, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None, attn_type="aa"):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''
        super(SoftPoolingGATEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, n_head, attn_dropout,
                                                    pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                    args=None, bn=bn, dropout=dropout, attn_type=attn_type)

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
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                num_layers, n_head, self.pred_input_dim, hidden_dim,
                embedding_dim, attn_dropout, attn_mask=False, attn_type=attn_type
            )
            conv_block2 = nn.ModuleList()
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
                cur_attn_mask = False  # old False
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_num_layers, n_head, assign_input_dim, assign_hidden_dim, assign_dim, attn_dropout, cur_attn_mask,
                attn_type=attn_type)
            assign_conv_block = nn.ModuleList()
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

        if use_diffpool and use_deepinf:
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

        # out, _ = torch.max(embedding_tensor, dim=1)
        out = torch.sum(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        first_assignment_mat = None

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

            # out, _ = torch.max(embedding_tensor, dim=1)
            out = torch.sum(embedding_tensor, dim=1)

            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        ypred = self.pred_model(output)

        return ypred, first_assignment_mat
