import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, attn_mask=True, bias=True, attn_type="aa"):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(self.n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_src_bias = Parameter(torch.Tensor(1))
        self.a_dst_bias = Parameter(torch.Tensor(1))
        init.constant_(self.a_src_bias, 0)
        init.constant_(self.a_dst_bias, 0)

        self.attn_type = attn_type
        self.attn_mask = attn_mask
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        n = adj.size()[1]
        if len(h.shape) == 3:
            h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        else:
            h_prime = torch.matmul(h, self.w)  # bs x n_head x n x f_out
        if self.attn_type == "aa":  # additive attention
            attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
            attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
            attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3,
                                                                                           2)  # bs x n_head x n x n
        elif self.attn_type == "da":  # dot attention
            attn_src = torch.matmul(torch.tanh(h_prime), self.a_src) + self.a_src_bias  # bs x n_head x n x 1
            attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst) + self.a_dst_bias  # bs x n_head x n x 1
            attn = attn_src.expand(-1, -1, -1, n) * attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3,
                                                                                           2)  # bs x n_head x n x n
        else:
            raise NotImplementedError

        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(1)  # bs x 1 x n x n

        if self.attn_mask:
            attn.data.masked_fill_(mask.bool(), float("-inf"))

        attn = self.softmax(attn)  # bs x n_head x n x n
        if self.training:
            attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y


class GATEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers, n_head, attn_dropout,
                 pred_hidden_dims=[], concat=True, bn=False, dropout=0.0, args=None):  # original default bn is True
        super(GATEncoderGraph, self).__init__()
        self.concat = concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1
        self.dropout = dropout

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            num_layers, n_head, input_dim, hidden_dim, embedding_dim, attn_dropout, True)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, num_layers, n_head, input_dim, hidden_dim, emb_dim, attn_dropout, attn_mask, attn_type):

        conv_first = BatchMultiHeadGraphAttention(n_head, f_in=input_dim,
                                                  f_out=hidden_dim, attn_dropout=attn_dropout, attn_mask=attn_mask,
                                                  attn_type=attn_type)

        conv_block = nn.ModuleList([
            BatchMultiHeadGraphAttention(n_head=n_head, f_in=hidden_dim, f_out=hidden_dim, attn_dropout=attn_dropout,
                                         attn_mask=attn_mask, attn_type=attn_type)
            for _ in range(num_layers - 2)
        ])

        conv_last = BatchMultiHeadGraphAttention(n_head, f_in=hidden_dim,
                                                 f_out=emb_dim, attn_dropout=attn_dropout, attn_mask=attn_mask,
                                                 attn_type=attn_type)

        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                # pred_layers.append(self.act)
                pred_layers.append(nn.ReLU())
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        # bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        bn_module = nn.BatchNorm2d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''
        # print("conv first", conv_first.w.shape, "x", x.shape)
        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x_first = x

        x_all = [x]

        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x_all.append(x)

        if conv_last is not None:
            x = conv_last(x, adj)
            x_all.append(x)
        # x_tensor: [batch_size x head x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=3)

        # remove heads
        x_tensor = torch.mean(x_tensor, dim=1)

        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor, x_first

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        # out, _ = torch.max(x, dim=1)
        out = torch.sum(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            # out, _ = torch.max(x, dim=1)
            out = torch.sum(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        if self.conv_last is not None:
            x = self.conv_last(x, adj)
        # out, _ = torch.max(x, dim=1)
        out = torch.sum(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred
