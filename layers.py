import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, attn_mask=True, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        # self.n_head = 1
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(self.n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        # self.a_src = Parameter(torch.Tensor(n_head, f_out, 8))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))
        # self.a_dst = Parameter(torch.Tensor(n_head, f_out, 8))
        self.a_src_bias = Parameter(torch.Tensor(1))
        self.a_dst_bias = Parameter(torch.Tensor(1))
        init.constant_(self.a_src_bias, 0)
        init.constant_(self.a_dst_bias, 0)

        self.w_bi = Parameter(torch.Tensor(1, f_out, f_out))

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
        init.xavier_uniform_(self.w_bi)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        n = adj.size()[1]
        # print("h", h.shape)
        if len(h.shape) == 3:
            h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        else:
            h_prime = torch.matmul(h, self.w)  # bs x n_head x n x f_out
        f_out = h_prime.size()[-1]
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        # attn_src = torch.matmul(torch.tanh(h_prime), self.a_src) + self.a_src_bias  # bs x n_head x n x 1
        # attn_src = torch.matmul(h_prime, self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        # attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst) + self.a_dst_bias  # bs x n_head x n x 1
        # attn_dst = torch.matmul(h_prime, self.a_dst)  # bs x n_head x n x 1
        attn_go = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3,
                                                                                       2)  # bs x n_head x n x n
        attn_2_order = attn_src.expand(-1, -1, -1, n) * attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)  # bs x n_head x n x n
        # attn_2_order_scale = attn_src.expand(-1, -1, -1, n) * attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)/np.sqrt(f_out)  # bs x n_head x n x n
        attn_2_order_scale = attn_src.expand(-1, -1, -1, n) * attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2) * np.sqrt(f_out)  # bs x n_head x n x n

        attn_half = attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)

        # attn = torch.einsum("abce,abde->abcd", h_prime, h_prime)  # weibo AUC: 0.8251 Prec: 0.4869 Rec: 0.7387 F1: 0.5869
        # attn = torch.einsum("abce,abde->abcd", torch.tanh(h_prime), torch.tanh(h_prime))  # weibo AUC: 0.8245 Prec: 0.4834 Rec: 0.7556 F1: 0.5896
        # attn_sdp = torch.einsum("abce,abde->abcd", h_prime, h_prime)/np.sqrt(h_prime.size()[-1])  # AUC: 0.8280 Prec: 0.4885 Rec: 0.7489 F1: 0.5913
        # attn = torch.einsum("abce,abde->abcd", torch.tanh(h_prime), torch.tanh(h_prime))/np.sqrt(h_prime.size()[-1])  # weibo AUC: 0.8219 Prec: 0.4836 Rec: 0.7456 F1: 0.5866
        # attn = attn_go * torch.sigmoid(attn_sdp)  # weibo lr=0.01 AUC: 0.8324 Prec: 0.4911 Rec: 0.7550 F1: 0.5951
        # attn = attn_go * torch.sigmoid(attn_sdp)  # weibo lr=0.1 AUC: 0.8322 Prec: 0.4963 Rec: 0.7413 F1: 0.5946

        # h_norm = torch.norm(h_prime, dim=3).unsqueeze(3) + 1e-6
        # h_scale = h_prime/h_norm
        # attn = torch.einsum("abce,abde->abcd", h_scale, h_scale)  # cosine sim weibo: AUC: 0.8115 Prec: 0.4677 Rec: 0.7342 F1: 0.5714

        # attn = attn_go + torch.sigmoid(attn_sdp)  # weibo AUC: 0.8238 Prec: 0.4808 Rec: 0.7495 F1: 0.5858
        # attn_with_ego = torch.einsum("abe,abde->abd", h_prime[:, :, -1, :], h_prime).unsqueeze(3).expand(-1, -1, -1, n).permute(0, 1, 3, 2)  #todo -1 is for weibo
        # attn_with_ego = attn_with_ego/np.sqrt(h_prime.size()[-1])
        # attn = attn_go * torch.sigmoid(attn_with_ego)
        attn = attn_go
        # attn = attn_go * np.sqrt(f_out)
        # attn = attn_half
        # attn = attn_2_order
        # attn = attn_2_order_scale

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

    def forward_old4(self, h, adj):  # weibo bs = 256 AUC: 0.8331 Prec: 0.5027 Rec: 0.7336 F1: 0.5966
        n = adj.size()[1]
        # print("h", h.shape)
        if len(h.shape) == 3:
            h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        else:
            h_prime = torch.matmul(h, self.w)  # bs x n_head x n x f_out
        # h_expand = h_prime.unsqueeze(3).expand(-1, -1, -1, n, -1)
        # print("h_prime", h_prime.shape)
        h_dot = torch.einsum("abce,abde->abcde", h_prime, h_prime)
        # h_dot = torch.einsum("abce,abde->abcde", torch.tanh(h_prime), torch.tanh(h_prime))
        # attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        # attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        # attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3,
        #                                                                                2)  # bs x n_head x n x n
        # print("----", h_dot.shape, self.a_src.shape)
        if self.a_src.shape[0] == h_dot.shape[1]:
            attn = torch.einsum("abcde,bef->abcdf", h_dot, self.a_src).squeeze(4)
        else:
            attn = torch.matmul(h_dot, self.a_src).squeeze(4)
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

    def forward_old3(self, h, adj):  # weibo AUC: 0.8309 Prec: 0.4980 Rec: 0.7366 F1: 0.5942
        n = adj.size()[1]
        bs = adj.size()[0]
        # print("h", h.shape)
        if len(h.shape) == 3:
            h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        else:
            h_prime = torch.matmul(h, self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 8
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 8
        attn_src = attn_src.view(-1, n, 8)
        attn_dst = attn_dst.view(-1, n, 8)
        attn = torch.bmm(attn_src, attn_dst.permute(0, 2, 1))  # (bs*n_head) x n x n
        attn = attn.view(bs, -1, n, n)
        # print("n_head", self.n_head, attn.shape)

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

    # def forward(self, h, adj):  # weibo AUC: 0.8299 Prec: 0.4970 Rec: 0.7343 F1: 0.5928
    def forward_old2(self, h, adj):  # tanh before attn weibo: AUC: 0.8201 Prec: 0.4803 Rec: 0.7423 F1: 0.5832
        n = adj.size()[1]
        if len(h.shape) == 3:
            h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        else:
            h_prime = torch.matmul(h, self.w)  # bs x n_head x n x f_out

        attn_left = torch.matmul(h_prime, self.w_bi).squeeze(1)  # bs x n x f_out
        # attn_left = torch.matmul(torch.tanh(h_prime), self.w_bi).squeeze(1)  # bs x n x f_out
        # print("h_prime shape", h_prime.shape)
        attn = torch.bmm(attn_left, h_prime.squeeze(1).permute(0, 2, 1)).unsqueeze(1)  # bs x n x n
        # attn = torch.bmm(attn_left, torch.tanh(h_prime).squeeze(1).permute(0, 2, 1)).unsqueeze(1)  # bs x n x n
        # attn = torch.tanh(attn)  # weibo AUC: 0.8234 Prec: 0.4820 Rec: 0.7455 F1: 0.5854

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

    def forward_old1(self, h, adj):
        n = adj.size()[1]
        # print("h", h.shape)
        if len(h.shape) == 3:
            h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        else:
            h_prime = torch.matmul(h, self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3,
                                                                                       2)  # bs x n_head x n x n
        # attn = attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)  # half: weibo AUC: 0.8243 Prec: 0.4777 Rec: 0.7573 F1: 0.5858

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
        # self.act = nn.ELU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        # print(self.pred_input_dim, hidden_dim, num_layers, embedding_dim)

        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, num_layers, n_head, input_dim, hidden_dim, emb_dim, attn_dropout, attn_mask):

        conv_first = BatchMultiHeadGraphAttention(n_head, f_in=input_dim,
                                                  f_out=hidden_dim, attn_dropout=attn_dropout, attn_mask=attn_mask)

        conv_block = nn.ModuleList([
            BatchMultiHeadGraphAttention(n_head=n_head, f_in=hidden_dim, f_out=hidden_dim, attn_dropout=attn_dropout,
                                         attn_mask=attn_mask)
            for _ in range(num_layers - 2)
        ])

        conv_last = BatchMultiHeadGraphAttention(n_head, f_in=hidden_dim,
                                                 f_out=emb_dim, attn_dropout=attn_dropout, attn_mask=attn_mask)

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
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
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
        # x = self.act(x)
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
