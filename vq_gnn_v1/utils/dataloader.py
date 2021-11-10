from math import ceil
import torch
from torch_sparse import SparseTensor, coalesce
import time

class OurDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data, batch_size, gnn_type='GCN', sampler_type='edge', walk_length=None,
                 recovery_flag=True, train_flag=True, cont_sliding_window=1, **kwargs):
        self.sampler_type = sampler_type
        self.gnn_type = gnn_type
        self.recovery_flag = recovery_flag
        self.walk_length = walk_length
        self.train_flag = train_flag
        self.cont_sliding_window = cont_sliding_window

        self.x = data.x
        self.adj_t = data.adj_t
        self.deg = data.deg
        self.deg_inv = data.deg_inv
        self.N = self.x.shape[0]
        self.num_batches = (self.N // batch_size) + 1

        if sampler_type == 'edge':
            self.batch_size = batch_size // 2
        elif sampler_type == 'rw':
            self.batch_size = batch_size // (self.walk_length + 1)
        elif sampler_type == 'cont' :
            self.batch_size = batch_size // self.cont_sliding_window
        else:
            self.batch_size = batch_size

        super(OurDataLoader, self).__init__(range(self.N), collate_fn=self.__collate__,
                                            batch_size=self.batch_size, **kwargs)

    def __collate__(self, idx):
        idx = torch.tensor(idx)
        node_idx_list = []

        if self.sampler_type == 'node':
            node_idx_list.append(idx)

        elif self.sampler_type == 'edge':
            node_idx_list.append(self.adj_t.random_walk(idx, 1).view(-1).unique())

        elif self.sampler_type == 'rw':
            node_idx_list.append(self.adj_t.random_walk(idx, self.walk_length).view(-1).unique())

        elif self.sampler_type == 'cont':
            node_idx = idx
            node_idx_list.append(node_idx)
            for _ in range(self.walk_length):
                node_idx = torch.cat([node_idx] * 3, 0)
                node_idx = self.adj_t.random_walk(node_idx, 1)[:, 1].unique()[:self.batch_size]
                node_idx_list.append(node_idx)

            if self.cont_sliding_window > 1 :
                node_idx_list_holder = []
                for i in range(len(node_idx_list)-self.cont_sliding_window+1) :
                    node_idx_list_holder.append(torch.cat(node_idx_list[i:i+self.cont_sliding_window]).unique())
                node_idx_list = node_idx_list_holder
        else:
            raise ValueError('Sampler type not supported!')

        result_list = []
        for node_idx in node_idx_list:

            x = self.x[node_idx]
            deg_inv = self.deg_inv[node_idx]
            A_BN = self.adj_t[node_idx]

            if self.recovery_flag and self.train_flag:
                A_BB = self.adj_t.saint_subgraph(node_idx)[0].coo()
            else:
                A_BB = None

            if self.gnn_type != 'GCN' and self.train_flag:
                # start = time.time()
                A_NB = self.deg[node_idx].view(-1, 1) * A_BN * self.deg_inv.view(1, -1)
                A_NB_v = A_NB.coo()[2]
                # print(time.time()-start)

            else:
                A_NB_v = None
            A_BN = A_BN.coo()

            result_list.append((x, deg_inv, A_BN, A_BB, A_NB_v, node_idx))

        return result_list

# class OurDataLoader(torch.utils.data.DataLoader):
#     def __init__(self, data, gnn_type='GCN', sampler_type='edge', walk_length=None, recovery_flag=True,
#                  train_flag=True, **kwargs):
#         self.sampler_type = sampler_type
#         self.gnn_type = gnn_type
#         self.recovery_flag = recovery_flag
#         self.walk_length = walk_length
#         self.train_flag = train_flag
#
#         self.x = data.x
#         self.adj_t = data.adj_t
#         self.deg = data.deg
#         self.deg_inv = data.deg_inv
#         self.N = self.x.shape[0]
#
#         super(OurDataLoader, self).__init__(range(self.N), collate_fn=self.__collate__, **kwargs)
#
#     def __collate__(self, idx):
#         idx = torch.tensor(idx)
#
#         if self.sampler_type == 'node':
#             node_idx = idx
#
#         elif self.sampler_type == 'edge':
#             node_idx = self.adj_t.random_walk(idx, 1)
#             node_idx = node_idx.view(-1).unique()
#
#         elif self.sampler_type == 'rw':
#             node_idx = self.adj_t.random_walk(idx, self.walk_length)
#             node_idx = node_idx.view(-1).unique()
#
#         else:
#             raise ValueError('Sampler type not supported!')
#
#         x = self.x[node_idx]
#         deg_inv = self.deg_inv[node_idx]
#         A_BN = self.adj_t[node_idx]
#
#         if self.recovery_flag and self.train_flag:
#             A_BB = self.adj_t.saint_subgraph(node_idx)[0].coo()
#         else:
#             A_BB = None
#
#         if self.gnn_type != 'GCN' and self.train_flag :
#             A_NB = self.deg[node_idx].view(-1, 1) * A_BN * self.deg_inv.view(1, -1)
#             A_NB_v = A_NB.coo()[2]
#         else:
#             A_NB_v = None
#
#         A_BN = A_BN.coo()
#
#         return x, deg_inv, A_BN, A_BB, A_NB_v, node_idx

import pdb
def mapper(batch, c, num_M, gnn_type, device):
    deg_inv, A_BN, A_BB, A_NB_v, batch_idx = batch
    num_B = batch_idx.shape[0]

    rows, cols, values = [], [], []
    A_BN_r, A_BN_c, A_BN_v = A_BN
    A_BN_c_mapped = c[A_BN_c].to(torch.long) + num_B
    rows.append(A_BN_r), cols.append(A_BN_c_mapped), values.append(A_BN_v)

    if A_NB_v is not None:
        rows.append(A_BN_c_mapped), cols.append(A_BN_r), values.append(A_NB_v)

    if A_BB is not None:
        A_BB_r, A_BB_c, A_BB_v = A_BB
        rows.append(A_BB_r), cols.append(A_BB_c), values.append(A_BB_v)

        A_BB_c_org = batch_idx[A_BB_c]
        A_BB_c_mapped = c[A_BB_c_org].to(torch.long) + num_B
        A_BB_v_neg = -1.0 * A_BB_v
        rows.append(A_BB_r), cols.append(A_BB_c_mapped), values.append(A_BB_v_neg)

        if A_NB_v is not None:
            A_BB_r_org = batch_idx[A_BB_r]
            A_BB_r_mapped = c[A_BB_r_org] + num_B
            rows.append(A_BB_r_mapped), cols.append(A_BB_c), values.append(A_BB_v_neg)

    rows = torch.cat(rows)
    cols = torch.cat(cols)
    values = torch.cat(values)

    dim = num_B + num_M
    idx_input, value_input = coalesce(torch.stack([rows, cols], dim=0), values, m=dim, n=dim)

    slicing = value_input > 0
    row_input = idx_input[0][slicing]
    col_input = idx_input[1][slicing]
    value_input = value_input[slicing]

    if gnn_type != 'SAGE':
        self_loops = torch.arange(batch_idx.shape[0]).to(device)
        row_input, col_input = torch.cat([row_input, self_loops]), torch.cat([col_input, self_loops])
        value_input = torch.cat([value_input, deg_inv])

    adj_input = SparseTensor(row=row_input, col=col_input, value=value_input, sparse_sizes=(dim, dim))

    if gnn_type == 'GCN':
        adj_input = adj_input.to_symmetric()

    return adj_input
