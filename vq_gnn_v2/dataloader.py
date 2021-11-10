import torch
from torch import Tensor
from typing import List

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch_geometric.utils.subgraph as subgraph

import pdb

class OurDataLoader(torch.utils.data.DataLoader):
    def __init__(self, data, cluster_indices, batch_size, gnn_type='GCN', sampler_type='node', walk_length=None,
                 recovery_flag=True, train_flag=True, cont_sliding_window=1, **kwargs):
        self.sampler_type = sampler_type
        self.gnn_type = gnn_type
        self.recovery_flag = recovery_flag = True # always on in this B+B' version
        self.walk_length = walk_length
        self.train_flag = train_flag
        self.cont_sliding_window = cont_sliding_window

        self.N = data.num_nodes
        self.adj_t = data.adj_t
        row, col, self.edge_w = self.adj_t.coo()
        self.edge_index = torch.stack([row, col])

        self.batch_size = batch_size

        if sampler_type == 'cluster' :
            if self.train_flag :
                num_sub_edge = 0
                for i in range(len(cluster_indices)) :
                    sub_edge_index, _ = subgraph(cluster_indices[i], self.edge_index)
                    num_sub_edge += sub_edge_index.shape[1]
                num_total_edge = self.edge_index.shape[1]
                print(f'inter over intra: ', (num_total_edge - num_sub_edge)/num_sub_edge)

            super(OurDataLoader, self).__init__(cluster_indices, collate_fn=self.__collate_cluster__,
                                                batch_size=self.batch_size, **kwargs)
        else :
            if sampler_type == 'edge':
                self.batch_size = batch_size // 2
            elif sampler_type == 'rw':
                self.batch_size = batch_size // (self.walk_length + 1)
            elif sampler_type == 'cont':
                self.batch_size = batch_size // self.cont_sliding_window
            else:
                self.batch_size = batch_size

            super(OurDataLoader, self).__init__(range(self.N), collate_fn=self.__collate__,
                                                batch_size=self.batch_size, **kwargs)

    def __collate_cluster__(self, batches: List[Tensor]):
        idx = torch.cat(batches, dim=0)
        idx = torch.tensor(idx)
        node_idx_list = [idx]
        result_list = []

        for node_idx in node_idx_list :
            result_list.append((self._k_hop_subgraph(node_idx), node_idx))

        return result_list

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
                node_idx = torch.cat([node_idx] * 3)
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
        for node_idx in node_idx_list :
            result_list.append((self._k_hop_subgraph(node_idx), node_idx))

        return result_list

    def _k_hop_subgraph(self, node_idx, num_hops=1, relabel_nodes=True):

        num_nodes = self.N
        row, col = self.edge_index
        edge_index = self.edge_index

        node_mask = row.new_empty(num_nodes, dtype=torch.bool)
        edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device).flatten()
        else:
            node_idx = node_idx.to(row.device)
        subsets = [node_idx]

        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])

        subset, inv = torch.cat(subsets).unique(return_inverse=True, sorted=False)
        inv = inv[:node_idx.numel()]

        # rearrange the subset to make batch nodes in the beginning of the array
        rearrange_mask = row.new_full((subset.size(0),), True, dtype=torch.bool)
        rearrange_mask[inv] = False
        subset = torch.cat([subset[inv],subset[rearrange_mask]])

        # assert torch.equal(inv, torch.arange(len(node_idx)))
        assert torch.equal(node_idx, subset[:len(node_idx)])

        node_mask.fill_(False)
        if self.train_flag :
            node_mask[subset] = True
            edge_mask = node_mask[row] & node_mask[col]

        # TODO: double check here
        else :
            node_mask[node_idx] = True
            edge_mask = node_mask[row]

        edge_index = edge_index[:, edge_mask]

        if relabel_nodes:
            node_idx = row.new_full((num_nodes,), -1)
            node_idx[subset] = torch.arange(subset.size(0), device=row.device)
            edge_index = node_idx[edge_index]

        edge_w = self.edge_w[edge_mask]
        return subset, edge_index, edge_w




if __name__ == '__main__' :
    # dataset = PygNodePropPredDataset(name='ogbn-arxiv',
    #                                  transform=T.ToSparseTensor(),
    #                                  root='/cmlscratch/kong/datasets/ogb')
    # data = dataset[0]
    # data.adj_t = data.adj_t.to_symmetric()
    # data = norm_adj(data)
    #
    # test_loader = OurDataLoader(data, batch_size=1000, walk_length=5, sampler_type='node', shuffle=True,
    #                             num_workers=8, drop_last=True)
    #
    # import time
    # total = 0
    # for _ in range(10) :
    #     start = time.time()
    #     for i, batches in enumerate(test_loader) :
    #         if i >= 16 :
    #             break
    #         for batch in batches :
    #             # subset, edge_index, edge_w = batch
    #             pass
    #     epoch_time = time.time()-start
    #     print(f'Epoch time:{epoch_time}')
    #     total += epoch_time
    # print(f'Avg:{total/10}')


    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor(),
                                     root='/cmlscratch/kong/datasets/ogb')
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()

    num_parts, batch_size = 16, 1
    perm, ptr = metis(data.adj_t, num_parts=num_parts, log=True)
    data = permute(data, perm, log=True)
    n_id = torch.arange(data.num_nodes)
    batches = n_id.split((ptr[1:] - ptr[:-1]).tolist())
    batches = [(i, batches[i]) for i in range(len(batches))]

    row, col, edge_w = data.adj_t.coo()
    edge_index = torch.stack([row, col])

    num_sub_edge = 0
    for i in range(len(batches)):
        idx = batches[i][1]
        sub_edge_index, _ = subgraph(idx, edge_index)
        num_sub_edge += sub_edge_index.shape[1]

        print(f'num_node:{len(idx)}, num_edge:{sub_edge_index.shape[1]}')
    num_total_edge = edge_index.shape[1]
    print(f'inter over intra: ', (num_total_edge - num_sub_edge) / num_sub_edge)
    print(f'total:{num_total_edge}, sub:{num_sub_edge}')

