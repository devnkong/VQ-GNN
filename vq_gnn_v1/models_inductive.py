import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import dense_to_sparse, to_undirected

# from nearest_embed import NearestEmbed, NearestEmbedEMA

from vq import VectorQuantizer, VectorQuantizerEMA
from convs import OurGCNConv, OurGATConv, OurSAGEConv, Transformer

import pdb
from sklearn.cluster import KMeans, MiniBatchKMeans
import math

from torch_sparse import coalesce
from utils.dataloader import mapper


class LowRankGNNBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_M, num_D, num_N, num_branch, cluster, kmeans_iter,
                 EMA_flag, kmeans_init, use_gcn, commitment_cost, grad_normalize_scale, hook_flag, warm_up_flag,
                 momentum, conv_type, transformer_flag, val_num_N=0, test_num_N=0):
        super(LowRankGNNBlock, self).__init__()

        self.num_M, self.num_D, self.num_N, self.EMA_flag = num_M, num_D, num_N, EMA_flag
        self.commitment_cost = commitment_cost
        self.hook_flag = hook_flag
        self.grad_normalize_scale = grad_normalize_scale
        self.conv_type = conv_type
        self.transformer_flag = transformer_flag

        # C init
        # c1 = -torch.ones(num_N, dtype=torch.long)
        # c = torch.randint(0, self.num_M, (self.num_N,), dtype=torch.long)
        c = torch.randint(0, self.num_M, (self.num_N,), dtype=torch.short)
        self.register_buffer('c_indices', c)


        self.val_num_N, self.test_num_N = val_num_N, test_num_N
        c = torch.randint(0, self.num_M, (self.val_num_N,), dtype=torch.short)
        self.register_buffer('c_indices_val', c)
        c = torch.randint(0, self.num_M, (self.test_num_N,), dtype=torch.short)
        self.register_buffer('c_indices_test', c)


        if self.transformer_flag :
            self.conv = Transformer(self.num_D+1)
            self.transformer_k = torch.nn.Linear(num_D, num_D)
        else :
            if self.conv_type != 'GAT' :
                self.conv = OurGCNConv(in_channels, in_channels, normalize=False)
            else :
                self.conv = OurGATConv(in_channels+1, in_channels+1, bias=False, add_self_loops=False)

        if EMA_flag :
            # self.emb = NearestEmbedEMA(self.num_M, self.num_D)
            add_flag = True if self.transformer_flag or self.conv_type == 'GAT' else False
            self.vq = VectorQuantizerEMA(self.num_M, self.num_D, commitment_cost=self.commitment_cost,
                                         grad_normalize_scale=grad_normalize_scale, warm_up_flag=warm_up_flag,
                                         momentum=momentum, add_flag=add_flag)
        else :
            raise ValueError('Not EMA vq not studied')
            # self.vq = VectorQuantizer(self.num_M, self.num_D, commitment_cost=self.commitment_cost)

        self.kmeans_init = kmeans_init
        self.grad_kmeans_init = kmeans_init
        self.inited = False

        self.ln = torch.nn.LayerNorm(in_channels, elementwise_affine=False)


    def reset_parameters(self):
        self.conv.reset_parameters()

    def hook(self, grad):
        # grad_use = grad[:, :self.num_D]
        grad_use = grad

        if self.grad_kmeans_init :
            X_B_cpu = torch.nn.BatchNorm1d(self.X_B.shape[1], affine=False, track_running_stats=False)\
                (self.X_B).detach().cpu()
            grad_cpu = torch.nn.BatchNorm1d(grad_use.shape[1], affine=False, track_running_stats=False, eps=1e-24)\
                (grad_use).detach().cpu()

            # reassignment_ratio=0.01
            k_obj = MiniBatchKMeans(n_clusters=self.num_M, init='k-means++', batch_size=400, n_init=10,
                                    init_size=4000, reassignment_ratio=0.3).fit(
                                    torch.cat([X_B_cpu, grad_cpu], dim=1))

            k_centroids = torch.tensor(k_obj.cluster_centers_, dtype=torch.float).to(self.X_B.device)
            k_counts = torch.tensor(k_obj.counts_, dtype=torch.long).to(self.X_B.device)

            self.vq.kmeans_init(k_centroids, k_counts)
            self.c_indices = -1
            self.c_indices[self.batch_indices] = torch.tensor(k_obj.labels_, dtype=torch.short).to(self.X_B.device)
            self.grad_kmeans_init = False

            # to reset X_output
            encoding_indices, encodings = self.vq.update(self.X_B, grad_use)
            self.c_indices[self.batch_indices] = encoding_indices.squeeze().to(torch.short)

        else :
            # grad_before = self.vq.get_grad()[self.c_indices[self.batch_indices].to(torch.long)]
            # grad_diff_before = grad_before - grad
            # self.grad_error_before = torch.norm(grad_diff_before, dim=1).mean().item()

            vq_steps = 1
            for _ in range(vq_steps) :
                encoding_indices, encodings = self.vq.update(self.X_B, grad_use)

            self.c_indices[self.batch_indices] = encoding_indices.squeeze().to(torch.short)

            grad_diff = grad_use - self.vq.get_grad()[encoding_indices.squeeze()]
            self.grad_error_after = torch.norm(grad_diff, dim=1).mean().item()
            self.grad_norm = torch.norm(grad_use, dim=1).mean().item()

            # grad_error_by_cluster = (encodings / (1e-6 + encodings.sum(dim=0, keepdim=True))).t() @ \
            #                         (torch.norm(grad_diff,dim=1)/torch.norm(grad,dim=1)).unsqueeze(1)
            # self.grad_error_by_cluster = grad_error_by_cluster.squeeze()


            X_B_diff = self.X_B - self.vq.get_codebook()[encoding_indices.squeeze()]
            self.vq_backward_error = torch.norm(X_B_diff, dim=1).mean().item()

            # feat_error_by_cluster = (encodings / (1e-6 + encodings.sum(dim=0, keepdim=True))).t() @ \
            #                         (torch.norm(X_B_diff,dim=1)/torch.norm(self.X_B,dim=1)).unsqueeze(1)
            # self.feat_error_by_cluster = feat_error_by_cluster.squeeze()

        return grad

    def M_hook(self, grad):
        self.M_grad_norm = torch.norm(grad, dim=1).mean().item()

    def ln_hook(self, grad):
        self.ln_grad_norm = torch.norm(grad, dim=1).mean().item()

    def X_B_hook(self, grad):
        self.X_B_grad_norm = torch.norm(grad, dim=1).mean().item()

    def X_bar_hook(self, grad):
        self.X_bar_grad_norm = torch.norm(grad, dim=1).mean().item()

    def init(self, X_B, batch_indices):
        encoding_indices = self.vq.feature_update(X_B)
        self.c_indices[batch_indices] = encoding_indices.squeeze().to(torch.short)

    def forward(self, X_B, batch_A, warm_up_rate, unlabeled):
        self.X_B = X_B.detach()
        self.batch_indices = batch_indices = batch_A[-1]

        if self.training : # init
            if self.kmeans_init : # reassignment_ratio=0.01, batch_size=400, n_init=10,
                k_obj = MiniBatchKMeans(n_clusters=self.num_M, init='k-means++', batch_size=400, n_init=10,
                                        init_size=4000, reassignment_ratio=0.3).fit(
                        torch.nn.BatchNorm1d(X_B.shape[1],affine=False,track_running_stats=False)(X_B).detach().cpu())
                k_centroids = torch.tensor(k_obj.cluster_centers_, dtype=torch.float).to(X_B.device)
                k_counts = torch.tensor(k_obj.counts_, dtype=torch.long).to(X_B.device)

                # print(f'min:{torch.min(k_counts)}')

                self.vq.feature_kmeans_init(k_centroids, k_counts)
                self.c_indices[batch_indices] = torch.tensor(k_obj.labels_, dtype=torch.long).to(X_B.device)
                self.kmeans_init = False

            elif not self.inited or unlabeled :
                self.init(X_B, batch_indices)
            else :
                pass

        adj_input = None
        if self.transformer_flag :
            pass
        else :
            adj_input = mapper(batch_A, self.c_indices, self.num_M, self.conv_type, X_B.device)


        X_bar = self.vq.get_codebook()

        # X_B.requires_grad_()
        # X_B.register_hook(self.X_B_hook)
        # X_bar.requires_grad_()
        # X_bar.register_hook(self.X_bar_hook)


        X_input = torch.cat([X_B, X_bar*warm_up_rate], dim=0)

        # projection before transformer
        if self.transformer_flag :
            X_input = self.ln(X_input)
            X_input = self.transformer_k(X_input)

        if self.transformer_flag or self.conv_type == 'GAT' :         # X_input cat tensor.ones
            X_input = torch.cat([X_input, torch.ones(X_input.shape[0],1).to(X_input.device)], dim=1)

        # Conv
        if self.transformer_flag :
            X_output = self.conv(X_input[:X_B.shape[0]], X_input[X_B.shape[0]:])
        else :
            X_output = self.conv(X_input, adj_input)

        X_output_B, X_output_M = X_output[:X_B.shape[0]], X_output[X_B.shape[0]:]

        if self.inited and self.hook_flag and self.training and not unlabeled :
            # X_output.requires_grad_()

            X_output_B.requires_grad_()
            X_output_B.register_hook(self.hook)

            # X_output_M.requires_grad_()
            # X_output_M.register_hook(self.M_hook)


        if self.transformer_flag or self.conv_type == 'GAT':#use the added one dim to normalize X_out_B
            X_output_B = X_output_B[:, :self.num_D] / (X_output_B[:, self.num_D].unsqueeze(1) + 1e-16)

        # X_output_B_ln = self.ln(X_output_B)
        X_output_B_ln = X_output_B

        # if self.inited and self.hook_flag and self.training and not unlabeled:
        #     X_output_B_ln.register_hook(self.ln_hook)

        # quantized = X_bar[self.c_indices[batch_indices].to(torch.long)]
        # commit_loss = self.commitment_cost*F.mse_loss(X_B, quantized)
        commit_loss = 0

        # info_backward = torch.sum(X_output_M[:, :self.num_D] * self.vq.get_grad())
        info_backward = torch.sum(X_output_M * self.vq.get_grad() * warm_up_rate)

        # self.vq_get_grad_norm = torch.norm(self.vq.get_grad(), dim=1).mean().item()
        # statistics record
        # error = torch.norm(quantized-X_B, dim=1).mean().item()
        # quantized_norm = torch.norm(quantized, dim=1).mean().item()
        error = quantized_norm = 0

        return X_output_B_ln, error, torch.norm(X_B, dim=1).mean().item(), \
               quantized_norm, commit_loss, info_backward, X_B

    def inference(self, X_B, batch_A, warm_up_rate, unlabeled, infer):
        self.X_B = X_B.detach()
        self.batch_indices = batch_indices = batch_A[-1]

        encoding_indices = self.vq.feature_update(self.X_B)
        if infer == 'val':
            self.c_indices_val[self.batch_indices] = encoding_indices.squeeze().to(torch.short)
            c_indices = self.c_indices_val
        elif infer == 'test':
            self.c_indices_test[self.batch_indices] = encoding_indices.squeeze().to(torch.short)
            c_indices = self.c_indices_test
        else:
            raise ValueError('val or test')

        adj_input = None
        if self.transformer_flag :
            pass
        else :
            adj_input = mapper(batch_A, c_indices, self.num_M, self.conv_type, X_B.device)
        X_bar = self.vq.get_codebook()
        X_input = torch.cat([X_B, X_bar*warm_up_rate], dim=0)

        # projection before transformer
        if self.transformer_flag :
            X_input = self.ln(X_input)
            X_input = self.transformer_k(X_input)

        if self.transformer_flag or self.conv_type == 'GAT' :         # X_input cat tensor.ones
            X_input = torch.cat([X_input, torch.ones(X_input.shape[0],1).to(X_input.device)], dim=1)

        # Conv
        if self.transformer_flag :
            X_output = self.conv(X_input[:X_B.shape[0]], X_input[X_B.shape[0]:])
        else :
            X_output = self.conv(X_input, adj_input)

        X_output_B, X_output_M = X_output[:X_B.shape[0]], X_output[X_B.shape[0]:]

        if self.inited and self.hook_flag and self.training and not unlabeled :
            X_output_B.requires_grad_()
            X_output_B.register_hook(self.hook)

        if self.transformer_flag or self.conv_type == 'GAT':#use the added one dim to normalize X_out_B
            X_output_B = X_output_B[:, :self.num_D] / (X_output_B[:, self.num_D].unsqueeze(1) + 1e-16)
        X_output_B_ln = X_output_B
        commit_loss = 0
        info_backward = torch.sum(X_output_M * self.vq.get_grad() * warm_up_rate)
        error = quantized_norm = 0

        return X_output_B_ln, error, torch.norm(X_B, dim=1).mean().item(), \
               quantized_norm, commit_loss, info_backward, X_B

class LowRankGNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout,
                 num_M, num_D, num_N, num_branch, cluster, ln_para, no_second_fc,
                 kmeans_iter, EMA_flag, split, kmeans_init, dropbranch, skip, use_gcn, commitment_cost,
                 grad_normalize_scale, hook, weight_ahead, warm_up_flag, momentum, conv_type, transformer_flag, val_num_N, test_num_N) :
        super(LowRankGNNLayer, self).__init__()

        # weight ahead disabled
        self.weight_ahead = weight_ahead
        if self.weight_ahead :
            if out_channels % num_D != 0 : raise ValueError('Cannot fully split')
            self.num_branch = int(out_channels / num_D)
        else :
            if in_channels % num_D != 0: raise ValueError('Cannot fully split')
            self.num_branch = int(in_channels / num_D)

        self.out_channels = out_channels
        self.no_second_fc = no_second_fc
        self.EMA_flag = EMA_flag
        self.split = split
        self.num_D = num_D
        self.dropbranch= dropbranch
        self.skip = skip
        self.conv_type = conv_type
        self.transformer_flag = transformer_flag

        self.linear_k, self.linear_v, self.gnn_block, self.transformer_block = \
            torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList()
        for i in range(self.num_branch):
            # if not self.split :
            #     self.linear_k.append(torch.nn.Linear(in_channels, num_D))

            if transformer_flag :
                self.transformer_block.append(LowRankGNNBlock(num_D, out_channels, num_M, num_D, num_N, self.num_branch,
                                                      cluster, kmeans_iter, EMA_flag, kmeans_init,
                                                      False, commitment_cost, grad_normalize_scale,
                                                      hook, warm_up_flag, momentum, conv_type, True))

            if no_second_fc :
                self.gnn_block.append(LowRankGNNBlock(num_D, out_channels, num_M, num_D, num_N, self.num_branch,
                                                      cluster, kmeans_iter, EMA_flag, kmeans_init,
                                                      use_gcn, commitment_cost, grad_normalize_scale,
                                                      hook, warm_up_flag, momentum, conv_type, False, val_num_N, test_num_N))
            else :
                raise ValueError('second fc not studied')
                # self.gnn_block.append(LowRankGNNBlock(num_D, num_D, num_M, num_D, num_N, self.num_branch,
                #                                       cluster, kmeans_iter, EMA_flag, kmeans_init,
                #                                       use_gcn, commitment_cost, grad_normalize_scale,
                #                                       hook, warm_up_flag, momentum, conv_type, False))
                # self.linear_v.append(torch.nn.Linear(num_D, out_channels))

        if self.skip :
            self.linear_skip = torch.nn.Linear(in_channels, out_channels)

        self.gnn_transform = torch.nn.Linear(in_channels, out_channels)
        # self.transform = torch.nn.utils.spectral_norm ( torch.nn.Linear(in_channels, out_channels) )
        self.batch_norm = torch.nn.BatchNorm1d(out_channels, affine=False)

        if self.conv_type == 'SAGE' :
            self.fc_sage = torch.nn.Linear(in_channels, out_channels)

        if self.transformer_flag :
            self.transformer_res = torch.nn.Linear(in_channels, out_channels)
            self.transformer_v = torch.nn.Linear(in_channels, out_channels)

    # def reset_parameters(self):
    #     for conv in self.convs:
    #         conv.reset_parameters()
    #     for bn in self.bns:
    #         bn.reset_parameters()

    def forward(self, x, batch_A, warm_up_rate, unlabeled, infer=None):

        errors, X_B_norms, quantized_norms = [], [], []
        losses, info_backwards = 0, 0
        hookeds = []

        if self.training and self.dropbranch > 0 :
            num_branch = int(self.num_branch*(1-self.dropbranch))
            branch_idx = torch.randperm(self.num_branch)[:num_branch]
        else :
            num_branch = self.num_branch
            branch_idx = torch.arange(num_branch)

        x_input, x_output = x, 0
        x_hiddens = []
        for i in branch_idx :
            x_hidden = x_input[:, self.num_D*i:self.num_D*(i+1)]
            if infer is None :
                x_hidden, error, X_B_norm, quantized_norm, loss, info_backward, hooked =\
                    self.gnn_block[i](x_hidden, batch_A, warm_up_rate, unlabeled)
            else :
                x_hidden, error, X_B_norm, quantized_norm, loss, info_backward, hooked =\
                    self.gnn_block[i].inference(x_hidden, batch_A, warm_up_rate, unlabeled, infer)

            if self.training :
                losses += loss
                info_backwards += info_backward

            x_hiddens.append(x_hidden)
            hookeds.append(hooked)
            errors.append(error)
            X_B_norms.append(X_B_norm)
            quantized_norms.append(quantized_norm)

        x_output_gnn = self.gnn_transform(torch.cat(x_hiddens, dim=1))
        if self.conv_type == 'SAGE' :
            x_output_gnn = x_output_gnn + self.fc_sage(x_input)
        x_output = x_output + x_output_gnn

        if self.transformer_flag :
            x_hiddens = []
            for i in branch_idx:
                x_hidden = x_input[:, self.num_D * i:self.num_D * (i + 1)]
                x_hidden, error, X_B_norm, quantized_norm, loss, info_backward, hooked = \
                    self.transformer_block[i](x_hidden, batch_A, warm_up_rate, unlabeled)

                if self.training:
                    losses += loss
                    info_backwards += info_backward

                x_hiddens.append(x_hidden)
                hookeds.append(hooked)

                # errors.append(error)
                # X_B_norms.append(X_B_norm)
                # quantized_norms.append(quantized_norm)

            x_output_trans = self.transformer_v(torch.cat(x_hiddens, dim=1))
            x_output_trans_res = self.transformer_res(x_input)
            x_output = x_output + x_output_trans + x_output_trans_res

        if self.skip :
            x_output += self.linear_skip(x)

        return x_output, errors, X_B_norms, quantized_norms, losses, info_backwards, hookeds


class LowRankGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout,
                 num_M, num_D, num_N, num_branch=0, cluster='vq', ln_para=True, no_second_fc=False,
                 kmeans_iter=100, EMA_flag=True, split=True, kmeans_init=False, dropbranch=0, skip=True,
                 use_gcn=False, commitment_cost=0.5, grad_scale=(1,1), act='relu', weight_ahead=False,
                 bn_flag=False, warm_up_flag=False, momentum=0.1, conv_type='GCN', transformer_flag=False,
                 alpha_dropout_flag=False, val_num_N=0, test_num_N=0):
        super(LowRankGNN, self).__init__()

        self.num_layers = num_layers
        self.skip = skip
        self.dropout = dropout
        self.bn_flag = bn_flag
        self.alpha_dropout_flag = alpha_dropout_flag

        if self.alpha_dropout_flag :
            self.alpha_dropout = torch.nn.AlphaDropout(p=self.dropout)

        self.convs, self.batch_norms = torch.nn.ModuleList(), torch.nn.ModuleList()
        self.convs.append(
            LowRankGNNLayer(in_channels, hidden_channels, dropout,
                            num_M, num_D, num_N, num_branch=num_branch,
                            cluster=cluster, ln_para=ln_para, no_second_fc=no_second_fc,
                            kmeans_iter=kmeans_iter, EMA_flag=EMA_flag, split=split,
                            kmeans_init=kmeans_init, dropbranch=0, skip=skip,
                            use_gcn=use_gcn, commitment_cost=commitment_cost, grad_normalize_scale=grad_scale,
                            hook=True, weight_ahead=weight_ahead, warm_up_flag=warm_up_flag, momentum=momentum,
                            conv_type=conv_type, transformer_flag=transformer_flag, val_num_N=val_num_N, test_num_N=test_num_N))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels, affine=False))

        for _ in range(num_layers - 2):
            self.convs.append(
                LowRankGNNLayer(hidden_channels, hidden_channels, dropout,
                                num_M, num_D, num_N, num_branch=num_branch,
                                cluster=cluster, ln_para=ln_para, no_second_fc=no_second_fc,
                                kmeans_iter=kmeans_iter, EMA_flag=EMA_flag, split=split,
                                kmeans_init=kmeans_init, dropbranch=dropbranch, skip=skip,
                                use_gcn=use_gcn, commitment_cost=commitment_cost, grad_normalize_scale=grad_scale,
                                hook=True, weight_ahead=weight_ahead, warm_up_flag=warm_up_flag, momentum=momentum,
                                conv_type=conv_type, transformer_flag=transformer_flag, val_num_N=val_num_N, test_num_N=test_num_N))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels, affine=False))

        self.convs.append(
            LowRankGNNLayer(hidden_channels, out_channels, dropout,
                            num_M, num_D, num_N, num_branch=num_branch,
                            cluster=cluster, ln_para=ln_para, no_second_fc=no_second_fc,
                            kmeans_iter=kmeans_iter, EMA_flag=EMA_flag, split=split,
                            kmeans_init=kmeans_init, dropbranch=dropbranch, skip=skip,
                            use_gcn=use_gcn, commitment_cost=commitment_cost, grad_normalize_scale=grad_scale,
                            hook=True, weight_ahead=weight_ahead, warm_up_flag=warm_up_flag, momentum=momentum,
                            conv_type=conv_type, transformer_flag=transformer_flag, val_num_N=val_num_N, test_num_N=test_num_N))


        self.transform = torch.nn.Linear(out_channels, out_channels)

        self.ln = torch.nn.LayerNorm(hidden_channels, elementwise_affine=False)

        if act == 'relu' :
            self.act_f = F.relu
        elif act == 'elu' :
            self.act_f = F.elu
        elif act == 'leaky_gelu' :
            self.act_f = lambda x : 0.1*x + 0.9*F.gelu(x)
        else :
            raise ValueError('Activation not supported!')

        # self.dropout = dropout

    # def reset_parameters(self):
    #     for conv in self.convs:
    #         conv.reset_parameters()
    #     for bn in self.bns:
    #         bn.reset_parameters()

    def forward(self, batch, warm_up_rate=1, unlabeled=False, infer=None):
        losses_full, info_backwards_full = 0, 0
        errors_full, X_B_norms_full, quantized_norms_full = [], [], []

        x, batch_A = batch
        for i, conv in enumerate(self.convs[:-1]):

            x, errors, X_B_norms, quantized_norms, losses, info_backwards, _ = \
                conv(x, batch_A, warm_up_rate, unlabeled, infer)

            # x = self.ln(x)
            if self.bn_flag :
                x = self.batch_norms[i](x)
            x = self.act_f(x)

            if self.alpha_dropout_flag :
                x = self.alpha_dropout(x)
            else :
                x = F.dropout(x, p=self.dropout, training=self.training)
            # if self.skip :
            #     x = torch.nn.LayerNorm(x.shape[1], elementwise_affine=False)(x)
            losses_full += losses
            info_backwards_full += info_backwards

            errors_full.append(errors)
            X_B_norms_full.append(X_B_norms)
            quantized_norms_full.append(quantized_norms)


        x, errors, X_B_norms, quantized_norms, losses, info_backwards, _ = \
            self.convs[-1](x, batch_A, warm_up_rate, unlabeled, infer)
        losses_full += losses
        info_backwards_full += info_backwards

        errors_full.append(errors)
        X_B_norms_full.append(X_B_norms)
        quantized_norms_full.append(quantized_norms)

        self.errors, self.X_B_norms, self.quantized_norms = errors_full, X_B_norms_full, quantized_norms_full

        return x, losses_full, info_backwards_full


    def inference(self, x, A):

        for i, conv in enumerate(self.convs[:-1]):

            if self.skip :
                x = conv.gnn_transform(conv.gnn_block[0].conv(x, A)) + conv.linear_skip(x)
            else :
                x = conv.gnn_transform(conv.gnn_block[0].conv(x, A))
            if self.bn_flag :
                x = self.batch_norms[i](x)
            x = self.act_f(x)

        conv = self.convs[-1]
        if self.skip:
            x = conv.gnn_transform(conv.gnn_block[0].conv(x, A)) + conv.linear_skip(x)
        else:
            x = conv.gnn_transform(conv.gnn_block[0].conv(x, A))

        return x

    def init(self, batch, layer_idx):
        x, batch_A = batch
        for i, conv in enumerate(self.convs[:layer_idx]):
            x, _,_,_,_,_,_ = conv(x, batch_A, 1, False)
            x = self.act_f(x)

    # def grad_init(self, X_B, A, batch_indices, layer_idx, loss_func, deg=None, unlabeled=False):
    #     x = X_B
    #
    #     x_for_grad = None
    #     info_backward_for_grad = 0
    #
    #     for i, conv in enumerate(self.convs[:-1]):
    #         x, _, _, _, losses, info_backwards, hooked = conv(x, A, batch_indices, 1, deg, unlabeled)
    #         x = self.act_f(x)
    #
    #         if layer_idx == i+1 :
    #             x_for_grad = hooked
    #         if layer_idx <= i+1 :
    #             info_backward_for_grad += info_backwards
    #
    #
    #     x,  _, _, _, losses, info_backwards, hooked = self.convs[-1](x, A, batch_indices, 1, deg, unlabeled)
    #     if layer_idx == len(self.convs) :
    #         x_for_grad = hooked
    #     if layer_idx <= len(self.convs):
    #         info_backward_for_grad += info_backwards
    #
    #     loss = loss_func(x)
    #     for x in x_for_grad :
    #         torch.autograd.grad(outputs=loss+info_backward_for_grad, inputs=x, retain_graph=True, allow_unused=True)


class LowRankGNN1Layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout,
                 num_M, num_D, num_N, num_branch=0, cluster='vq', ln_para=True, no_second_fc=False,
                 kmeans_iter=100, EMA_flag=True, split=True, kmeans_init=False, dropbranch=0):
        super(LowRankGNN1Layer, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(LowRankGNNLayer(in_channels, out_channels, dropout,
                                          num_M, num_D, num_N, num_branch=num_branch, cluster=cluster,
                                          ln_para=ln_para, no_second_fc=no_second_fc, kmeans_iter=kmeans_iter,
                                            EMA_flag=EMA_flag, split=split, kmeans_init=kmeans_init, dropbranch=dropbranch))


    def forward(self, X_B, A, batch_indices):
        x = X_B
        x, errors, X_B_norms, quantized_norms, losses = self.convs[-1](x, A, batch_indices)
        self.errors = errors
        self.X_B_norms = X_B_norms
        self.quantized_norms = quantized_norms

        return x, losses



if __name__ == "__main__" :
    num_B = 1024
    num_D = 128
    num_M = 16


    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor(),
                                     root='/cmlscratch/kong/datasets/ogb')
    data = dataset[0]
    adj_t = data.adj_t.to_symmetric()

    model = LowRankGNN(num_D,num_D,None,None,None, num_M, num_D, data.x.shape[0])

    batch_indices = torch.arange(num_B)
    X_B = data.x[batch_indices]
    X_bar = torch.randn((num_M, num_D))

    model(X_B, adj_t, batch_indices)
