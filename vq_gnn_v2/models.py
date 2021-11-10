import torch
import torch.nn.functional as F
from vq import VectorQuantizer, VectorQuantizerEMA
from convs import OurGCNConv, OurGATConv, OurSAGEConv, Transformer

import pdb
from sklearn.cluster import KMeans, MiniBatchKMeans
import math


class LowRankGNNBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_M, num_D, num_N, num_branch, cluster, kmeans_iter,
                 EMA_flag, kmeans_init, use_gcn, commitment_cost, grad_normalize_scale, hook_flag, warm_up_flag,
                 momentum, conv_type, transformer_flag):
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

        add_flag = False
        self.vq = VectorQuantizerEMA(self.num_M, self.num_D, commitment_cost=self.commitment_cost,
                                     grad_normalize_scale=grad_normalize_scale, warm_up_flag=warm_up_flag,
                                     momentum=momentum, add_flag=add_flag)

        self.kmeans_init = kmeans_init
        self.grad_kmeans_init = kmeans_init
        self.inited = False

    def hook(self, grad):
        grad_use = grad
        vq_steps = 1

        for _ in range(vq_steps) :
            encoding_indices, encodings = self.vq.update(self.X_B, grad_use)

        self.c_indices[self.batch_indices] = encoding_indices.squeeze().to(torch.short)

        # logging
        # grad_diff = grad_use - self.vq.get_grad()[encoding_indices.squeeze()]
        # self.grad_error_after = torch.norm(grad_diff, dim=1).mean().item()
        # self.grad_norm = torch.norm(grad_use, dim=1).mean().item()

        X_B_diff = self.X_B - self.vq.get_codebook()[encoding_indices.squeeze()]
        self.vq_backward_error = torch.norm(X_B_diff, dim=1).mean().item()

        return grad

    # def M_hook(self, grad):
    #     self.M_grad_norm = torch.norm(grad, dim=1).mean().item()

    def init(self, X_B, batch_indices):
        encoding_indices = self.vq.feature_update(X_B)
        self.c_indices[batch_indices] = encoding_indices.squeeze().to(torch.short)


class LowRankGNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout,
                 num_M, num_D, num_N, num_branch, cluster, ln_para, no_second_fc,
                 kmeans_iter, EMA_flag, split, kmeans_init, dropbranch, skip, use_gcn, commitment_cost,
                 grad_normalize_scale, hook, weight_ahead, warm_up_flag, momentum, conv_type, transformer_flag) :
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
        self.num_M = num_M

        if self.conv_type != 'GAT':
            self.conv = OurGCNConv(in_channels, in_channels, normalize=False)
        else:
            # self.conv = OurGATConv(num_D + 1, num_D + 1, bias=False, add_self_loops=False)
            self.conv = OurGATConv(in_channels+1, in_channels+1, bias=False, add_self_loops=False)

        self.linear_k, self.linear_v, self.gnn_block, self.transformer_block = \
            torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList()
        for i in range(self.num_branch):
            # if not self.split :
            #     self.linear_k.append(torch.nn.Linear(in_channels, num_D))

            if transformer_flag :
                self.transformer_block.append(LowRankGNNBlock(None, None, num_M, num_D, num_N, self.num_branch,
                                                      cluster, kmeans_iter, EMA_flag, kmeans_init,
                                                      False, commitment_cost, grad_normalize_scale,
                                                      hook, warm_up_flag, momentum, conv_type, True))

            if no_second_fc :
                self.gnn_block.append(LowRankGNNBlock(None, None, num_M, num_D, num_N, self.num_branch,
                                                      cluster, kmeans_iter, EMA_flag, kmeans_init,
                                                      use_gcn, commitment_cost, grad_normalize_scale,
                                                      hook, warm_up_flag, momentum, conv_type, False))
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

    def forward(self, x, batch_A, warm_up_rate, unlabeled):

        errors, X_B_norms, quantized_norms = [], [], []
        losses, info_backwards = 0, 0
        hookeds = []

        if self.training and self.dropbranch > 0 :
            num_branch = int(self.num_branch*(1-self.dropbranch))
            branch_idx = torch.randperm(self.num_branch)[:num_branch]
        else :
            num_branch = self.num_branch
            branch_idx = torch.arange(num_branch)

        batch_idx, subset, adj = batch_A
        first_order_idx = subset[len(batch_idx):]

        x_first_order, grad_first_order = [], []
        for i in branch_idx :
            self.gnn_block[i].X_B = x[:,self.num_D*i:self.num_D*(i+1)].detach()
            self.gnn_block[i].batch_indices = batch_idx

            if not self.gnn_block[i].inited or unlabeled :
                self.gnn_block[i].init(x[:,self.num_D*i:self.num_D*(i+1)], batch_idx)

            first_order_c_idx = self.gnn_block[i].c_indices[first_order_idx].to(torch.long)
            codebook = self.gnn_block[i].vq.get()[first_order_c_idx]
            x_first_order.append(codebook[:, :self.num_D])
            grad_first_order.append(codebook[:, self.num_D:])

        x_first_order, grad_first_order = torch.cat(x_first_order, dim=1), torch.cat(grad_first_order, dim=1)
        x_input = torch.cat([x, x_first_order])

        if self.conv_type == 'GAT' :   # X_input cat tensor.ones
            x_input = torch.cat([x_input, torch.ones(x_input.shape[0],1).to(x_input.device)], dim=1)

        x_output = self.conv(x_input, adj)

        for i in branch_idx : # add VQ hook
            x_output_B = x_output[:x.shape[0], i*self.num_D:(i+1)*self.num_D]
            if self.gnn_block[i].inited and self.training and not unlabeled:
                x_output_B.requires_grad_()
                x_output_B.register_hook(self.gnn_block[i].hook)

        if self.conv_type == 'GAT':# use the added one dim to normalize X_out_B
            x_output[:x.shape[0], :-1] /= (x_output[:x.shape[0], -1].unsqueeze(1) + 1e-16)
            x_output = x_output[:, :-1]

        for _ in branch_idx :
            if self.training :
                losses += 0
            errors.append(0)
            X_B_norms.append(0)
            quantized_norms.append(0)

        info_backward = torch.sum(x_output[x.shape[0]:] * grad_first_order * warm_up_rate)
        if self.training:
            info_backwards += info_backward

        x_output = self.gnn_transform(x_output[:x.shape[0]])
        if self.conv_type == 'SAGE' :
            x_output = x_output + self.fc_sage(x)

        # if self.transformer_flag :
        #     x_hiddens = []
        #     for i in branch_idx:
        #         x_hidden = x_input[:, self.num_D * i:self.num_D * (i + 1)]
        #         x_hidden, error, X_B_norm, quantized_norm, loss, info_backward, hooked = \
        #             self.transformer_block[i](x_hidden, batch_A, warm_up_rate, unlabeled)
        #
        #         if self.training:
        #             losses += loss
        #             info_backwards += info_backward
        #
        #         x_hiddens.append(x_hidden)
        #         hookeds.append(hooked)
        #
        #         # errors.append(error)
        #         # X_B_norms.append(X_B_norm)
        #         # quantized_norms.append(quantized_norm)
        #
        #     x_output_trans = self.transformer_v(torch.cat(x_hiddens, dim=1))
        #     x_output_trans_res = self.transformer_res(x_input)
        #     x_output = x_output + x_output_trans + x_output_trans_res

        if self.skip :
            x_output += self.linear_skip(x)

        return x_output, errors, X_B_norms, quantized_norms, losses, info_backwards, hookeds


class LowRankGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout,
                 num_M, num_D, num_N, num_branch=0, cluster='vq', ln_para=True, no_second_fc=False,
                 kmeans_iter=100, EMA_flag=True, split=True, kmeans_init=False, dropbranch=0, skip=True,
                 use_gcn=False, commitment_cost=0.5, grad_scale=(1,1), act='relu', weight_ahead=False,
                 bn_flag=False, warm_up_flag=False, momentum=0.1, conv_type='GCN', transformer_flag=False,
                 alpha_dropout_flag=False):
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
                            conv_type=conv_type, transformer_flag=transformer_flag))
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
                                conv_type=conv_type, transformer_flag=transformer_flag))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_channels, affine=False))

        self.convs.append(
            LowRankGNNLayer(hidden_channels, out_channels, dropout,
                            num_M, num_D, num_N, num_branch=num_branch,
                            cluster=cluster, ln_para=ln_para, no_second_fc=no_second_fc,
                            kmeans_iter=kmeans_iter, EMA_flag=EMA_flag, split=split,
                            kmeans_init=kmeans_init, dropbranch=dropbranch, skip=skip,
                            use_gcn=use_gcn, commitment_cost=commitment_cost, grad_normalize_scale=grad_scale,
                            hook=True, weight_ahead=weight_ahead, warm_up_flag=warm_up_flag, momentum=momentum,
                            conv_type=conv_type, transformer_flag=transformer_flag))


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

    def forward(self, batch, warm_up_rate=1, unlabeled=False):
        losses_full, info_backwards_full = 0, 0
        errors_full, X_B_norms_full, quantized_norms_full = [], [], []

        x, batch_A = batch
        for i, conv in enumerate(self.convs[:-1]):

            x, errors, X_B_norms, quantized_norms, losses, info_backwards, _ = \
                conv(x, batch_A, warm_up_rate, unlabeled)

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
            self.convs[-1](x, batch_A, warm_up_rate, unlabeled)
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
