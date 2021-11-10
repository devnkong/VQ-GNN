# from comet_ml import Experiment

import argparse
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from utils.logger import Logger, AverageValueMeter, exp_log
from utils.dataloader import OurDataLoader
from utils.parser import parse

from models import LowRankGNN, LowRankGNN1Layer
from math import ceil

import time
from datetime import timedelta

from torch.optim.lr_scheduler import StepLR, ExponentialLR
from utils.scheduler import GradualWarmupScheduler
import pdb

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

def prepare(batch, device):
    x, deg_inv, A_BN, A_BB, A_NB_v, batch_idx = batch

    x, deg_inv = x.to(device), deg_inv.to(device)
    row, col, value = A_BN[0].to(device), A_BN[1].to(device), A_BN[2].to(device)
    A_BN = row, col, value

    if A_BB is not None:
        row, col, value = A_BB[0].to(device), A_BB[1].to(device), A_BB[2].to(device)
        A_BB = row, col, value
    A_NB_v = A_NB_v.to(device) if A_NB_v is not None else None

    batch = x, (deg_inv, A_BN, A_BB, A_NB_v, batch_idx)

    return batch

def train(model, data, batch_size, train_bool, optimizer, device, commitment_cost, use_gcn, warm_up_rate,
          ce_only, exp_log_f, exp_flag, conv_type, clip, num_layers, loader, predictor):
    batch_forward_time_meter = AverageValueMeter()
    batch_backward_time_meter = AverageValueMeter()
    loss_cls_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    model.train()

    for i, batches in enumerate(loader) :

        for j, batch in enumerate(batches):

            A_BB = batch[3]
            assert(A_BB is not None)
            src, dst, _ = A_BB

            batch_idx = batch[-1]

            batch = prepare(batch, device)
            optimizer.zero_grad()

            out, vq_losses, info_backward = model(batch, warm_up_rate)

            pos_out = predictor(out[src], out[dst])
            pos_loss = -torch.log(pos_out + 1e-15).mean()
            # Just do some trivial random sampling.
            dst_neg = torch.randint(0, batch_idx.shape[0], src.shape,
                                    dtype=torch.long, device=device)
            neg_out = predictor(out[src], out[dst_neg])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss_pre = pos_loss + neg_loss

            if commitment_cost > 0 :
                loss = loss_pre + info_backward + vq_losses
            else :
                loss = loss_pre + info_backward
            if ce_only :
                loss = loss_pre

            start = time.time()
            loss.backward()
            batch_backward_time_meter.add(time.time()-start)

            if clip is not None :
                for i in range(num_layers):
                    torch.nn.utils.clip_grad_norm_(model.convs[i].gnn_transform.parameters(), clip[0])
                    if conv_type == 'GAT' : # TODO: transformer case
                        torch.nn.utils.clip_grad_norm_(model.convs[i].gnn_block.parameters(), clip[1])

            a_grad_norms, w_grad_norms = [], []
            for i in range(3):
                if conv_type == 'GAT' :
                    gat_conv = model.convs[i].gnn_block[0].conv
                    a_grad_norm = torch.norm(torch.cat([gat_conv.att_l.grad.view(-1), gat_conv.att_r.grad.view(-1)])).item()
                    a_grad_norms.append(a_grad_norm)

                transform = model.convs[i].gnn_transform
                w_grads = []
                for param in transform.parameters():
                    w_grads.append(param.grad.view(-1))
                w_grad_norm = torch.norm(torch.cat(w_grads)).item()
                w_grad_norms.append(w_grad_norm)

            model.a_grad_norms, model.w_grad_norms = a_grad_norms, w_grad_norms

            if exp_flag :
                exp_log_f()

            if j == 0 and len(batches) > 1 :
                pass
            else :
                optimizer.step()

            loss_meter.add(loss.item())
            loss_cls_meter.add(loss_pre.item())
            print(torch.cuda.max_memory_allocated(device=device)/1e+6)

    return loss_meter.value()[0], loss_cls_meter.value()[0], batch_forward_time_meter.value()[0], \
           batch_backward_time_meter.value()[0]


def init(model, device, loader):
    model.train()

    for layer_idx in range(1, model.num_layers+1) :
        print(layer_idx)

        with torch.no_grad() :
            for i, batch in enumerate(loader):
                batch = batch[0]
                batch = prepare(batch, device)
                model.init(batch, layer_idx)

                print(torch.cuda.max_memory_allocated(device=device) / 1e+6)

    for layer in model.convs :
        for gnn_block in layer.gnn_block :
            gnn_block.inited = True

    # for layer_idx in reversed(range(1, model.num_layers+1)) :
    #     print(layer_idx)
    #
    #     rand_idx = torch.randperm(data.num_nodes)
    #     num_batches = ceil(data.num_nodes / batch_size)
    #     for i in range(num_batches) :
    #
    #         if (i+1)*batch_size > data.num_nodes :
    #             batch_idx = rand_idx[i * batch_size:data.num_nodes]
    #         else :
    #             batch_idx = rand_idx[i*batch_size:(i+1)*batch_size]
    #
    #         loss_func = lambda x: F.cross_entropy(x[train_bool[batch_idx]],
    #                                               data.y.squeeze(1)[batch_idx][train_bool[batch_idx]].to(device))
    #
    #         if use_gcn :
    #             if conv_type.startswith('SAGE'):
    #                 A = (data.adj_t[batch_idx].to(device), data.adj_t[:, batch_idx].to(device))
    #             else:
    #                 A = data.adj_t[batch_idx].to(device)
    #             model.grad_init(data.x[batch_idx].to(device), A, batch_idx,
    #                         layer_idx, loss_func, data.deg[batch_idx].to(device))
    #         else :
    #             if conv_type.startswith('degree-GAT') :
    #                 A = (data.adj_t[batch_idx].to(device), data.adj_t[:, batch_idx].to(device))
    #                 model.grad_init(data.x[batch_idx].to(device), A, batch_idx,
    #                                 layer_idx, loss_func, data.deg[batch_idx].to(device))
    #             else :
    #                 model.grad_init(data.x[batch_idx].to(device), data.adj_t[batch_idx].to(device), batch_idx,
    #                             layer_idx, loss_func)

    for layer in model.convs :
        for gnn_block in layer.gnn_block :
            gnn_block.kmeans_init = False
            gnn_block.grad_kmeans_init = False


# citation2
@torch.no_grad()
def test_citation2(model, data, split_idx, evaluator, batch_size, device, use_gcn, conv_type, loader,
                   split_edge, predictor, dataset):
    model.eval()
    predictor.eval()

    outs = []
    for i, batch in enumerate(loader):
        batch = batch[0]
        batch = prepare(batch, device)
        out, _, _ = model(batch)
        outs.append(out)

    outs = torch.cat(outs, dim=0)

    def test_split(split):
        source = split_edge[split]['source_node'].to(device)
        target = split_edge[split]['target_node'].to(device)
        target_neg = split_edge[split]['target_node_neg'].to(device)

        pos_preds = []
        for perm in torch.utils.data.DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(outs[src], outs[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in torch.utils.data.DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(outs[src], outs[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr

@torch.no_grad()
def test_collab(model, data, split_idx, evaluator, batch_size, device, use_gcn, conv_type, loader,
                split_edge, predictor, dataset):
    model.eval()
    predictor.eval()

    outs = []
    for i, batch in enumerate(loader):
        batch = batch[0]
        batch = prepare(batch, device)
        out, _, _ = model(batch)
        outs.append(out)

    h = outs = torch.cat(outs, dim=0)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in torch.utils.data.DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in torch.utils.data.DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in torch.utils.data.DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    # h = model(data.x, data.full_adj_t)

    pos_test_preds = []
    for perm in torch.utils.data.DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in torch.utils.data.DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    if dataset == 'collab' :
        evaluator.K = K = 50
    elif dataset == 'ppa' :
        evaluator.K = K = 100
    else :
        raise ValueError('not work')
    train_hits = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_valid_pred,
    })[f'hits@{K}']
    valid_hits = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })[f'hits@{K}']
    test_hits = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })[f'hits@{K}']


    return train_hits, valid_hits, test_hits



def main():

    args = parse()
    # if args.exp :
    #     experiment = Experiment(
    #         api_key="",
    #         project_name="",
    #         workspace="",
    #     )
    #     experiment.set_name(args.exp_name)
    #     experiment.log_code(folder='.')
    #     experiment.add_tag(args.exp_tag)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.dataset == 'citation2' :
        dataset = PygLinkPropPredDataset(name=f'ogbl-{args.dataset}',
                                         transform=T.ToSparseTensor(),
                                         root='/cmlscratch/kong/datasets/ogb')
        data = dataset[0]
    elif args.dataset == 'collab' :
        dataset = PygLinkPropPredDataset(name=f'ogbl-{args.dataset}',
                                         root='/cmlscratch/kong/datasets/ogb')
        data = dataset[0]
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        data = T.ToSparseTensor()(data)
    else :
        raise ValueError('Dataset not supported!')

    if args.test_batch_size <= 0:
        args.test_batch_size = data.x.shape[0]

    if args.dataset == 'citation2' :
        data.adj_t = data.adj_t.to_symmetric()

    if args.conv_type == 'GCN':
        deg = data.adj_t.sum(dim=1).to(torch.float) + 1
        data.deg = deg
        deg_inv_sqrt = deg.pow(-1 / 2)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv = deg.pow(-1)
        data.deg_inv = deg_inv
        data.adj_t = deg_inv_sqrt.view(-1, 1) * data.adj_t * deg_inv_sqrt.view(1, -1)
    elif args.conv_type == 'SAGE':
        deg = data.adj_t.sum(dim=1).to(torch.float)
        data.deg = deg
        deg_inv = deg.pow(-1)
        data.deg_inv = deg_inv
        deg_inv[deg_inv == float('inf')] = 0
        data.adj_t = deg_inv.view(-1, 1) * data.adj_t
    elif args.conv_type == 'GAT':
        deg = data.adj_t.sum(dim=1).to(torch.float) + 1
        data.deg = deg
        deg_inv = deg.pow(-1)
        data.deg_inv = deg_inv
        deg_inv[deg_inv == float('inf')] = 0
        data.adj_t = deg_inv.view(-1, 1) * data.adj_t
    else:
        raise ValueError('GNN conv type not supported')

    train_loader = OurDataLoader(data, gnn_type=args.conv_type, sampler_type=args.sampler_type,
                                 walk_length=args.walk_length, recovery_flag=args.recovery_flag,
                                 batch_size=args.batch_size, cont_sliding_window=args.cont_sliding_window,
                                 shuffle=True, num_workers=args.num_workers)

    test_loader = OurDataLoader(data, gnn_type=args.conv_type, sampler_type='node',train_flag = False,                                          batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.num_workers)

    evaluator = Evaluator(name=f'ogbl-{args.dataset}')
    logger = Logger(args.runs, args)

    num_N = data.x.shape[0]
    if args.split :
        if data.num_features % args.num_D != 0 or args.hidden_channels % args.num_D != 0 :
            raise ValueError('Cannot fully split original features')
        # args.num_branch = int(data.num_features/args.num_D)

    if args.batch_size <= 0:
        args.batch_size = data.num_nodes

    if args.num_layers == 1 :
        model = LowRankGNN1Layer(data.num_features, args.hidden_channels, dataset.num_classes,
                                 args.num_layers, args.dropout, args.num_M, args.num_D, num_N,
                                 args.num_branch, args.cluster, args.ln_para, args.no_second_fc,
                                 args.kmeans_iter, args.EMA, args.split, args.kmeans_init,
                                 args.dropbranch, args.skip, args.use_gcn, args.commitment_cost).to(device)

    else :
        model = LowRankGNN(data.num_features, args.hidden_channels, args.hidden_channels,
                           args.num_layers, args.dropout, args.num_M, args.num_D, num_N,
                           args.num_branch, args.cluster, args.ln_para, args.no_second_fc,
                           args.kmeans_iter, args.EMA, args.split, args.kmeans_init,
                           args.dropbranch, args.skip, args.use_gcn, args.commitment_cost,
                           args.grad_scale, args.act, args.weight_ahead, args.bn_flag,
                           args.warm_up, args.momentum, args.conv_type, args.transformer_flag).to(device)
        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                  args.num_layers, args.dropout).to(device)


    split_edge = dataset.get_edge_split()
    if args.dataset == 'citation2':
        # We randomly pick some training samples that we want to evaluate on:
        torch.manual_seed(12345)
        idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
        split_edge['eval_train'] = {
            'source_node': split_edge['train']['source_node'][idx],
            'target_node': split_edge['train']['target_node'][idx],
            'target_node_neg': split_edge['valid']['target_node_neg'],
        }


    for run in range(args.runs):
        # exp_log_f = lambda x=None : exp_log(experiment, model, args.num_layers, args.num_D, args.num_M,
        #                                     args.use_gcn, args.conv_type)

        # model.reset_parameters()
        init(model, device, test_loader)
        print('init done')

        # start = time.time()
        # result = test(model, data, split_idx, evaluator, args.test_batch_size, device, args.use_gcn,
        #               args.conv_type, test_loader)
        # elapsed_inference = str(timedelta(seconds=time.time() - start))
        # print(f'inference time: {elapsed_inference}')

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer = torch.optim.RMSprop(list(model.parameters())+list(predictor.parameters()),
                                        lr=args.lr, alpha=0.99)

        for epoch in range(1, 1 + args.epochs):
            if args.sche :
                for g in optimizer.param_groups:
                    g['lr'] = args.lr * epoch /200 if epoch < 200 else args.lr
            start = time.time()
            if args.warm_up and epoch <= args.warm_up_epochs :
                warm_up_rate = epoch/args.warm_up_epochs
            else :
                warm_up_rate = 1

            loss, loss_cls, batch_forward_time, batch_backward_time = \
                train(model, data, args.batch_size, None, optimizer, device, args.commitment_cost,
                      args.use_gcn, warm_up_rate, args.ce_only, None, args.exp, args.conv_type,
                      args.clip, args.num_layers, train_loader, predictor)

            # loss, loss_cls, batch_forward_time, batch_backward_time = \
            #     separate_train(model, data, args.batch_size, train_bool, optimizer, device, args.commitment_cost,
            #           args.use_gcn, warm_up_rate, args.ce_only, exp_log_f, args.exp, args.conv_type,
            #           args.clip, args.num_layers, train_idx)

            elapsed = str(timedelta(seconds=time.time() - start))
            start = time.time()

            if args.dataset == 'citation2'  :
                test = test_citation2
            elif args.dataset == 'collab' or args.dataset == 'ppa' :
                test = test_collab
            else :
                raise ValueError('dataset not supported')

            result = test(model, data, None, evaluator, args.test_batch_size, device, args.use_gcn,
                          args.conv_type, test_loader, split_edge, predictor, args.dataset)
            elapsed_inference = str(timedelta(seconds=time.time() - start))
            print(f'Epoch time:{elapsed}, inference time:{elapsed_inference}, '
                  f'batch_forward_time:{batch_forward_time:.2f}, '
                  f'batch_backward_time:{batch_backward_time:.2f}')

            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1}, '
                      f'Epoch: {epoch}, '
                      f'Loss: {loss:.4f}, '
                      f'Loss Cls: {loss_cls:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

                # if args.exp:
                #     experiment.log_metric('train_acc', train_acc)
                #     experiment.log_metric('valid_acc', valid_acc)
                #     experiment.log_metric('test_acc', test_acc)
                #     experiment.log_metric('train_loss', loss_cls)

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()