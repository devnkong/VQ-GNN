# from comet_ml import Experiment

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Flickr, Yelp, PPI, Reddit, GNNBenchmarkDataset

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import Batch

from utils.logger import Logger, AverageValueMeter, exp_log
from utils.dataloader import OurDataLoader
from utils.parser import parse

from models_inductive import LowRankGNN, LowRankGNN1Layer
from math import ceil
import time
from datetime import timedelta

from torch.optim.lr_scheduler import StepLR, ExponentialLR
from utils.scheduler import GradualWarmupScheduler

import pdb
import copy
import os

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
          ce_only, exp_log_f, exp_flag, conv_type, clip, num_layers, loader, test_f, experiment):
    batch_forward_time_meter = AverageValueMeter()
    batch_backward_time_meter = AverageValueMeter()
    loss_cls_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    model.train()
    if data.y.dim() > 1:
        if data.y.shape[-1] > 1 :
            y, criterion = data.y, torch.nn.BCEWithLogitsLoss()
        else :
            y, criterion = data.y.squeeze(1), torch.nn.CrossEntropyLoss()
    else :
        y, criterion  = data.y, torch.nn.CrossEntropyLoss()

    for i, batches in enumerate(loader) :

        for j, batch in enumerate(batches) :

            batch_idx = batch[-1]
            # if current batch has no training sample, continue
            if torch.sum(train_bool[batch_idx]).item() <= 0 :
                continue

            batch = prepare(batch, device)
            optimizer.zero_grad()

            start = time.time()
            out, vq_losses, info_backward = model(batch, warm_up_rate)

            batch_forward_time_meter.add(time.time()-start)
            out = out[train_bool[batch_idx]]


            score = compute_micro_f1(out, y[batch_idx][train_bool[batch_idx]].to(device))
            print(f'Batch {i}, train acc:{score}')
            loss_cls = criterion(out, y[batch_idx][train_bool[batch_idx]].to(device))

            if commitment_cost > 0 :
                loss = loss_cls + info_backward + vq_losses
            else :
                loss = loss_cls + info_backward
            if ce_only :
                loss = loss_cls

            # print('Train loss:', loss.item())

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
            loss_cls_meter.add(loss_cls.item())
            # print(torch.cuda.max_memory_allocated(device=device)/1e+6)

            # train_acc, valid_acc, test_acc = test_f()
            # experiment.log_metric('curve_train_acc', train_acc)
            # experiment.log_metric('curve_valid_acc', valid_acc)
            # experiment.log_metric('curve_test_acc', test_acc)

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

def compute_micro_f1(logits, y, mask=None) -> float:
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.

@torch.no_grad()
def test(model, data, device, loader, dataset, split_idx, evaluator):
    model.eval()

    outs = []
    for i, batch in enumerate(loader):
        batch = batch[0]
        batch = prepare(batch, device)
        out, _, _ = model(batch)
        outs.append(out)

    outs = torch.cat(outs, dim=0)

    if dataset in {'arxiv', 'products'} :
        y_pred = outs.argmax(dim=-1, keepdim=True)
        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']].to(device),
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']].to(device),
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']].to(device),
            'y_pred': y_pred[split_idx['test']],
        })['acc']
    else :
        train_acc  = compute_micro_f1(outs, data.y.to(device), data.train_mask)
        valid_acc = compute_micro_f1(outs, data.y.to(device), data.val_mask)
        test_acc = compute_micro_f1(outs, data.y.to(device), data.test_mask)

    return train_acc, valid_acc, test_acc

@torch.no_grad()
def test_inference(model, data, device, loader, infer):
    model.eval()

    outs = []
    for i, batch in enumerate(loader):
        batch = batch[0]
        batch = prepare(batch, device)
        out, _, _ = model(batch, infer=infer)
        outs.append(out)

    outs = torch.cat(outs, dim=0)

    result  = compute_micro_f1(outs, data.y.to(device))
    return result


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

    if args.dataset in {'arxiv', 'products'} :
        dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}',
                                         transform=T.ToSparseTensor(),
                                         root=os.path.join(args.data_root, 'ogb'))
    else :
        def inductive_data(dataset):
            data = Batch.from_data_list(dataset)
            data.batch, data.ptr = None, None
            data['train_mask'] = torch.ones(data.num_nodes, dtype=torch.bool)
            return data

        if args.dataset == 'flickr' :
            dataset = Flickr(root=os.path.join(args.data_root, 'graph', args.dataset),
                             transform=T.ToSparseTensor())
        elif args.dataset == 'yelp' :
            dataset = Yelp(root=os.path.join(args.data_root, 'graph', args.dataset),
                           transform=T.ToSparseTensor())
        elif args.dataset == 'reddit' :
            dataset = Reddit(root=os.path.join(args.data_root, 'graph', args.dataset),
                             transform=T.ToSparseTensor())
        elif args.dataset == 'ppi' :
            print('PPI loaded')
            dataset = PPI(root=os.path.join(args.data_root, 'graph', args.dataset),
                          transform=T.ToSparseTensor(), split='train')
            val_dataset = PPI(root=os.path.join(args.data_root, 'graph', args.dataset),
                          transform=T.ToSparseTensor(), split='val')
            test_dataset = PPI(root=os.path.join(args.data_root, 'graph', args.dataset),
                          transform=T.ToSparseTensor(), split='test')
            data, val_data, test_data = inductive_data(dataset), inductive_data(val_dataset), inductive_data(test_dataset)

        elif args.dataset == 'cluster' :
            print('CLUSTER loaded')
            kwargs = {'root': os.path.join(args.data_root, 'graph', args.dataset), 'name': 'CLUSTER',
                      'transform': T.ToSparseTensor()}
            dataset = GNNBenchmarkDataset(split='train', **kwargs)
            val_dataset = GNNBenchmarkDataset(split='val', **kwargs)
            test_dataset = GNNBenchmarkDataset(split='test', **kwargs)
            data, val_data, test_data = inductive_data(dataset), inductive_data(val_dataset), inductive_data(test_dataset)

            pdb.set_trace()
        else :
            raise ValueError('Dataset not supported!')

    # inference
    # data_infer = copy.deepcopy(data)
    # data_infer.adj_t = data_infer.adj_t.set_diag()
    # deg = data_infer.adj_t.sum(dim=1).to(torch.float)
    # deg_inv_sqrt = deg.pow(-1 / 2)
    # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    # data_infer.adj_t = deg_inv_sqrt.view(-1, 1) * data_infer.adj_t * deg_inv_sqrt.view(1, -1)
    # data_infer = data_infer.to(device)

    def norm_adj(data) :
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

        return data

    if args.dataset not in {'ppi', 'cluster'}:
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        data = norm_adj(data)
    else :
        data.adj_t, val_data.adj_t, test_data.adj_t = data.adj_t.to_symmetric(), val_data.adj_t.to_symmetric(),test_data.adj_t.to_symmetric()
        data, val_data, test_data = norm_adj(data), norm_adj(val_data), norm_adj(test_data)


    if args.batch_size <= 0:
        args.batch_size = data.num_nodes
    if args.test_batch_size <= 0 :
        raise ValueError('No!')

    if args.dataset in {'arxiv', 'products'} :
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name=f'ogbn-{args.dataset}')
        train_bool = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_bool[split_idx['train']] = True
    else :
        train_bool = data.train_mask
        split_idx, evaluator = None, None

    logger = Logger(args.runs, args)

    num_N = data.x.shape[0]
    if args.split :
        if data.num_features % args.num_D != 0 :
            padding_dim = args.num_D - data.num_features % args.num_D
            data.x = torch.cat([data.x, torch.zeros((num_N, padding_dim))], dim=-1)
            if args.dataset in {'ppi', 'cluster'} :
                val_data.x = torch.cat([val_data.x, torch.zeros((val_data.x.shape[0], padding_dim))], dim=-1)
                test_data.x = torch.cat([test_data.x, torch.zeros((test_data.x.shape[0], padding_dim))], dim=-1)
                test_loader_val = OurDataLoader(val_data, gnn_type=args.conv_type, sampler_type='node',
                                                train_flag=False,
                                                batch_size=args.test_batch_size, shuffle=False,
                                                num_workers=args.num_workers)
                test_loader_test = OurDataLoader(test_data, gnn_type=args.conv_type, sampler_type='node',
                                                 train_flag=False,
                                                 batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=args.num_workers)
        if args.hidden_channels % args.num_D != 0 :
            raise ValueError('Cannot fully split hidden features')

    train_loader = OurDataLoader(data, gnn_type=args.conv_type, sampler_type=args.sampler_type,
                                 walk_length=args.walk_length, recovery_flag=args.recovery_flag,
                                 batch_size=args.batch_size, cont_sliding_window=args.cont_sliding_window,
                                 shuffle=True, num_workers=args.num_workers)

    test_loader = OurDataLoader(data, gnn_type=args.conv_type, sampler_type='node',train_flag = False,                                          batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.num_workers)

    if args.num_layers == 1 :
        model = LowRankGNN1Layer(data.num_features, args.hidden_channels, dataset.num_classes,
                                 args.num_layers, args.dropout, args.num_M, args.num_D, num_N,
                                 args.num_branch, args.cluster, args.ln_para, args.no_second_fc,
                                 args.kmeans_iter, args.EMA, args.split, args.kmeans_init,
                                 args.dropbranch, args.skip, args.use_gcn, args.commitment_cost).to(device)

    else :
        model = LowRankGNN(data.num_features, args.hidden_channels, dataset.num_classes,
                           args.num_layers, args.dropout, args.num_M, args.num_D, num_N,
                           args.num_branch, args.cluster, args.ln_para, args.no_second_fc,
                           args.kmeans_iter, args.EMA, args.split, args.kmeans_init,
                           args.dropbranch, args.skip, args.use_gcn, args.commitment_cost,
                           args.grad_scale, args.act, args.weight_ahead, args.bn_flag,
                           args.warm_up, args.momentum, args.conv_type, args.transformer_flag,
                           args.alpha_dropout_flag, val_data.num_nodes, test_data.num_nodes).to(device)


    for run in range(args.runs):
        # exp_log_f = lambda x=None : exp_log(experiment, model, args.num_layers, args.num_D, args.num_M,
        #                                     args.use_gcn, args.conv_type)
        test_f = lambda x=None : test(model, data, device, test_loader, args.dataset, split_idx, evaluator)

        # model.reset_parameters()
        init(model, device, test_loader)
        print('init done')

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99)

        for epoch in range(1, 1 + args.epochs):
            if args.sche :
                for g in optimizer.param_groups:
                    g['lr'] = args.lr * epoch /200 if epoch < 200 else args.lr
            if args.warm_up and epoch <= args.warm_up_epochs :
                warm_up_rate = epoch/args.warm_up_epochs
            else :
                warm_up_rate = 1

            start = time.time()

            loss, loss_cls, batch_forward_time, batch_backward_time = \
                train(model, data, args.batch_size, train_bool, optimizer, device, args.commitment_cost,
                      args.use_gcn, warm_up_rate, args.ce_only, None, args.exp, args.conv_type,
                      args.clip, args.num_layers, train_loader, test_f, None)

            # # loss, loss_cls, batch_forward_time, batch_backward_time = \
            # #     separate_train(model, data, args.batch_size, train_bool, optimizer, device, args.commitment_cost,
            # #           args.use_gcn, warm_up_rate, args.ce_only, exp_log_f, args.exp, args.conv_type,
            # #           args.clip, args.num_layers, train_idx)

            if args.dataset not in {'ppi', 'cluster'} :
                elapsed = str(timedelta(seconds=time.time() - start))
                result = test(model, data, device, test_loader, args.dataset, split_idx, evaluator)

                elapsed_inference = str(timedelta(seconds=time.time() - start))
                print(f'Epoch time:{elapsed}, inference time:{elapsed_inference}, '
                      f'batch_forward_time:{batch_forward_time:.2f}, '
                      f'batch_backward_time:{batch_backward_time:.2f}')
            else :
                train_acc = test_inference(model, data, device, test_loader, None)
                valid_acc = test_inference(model, val_data, device, test_loader_val, 'val')
                test_acc = test_inference(model, test_data, device, test_loader_test, 'test')
                result = train_acc, valid_acc, test_acc
                logger.add_result(run, result)

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
