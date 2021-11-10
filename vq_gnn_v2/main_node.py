# from comet_ml import Experiment

import torch
from models import LowRankGNN
from dataloader import OurDataLoader
from utils.logger import Logger, AverageValueMeter, exp_log
from utils.parser import parse
from utils.misc import compute_micro_f1, prepare_batch_input, get_data

import time
from datetime import timedelta

# TODO: GAT, Transformer, current version works
# TODO: first layer no vq, maybe not this time
# TODO: non-stochastic test, seems unnecessary

def init(data, model, device, loader):
    model.train()

    for layer_idx in range(1, model.num_layers+1) :
        print(layer_idx)

        with torch.no_grad() :
            for i, batches in enumerate(loader):
                batch = batches[0]
                batch_input, _ = prepare_batch_input(data.x, batch, device)
                model.init(batch_input, layer_idx)
                print(torch.cuda.max_memory_allocated(device=device) / 1e+6)

    for layer in model.convs :
        for gnn_block in layer.gnn_block :
            gnn_block.inited = True

    for layer in model.convs :
        for gnn_block in layer.gnn_block :
            gnn_block.kmeans_init = False
            gnn_block.grad_kmeans_init = False

def train(model, data, batch_size, train_bool, optimizer, device, commitment_cost, use_gcn, warm_up_rate,
          ce_only, exp_log_f, exp_flag, conv_type, clip, num_layers, loader, test_f, experiment):
    batch_forward_time_meter = AverageValueMeter()
    batch_backward_time_meter = AverageValueMeter()
    loss_cls_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    num_B_meter = AverageValueMeter()
    num_B_prime_meter = AverageValueMeter()

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

            optimizer.zero_grad()

            start = time.time()
            batch_input, (num_B, num_B_prime) = prepare_batch_input(data.x, batch, device)
            num_B_meter.add(num_B), num_B_prime_meter.add(num_B_prime)

            out, vq_losses, info_backward = model(batch_input, warm_up_rate)

            batch_forward_time_meter.add(time.time()-start)
            out = out[train_bool[batch_idx]]

            score = compute_micro_f1(out, y[batch_idx][train_bool[batch_idx]].to(device))
            print(f'Batch {i}.{j}, train acc:{score}')

            loss_cls = criterion(out, y[batch_idx][train_bool[batch_idx]].to(device))
            if commitment_cost > 0 :
                loss = loss_cls + info_backward + vq_losses
            else :
                loss = loss_cls + info_backward
            if ce_only :
                loss = loss_cls

            start = time.time()
            loss.backward()
            batch_backward_time_meter.add(time.time()-start)

            # if clip is not None :
            #     for i in range(num_layers):
            #         torch.nn.utils.clip_grad_norm_(model.convs[i].gnn_transform.parameters(), clip[0])
            #         if conv_type == 'GAT' : # TODO: transformer case
            #             torch.nn.utils.clip_grad_norm_(model.convs[i].gnn_block.parameters(), clip[1])
            #
            # a_grad_norms, w_grad_norms = [], []
            # for i in range(num_layers):
            #     if conv_type == 'GAT' :
            #         gat_conv = model.convs[i].gnn_block[0].conv
            #         a_grad_norm = torch.norm(torch.cat([gat_conv.att_l.grad.view(-1), gat_conv.att_r.grad.view(-1)])).item()
            #         a_grad_norms.append(a_grad_norm)
            #
            #     transform = model.convs[i].gnn_transform
            #     w_grads = []
            #     for param in transform.parameters():
            #         w_grads.append(param.grad.view(-1))
            #     w_grad_norm = torch.norm(torch.cat(w_grads)).item()
            #     w_grad_norms.append(w_grad_norm)
            # model.a_grad_norms, model.w_grad_norms = a_grad_norms, w_grad_norms

            # assert j == 0
            if len(batches) > 1 and j == 0 :
                pass
            else :
                optimizer.step()

            loss_meter.add(loss.item())
            loss_cls_meter.add(loss_cls.item())

    return loss_meter.value()[0], loss_cls_meter.value()[0], batch_forward_time_meter.value()[0], \
           batch_backward_time_meter.value()[0], num_B_meter.value()[0], num_B_prime_meter.value()[0]


@torch.no_grad()
def test(model, data, device, loader, evaluator):
    model.eval()

    outs = []
    for i, batches in enumerate(loader):
        batch = batches[0]
        batch_input, _ = prepare_batch_input(data.x, batch, device)
        out, _, _ = model(batch_input)
        outs.append(out)
    outs = torch.cat(outs, dim=0)

    if evaluator is not None :
        y_pred = outs.argmax(dim=-1, keepdim=True)
        train_acc = evaluator.eval({
            'y_true': data.y[data.train_mask].to(device),
            'y_pred': y_pred[data.train_mask],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': data.y[data.val_mask].to(device),
            'y_pred': y_pred[data.val_mask],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': data.y[data.test_mask].to(device),
            'y_pred': y_pred[data.test_mask],
        })['acc']
    else :
        train_acc  = compute_micro_f1(outs, data.y.to(device), data.train_mask)
        valid_acc = compute_micro_f1(outs, data.y.to(device), data.val_mask)
        test_acc = compute_micro_f1(outs, data.y.to(device), data.test_mask)

    return train_acc, valid_acc, test_acc

@torch.no_grad()
def test_inference(model, data, device, loader):
    model.eval()

    outs = []
    for i, batches in enumerate(loader):
        batch = batches[0]
        batch_input, _ = prepare_batch_input(data.x, batch, device)
        out, _, _ = model(batch_input)
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

    data, val_data, test_data, dataset, evaluator, cluster_indices = get_data(args)

    if val_data is not None and test_data is not None :
        assert cluster_indices is None
        test_loader_val = OurDataLoader(val_data, cluster_indices, gnn_type=args.conv_type,
                                        sampler_type='node', train_flag=False,
                                        batch_size=val_data.num_nodes, shuffle=False,
                                        num_workers=args.num_workers)
        test_loader_test = OurDataLoader(test_data, cluster_indices, gnn_type=args.conv_type,
                                        sampler_type='node', train_flag=False,
                                        batch_size=test_data.num_nodes, shuffle=False,
                                        num_workers=args.num_workers)

    train_bool = data.train_mask
    num_N = data.num_nodes

    if args.batch_size <= 0:
        args.batch_size = num_N
    if args.test_batch_size <= 0 :
        args.test_batch_size = num_N

    train_loader = OurDataLoader(data, cluster_indices, gnn_type=args.conv_type, sampler_type=args.sampler_type,
                                 walk_length=args.walk_length, recovery_flag=args.recovery_flag,
                                 batch_size=args.batch_size, cont_sliding_window=args.cont_sliding_window,
                                 shuffle=True, num_workers=args.num_workers)

    if cluster_indices is not None : test_sampler_type = 'cluster'
    else : test_sampler_type = 'node'
    test_loader = OurDataLoader(data, cluster_indices, gnn_type=args.conv_type, sampler_type=test_sampler_type,
                                train_flag=False, batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.num_workers)
    logger = Logger(args.runs, args)


    model = LowRankGNN(data.num_features, args.hidden_channels, dataset.num_classes,
                       args.num_layers, args.dropout, args.num_M, args.num_D, num_N,
                       args.num_branch, args.cluster, args.ln_para, args.no_second_fc,
                       args.kmeans_iter, args.EMA, args.split, args.kmeans_init,
                       args.dropbranch, args.skip, args.use_gcn, args.commitment_cost,
                       args.grad_scale, args.act, args.weight_ahead, args.bn_flag,
                       args.warm_up, args.momentum, args.conv_type, args.transformer_flag,
                       args.alpha_dropout_flag).to(device)


    for run in range(args.runs):
        # exp_log_f = lambda x=None : exp_log(experiment, model, args.num_layers, args.num_D, args.num_M,
        #                                     args.use_gcn, args.conv_type)
        # test_f = lambda x=None : test(model, data, device, test_loader, evaluator)

        # model.reset_parameters()
        init(data, model, device, test_loader)
        print('init done')

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99)
        total_train_time, total_inference_time = 0, 0

        x, y = [], []
        for epoch in range(1, 1 + args.epochs):
            if args.sche :
                for g in optimizer.param_groups:
                    g['lr'] = args.lr * epoch /200 if epoch < 200 else args.lr
            if args.warm_up and epoch <= args.warm_up_epochs :
                warm_up_rate = epoch/args.warm_up_epochs
            else :
                warm_up_rate = 1

            start = time.time()
            loss, loss_cls, batch_forward_time, batch_backward_time, num_B, num_B_prime = \
                train(model, data, args.batch_size, train_bool, optimizer, device, args.commitment_cost,
                      args.use_gcn, warm_up_rate, args.ce_only, None, False, args.conv_type,
                      args.clip, args.num_layers, train_loader, None, None)

            mem = torch.cuda.max_memory_allocated(device=device) / 1e+6

            if args.dataset not in {'ppi', 'cluster'} :
                elapsed_train = time.time() - start
                total_train_time += elapsed_train
                result = test(model, data, device, test_loader, evaluator)
                elapsed_inference = time.time() - start - elapsed_train
                total_inference_time += elapsed_inference

                print(f'avg train:{total_train_time/epoch:.2f}, '
                      f'avg inference:{total_inference_time/epoch:.2f}, '
                      f'Epoch train time:{str(timedelta(seconds=elapsed_train))}, '
                      f'inference time:{str(timedelta(seconds=elapsed_inference))}, ')
            else :
                train_acc = test_inference(model, data, device, test_loader)
                valid_acc = test_inference(model, val_data, device, test_loader_val)
                test_acc = test_inference(model, test_data, device, test_loader_test)
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
                #     experiment.log_metric('train_acc', train_acc, epoch)
                #     experiment.log_metric('valid_acc', valid_acc, epoch)
                #     experiment.log_metric('test_acc', test_acc, epoch)
                #     experiment.log_metric('train_loss', loss_cls, epoch)
                #     experiment.log_metric('num_B', num_B, epoch)
                #     experiment.log_metric('num_B\'', num_B_prime, epoch)
                #     experiment.log_metric('B/B\'', num_B/num_B_prime, epoch)
                #     experiment.log_metric('mem', mem, epoch)

                #     x.append(total_train_time)
                #     y.append(valid_acc)

        logger.print_statistics(run)
    logger.print_statistics()

if __name__ == "__main__":
    main()
