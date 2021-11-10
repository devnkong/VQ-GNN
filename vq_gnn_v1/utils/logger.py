import torch
import math
import numpy as np


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


class AverageValueMeter(object):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value * n
        if n <= 0:
            raise ValueError("Cannot use a non-positive weight for the running stat.")
        elif self.n == 0:
            self.mean = 0.0 + value  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + n * (value - self.mean_old) / float(self.n + n)
            self.m_s += n * (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n + n - 1.0))
        self.var = self.std ** 2

        self.n += n

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan



def exp_log(experiment, model, num_layers, num_D, num_M, use_gcn, conv_type) :
    for i in range(num_layers):
        experiment.log_metric(f'train_vq_errors_before_l{i}', model.errors[i][0])
        experiment.log_metric(f'train_quantized_norms_l{i}', model.quantized_norms[i][0])
        experiment.log_metric(f'train_vq_errors_rate_before_l{i}', model.errors[i][0] / (model.X_B_norms[i][0] + 1e-24))
        experiment.log_metric(f'train_X_B_norms_l{i}', model.X_B_norms[i][0])

        # experiment.log_metric(f'train_grad_errors_before_l{i}', model.convs[i].gnn_block[0].grad_error_before)
        experiment.log_metric(f'train_grad_errors_after_l{i}', model.convs[i].gnn_block[0].grad_error_after)
        experiment.log_metric(f'train_grad_norms_l{i}', model.convs[i].gnn_block[0].grad_norm)

        # experiment.log_metric(f'train_grad_errors_rate_before_l{i}',
        #                       model.convs[i].gnn_block[0].grad_error_before /
        #                       (model.convs[i].gnn_block[0].grad_norm + 1e-24) )
        experiment.log_metric(f'train_grad_errors_rate_after_l{i}',
                              model.convs[i].gnn_block[0].grad_error_after /
                              (model.convs[i].gnn_block[0].grad_norm + 1e-24) )

        # experiment.log_metric(f'train_vq_get_grad_norms_l{i}', model.convs[i].gnn_block[0].vq_get_grad_norm)
        # experiment.log_metric(f'train_M_grad_norms_l{i}', model.convs[i].gnn_block[0].M_grad_norm)
        # experiment.log_metric(f'train_ln_grad_norms_l{i}', model.convs[i].gnn_block[0].ln_grad_norm)

        # if i >=1 :
        # experiment.log_metric(f'train_X_B_grad_norms_l{i}', model.convs[i].gnn_block[0].X_B_grad_norm)
        # experiment.log_metric(f'train_X_bar_grad_norms_l{i}', model.convs[i].gnn_block[0].X_bar_grad_norm)

        experiment.log_metric(f'train_vq_errors_rate_after_l{i}',
                              model.convs[i].gnn_block[0].vq_backward_error / (model.X_B_norms[i][0] + 1e-24))
        experiment.log_metric(f'train_vq_errors_after_l{i}',
                              model.convs[i].gnn_block[0].vq_backward_error)


        for j in range(num_D) :
            experiment.log_metric(f'train_EMA_feat_mean_l{i}_d{j}',
                                  model.convs[i].gnn_block[0].vq.mean.squeeze()[j])
            experiment.log_metric(f'train_EMA_feat_std_l{i}_d{j}',
                                  model.convs[i].gnn_block[0].vq.std.squeeze()[j])
            experiment.log_metric(f'train_EMA_grad_mean_l{i}_d{j}',
                                  model.convs[i].gnn_block[0].vq.mean.squeeze()[num_D+j])
            experiment.log_metric(f'train_EMA_grad_std_l{i}_d{j}',
                                  model.convs[i].gnn_block[0].vq.std.squeeze()[num_D+j])

            experiment.log_metric(f'train_EMA_feat_mean_running_l{i}_d{j}',
                                  model.convs[i].gnn_block[0].vq.running_mean.squeeze()[j])
            experiment.log_metric(f'train_EMA_feat_std_running_l{i}_d{j}',
                                  model.convs[i].gnn_block[0].vq.running_std.squeeze()[j])
            experiment.log_metric(f'train_EMA_grad_mean_running_l{i}_d{j}',
                                  model.convs[i].gnn_block[0].vq.running_mean.squeeze()[num_D+j])
            experiment.log_metric(f'train_EMA_grad_std_running_l{i}_d{j}',
                                  model.convs[i].gnn_block[0].vq.running_std.squeeze()[num_D+j])

        if conv_type == 'GAT' :
            experiment.log_metric(f'train_EMA_grad_mean_l{i}_extra',
                                  model.convs[i].gnn_block[0].vq.mean.squeeze()[num_D*2])
            experiment.log_metric(f'train_EMA_grad_std_l{i}_extra',
                                  model.convs[i].gnn_block[0].vq.std.squeeze()[num_D*2])
            experiment.log_metric(f'train_EMA_grad_mean_running_l{i}_extra',
                                  model.convs[i].gnn_block[0].vq.running_mean.squeeze()[num_D*2])
            experiment.log_metric(f'train_EMA_grad_std_running_l{i}_extra',
                                  model.convs[i].gnn_block[0].vq.running_std.squeeze()[num_D*2])

        # experiment.log_metric(f'train_EMA_feat_zero_rate_l{i}',
        #                       model.convs[i].gnn_block[0].vq.feat_zero_rate.item())
        # experiment.log_metric(f'train_EMA_grad_zero_rate_l{i}',
        #                       model.convs[i].gnn_block[0].vq.grad_zero_rate.item())

        # labels, info_counts = torch.unique(model.convs[i].gnn_block[0].c_indices[1], sorted=True,
        #                                    return_counts=True)
        # info_counts = torch.cat([info_counts.cpu(), torch.zeros(num_M - labels.shape[0])])
        #
        # info_counts = torch.tensor(info_counts, dtype=torch.float)
        # info_counts_mean = torch.mean(info_counts).item()
        # info_counts_max = torch.max(info_counts).item()
        # info_counts_min = torch.min(info_counts).item()
        # info_counts_median = torch.median(info_counts).item()
        # info_counts_std = torch.std(info_counts).item()
        #
        # experiment.log_metric(f'info_counts_mean_l{i}', info_counts_mean)
        # experiment.log_metric(f'info_used_num_l{i}', labels.shape[0])
        # experiment.log_metric(f'info_counts_max_l{i}', info_counts_max)
        # experiment.log_metric(f'info_counts_min_l{i}', info_counts_min)
        # experiment.log_metric(f'info_counts_median_l{i}', info_counts_median)
        # experiment.log_metric(f'info_counts_std_l{i}', info_counts_std)

        # grad_error_by_cluster = model.convs[i].gnn_block[0].grad_error_by_cluster
        # grad_error_by_cluster_mean = torch.mean(grad_error_by_cluster).item()
        # grad_error_by_cluster_max = torch.max(grad_error_by_cluster).item()
        # grad_error_by_cluster_min = torch.min(grad_error_by_cluster).item()
        # grad_error_by_cluster_median = torch.median(grad_error_by_cluster).item()
        # grad_error_by_cluster_std = torch.std(grad_error_by_cluster).item()
        #
        # experiment.log_metric(f'grad_error_by_cluster_mean_l{i}', grad_error_by_cluster_mean)
        # experiment.log_metric(f'grad_error_by_cluster_max_l{i}', grad_error_by_cluster_max)
        # experiment.log_metric(f'grad_error_by_cluster_min_l{i}', grad_error_by_cluster_min)
        # experiment.log_metric(f'grad_error_by_cluster_median_l{i}', grad_error_by_cluster_median)
        # experiment.log_metric(f'grad_error_by_cluster_std_l{i}', grad_error_by_cluster_std)

        # feat_error_by_cluster = model.convs[i].gnn_block[0].feat_error_by_cluster
        # feat_error_by_cluster_mean = torch.mean(feat_error_by_cluster).item()
        # feat_error_by_cluster_max = torch.max(feat_error_by_cluster).item()
        # feat_error_by_cluster_min = torch.min(feat_error_by_cluster).item()
        # feat_error_by_cluster_median = torch.median(feat_error_by_cluster).item()
        # feat_error_by_cluster_std = torch.std(feat_error_by_cluster).item()
        #
        # experiment.log_metric(f'feat_error_by_cluster_mean_l{i}', feat_error_by_cluster_mean)
        # experiment.log_metric(f'feat_error_by_cluster_max_l{i}', feat_error_by_cluster_max)
        # experiment.log_metric(f'feat_error_by_cluster_min_l{i}', feat_error_by_cluster_min)
        # experiment.log_metric(f'feat_error_by_cluster_median_l{i}', feat_error_by_cluster_median)
        # experiment.log_metric(f'feat_error_by_cluster_std_l{i}', feat_error_by_cluster_std)

        # feat_cen_dists, grad_cen_dists = model.convs[i].gnn_block[0].vq.get_embedding_for_record()
        #
        # feat_cen_dists_mean = torch.mean(feat_cen_dists).item()
        # feat_cen_dists_max = torch.max(feat_cen_dists).item()
        # feat_cen_dists_min = torch.min(feat_cen_dists).item()
        # feat_cen_dists_median = torch.median(feat_cen_dists).item()
        # feat_cen_dists_std = torch.std(feat_cen_dists).item()
        #
        # # centroids distances with each other
        # experiment.log_metric(f'feat_cen_dists_mean_l{i}', feat_cen_dists_mean)
        # experiment.log_metric(f'feat_cen_dists_max_l{i}', feat_cen_dists_max)
        # experiment.log_metric(f'feat_cen_dists_min_l{i}', feat_cen_dists_min)
        # experiment.log_metric(f'feat_cen_dists_median_l{i}', feat_cen_dists_median)
        # experiment.log_metric(f'feat_cen_dists_std_l{i}', feat_cen_dists_std)

        # grad_cen_dists_mean = torch.mean(grad_cen_dists).item()
        # grad_cen_dists_max = torch.max(grad_cen_dists).item()
        # grad_cen_dists_min = torch.min(grad_cen_dists).item()
        # grad_cen_dists_median = torch.median(grad_cen_dists).item()
        # grad_cen_dists_std = torch.std(grad_cen_dists).item()

        # centroids distances with each other
        # experiment.log_metric(f'grad_cen_dists_mean_l{i}', grad_cen_dists_mean)
        # experiment.log_metric(f'grad_cen_dists_max_l{i}', grad_cen_dists_max)
        # experiment.log_metric(f'grad_cen_dists_min_l{i}', grad_cen_dists_min)
        # experiment.log_metric(f'grad_cen_dists_median_l{i}', grad_cen_dists_median)
        # experiment.log_metric(f'grad_cen_dists_std_l{i}', grad_cen_dists_std)

        if conv_type == 'GAT' :
            experiment.log_metric(f'a-grad-norm-l{i}', model.a_grad_norms[i])
        experiment.log_metric(f'w-grad-norm-l{i}', model.w_grad_norms[i])

        # experiment.log_metric(f'feat_cen_norm_l{i}', model.convs[i].gnn_block[0].vq.get_feat_cen_norm())
        # experiment.log_metric(f'grad_cen_norm_l{i}', model.convs[i].gnn_block[0].vq.get_grad_cen_norm())