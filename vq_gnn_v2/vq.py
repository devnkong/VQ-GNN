import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F

import pdb


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.5, holistic_cost=0.1):
        super        (VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost
        self._holistic_cost = holistic_cost

    def kmeans_init(self, kmeans_centroids):
        self._embedding.weight = nn.Parameter(kmeans_centroids)

    def get_codebook(self):
        return self._embedding.weight.data

    def forward(self, inputs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = self._holistic_cost * (q_latent_loss + self._commitment_cost * e_latent_loss)

        quantized = inputs + (quantized - inputs).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, encodings, encoding_indices
        # return loss, quantized.contiguous(), perplexity, encodings, encoding_indices


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.5, decay=0.99, epsilon=1e-24,
                 grad_normalize_scale=(1,1), warm_up_flag=False, momentum=0.1, add_flag=False):
        super(VectorQuantizerEMA, self).__init__()

        self.add_flag = add_flag
        added_dim = 1 if add_flag  else 0

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._warm_up_flag = warm_up_flag

        self.register_buffer('_embedding', torch.randn(self._num_embeddings, self._embedding_dim*2+added_dim))
        self.register_buffer('_embedding_output', torch.zeros(self._num_embeddings, self._embedding_dim*2+added_dim))

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.zeros(self._num_embeddings, self._embedding_dim*2+added_dim))

        if self._warm_up_flag :
            self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        # variance of gradient is very small, eps needs to be small
        self.batch_norm_feat = torch.nn.BatchNorm1d(embedding_dim, affine=False)
        self.batch_norm_grad = torch.nn.BatchNorm1d(embedding_dim+added_dim, eps=self._epsilon, affine=False,
                                                    momentum=momentum)

        self.grad_normalize_scale = grad_normalize_scale
        if type(self.grad_normalize_scale) is not list :
            raise ValueError('grad scale type wrong!')
        self._embedding.data[:, self._embedding_dim:self._embedding_dim*2] *= self.grad_normalize_scale[0]
        self._ema_w.data[:, self._embedding_dim:self._embedding_dim*2] *= self.grad_normalize_scale[0]

        if self.add_flag :
            self._embedding.data[:, self._embedding_dim*2] *= self.grad_normalize_scale[1]
            self._ema_w.data[:, self._embedding_dim*2] *= self.grad_normalize_scale[1]

        self.bn_inited = False

    def feature_kmeans_init(self, kmeans_centroids, kmeans_counts):
        self._embedding.data[:, :self._embedding_dim] = kmeans_centroids
        self._ema_cluster_size.data = kmeans_counts
        self._ema_w.data[:, :self._embedding_dim] = kmeans_centroids * kmeans_counts.unsqueeze(1)


    def kmeans_init(self, kmeans_centroids, kmeans_counts):
        self._embedding.data = kmeans_centroids
        self._ema_cluster_size.data = kmeans_counts
        self._ema_w.data = kmeans_centroids * kmeans_counts.unsqueeze(1)

        self._embedding.data[:, self._embedding_dim:self._embedding_dim*2] *= self.grad_normalize_scale[0]
        self._ema_w.data[:, self._embedding_dim:self._embedding_dim*2] *= self.grad_normalize_scale[0]

        if self.add_flag :
            self._embedding.data[:, self._embedding_dim*2] *= self.grad_normalize_scale[1]
            self._ema_w.data[:, self._embedding_dim*2] *= self.grad_normalize_scale[1]

    def get(self):
        return self._embedding_output

    def get_codebook(self):
        return self._embedding_output[:, :self._embedding_dim]

    def get_grad(self):
        return self._embedding_output[:, self._embedding_dim:]

    def get_feat_cen_norm(self) :
        center = torch.mean(self._embedding[:, :self._embedding_dim], dim=0)
        return torch.norm(center).item()

    def get_grad_cen_norm(self) :
        center = torch.mean(self._embedding[:, self._embedding_dim: ], dim=0)
        return torch.norm(center).item()

    def get_embedding_for_record(self):

        emb = self._embedding[:, :self._embedding_dim]
        feat_distances = (torch.sum(emb ** 2, dim=1, keepdim=True)
                     + torch.sum(emb ** 2, dim=1)
                     - 2 * torch.matmul(emb, emb.t()))
        feat_distances = torch.sqrt(feat_distances.clamp_(min=0))
        triu_idx = feat_distances.triu().nonzero().t()
        feat_distances = feat_distances[triu_idx[0], triu_idx[1]]

        emb = self._embedding[:, self._embedding_dim:]
        grad_distances = (torch.sum(emb ** 2, dim=1, keepdim=True)
                          + torch.sum(emb ** 2, dim=1)
                          - 2 * torch.matmul(emb, emb.t()))
        grad_distances = torch.sqrt(grad_distances.clamp_(min=0))
        triu_idx = grad_distances.triu().nonzero().t()
        grad_distances = grad_distances[triu_idx[0], triu_idx[1]]

        return feat_distances, grad_distances

    def _get_feat_embed(self):
        return self._embedding[:, :self._embedding_dim]

    def feature_update(self, X_B):
        inputs = X_B
        inputs_normalized = self.batch_norm_feat(inputs)
        embedding_normalized = self._get_feat_embed()

        # Calculate distances
        distances = (torch.sum(inputs_normalized ** 2, dim=1, keepdim=True)
                     + torch.sum(embedding_normalized ** 2, dim=1)
                     - 2 * torch.matmul(inputs_normalized, embedding_normalized.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=X_B.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size.data = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)


            # Laplace smoothing of the cluster size
            if self._warm_up_flag :
                n = torch.sum(self._ema_cluster_size.data)
                self._ema_cluster_size.data = (
                        (self._ema_cluster_size + 1e-5)
                        / (n + self._num_embeddings * 1e-5) * n)

            if torch.count_nonzero(self._ema_cluster_size) != self._ema_cluster_size.shape[0] :
                raise ValueError('Bad Init!')

            dw = torch.matmul(encodings.t(), inputs_normalized)

            self._ema_w.data[:, :self._embedding_dim] = self._ema_w[:, :self._embedding_dim] * self._decay \
                                                        + (1 - self._decay) * dw
            self._embedding.data[:, :self._embedding_dim] = self._ema_w[:, :self._embedding_dim] /\
                                                            self._ema_cluster_size.unsqueeze(1)

            running_std = torch.sqrt(self.batch_norm_feat.running_var + 1e-5).unsqueeze(dim=0)
            running_mean = self.batch_norm_feat.running_mean.unsqueeze(dim=0)
            self._embedding_output.data[:, :self._embedding_dim] = self._get_feat_embed()*running_std + running_mean

        return encoding_indices

    def update(self, X_B, grad):

        inputs = torch.cat([X_B, grad], dim=1)

        mean = torch.mean(inputs, dim=0, keepdim=True).detach()
        std = torch.sqrt(torch.var(inputs, dim=0, keepdim=True) + self._epsilon).detach()
        self.mean = mean
        self.std = std

        self.feat_zero_rate = torch.sum(torch.abs(inputs[:, 0]) < std[0][0] * 1e-5)/X_B.shape[0]
        self.grad_zero_rate = torch.sum(inputs[:, self._embedding_dim] < std[0][self._embedding_dim] * 1e-5)/X_B.shape[0]

        if not self.bn_inited :
            self.batch_norm_feat.running_mean.data = torch.mean(X_B, dim=0).detach()
            self.batch_norm_feat.running_var.data = torch.var(X_B, dim=0).detach()
            self.batch_norm_grad.running_mean.data = torch.mean(grad, dim=0).detach()
            self.batch_norm_grad.running_var.data = torch.var(grad, dim=0).detach()
            self.bn_inited = True

        inputs_normalized = torch.cat([self.batch_norm_feat(X_B), self.batch_norm_grad(grad)], dim=1)
        inputs_normalized[:, self._embedding_dim:self._embedding_dim*2] *= self.grad_normalize_scale[0]

        if self.add_flag :
            inputs_normalized[:, self._embedding_dim*2] *= self.grad_normalize_scale[1]

        # Calculate distances
        distances = (torch.sum(inputs_normalized ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding ** 2, dim=1)
                     - 2 * torch.matmul(inputs_normalized, self._embedding.t()))
        # distances = F.cosine_similarity(inputs_normalized, self._embedding)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=X_B.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size.data = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)


            # Laplace smoothing of the cluster size
            if self._warm_up_flag :
                n = torch.sum(self._ema_cluster_size.data)
                self._ema_cluster_size.data = (
                        (self._ema_cluster_size + 1e-5)
                        / (n + self._num_embeddings * 1e-5) * n)

            if torch.count_nonzero(self._ema_cluster_size) != self._ema_cluster_size.shape[0] :
                raise ValueError('Bad Init!')

            dw= torch.matmul(encodings.t(), inputs_normalized)

            self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw
            self._embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

            output_data = self._embedding.data.detach().clone()

            output_data[:, self._embedding_dim:self._embedding_dim*2] /= self.grad_normalize_scale[0] + self._epsilon
            if self.add_flag :
                output_data[:, self._embedding_dim*2] /= self.grad_normalize_scale[1] + self._epsilon

            running_var = torch.cat([self.batch_norm_feat.running_var+1e-5, self.batch_norm_grad.running_var+self._epsilon])
            running_std = torch.sqrt(running_var).unsqueeze(dim=0)

            running_mean = torch.cat([self.batch_norm_feat.running_mean, self.batch_norm_grad.running_mean])
            running_mean = running_mean.unsqueeze(dim=0)
            self._embedding_output.data = output_data*running_std + running_mean

            if self.grad_normalize_scale[0] == 0 :
                self._embedding_output.data[:, self._embedding_dim:] *= 0
            self.running_mean = running_mean
            self.running_std = running_std

        return encoding_indices, encodings