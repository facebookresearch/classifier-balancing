"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch
import torch.nn as nn
from torch.autograd.function import Function

import pdb

class DiscCentroidsLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(DiscCentroidsLoss, self).__init__()
        self.num_classes = num_classes
        self.centroids = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.disccentroidslossfunc = DiscCentroidsLossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        batch_size = feat.size(0)
        
        # calculate attracting loss

        feat = feat.view(batch_size, -1)
        # To check the dim of centroids and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss_attract = self.disccentroidslossfunc(feat, label, self.centroids, batch_size_tensor).squeeze()
        
        # calculate repelling loss

        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centroids, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, feat, self.centroids.t())

        classes = torch.arange(self.num_classes).long().cuda()
        labels_expand = label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expand.eq(classes.expand(batch_size, self.num_classes))

        distmat_neg = distmat
        distmat_neg[mask] = 0.0
        # margin = 50.0
        margin = 10.0
        loss_repel = torch.clamp(margin - distmat_neg.sum() / (batch_size * self.num_classes), 0.0, 1e6)

        # loss = loss_attract + 0.05 * loss_repel
        loss = loss_attract + 0.01 * loss_repel

        return loss


class DiscCentroidsLossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centroids, batch_size):
        ctx.save_for_backward(feature, label, centroids, batch_size)
        centroids_batch = centroids.index_select(0, label.long())
        return (feature - centroids_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centroids, batch_size = ctx.saved_tensors
        centroids_batch = centroids.index_select(0, label.long())
        diff = centroids_batch - feature
        # init every iteration
        counts = centroids.new_ones(centroids.size(0))
        ones = centroids.new_ones(label.size(0))
        grad_centroids = centroids.new_zeros(centroids.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centroids.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centroids = grad_centroids/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centroids / batch_size, None


    
def create_loss (feat_dim=512, num_classes=1000):
    print('Loading Discriminative Centroids Loss.')
    return DiscCentroidsLoss(num_classes, feat_dim)
