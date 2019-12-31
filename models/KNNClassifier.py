"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import torch
import torch.nn as nn
import numpy as np
import pickle
from os import path

class KNNClassifier(nn.Module):
    def __init__(self, feat_dim=512, num_classes=1000, feat_type='cl2n', dist_type='l2'):
        super(KNNClassifier, self).__init__()
        assert feat_type in ['un', 'l2n', 'cl2n'], "feat_type is wrong!!!"
        assert dist_type in ['l2', 'cos'], "dist_type is wrong!!!"
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.centroids = torch.randn(num_classes, feat_dim)
        self.feat_mean = torch.randn(feat_dim)
        self.feat_type = feat_type
        self.dist_type = dist_type
        self.initialized = False
    
    def update(self, cfeats):
        mean = cfeats['mean']
        centroids = cfeats['{}cs'.format(self.feat_type)]

        mean = torch.from_numpy(mean)
        centroids = torch.from_numpy(centroids)
        self.feat_mean.copy_(mean)
        self.centroids.copy_(centroids)
        if torch.cuda.is_available():
            self.feat_mean = self.feat_mean.cuda()
            self.centroids = self.centroids.cuda()
        self.initialized = True

    def forward(self, inputs, *args):
        centroids = self.centroids
        feat_mean = self.feat_mean

        # Feature transforms
        if self.feat_type == 'cl2n':
            inputs = inputs - feat_mean
            #centroids = centroids - self.feat_mean

        if self.feat_type in ['l2n', 'cl2n']:
            norm_x = torch.norm(inputs, 2, 1, keepdim=True)
            inputs = inputs / norm_x

            #norm_c = torch.norm(centroids, 2, 1, keepdim=True)
            #centroids = centroids / norm_c
        
        # Logit calculation
        if self.dist_type == 'l2':
            logit = self.l2_similarity(inputs, centroids)
        elif self.dist_type == 'cos':
            logit = self.cos_similarity(inputs, centroids)
        
        return logit, None

    def l2_similarity(self, A, B):
        # input A: [bs, fd] (batch_size x feat_dim)
        # input B: [nC, fd] (num_classes x feat_dim)
        feat_dim = A.size(1)

        AB = torch.mm(A, B.t())
        AA = (A**2).sum(dim=1, keepdim=True)
        BB = (B**2).sum(dim=1, keepdim=True)
        dist = AA + BB.t() - 2*AB

        return -dist
    
    def cos_similarity(self, A, B):
        feat_dim = A.size(1)
        AB = torch.mm(A, B.t())
        AB = AB / feat_dim
        return AB


def create_model(feat_dim, num_classes=1000, feat_type='cl2n', dist_type='l2',
                 log_dir=None, test=False, *args):
    print('Loading KNN Classifier')
    print(feat_dim, num_classes, feat_type, dist_type, log_dir, test)
    clf = KNNClassifier(feat_dim, num_classes, feat_type, dist_type)

    if log_dir is not None:
        fname = path.join(log_dir, 'cfeats.pkl')
        if path.exists(fname):
            print('===> Loading features from %s' % fname)
            with open(fname, 'rb') as f:
                data = pickle.load(f)
            clf.update(data)
    else:
        print('Random initialized classifier weights.')
    
    return clf


if __name__ == "__main__":
    cens = np.eye(4)
    mean = np.ones(4)
    xs = np.array([
        [0.9, 0.1, 0.0, 0.0],
        [0.2, 0.1, 0.1, 0.6],
        [0.3, 0.3, 0.4, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25]
    ])
    xs = torch.Tensor(xs)

    classifier = KNNClassifier(feat_dim=4, num_classes=4, 
                               feat_type='un')
    classifier.update(mean, cens)
    import pdb; pdb.set_trace()
    logits, _ = classifier(xs)
