"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pylab
import torch
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils import mic_acc_cal, shot_acc

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='')
parser.add_argument('--type', type=str, default='test')
args = parser.parse_args()
# ----------------------------------------------------------------------------------

root = args.root
train_file = 'trainfeat_all.pkl'
test_file = '{}feat_all.pkl'.format(args.type)

# load data
with open(os.path.join(root, train_file), 'rb') as f:
    trainset = pickle.load(f)
if args.type == 'train':
    testset = trainset
else:
    with open(os.path.join(root, test_file), 'rb') as f:
        testset = pickle.load(f)
testsize = len(testset['feats'])
batch_size = 512

# Calculate centriods
centroids = []
c_labels = []
for i in np.unique(trainset['labels']):
    c_labels.append(i)
    centroids.append(np.mean(trainset['feats'][trainset['labels']==i], axis=0))
centroids = torch.Tensor(np.stack(centroids))
c_labels = np.array(c_labels)

# ----------------------------------------------------------------------------------
    
# load weight
x = torch.load(os.path.join(root, 'final_model_checkpoint.pth'), map_location=torch.device('cpu'))
weights = x['state_dict_best']['classifier']['module.fc.weight'].cpu()
bias = x['state_dict_best']['classifier']['module.fc.bias'].cpu()

def cos_similarity(A, B):
    feat_dim = A.size(1)

    normB = torch.norm(B, 2, 1, keepdim=True)
    B = B / normB
    AB = torch.mm(A, B.t())

    return AB

def linear_classifier(inputs, weights, bias):
    return torch.addmm(bias, inputs, weights.t())

def logits2preds(logits, labels):
    _, nns = logits.max(dim=1)
    preds = np.array([labels[i] for i in nns])
    return preds

def preds2accs(preds, testset, trainset):
    top1_all = mic_acc_cal(preds, testset['labels'])
    many, median, low, cls_accs = shot_acc(preds, testset['labels'], trainset['labels'], acc_per_cls=True)
    top1_all = np.mean(cls_accs)
    print("{:.2f} \t {:.2f} \t {:.2f} \t {:.2f}".format(
        many * 100, median*100, low*100, top1_all*100))

def dotproduct_similarity(A, B):
    feat_dim = A.size(1)
    AB = torch.mm(A, B.t())

    return AB

def forward(weights):
    total_logits = []
    for i in range(testsize // batch_size + 1):
        # if i%10 == 0:
        #     print('{}/{}'.format(i, testsize // batch_size + 1))
        feat = testset['feats'][batch_size*i:min(batch_size*(i+1), testsize)]
        feat = torch.Tensor(feat)

        logits = dotproduct_similarity(feat, weights)
        total_logits.append(logits)

    total_logits = torch.cat(total_logits)
    return total_logits

def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws


for p in np.linspace(0,2,21):
    ws = pnorm(weights, p)
    logits = forward(ws)
    preds = logits2preds(logits, c_labels)
    preds2accs(preds, testset, trainset)
