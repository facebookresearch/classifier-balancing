"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import json
from tqdm import tqdm

root = '/datasets01_101/imagenet_full_size/061417'
split2txt = {
    'train': 'ImageNet_train.txt',
    'val': 'ImageNet_val.txt',
    # 'test': 'ImageNet_test.txt',
}

def convert(split, txt_file):
    clsnames = os.listdir(os.path.join(root, split))
    clsnames.sort()

    lines = []
    for i, name in enumerate(clsnames):
        imgs = os.listdir(os.path.join(root, split, name))
        imgs.sort()
        for img in imgs:
            lines.append(os.path.join(split, name, img) + ' ' + str(i) + '\n')
    
    with open(txt_file, 'w') as f:
        f.writelines(lines)

for k, v in split2txt.items():
    print('===> Converting {} to {}'.format(k, v))
    convert(k, v)
