"""Find which COPE checkpoint in a folder belongs to a dataset.

Released checkpoints may have arbitrary file names, so we identify them by content:
  * number of classes of the classifier  -> must equal the number of training ids
    of the dataset (Occluded-Duke: 702, P-DukeMTMC-reID: 665, Market-1501: 751, MSMT17: 1041)
  * length of the ViT positional embedding -> stride 16 (129 tokens) or 12 (211 tokens)

Prints two lines:  <checkpoint path>  and  <stride>
Exit code 1 if nothing (or more than one candidate without a keyword hint) matches.

Usage:
    python tools/find_checkpoint.py --dir weights --num_classes 702 [--keyword duke]
"""
import argparse
import glob
import os.path as osp
import sys

import torch


def inspect(path):
    try:
        sd = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # very old torch without weights_only
        sd = torch.load(path, map_location='cpu')
    except Exception:
        return None
    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']
    if not isinstance(sd, dict):
        return None
    ncls, npos = None, None
    for k, v in sd.items():
        if not hasattr(v, 'shape'):
            continue
        if k.endswith('classifier.weight'):
            ncls = int(v.shape[0])
        if k.endswith('image_encoder.positional_embedding'):
            npos = int(v.shape[0])
    if ncls is None:
        return None
    stride = {129: 16, 211: 12}.get(npos, 16)
    return ncls, stride


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--num_classes', type=int, required=True)
    ap.add_argument('--keyword', default='', help='substring of file name used to break ties')
    a = ap.parse_args()

    files = sorted(glob.glob(osp.join(a.dir, '**', '*.pth'), recursive=True) +
                   glob.glob(osp.join(a.dir, '**', '*.pt'), recursive=True))
    cands = []
    for f in files:
        info = inspect(f)
        if info is None:
            continue
        print('  [scan] {:<70s} classes={:<5d} stride={}'.format(f, info[0], info[1]), file=sys.stderr)
        if info[0] == a.num_classes:
            cands.append((f, info[1]))
    if len(cands) > 1 and a.keyword:
        kw = [c for c in cands if a.keyword.lower() in osp.basename(c[0]).lower()]
        if kw:
            cands = kw
    if len(cands) != 1:
        print('[find_checkpoint] {} candidate(s) with {} classes in {}: {}'.format(
            len(cands), a.num_classes, a.dir, [c[0] for c in cands]), file=sys.stderr)
        sys.exit(1)
    print(cands[0][0])
    print(cands[0][1])
