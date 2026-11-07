"""Sanity-check a dataset folder BEFORE launching a long training run.

It instantiates the same dataset class that train_cope.py uses, prints the
statistics, and verifies that every training image has its human-parsing mask
(.npy) and that the masks can be loaded. Test-set masks are NOT needed by COPE.

Usage (from the repo root):
    python tools/check_dataset.py --config_file configs/OCC_Duke/cope.yml
    python tools/check_dataset.py --config_file configs/OCC_Duke/cope.yml DATASETS.ROOT_DIR /data/reid
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from config import cfg  # noqa: E402
from cope.dataloader import FACTORY  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--n_load', type=int, default=20, help='number of masks to actually np.load')
    ap.add_argument('opts', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cfg.merge_from_file(a.config_file)
    cfg.merge_from_list(a.opts)

    ds = FACTORY[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    print('num_train_pids = {}, num_train_cams = {}'.format(ds.num_train_pids, ds.num_train_cams))
    labels = sorted({x[1] for x in ds.train})
    assert labels == list(range(len(labels))), 'train labels are not contiguous 0..N-1'

    missing = [x[4] for x in ds.train if not osp.exists(x[4])]
    print('train images: {}, missing masks: {}'.format(len(ds.train), len(missing)))
    for m in missing[:10]:
        print('   MISSING:', m)
    if missing:
        sys.exit('[ERROR] Some training masks are missing -> training will crash. Check masks folder layout.')

    for x in ds.train[:a.n_load]:
        m = np.load(x[4])
        assert m.ndim == 3 and m.shape[0] == 36, 'unexpected mask shape {} for {}'.format(m.shape, x[4])
    print('[OK] masks load correctly, example shape:', np.load(ds.train[0][4]).shape)
    print('[OK] query: {} images, gallery: {} images'.format(len(ds.query), len(ds.gallery)))
