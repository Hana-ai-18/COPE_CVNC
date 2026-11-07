"""Convert the ORIGINAL Occluded-REID release into the layout expected by COPE.

Original release (Zhuo et al., ICME 2018):
    OccludedREID/
        occluded_body_images/<pid>/<pid>_<idx>.tif   -> used as QUERY
        whole_body_images/<pid>/<pid>_<idx>.tif      -> used as GALLERY

Layout read by datasets/occ_reid.py (regex r'([-\\d]+)_c(\\d)' on the file name):
    <ROOT_DIR>/Occluded_REID/
        query/<pid>_c1_<idx>.tif
        bounding_box_test/<pid>_c2_<idx>.tif

Query and gallery get different camera ids (c1 / c2) so the standard
"same pid & same camera" filtering in eval_func removes nothing, which is the
usual Occluded-REID protocol (occluded queries vs. holistic gallery, 200 ids).

Usage:
    python tools/prepare_occluded_reid.py --src /path/to/OccludedREID --dst ./data/Occluded_REID [--copy]
By default symlinks are created (no extra disk space); use --copy to copy files.
"""
import argparse
import glob
import os
import os.path as osp
import shutil


def convert(src_dir, dst_dir, cam, copy):
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(osp.join(src_dir, '*', '*.tif')) + glob.glob(osp.join(src_dir, '*.tif')))
    if not files:
        raise RuntimeError('No .tif files found under {}'.format(src_dir))
    n = 0
    for f in files:
        name = osp.splitext(osp.basename(f))[0]          # e.g. 001_01
        parts = name.split('_')
        pid = int(parts[0])
        idx = '_'.join(parts[1:]) if len(parts) > 1 else str(n)
        out = osp.join(dst_dir, '{:03d}_c{}_{}.tif'.format(pid, cam, idx))
        if osp.lexists(out):
            os.remove(out)
        if copy:
            shutil.copy2(f, out)
        else:
            os.symlink(osp.abspath(f), out)
        n += 1
    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='original OccludedREID folder')
    ap.add_argument('--dst', required=True, help='output folder, e.g. ./data/Occluded_REID')
    ap.add_argument('--copy', action='store_true', help='copy files instead of symlinks')
    a = ap.parse_args()
    nq = convert(osp.join(a.src, 'occluded_body_images'), osp.join(a.dst, 'query'), 1, a.copy)
    ng = convert(osp.join(a.src, 'whole_body_images'), osp.join(a.dst, 'bounding_box_test'), 2, a.copy)
    print('query: {} images, gallery: {} images -> {}'.format(nq, ng, a.dst))
