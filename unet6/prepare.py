"""Validate every sample, report label distribution and save reusable splits."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
from .data import discover, read_mask, read_rgb, CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.')
    p.add_argument('--output', default='splits.json')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-ratio', type=float, default=.15)
    p.add_argument('--test-ratio', type=float, default=.15)
    args = p.parse_args()
    if Path(args.output).exists():
        p.error('Output already exists; choose another filename to preserve shared splits')
    if not (0 < args.val_ratio < 1 and 0 < args.test_ratio < 1 and
            args.val_ratio + args.test_ratio < 1):
        p.error('Require positive ratios with val + test < 1')
    root = Path(args.root)
    rows = discover(root)
    histograms = {}
    for i, row in enumerate(rows):
        pre, post = [read_rgb(root / row[k]) for k in ['pre', 'post']]
        mask = read_mask(root / row['mask'])
        if pre.shape != post.shape or mask.shape != pre.shape[:2]:
            raise ValueError(f'Shape mismatch: {row["id"]}')
        histograms[row['id']] = np.bincount(mask.ravel(), minlength=5)
        if (i+1) % 100 == 0:
            print(f'Validated {i+1}/{len(rows)}', flush=True)
    random.Random(args.seed).shuffle(rows)
    nv, nt = max(1, round(len(rows)*args.val_ratio)), max(1, round(len(rows)*args.test_ratio))
    if nv + nt >= len(rows):
        raise ValueError('Not enough samples for three nonempty splits')
    splits = dict(train=rows[nv+nt:], val=rows[:nv], test=rows[nv:nv+nt])
    statistics = {}
    for name, samples in splits.items():
        counts = np.sum([histograms[r['id']] for r in samples], axis=0)
        statistics[name] = dict(samples=len(samples), pixels=counts.tolist(),
            images_with_class=[int(sum(histograms[r['id']][c] > 0 for r in samples)) for c in range(5)])
    result = dict(seed=args.seed, classes=CLASSES, split_method='random_by_image_pair',
                  note='Same event; spatial independence cannot be guaranteed without geographic metadata.',
                  statistics=statistics, **splits)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(statistics, indent=2))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
