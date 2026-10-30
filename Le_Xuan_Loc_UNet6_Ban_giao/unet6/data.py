import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ['background', 'no_damage', 'minor_damage', 'major_damage', 'destroyed']
COLORS = np.array([[0, 0, 0], [40, 180, 70], [255, 220, 0],
                   [255, 130, 0], [220, 30, 40]], dtype=np.uint8)


def discover(root):
    root = Path(root)
    records = []
    for pre in sorted(root.rglob('*_pre_disaster.png')):
        if pre.parent.name != 'images':
            continue
        stem = pre.name.removesuffix('_pre_disaster.png')
        post = pre.with_name(stem + '_post_disaster.png')
        mask = pre.parent.parent / 'masks' / post.name
        if not post.is_file() or not mask.is_file():
            raise FileNotFoundError(f'Missing post image or mask for {pre}')
        records.append(dict(id=str(pre.parent.parent.relative_to(root) / stem),
                            pre=str(pre.relative_to(root)),
                            post=str(post.relative_to(root)),
                            mask=str(mask.relative_to(root))))
    if not records:
        raise ValueError(f'No pre/post pairs found under {root}')
    # Also reject orphan post images, rather than silently excluding them.
    posts = {str(p.relative_to(root)) for p in root.rglob('*_post_disaster.png')
             if p.parent.name == 'images'}
    if posts != {r['post'] for r in records}:
        raise ValueError('Orphan post images without matching pre images')
    return records


def read_mask(path):
    with Image.open(path) as im:
        mask = np.array(im)
    if mask.ndim != 2 or not np.isin(mask, np.arange(5)).all():
        raise ValueError(f'Expected a single-channel mask with labels 0..4: {path}')
    return mask.astype(np.int64)


def read_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    seen = set()
    for split in ['train', 'val', 'test']:
        if not data[split]:
            raise ValueError(f'Empty split: {split}')
        for row in data[split]:
            if row['id'] in seen:
                raise ValueError(f'Duplicate sample across splits: {row["id"]}')
            seen.add(row['id'])
    return data


class DamageDataset:
    def __init__(self, root, records, augment=False, size=None):
        self.root, self.records = Path(root), records
        self.augment, self.size = augment, size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        row = self.records[index]
        pre, post = [read_rgb(self.root / row[k]) for k in ['pre', 'post']]
        mask = read_mask(self.root / row['mask'])
        if pre.shape != post.shape or pre.shape[:2] != mask.shape:
            raise ValueError(f'Image/mask shape mismatch: {row["id"]}')
        if self.size is not None:
            dims = (self.size, self.size)
            pre = np.array(Image.fromarray(pre).resize(dims, Image.Resampling.BILINEAR))
            post = np.array(Image.fromarray(post).resize(dims, Image.Resampling.BILINEAR))
            mask = np.array(Image.fromarray(mask.astype('uint8')).resize(
                dims, Image.Resampling.NEAREST)).astype('int64')
        x = np.concatenate([pre, post], axis=2)
        if self.augment:
            k = random.randrange(4)
            x, mask = np.rot90(x, k), np.rot90(mask, k)
            if random.random() < 0.5:
                x, mask = np.flip(x, axis=1), np.flip(mask, axis=1)
        # Same fixed normalization for training and inference; no dataset leakage.
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float() / 255.
        return x, torch.from_numpy(mask.copy()).long(), row['id']
