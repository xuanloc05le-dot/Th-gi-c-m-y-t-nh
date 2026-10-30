import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .data import load_manifest, read_rgb, read_mask, COLORS
from .runtime import device_for, load_model, infer_pair, save_json
from .metrics import confusion_update, scores
from .visualize import comparison


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.')
    p.add_argument('--splits', default='splits.json')
    p.add_argument('--checkpoint', default='runs/unet6/best.pt')
    p.add_argument('--output', default='runs/unet6/test')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='auto')
    p.add_argument('--examples', type=int, default=8)
    args = p.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    model, saved = load_model(args.checkpoint, device)
    checksum = hashlib.sha256(Path(args.splits).read_bytes()).hexdigest()
    if checksum != saved['config']['splits_sha256']:
        raise ValueError('Splits differ from the manifest used for training')
    rows = load_manifest(args.splits)[args.split]
    cm = np.zeros((5, 5), dtype=np.int64)
    for i, row in enumerate(rows):
        mask = read_mask(root / row['mask'])
        pred = infer_pair(model, root / row['pre'], root / row['post'], device, saved['config']['size'])
        confusion_update(cm, mask, pred)
        if i < args.examples:
            name = row['id'].replace('/', '__')
            Image.fromarray(pred).save(out / f'{name}_labels.png')
            Image.fromarray(COLORS[pred]).save(out / f'{name}_color.png')
            comparison(read_rgb(root / row['pre']), read_rgb(root / row['post']), mask,
                       pred, out / f'{name}_comparison.png')
        if (i+1) % 20 == 0:
            print(f'Evaluated {i+1}/{len(rows)}', flush=True)
    result = scores(cm)
    result.update(split=args.split, samples=len(rows), checkpoint=str(args.checkpoint),
                  checkpoint_epoch=saved['epoch'], debug_training=bool(saved['config'].get('limit')),
                  device_used=str(device),
                  resolution='original mask resolution')
    save_json(out / 'metrics.json', result)
    with (out / 'per_class.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['per_class'][0]))
        writer.writeheader(); writer.writerows(result['per_class'])
    np.savetxt(out / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    import matplotlib.pyplot as plt
    from .data import CLASSES
    totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(5), CLASSES, rotation=35, ha='right')
    ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel('Prediction'); ax.set_ylabel('Ground truth')
    ax.set_title('Confusion matrix (row-normalized, pixel counts)')
    for row in range(5):
        for col in range(5):
            ax.text(col, row, f'{normalized[row, col]:.1%}\n{cm[row, col]:,}',
                    ha='center', va='center', fontsize=8,
                    color='white' if normalized[row, col] > .5 else 'black')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    print(result)


if __name__ == '__main__':
    main()
