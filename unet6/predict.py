import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .data import COLORS, CLASSES, read_rgb
from .runtime import device_for, load_model, infer_pair, save_json
from .visualize import comparison


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pre', required=True)
    p.add_argument('--post', required=True)
    p.add_argument('--checkpoint', default='runs/unet6/best.pt')
    p.add_argument('--output', default='runs/prediction')
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    model, saved = load_model(args.checkpoint, device)
    pred = infer_pair(model, args.pre, args.post, device, saved['config']['size'])
    Image.fromarray(pred).save(out / 'labels.png')
    Image.fromarray(COLORS[pred]).save(out / 'color.png')
    Image.fromarray(((pred > 0)*255).astype('uint8')).save(out / 'buildings.png')
    post = read_rgb(args.post)
    overlay = post.copy()
    selected = pred > 0
    overlay[selected] = (.55*post[selected] + .45*COLORS[pred[selected]]).astype('uint8')
    Image.fromarray(overlay).save(out / 'overlay.png')
    comparison(read_rgb(args.pre), post, None, pred, out / 'comparison.png')
    save_json(out / 'pixel_counts.json', dict(zip(CLASSES, np.bincount(pred.ravel(), minlength=5).tolist())))
    print(f'Saved predictions to {out}')


if __name__ == '__main__':
    main()
