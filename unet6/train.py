import argparse
import csv
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import DamageDataset, load_manifest, CLASSES
from .metrics import confusion_update, scores
from .model import UNet6
from .runtime import device_for, save_json, seed_all


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    cm = np.zeros((5, 5), dtype=np.int64)
    total_loss, total = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; inspect inputs and configuration')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.shape[0]
            total += x.shape[0]
            confusion_update(cm, y.detach().cpu().numpy(), logits.detach().argmax(1).cpu().numpy())
    return total_loss / total, scores(cm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='.')
    p.add_argument('--splits', default='splits.json')
    p.add_argument('--output', default='runs/unet6')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--base', type=int, default=32)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--device', default='auto')
    p.add_argument('--weighted-loss', action='store_true')
    p.add_argument('--limit', type=int, help='Debug only: first N samples per train/val split')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.base, args.patience) < 1 or args.size < 32 or args.lr <= 0:
        p.error('Invalid training configuration')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        p.error('Output directory must be empty to avoid overwriting an experiment')
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = device_for(args.device)
    manifest = load_manifest(args.splits)
    train_rows, val_rows = manifest['train'], manifest['val']
    if args.limit:
        train_rows, val_rows = train_rows[:args.limit], val_rows[:args.limit]
    train = DamageDataset(args.root, train_rows, augment=True, size=args.size)
    val = DamageDataset(args.root, val_rows, size=args.size)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)
    weights = None
    if args.weighted_loss:
        # Count only the training samples actually used in this run.
        histogram = np.zeros(5, dtype=np.int64)
        raw = DamageDataset(args.root, train_rows, size=args.size)
        for _, mask, _ in raw:
            histogram += np.bincount(mask.numpy().ravel(), minlength=5)
        w = np.zeros(5, dtype=np.float32)
        present = histogram > 0
        w[present] = 1 / np.sqrt(histogram[present])
        w[present] /= w[present].mean()
        weights = torch.tensor(w, device=device)
    config = vars(args).copy()
    config.update(device_used=str(device), classes=CLASSES,
                  class_weights=weights.cpu().tolist() if weights is not None else None,
                  train_samples=len(train), val_samples=len(val),
                  torch_version=str(torch.__version__),
                  normalization='RGB / 255, channel order pre RGB then post RGB',
                  splits_sha256=hashlib.sha256(Path(args.splits).read_bytes()).hexdigest())
    save_json(out / 'config.json', config)
    save_json(out / 'splits_used.json', manifest)
    model = UNet6(args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=.5)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best, stale, history = -1., 0, []
    print(f'Device={device}; train={len(train)}, val={len(val)}; debug={bool(args.limit)}', flush=True)
    for epoch in range(1, args.epochs+1):
        start = time.monotonic()
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device)
        score = val_metrics['miou_damage']
        if score is None:
            raise RuntimeError('Validation has no defined damage IoU')
        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                   train_miou_damage=train_metrics['miou_damage'], val_miou_damage=score,
                   val_dice_damage=val_metrics['macro_dice_damage'],
                   lr=optimizer.param_groups[0]['lr'], seconds=time.monotonic()-start)
        history.append(row)
        with (out / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        scheduler.step(score)
        saved = dict(model=model.state_dict(), config=config, epoch=epoch,
                     val_metrics=val_metrics)
        torch.save(saved, out / 'last.pt')
        if score > best:
            best, stale = score, 0
            torch.save(saved, out / 'best.pt')
            save_json(out / 'best_val_metrics.json', val_metrics)
        else:
            stale += 1
        print(row, flush=True)
        if stale >= args.patience:
            print('Early stopping', flush=True)
            break
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in ['train_loss', 'val_loss']:
        axes[0].plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    for key in ['train_miou_damage', 'val_miou_damage']:
        axes[1].plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    for ax in axes:
        ax.legend(); ax.set_xlabel('Epoch'); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / 'learning_curves.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
