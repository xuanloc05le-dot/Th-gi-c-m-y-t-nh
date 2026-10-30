import json
from pathlib import Path
import random
import numpy as np
import torch
from .model import UNet6


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name='auto'):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def load_model(checkpoint, device):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = UNet6(base=saved['config']['base']).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    return model, saved


def infer_pair(model, pre_path, post_path, device, size=None):
    """Return uint8 HxW labels 0..4 at original resolution."""
    from PIL import Image
    from torch.nn import functional as F
    from .data import read_rgb
    pre, post = read_rgb(pre_path), read_rgb(post_path)
    if pre.shape != post.shape:
        raise ValueError('Pre/post images must have identical dimensions')
    if min(pre.shape[:2]) < 32:
        raise ValueError('Image dimensions must be at least 32 pixels')
    original = pre.shape[:2]
    if size is not None:
        pre, post = [np.array(Image.fromarray(a).resize((size, size), Image.Resampling.BILINEAR))
                     for a in [pre, post]]
    x = torch.from_numpy(np.concatenate([pre, post], axis=2).transpose(2, 0, 1).copy()).float()
    with torch.inference_mode():
        logits = model(x.unsqueeze(0).to(device) / 255.)
        logits = F.interpolate(logits, size=original, mode='bilinear', align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype('uint8')
