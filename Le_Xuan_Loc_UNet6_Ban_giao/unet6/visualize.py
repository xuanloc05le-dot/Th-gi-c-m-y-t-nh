from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .data import COLORS, CLASSES


def comparison(pre, post, truth, prediction, output):
    items = [(pre, 'Pre-disaster'), (post, 'Post-disaster')]
    if truth is not None:
        items.append((COLORS[truth], 'Ground truth'))
    items.append((COLORS[prediction], 'U-Net prediction'))
    fig, axes = plt.subplots(1, len(items), figsize=(4*len(items), 4.8))
    for ax, (data, title) in zip(axes, items):
        ax.imshow(data); ax.set_title(title); ax.axis('off')
    fig.legend(handles=[Patch(color=color/255., label=name) for color, name in zip(COLORS, CLASSES)],
               loc='lower center', ncol=5)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(Path(output), dpi=140)
    plt.close(fig)
