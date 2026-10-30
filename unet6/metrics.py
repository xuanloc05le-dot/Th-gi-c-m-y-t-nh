import numpy as np
from .data import CLASSES


def confusion_update(cm, truth, prediction):
    truth, prediction = np.asarray(truth).ravel(), np.asarray(prediction).ravel()
    if truth.shape != prediction.shape:
        raise ValueError('Prediction and target shapes differ')
    cm += np.bincount(5 * truth.astype('int64') + prediction.astype('int64'),
                      minlength=25).reshape(5, 5)


def scores(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp = cm.diagonal()
    support, predicted = cm.sum(1), cm.sum(0)
    union = support + predicted - tp
    def divide(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
    iou = divide(tp, union)
    dice = divide(2*tp, support+predicted)
    def mean(values):
        return float(np.nanmean(values)) if np.isfinite(values).any() else None
    def scalar(v):
        return float(v) if np.isfinite(v) else None
    building_tp = cm[1:, 1:].sum()
    building_fp = cm[0, 1:].sum()
    building_fn = cm[1:, 0].sum()
    building_union = building_tp + building_fp + building_fn
    building_denom = 2*building_tp + building_fp + building_fn
    return dict(per_class=[dict(id=i, name=name, support_pixels=int(support[i]),
                                iou=scalar(iou[i]), dice_f1=scalar(dice[i]),
                                precision=scalar(divide(tp, predicted)[i]),
                                recall=scalar(divide(tp, support)[i]))
                           for i, name in enumerate(CLASSES)],
                miou_all=mean(iou), miou_damage=mean(iou[1:]),
                macro_dice_all=mean(dice), macro_dice_damage=mean(dice[1:]),
                building_iou=float(building_tp/building_union) if building_union else None,
                building_dice=float(2*building_tp/building_denom) if building_denom else None,
                confusion_matrix=cm.astype('int64').tolist())
