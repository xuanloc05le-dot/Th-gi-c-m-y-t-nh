import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from unet6.data import DamageDataset
from unet6.metrics import confusion_update, scores
from unet6.model import UNet6
from unet6.runtime import load_model, infer_pair


class PipelineTests(unittest.TestCase):
    def test_metrics_known_counts(self):
        cm = np.zeros((5, 5), dtype=np.int64)
        confusion_update(cm, [0, 1, 1, 2], [0, 1, 2, 2])
        result = scores(cm)
        self.assertAlmostEqual(result['per_class'][1]['iou'], .5)
        self.assertAlmostEqual(result['per_class'][1]['dice_f1'], 2/3)
        self.assertIsNone(result['per_class'][4]['iou'])
        self.assertAlmostEqual(result['miou_damage'], .5)
        self.assertEqual(result['building_iou'], 1.)
        # A false positive in a class absent from GT must count as zero, not be ignored.
        confusion_update(cm, [0], [4])
        self.assertEqual(scores(cm)['per_class'][4]['iou'], 0.)
        self.assertEqual(scores(cm)['building_iou'], .75)
        self.assertAlmostEqual(scores(cm)['building_dice'], 6/7)

    def test_pair_order_and_joint_augmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = (np.indices((32, 32))[0] % 5).astype('uint8')
            pre = np.repeat((mask*40)[..., None], 3, axis=2)
            post = pre + 10
            for name, data in [('pre.png', pre), ('post.png', post), ('mask.png', mask)]:
                Image.fromarray(data).save(root / name)
            dataset = DamageDataset(root, [dict(id='a', pre='pre.png', post='post.png', mask='mask.png')], True)
            for _ in range(8):
                x, y, _ = dataset[0]
                np.testing.assert_allclose(x[0].numpy()*255, y.numpy()*40, atol=1e-4)
                np.testing.assert_allclose(x[3].numpy()*255, y.numpy()*40+10, atol=1e-4)

    def test_backward_checkpoint_and_original_resolution(self):
        torch.set_num_threads(2)
        model = UNet6(base=2)
        x = torch.rand(2, 6, 33, 35)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 5, 33, 35))
        loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 5, (2, 33, 35)))
        loss.backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(dict(config=dict(base=2), model=model.state_dict()), root / 'model.pt')
            restored, _ = load_model(root / 'model.pt', torch.device('cpu'))
            image = np.zeros((41, 43, 3), dtype='uint8')
            Image.fromarray(image).save(root / 'pre.png')
            Image.fromarray(image).save(root / 'post.png')
            pred = infer_pair(restored, root / 'pre.png', root / 'post.png', torch.device('cpu'), 32)
            self.assertEqual(pred.shape, (41, 43))
            self.assertTrue(np.isin(pred, np.arange(5)).all())


if __name__ == '__main__':
    unittest.main()
