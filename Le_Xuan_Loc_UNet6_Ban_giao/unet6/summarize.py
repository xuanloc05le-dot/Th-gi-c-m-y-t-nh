"""Build a short report from actual saved results; never substitute example scores."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='runs/unet6')
    args = p.parse_args()
    run = Path(args.run)
    config = json.loads((run / 'config.json').read_text())
    metrics = json.loads((run / 'test' / 'metrics.json').read_text())
    history = list(csv.DictReader((run / 'history.csv').open()))
    if config.get('limit') or metrics.get('debug_training') or metrics['split'] != 'test':
        raise ValueError('A final report requires a full training run and test evaluation')
    names = ['Nền', 'Không hư hại', 'Hư hại nhẹ', 'Hư hại nặng', 'Phá hủy']
    def percentage(x):
        return 'Không xác định' if x is None else f'{x*100:.2f}%'
    lines = ['# Kết quả thực nghiệm U-Net 6 kênh — Lê Xuân Lộc (23001533)', '',
             'Các số liệu dưới đây được đọc từ tệp kết quả thực nghiệm thực tế.', '',
             f'- Dữ liệu train/validation: {config["train_samples"]}/{config["val_samples"]} cặp.',
             f'- Tập test: {metrics["samples"]} cặp; đánh giá theo pixel ở độ phân giải mask gốc.',
             f'- Thiết bị: {config["device_used"]}; PyTorch {config["torch_version"]}.',
             f'- Thiết bị đánh giá: {metrics.get("device_used", "không ghi nhận")}.',
             f'- Đầu vào: 6×{config["size"]}×{config["size"]}; U-Net base={config["base"]}.',
             f'- Batch size: {config["batch_size"]}; AdamW, learning rate ban đầu {config["lr"]}.',
             f'- Loss: cross entropy; trọng số lớp: {config["class_weights"]}.',
             f'- Số epoch đã chạy: {len(history)}/{config["epochs"]}; checkpoint được chọn ở epoch {metrics["checkpoint_epoch"]}.',
             '- Tiêu chí chọn checkpoint: mIoU bốn lớp thiệt hại trên validation.',
             f'- Tổng thời gian các epoch: {sum(float(r["seconds"]) for r in history)/60:.1f} phút.',
             f'- SHA-256 manifest: `{config["splits_sha256"]}`.', '',
             '| Lớp | IoU | Dice/F1 | Số pixel ground truth |',
             '|---|---:|---:|---:|']
    for name, row in zip(names, metrics['per_class']):
        lines.append(f'| {name} | {percentage(row["iou"])} | {percentage(row["dice_f1"])} | {row["support_pixels"]:,} |')
    lines += [f'| Macro 5 lớp | {percentage(metrics["miou_all"])} | {percentage(metrics["macro_dice_all"])} | — |',
              f'| Macro 4 lớp thiệt hại | {percentage(metrics["miou_damage"])} | {percentage(metrics["macro_dice_damage"])} | — |', '',
              'Đây là chỉ số phân đoạn theo pixel, không phải độ chính xác phân loại từng công trình. '
              'Lớp vắng ở cả ground truth và dự đoán bị loại khỏi macro; dự đoán sai vào lớp vắng vẫn được tính là lỗi.', '',
              'Dữ liệu hiện tại thuộc HURRICANE-IRMA. Bộ chia ngẫu nhiên theo cặp ảnh không chứng minh '
              'độc lập không gian hoặc khả năng tổng quát hóa sang sự kiện mới. '
              'Cần đọc kết quả từng lớp, đặc biệt các lớp thiệt hại hiếm, trước khi kết luận chất lượng.', '',
              'Ảnh so sánh nằm trong `test/*_comparison.png`; biểu đồ huấn luyện là `learning_curves.png`; '
              'confusion matrix là `test/confusion_matrix.png`. Mô hình bàn giao: `best.pt`.', '']
    damage = [row for row in metrics['per_class'][1:] if row['iou'] is not None]
    if damage:
        highest = max(damage, key=lambda row: row['iou'])
        lowest = min(damage, key=lambda row: row['iou'])
        lines += ['## Nhận xét từ số liệu', '',
                  f'Trong bốn lớp thiệt hại, IoU cao nhất thuộc lớp {names[highest["id"]]} '
                  f'({percentage(highest["iou"])}), thấp nhất thuộc lớp {names[lowest["id"]]} '
                  f'({percentage(lowest["iou"])}).', '']
    if 'building_iou' in metrics:
        lines += [f'Khi gộp các nhãn 1–4 thành vùng công trình, IoU đạt '
                  f'{percentage(metrics["building_iou"])} và Dice đạt '
                  f'{percentage(metrics["building_dice"])}. Đây là đánh giá vị trí công trình '
                  'suy ra từ mask 5 lớp; nhầm giữa hai mức thiệt hại vẫn được tính là đúng vùng công trình '
                  'trong phép đánh giá nhị phân này.', '']
    cm = metrics['confusion_matrix']
    building_pixels = sum(sum(row) for row in cm[1:])
    if building_pixels:
        missed = sum(row[0] for row in cm[1:]) / building_pixels
        lines += [f'Tỷ lệ pixel công trình trong ground truth bị dự đoán thành nền là '
                  f'{percentage(missed)}. Để phân tích nhầm mức thiệt hại, đọc các ô ngoài đường chéo '
                  'trong confusion matrix và đối chiếu ảnh minh họa.', '']
    output = run / 'KET_QUA.md'
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
