# U-Net 6 kênh — Lê Xuân Lộc (23001533)

Phần thực hiện trong đề tài phát hiện và phân đoạn mức độ thiệt hại công trình từ ảnh vệ tinh trước/sau thiên tai. Mô hình ghép **pre RGB rồi post RGB** thành 6 kênh, dự đoán mask 5 lớp bằng một U-Net. Dữ liệu đầu vào giữ nguyên; mọi kết quả được ghi vào `runs/`.

## Thí nghiệm trên dữ liệu được cung cấp

Đã huấn luyện đủ 20 epoch trong `runs/unet6_mps/`: ảnh 512×512, U-Net base=16, batch size 4, weighted cross entropy, GPU Apple MPS. Checkpoint tốt nhất ở epoch 20; đã đánh giá trên toàn bộ 118 mẫu test và chạy demo trên CPU. Đây là cấu hình riêng với cấu hình mặc định base=32/40 epoch trong hướng dẫn huấn luyện mới bên dưới. Xem [kết quả thực nghiệm](runs/unet6_mps/KET_QUA.md) và [báo cáo](BAO_CAO_LE_XUAN_LOC.md).

| Nhiệm vụ trên test | IoU | Dice/F1 |
|---|---:|---:|
| Vùng công trình, gộp nhãn 1–4 | 54,06% | 70,18% |
| Macro bốn lớp thiệt hại | 13,19% | 21,10% |

Lớp hư hại nhẹ và phá hủy còn có kết quả thấp; cần dùng bảng từng lớp trong `runs/unet6_mps/test/per_class.csv` khi nhận xét. Điểm vùng công trình không thay thế chất lượng phân biệt mức độ thiệt hại.

Để dự đoán bằng trọng số của lượt này:

```bash
source .venv/bin/activate
python -m unet6.predict --pre HURRICANE-IRMA/images/HURRICANE-IRMA_000026_pre_disaster.png --post HURRICANE-IRMA/images/HURRICANE-IRMA_000026_post_disaster.png --checkpoint runs/unet6_mps/best.pt --output runs/demo_trained
```

Máy hiện tại đã cài môi trường `.venv`. Khi chuyển sang máy khác, cài dependencies theo hướng dẫn dưới. `artifacts/dataset_example.png` là ảnh minh họa dữ liệu gốc; `artifacts/data_audit.json` lưu thống kê và kiểm tra trùng lặp cặp ảnh.

## Chạy trên máy

Python 3.9 trở lên. Nếu có GPU NVIDIA, cài bản PyTorch phù hợp với môi trường GPU. Mã tự chọn CUDA, Apple MPS hoặc CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unet6.prepare --root . --output splits.json
python -m unet6.train --root . --splits splits.json --output runs/unet6 --epochs 40 --batch-size 4 --weighted-loss
python -m unet6.evaluate --root . --splits splits.json --checkpoint runs/unet6/best.pt
python -m unet6.summarize --run runs/unet6
```

Nếu `splits.json` đã có thì bỏ qua bước `prepare`. Chia sẻ **chính file này** cho hai thành viên còn lại. Các đường dẫn trong file là tương đối so với `--root`, nên có thể chuyển sang Colab. Mã chuẩn bị không ghi đè bộ chia đã tồn tại.

Nếu hết bộ nhớ, giảm `--batch-size` xuống 2 hoặc 1. Chỉ giảm `--size` khi chấp nhận thay đổi độ phân giải thí nghiệm và ghi rõ trong báo cáo. U-Net mặc định dùng số kênh `[32,64,128,256,512]`; `--base 16` tạo bản nhỏ hơn. Không dùng kết quả chạy `--limit` làm kết quả thực nghiệm chính thức.

## Chạy Colab

Mở `Le_Xuan_Loc_UNet6_Colab.ipynb`, chọn runtime GPU, đặt thư mục dự án trên Google Drive, chỉnh `PROJECT_DIR` ở ô đầu tiên rồi chạy lần lượt. Thư mục đó cần chứa `unet6/`, `requirements.txt`, `splits.json` và `HURRICANE-IRMA/`. Notebook lưu checkpoint trực tiếp vào Drive. Mỗi lần thí nghiệm dùng một thư mục kết quả mới.

## Dữ liệu và nhãn

```text
HURRICANE-IRMA/
  images/<id>_pre_disaster.png
  images/<id>_post_disaster.png
  masks/<id>_post_disaster.png
  masks/<id>_pre_disaster.png  # có sẵn; không dùng làm nhãn 5 lớp
```

| ID | Lớp | Màu hiển thị |
|---|---|---|
| 0 | Nền | Đen |
| 1 | Không hư hại | Xanh lá |
| 2 | Hư hại nhẹ | Vàng |
| 3 | Hư hại nặng | Cam |
| 4 | Phá hủy | Đỏ |

Mask giữ nguyên giá trị nguyên 0–4, không chia cho 255 và không chuyển thành RGB. Ảnh chuyển float và chia 255. Khi tăng cường dữ liệu, cùng một phép xoay/lật áp dụng cho cả pre/post/mask. Chỉ train có augmentation. Resize mask dùng nearest-neighbor.

`prepare` kiểm tra toàn bộ cặp, kích thước và miền nhãn; xuất số pixel và số ảnh có mỗi lớp theo split. Chia theo cặp ảnh với seed 42, tỷ lệ gần 70/15/15. Do chỉ có một sự kiện HURRICANE-IRMA và chưa có tọa độ vùng ảnh, bộ chia này chưa chứng minh độc lập không gian hoặc khả năng tổng quát hóa sang thiên tai khác. Nếu nhóm có metadata địa lý, nên tạo bộ chia theo vùng trước khi chạy thực nghiệm chính thức.

## Huấn luyện và đánh giá

- CrossEntropyLoss; tùy chọn `--weighted-loss` tính trọng số nghịch đảo căn bậc hai số pixel **chỉ từ train**, chuẩn hóa về trung bình 1. Lớp không có trong train nhận trọng số 0 và cần được nêu rõ là không học được từ dữ liệu này.
- AdamW, learning rate ban đầu 0.001, weight decay 0.0001. Scheduler giảm LR khi mIoU validation không tăng; dừng sớm mặc định sau 10 epoch không cải thiện.
- Chọn `best.pt` bằng **mIoU validation của các lớp 1–4**. Test không tham gia chọn checkpoint hay chỉnh tham số. Các thí nghiệm so sánh nên dùng cùng manifest, kích thước ảnh và quy tắc đánh giá; ghi rõ khác biệt về kiến trúc và huấn luyện.
- IoU = TP/(TP+FP+FN), Dice/F1 = 2TP/(2TP+FP+FN). Tính từ confusion matrix cộng dồn pixel trên toàn bộ split, không lấy trung bình theo batch. Các chỉ số này là **theo pixel**, không phải theo từng công trình.
- Báo cáo theo lớp và macro trung bình cả 5 lớp hoặc chỉ 4 lớp thiệt hại. Lớp không xuất hiện ở cả nhãn thật và dự đoán có chỉ số `null`, được loại khỏi macro; lớp có dự đoán sai dù không có nhãn thật vẫn nhận IoU/Dice bằng 0.
- Đánh giá cuối dùng kích thước mask gốc. Nếu train ở kích thước khác, nội suy logits về kích thước gốc rồi mới argmax. Confusion matrix có hàng là nhãn thật, cột là dự đoán.
- Báo cáo thêm `building_iou`/`building_dice` bằng cách gộp nhãn 1–4 thành vùng công trình. Hai chỉ số này chỉ đánh giá vị trí; nhầm giữa các mức thiệt hại vẫn được tính đúng vị trí, nên cần đọc cùng chỉ số theo từng lớp.

Kết quả huấn luyện: `config.json`, bản sao `splits_used.json`, `history.csv`, `learning_curves.png`, `best.pt`, `last.pt`, `best_val_metrics.json`. `last.pt` chứa trọng số để suy luận, không chứa trạng thái optimizer để tiếp tục chính xác phiên huấn luyện. Kết quả đánh giá: `metrics.json`, `per_class.csv`, `confusion_matrix.csv`, `confusion_matrix.png` và ảnh so sánh pre/post/ground truth/prediction. `summarize` tạo `KET_QUA.md` từ số liệu thật, từ chối lượt chạy debug. Dấu kiểm SHA-256 của manifest bảo đảm đánh giá dùng cùng bộ chia khi huấn luyện.

## Dự đoán và tích hợp demo

```bash
python -m unet6.predict --pre HURRICANE-IRMA/images/HURRICANE-IRMA_000026_pre_disaster.png --post HURRICANE-IRMA/images/HURRICANE-IRMA_000026_post_disaster.png --checkpoint runs/unet6/best.pt --output runs/demo
```

Đầu ra gồm `labels.png` (nhãn 0–4), `color.png`, `buildings.png`, `overlay.png`, `comparison.png` và `pixel_counts.json`. Thống kê là **số pixel theo lớp**, chưa phải số công trình; công trình sát nhau có thể dính mask nên không dùng đếm vùng liên thông như số công trình thật.

Gọi trực tiếp từ chương trình demo:

```python
from unet6.runtime import device_for, load_model, infer_pair

device = device_for()
model, checkpoint = load_model('runs/unet6/best.pt', device)
mask = infer_pair(model, 'pre.png', 'post.png', device,
                  size=checkpoint['config']['size'])
# mask: numpy uint8 [H, W], giá trị 0..4
```

## Kiểm tra mã

```bash
python -m unittest discover -s tests -v
python -m unet6.train --epochs 1 --batch-size 2 --base 4 --size 64 --limit 4 --output runs/smoke
```

Kiểm tra gồm công thức metric với đáp án biết trước, đồng bộ augmentation và thứ tự kênh, lan truyền gradient, nạp checkpoint và trả mask đúng kích thước gốc. Lượt `smoke` chỉ kiểm tra quy trình chạy được, không phản ánh chất lượng mô hình.
