# COPE (CVPR 2026) – Hướng dẫn reproduce trên server

Bản code này là code chính thức của COPE (`github.com/Cecoming/COPE`) đã được rà soát.
**Toàn bộ hyper-parameter (batch size, learning rate, số epoch, M, N, K1, K2, stride, …) giữ nguyên như file config gốc.**
Chỉ sửa các lỗi khiến code crash hoặc gán nhãn sai (xem mục 9) và thêm 2 script tiện ích trong `tools/`.

Kết quả mục tiêu (bảng 1 của paper, dòng **COPE** = stride 16, đúng với config mặc định):

| Dataset | Train trên | Epoch | Rank-1 | mAP |
|---|---|---|---|---|
| Occluded-Duke | Occluded-Duke | 100 | 82.1 | 75.4 |
| P-DukeMTMC-reID | P-DukeMTMC-reID | 100 | 93.3 | 88.4 |
| Occluded-REID | Market-1501 | 60 | 88.7 | 88.2 |
| Partial-REID | Market-1501 | 100 (config) / 60 (paper) | 88.0 | 88.7 |
| Market-1501 | Market-1501 | 100 | 96.2 | 93.0 |
| MSMT17 | MSMT17 | 100 | 89.0 | 76.4 |

Dòng **COPE\*** trong paper (Occ-Duke 82.4 / 76.4) dùng stride 12: thêm `MODEL.STRIDE_SIZE "[12, 12]"` vào lệnh chạy (trọng số tác giả công bố chỉ có stride 16).

---

## 1. Yêu cầu

- Linux, 1 GPU NVIDIA (paper dùng 1× V100). Code chạy single-GPU, luôn dùng `cuda:0` → chọn GPU bằng `CUDA_VISIBLE_DEVICES`.
- Driver hỗ trợ CUDA 12.1 (kiểm tra bằng `nvidia-smi`). Nếu driver cũ, đổi `cu121` → `cu118` trong `requirements.txt`.
- Python 3.10.
- Server cần truy cập Internet **một lần** để tải trọng số CLIP ViT-B/16 (mục 4); nếu không có mạng, tải tay rồi copy lên.
- **Không giảm batch size** nếu bị thiếu VRAM: batch 64 = 16 ID × 4 ảnh là một phần của phương pháp (CICO áp occlusion theo vị trí trong nhóm 4 ảnh). Hãy dùng GPU lớn hơn.

## 2. Giải nén

```bash
cd ~/projects                 # thư mục bất kỳ
unzip COPE_reproduce.zip
cd COPE-main
```

## 3. Tạo môi trường ảo và cài thư viện

**Cách A – conda (khuyến nghị, giống README gốc):**
```bash
conda create -n cope python=3.10 -y
conda activate cope
pip install -r requirements.txt
```

**Cách B – venv (nếu server đã có python3.10):**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# mong đợi: 2.5.1+cu121 12.1 True <tên GPU>
```

## 4. Tải trước trọng số CLIP ViT-B/16

Code tự tải về `~/.cache/clip/ViT-B-16.pt` ở lần chạy đầu. Nên tải trước để biết chắc server có mạng:
```bash
python -c "import clip.clip as c; print(c._download(c._MODELS['ViT-B-16']))"
```
Nếu server không ra Internet: tải file `ViT-B-16.pt` (link trong `clip/clip.py`, biến `_MODELS`) trên máy khác rồi copy vào `~/.cache/clip/ViT-B-16.pt` của server.

## 5. Chuẩn bị dataset

### 5.1 Cấu trúc thư mục (tên thư mục phân biệt HOA/thường)

Mặc định code tìm dataset trong `./data` (biến `DATASETS.ROOT_DIR`). Chỉ cần chuẩn bị dataset bạn định chạy:

```text
data/
├── Occluded_Duke/
│   ├── bounding_box_train/  query/  bounding_box_test/
│   └── masks/pifpaf_maskrcnn_filtering/bounding_box_train/<tên_ảnh>.jpg.confidence_fields.npy
├── P-DukeMTMC-reid/                        # chú ý: "reid" viết thường
│   ├── train/{whole_body_images,occluded_body_images}/<pid>/*.jpg
│   ├── test/{occluded_body_images,whole_body_images}/<pid>/*.jpg
│   └── masks/pifpaf_maskrcnn_filtering/train/{whole_body_images,occluded_body_images}/<tên_ảnh>.npy
├── market1501/                             # dùng để train cho Market, Occluded-REID, Partial-REID
│   ├── bounding_box_train/  query/  bounding_box_test/
│   └── masks/pifpaf_maskrcnn_filtering/bounding_box_train/<tên_ảnh>.npy
├── Occluded_REID/                          # tạo bằng tools/prepare_occluded_reid.py (mục 5.3)
│   ├── query/<pid>_c1_<idx>.tif
│   └── bounding_box_test/<pid>_c2_<idx>.tif
├── Partial_REID/
│   ├── occluded_body_images/*.jpg
│   └── whole_body_images/*.jpg
└── MSMT17/
    ├── train/  test/  list_train.txt  list_val.txt  list_query.txt  list_gallery.txt
    └── masks/pifpaf/{train,test}/<pid>/<tên_ảnh>.jpg.confidence_fields.npy
```

Lưu ý:
- **Chỉ tập train cần mask** (nhãn human-parsing). Query/gallery không cần mask.
- Occluded-REID và Partial-REID **không có tập train riêng**: COPE train trên Market-1501, nên luôn cần `market1501/bounding_box_train` + mask của nó (kể cả khi chỉ chạy test).
- Nếu tên thư mục bạn tải về khác (ví dụ `Market-1501-v15.09.15`, `P-DukeMTMC-reID`), tạo symlink thay vì copy:
  ```bash
  ln -s /duong/dan/Market-1501-v15.09.15 data/market1501
  ln -s /duong/dan/P-DukeMTMC-reID       data/P-DukeMTMC-reid
  ```

### 5.2 Nhãn human-parsing (mask)

COPE dùng nhãn `pifpaf_maskrcnn_filtering` của BPBReID (Google Drive trong README của BPBReID: https://github.com/VlSomers/bpbreid#download-human-parsing-labels).
Giải nén và đặt thư mục `masks/` vào **đúng thư mục dataset tương ứng** như sơ đồ trên. Có thể tải bằng `gdown` trên server:
```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1IbCAbjj3XtV3_tFOsCuqBi79ZiDqNc1H
```
Với MSMT17, BPBReID không phát hành mask; tác giả COPE tự sinh nhãn `pifpaf` bằng pipeline của BPBReID (phải tự chạy script sinh nhãn).

### 5.3 Occluded-REID (chỉ khi dùng dataset này)

Bản gốc có dạng `occluded_body_images/<pid>/<pid>_<idx>.tif` và `whole_body_images/...`, còn code đọc dạng Market (`query/`, `bounding_box_test/`, tên chứa `_c<cam>`). Chuyển đổi (tạo symlink, không tốn dung lượng):
```bash
python tools/prepare_occluded_reid.py --src /duong/dan/OccludedREID --dst data/Occluded_REID
```

### 5.4 Kiểm tra dataset trước khi train (nên làm)

```bash
python tools/check_dataset.py --config_file configs/OCC_Duke/cope.yml
```
Script in thống kê, kiểm tra mọi ảnh train đều có mask và load được. Thấy `[OK]` là sẵn sàng.

## 6. Cấu hình đường dẫn

Hai cách (chọn một):

1. Sửa trong file yml, ví dụ `configs/OCC_Duke/cope.yml`:
   ```yaml
   DATASETS:
     ROOT_DIR: ('/data/reid')        # thư mục chứa Occluded_Duke/, market1501/, ...
   OUTPUT_DIR: './logs/occ_duke_cope'
   ```
2. Hoặc ghi đè trên dòng lệnh (không phải sửa file): thêm `DATASETS.ROOT_DIR /data/reid OUTPUT_DIR ./logs/occ_duke_run1` vào cuối lệnh.

Config có sẵn: `configs/OCC_Duke`, `configs/P_Duke`, `configs/OCC_ReID`, `configs/P_ReID`, `configs/Market1501`, `configs/MSMT17`.

## 7. Huấn luyện

Quy trình trong `train_cope.py`: **Giai đoạn 1** (60 epoch, Adam) pre-train prompt cho PBF bằng loss segmentation → **Giai đoạn 2** (100 hoặc 60 epoch, SGD) train toàn bộ COPE; đánh giá mỗi `EVAL_PERIOD` epoch (20) và ở epoch cuối.

```bash
conda activate cope        # hoặc: source .venv/bin/activate
cd COPE-main
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs

# Occluded-Duke (chạy nền, tắt SSH vẫn chạy)
nohup python train_cope.py --config_file configs/OCC_Duke/cope.yml > logs/occ_duke_stdout.txt 2>&1 &

# Theo dõi
tail -f logs/occ_duke_cope/train_log.txt
```

Các dataset khác: thay `configs/OCC_Duke` bằng `configs/P_Duke`, `configs/OCC_ReID`, … 

Tái sử dụng giai đoạn 1 (ví dụ chạy lại giai đoạn 2 cùng dataset): `SOLVER.STAGE1.PRETRAINED_PATH logs/occ_duke_cope/ViT-B-16pre_60.pth`.

Nếu server ít CPU, có thể giảm `DATALOADER.NUM_WORKERS 8` (chỉ ảnh hưởng tốc độ, không đổi mô hình).

## 8. Đánh giá (inference)

Với checkpoint bạn tự train:
```bash
python test_cope.py --config_file configs/OCC_Duke/cope.yml TEST.WEIGHT logs/occ_duke_cope/ViT-B-16_100.pth
```
Với trọng số tác giả (Google Drive: https://drive.google.com/drive/folders/1ekVtlmAv_3mkgQUIM9db6wqCrr0--rNB, stride 16):
```bash
gdown --folder https://drive.google.com/drive/folders/1ekVtlmAv_3mkgQUIM9db6wqCrr0--rNB -O weights/
python test_cope.py --config_file configs/OCC_Duke/cope.yml TEST.WEIGHT weights/<file_occ_duke>.pth OUTPUT_DIR logs/occ_duke_eval_official
```
Kết quả in ra đã gồm hậu xử lý PSS (K1=200, K2=5). Bật re-ranking (bảng 7 của paper, COPE+re-rank): thêm `TEST.RE_RANKING True`.

Lưu ý MSMT17: ma trận khoảng cách query×gallery (11 659 × 82 161) tính trên CPU/RAM → cần nhiều RAM (cỡ vài chục GB).

## 9. Log và kết quả

Mọi thứ nằm trong `OUTPUT_DIR` (mặc định `./logs/<dataset>_cope/`):

| File | Nội dung |
|---|---|
| `train_log.txt` | Toàn bộ config đã dùng, loss theo iteration, kết quả đánh giá mỗi 20 epoch |
| `ViT-B-16pre_60.pth` | Checkpoint sau giai đoạn 1 |
| `ViT-B-16_100.pth` (hoặc `_60`) | Checkpoint cuối giai đoạn 2 (dùng để test) |
| `test_log.txt` | Kết quả khi chạy `test_cope.py` (mAP, Rank-1/5/10, thời gian inference) |

Trích nhanh kết quả:
```bash
grep -E "Validation Results|mAP|Rank-1 " logs/occ_duke_cope/train_log.txt
grep -E "mAP|Rank-1 |Inference completed" logs/occ_duke_cope/test_log.txt
```
Ví dụ dòng log:
```text
... COPE INFO: Validation Results - Epoch: 100
... COPE INFO: mAP: 75.4%
... COPE INFO: CMC curve, Rank-1  :82.1%
```
Số dùng để so với paper là kết quả ở **epoch cuối**. Seed cố định (1234) và cuDNN deterministic, nhưng giữa các GPU/driver vẫn có thể lệch vài phần mười điểm.

## 10. Những gì đã sửa so với code gốc

Không thay đổi hyper-parameter hay công thức loss. Chỉ sửa:

1. `cope/loss/triplet_loss.py`: xoá dòng `from turtle import pd` (import thừa do IDE tự thêm) — làm crash ngay khi khởi chạy trên server không có `tkinter`.
2. `datasets/p_duke.py`: tập train P-DukeMTMC trước đây gán nhãn (relabel) **riêng** cho `whole_body_images` và `occluded_body_images`; nếu hai thư mục không có cùng tập ID thì cùng một người nhận hai nhãn khác nhau. Nay dùng một bảng nhãn chung (giống loader của BPBReID).
3. `datasets/*.py`: regex đọc pid/camera chỉ áp lên **tên file** thay vì cả đường dẫn (tránh đọc nhầm nếu đường dẫn server chứa dạng `_c1`, `2024_1`, …). Kết quả giống hệt khi đường dẫn "sạch".
4. `datasets/occ_duke.py`: nếu không thấy thư mục, code gốc âm thầm tải nhầm DukeMTMC-reID (link đã chết); nay báo lỗi rõ ràng.
5. `addmm_(1, -2, a, b)` (cú pháp cũ, deprecated) → `addmm_(a, b, beta=1, alpha=-2)`; tương đương toán học.
6. `requirements.txt`: ghim phiên bản (torch 2.5.1/cu121 như README gốc). Config: `ROOT_DIR` đổi từ `/home/xxx/...` thành `./data`.
7. Thêm `tools/check_dataset.py`, `tools/prepare_occluded_reid.py`.

Đã chạy thử end-to-end (giai đoạn 1 → giai đoạn 2 → đánh giá PSS → lưu/nạp checkpoint → `test_cope.py`) trên dữ liệu giả cho Occluded-Duke, P-Duke, Occluded-REID để chắc chắn không lỗi luồng chạy. Chưa thể chạy trên GPU/dataset thật, nên chưa xác nhận được con số cuối.

## 11. Các điểm code chính thức KHÁC paper (giữ nguyên để khớp trọng số tác giả)

| # | Paper | Code chính thức |
|---|---|---|
| a | Prompt CoCoOp: 4 token học được + `MLP(F_g)` cộng vào token | 16 token học được, **không** dùng đặc trưng ảnh (ảnh truyền vào nhưng chỉ dùng để lấy batch size); `SOLVER.CONTEXT_NUM` không được dùng |
| b | Text encoder đóng băng | Chỉ `eval()`; tham số text encoder vẫn `requires_grad=True` và được SGD cập nhật ở giai đoạn 2 (qua loss segmentation) |
| c | `L_seg` là cross-entropy trên nhãn parsing | BCE foreground/background × 10 |
| d | Hệ số các loss bằng 1 | `L_ce` × 0.5; `L_oc` chuẩn hoá khác công thức (4) (chia 2 và chia 4 nhóm) |
| e | Partial-REID train 60 epoch | `configs/P_ReID/cope.yml` để 100 epoch |
| f | Partial-REID (giao thức phổ biến: query = `partial_body_images`) | Code dùng `occluded_body_images` làm query |
| g | — | MSMT17 bật SIE camera (`SIE_COE 1.0`), các dataset khác không |

Vì trọng số và bảng kết quả của tác giả sinh ra từ code này, bản đóng gói giữ nguyên hành vi code. Nếu muốn bám sát paper ở điểm nào, sửa riêng và ghi rõ trong báo cáo.

## 12. Lỗi thường gặp

- `RuntimeError: '.../Occluded_Duke' is not available` → sai `ROOT_DIR` hoặc sai tên thư mục (phân biệt hoa thường).
- `IOError: Masks file ... does not exist` → thư mục `masks/` đặt sai chỗ; chạy `tools/check_dataset.py` để xem đường dẫn nó đang tìm.
- `urllib.error.URLError` khi khởi tạo model → server không tải được CLIP, làm theo mục 4.
- `CUDA out of memory` → dùng GPU nhiều VRAM hơn; không giảm `SOLVER.IMS_PER_BATCH`.
- Sai số lượng lớp khi `test_cope.py` nạp trọng số → dùng đúng config của dataset mà checkpoint được train.
