# COPE – Inference trên P-DukeMTMC-reID và Occluded-Duke

Làm lần lượt từ bước 1 đến bước 6. Mọi lệnh chạy trong thư mục `COPE-main/`.

> Muốn chạy tự động toàn bộ thì dùng `DUKE_ZIP=/duong/dan/DukeMTMC-reID.zip bash run_all.sh` (xem mục 7).

---

## Bước 1. Giải nén code

```bash
unzip COPE_reproduce.zip
cd COPE-main
mkdir -p data weights logs .cache_downloads
```

## Bước 2. Tạo môi trường và cài thư viện

Yêu cầu: Linux, GPU NVIDIA, driver hỗ trợ CUDA 12.1 (kiểm tra bằng `nvidia-smi`).

```bash
conda create -n cope -c conda-forge --override-channels python=3.10 -y
conda activate cope
pip install -r requirements.txt gdown
```

Nếu không có conda, dùng venv:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt gdown
```

Kiểm tra:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.5.1+cu121 True
```

> Nếu driver cũ, sửa `cu121` thành `cu118` trong `requirements.txt` rồi cài lại.

Tải trọng số CLIP ViT-B/16 (bắt buộc kể cả khi chỉ inference):

```bash
python -c "import clip.clip as c; print(c._download(c._MODELS['ViT-B-16']))"
```

## Bước 3. Tải P-DukeMTMC-reID

Nguồn: https://github.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets. File zip nằm ngay trong repo, tải thẳng được.

```bash
wget -O .cache_downloads/P-DukeMTMC-reid.zip \
  https://raw.githubusercontent.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets/master/P-DukeMTMC-reid.zip

unzip -q .cache_downloads/P-DukeMTMC-reid.zip -d data/P-DukeMTMC-reid
```

Kiểm tra (phải ra **12927** và **11216**):

```bash
find data/P-DukeMTMC-reid/train -name "*.jpg" | wc -l
find data/P-DukeMTMC-reid/test  -name "*.jpg" | wc -l
```

## Bước 4. Tạo Occluded-Duke

Nguồn: https://github.com/lightas/Occluded-DukeMTMC-Dataset. Repo này chỉ có danh sách tên ảnh và script chuyển đổi, **không có ảnh**. Ảnh được sinh từ file gốc **`DukeMTMC-reID.zip`**. Duke đã gỡ file này khỏi trang chính thức, nên phải xin bản copy từ thầy hoặc lab.

**4.1. Đặt `DukeMTMC-reID.zip` vào `.cache_downloads/`** (chọn một cách):

```bash
# Cách A: file đã có trên server
cp /duong/dan/DukeMTMC-reID.zip .cache_downloads/DukeMTMC-reID.zip

# Cách B: file nằm trên Google Drive (chia sẻ "Anyone with the link")
#   link: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
gdown "https://drive.google.com/uc?id=<FILE_ID>" -O .cache_downloads/DukeMTMC-reID.zip
```

**4.2. Chuyển đổi bằng script của tác giả:**

```bash
git clone --depth 1 https://github.com/lightas/Occluded-DukeMTMC-Dataset.git .cache_downloads/occduke
ln -sf "$(readlink -f .cache_downloads/DukeMTMC-reID.zip)" .cache_downloads/occduke/DukeMTMC-reID.zip
(cd .cache_downloads/occduke && python convert_duke_to_occduke.py DukeMTMC-reID.zip)

mv .cache_downloads/occduke/Occluded_Duke data/Occluded_Duke
rm -f data/Occluded_Duke/*.list
rm -rf .cache_downloads/occduke
```

> Script của tác giả bắt buộc file phải tên đúng là `DukeMTMC-reID.zip`. Bên trong zip phải có thư mục `DukeMTMC-reID/`.

Kiểm tra (phải ra **15618**, **2210**, **17661**):

```bash
ls data/Occluded_Duke/bounding_box_train | wc -l
ls data/Occluded_Duke/query | wc -l
ls data/Occluded_Duke/bounding_box_test | wc -l
```

Cấu trúc `data/` sau bước 3 và 4:

```text
data/
├── P-DukeMTMC-reid/
│   ├── train/{whole_body_images,occluded_body_images}/<pid>/*.jpg
│   └── test/{occluded_body_images,whole_body_images}/<pid>/*.jpg
└── Occluded_Duke/
    ├── bounding_box_train/
    ├── query/
    └── bounding_box_test/
```

## Bước 5. Tải checkpoint COPE

Nguồn: thư mục Google Drive công khai của tác giả (stride 16).

```bash
gdown --folder https://drive.google.com/drive/folders/1ekVtlmAv_3mkgQUIM9db6wqCrr0--rNB -O weights
find weights -name "*.pth"
```

> Nếu `gdown` báo lỗi do Drive giới hạn lượt tải, mở link trên trình duyệt, tải các file `.pth` rồi copy vào `weights/`.

Tìm checkpoint của từng dataset. Script nhận diện theo số lớp trong checkpoint, nên tên file không quan trọng: Occluded-Duke có **702** lớp, P-Duke có **665** lớp.

```bash
CKPT_OCC_DUKE=$(python tools/find_checkpoint.py --dir weights --num_classes 702 | head -1)
CKPT_P_DUKE=$(python tools/find_checkpoint.py --dir weights --num_classes 665 | head -1)
echo "$CKPT_OCC_DUKE"; echo "$CKPT_P_DUKE"
```

## Bước 6. Chạy inference

```bash
export CUDA_VISIBLE_DEVICES=0

# Occluded-Duke
python test_cope.py --config_file configs/OCC_Duke/cope.yml \
    TEST.WEIGHT "$CKPT_OCC_DUKE" DATASETS.ROOT_DIR ./data OUTPUT_DIR logs/occ_duke_inference

# P-DukeMTMC-reID
python test_cope.py --config_file configs/P_Duke/cope.yml \
    TEST.WEIGHT "$CKPT_P_DUKE" DATASETS.ROOT_DIR ./data OUTPUT_DIR logs/p_duke_inference
```

> Thiếu VRAM thì thêm `TEST.IMS_PER_BATCH 64` vào cuối lệnh. Tham số này không ảnh hưởng kết quả.

### Kết quả

```bash
grep -E "mAP|Rank-|Inference completed" logs/occ_duke_inference/test_log.txt
grep -E "mAP|Rank-|Inference completed" logs/p_duke_inference/test_log.txt
```

Mẫu output:

```text
... COPE INFO: Validation P_socre Results
... COPE INFO: mAP: xx.x%
... COPE INFO: CMC curve, Rank-1  :xx.x%
... COPE INFO: CMC curve, Rank-5  :xx.x%
... COPE INFO: CMC curve, Rank-10 :xx.x%
... COPE INFO: Inference completed in xx.xx seconds
```

So với paper (Rank-1 / mAP): **Occluded-Duke 82.1 / 75.4**, **P-Duke 93.3 / 88.4**.

Mỗi file `test_log.txt` ghi cả config đã dùng lẫn kết quả, tiện đưa vào báo cáo.

---

## 7. Chạy tự động bằng một lệnh (tương đương bước 2–6)

```bash
DUKE_ZIP=/duong/dan/DukeMTMC-reID.zip bash run_all.sh
# hoặc: DUKE_GDRIVE_ID=<FILE_ID> bash run_all.sh
# chạy nền: nohup env DUKE_ZIP=/duong/dan/DukeMTMC-reID.zip bash run_all.sh > logs/nohup.txt 2>&1 &
```

Script tự bỏ qua bước nào đã xong. Kết quả nằm trong:

- `logs/summary.txt`: bảng tổng hợp hai dataset
- `logs/<dataset>_inference/test_log.txt`: chi tiết từng dataset
- `logs/run_all_<thời_gian>.log`: log toàn bộ lần chạy

---

## Cấu trúc thư mục tổng

Sau khi chạy xong các bước:

```text
COPE-main/
├── run_all.sh                      # chạy tự động bước 2–6
├── test_cope.py                    # script inference
├── train_cope.py                   # script huấn luyện (không dùng)
├── requirements.txt
├── README_INFERENCE.md
├── configs/
│   ├── OCC_Duke/cope.yml           # config Occluded-Duke
│   ├── P_Duke/cope.yml             # config P-DukeMTMC-reID
│   └── ...                         # các dataset khác
├── tools/
│   ├── find_checkpoint.py          # tìm checkpoint theo số lớp
│   ├── check_dataset.py
│   └── prepare_occluded_reid.py
├── clip/  config/  cope/  datasets/  solver/  utils/   # mã nguồn COPE
│
├── .cache_downloads/               # file zip đã tải
│   ├── P-DukeMTMC-reid.zip
│   └── DukeMTMC-reID.zip
│
├── data/
│   ├── P-DukeMTMC-reid/
│   │   ├── train/
│   │   │   ├── whole_body_images/<pid>/*.jpg
│   │   │   └── occluded_body_images/<pid>/*.jpg
│   │   └── test/
│   │       ├── occluded_body_images/<pid>/*.jpg     # query
│   │       └── whole_body_images/<pid>/*.jpg        # gallery
│   └── Occluded_Duke/
│       ├── bounding_box_train/*.jpg
│       ├── query/*.jpg
│       └── bounding_box_test/*.jpg                  # gallery
│
├── weights/                        # checkpoint COPE tải từ Drive
│   └── *.pth
│
└── logs/
    ├── occ_duke_inference/test_log.txt
    ├── p_duke_inference/test_log.txt
    ├── summary.txt                 # chỉ có khi chạy run_all.sh
    └── run_all_<thời_gian>.log     # chỉ có khi chạy run_all.sh
```

Ngoài ra, trọng số CLIP được lưu ngoài project, tại `~/.cache/clip/ViT-B-16.pt`.
