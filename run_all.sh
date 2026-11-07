#!/usr/bin/env bash
# =============================================================================
#  COPE – chạy INFERENCE bằng MỘT lệnh:
#     bash run_all.sh
#
#  Script tự làm lần lượt (bước nào đã xong thì tự bỏ qua khi chạy lại):
#    1. Tạo môi trường Python 3.10 (conda nếu có, không thì venv) + cài thư viện
#    2. Tải trọng số CLIP ViT-B/16
#    3. Tải checkpoint COPE của tác giả (Google Drive, bằng gdown)
#    4. Tải + chuẩn bị dataset:
#         - P-DukeMTMC-reID : tải thẳng từ GitHub (không cần Google Drive)
#         - Occluded-Duke   : cần file gốc DukeMTMC-reID.zip (xem DUKE_ZIP / DUKE_GDRIVE_ID)
#    5. Tự nhận diện checkpoint nào thuộc dataset nào, chạy test_cope.py
#    6. Ghi bảng tổng hợp kết quả vào logs/summary.txt
#
#  Tuỳ chỉnh bằng biến môi trường, ví dụ:
#     DUKE_ZIP=/data/DukeMTMC-reID.zip bash run_all.sh
#     DUKE_GDRIVE_ID=1AbC...xyz        bash run_all.sh
#     DATASETS="p_duke"                bash run_all.sh      # chỉ chạy P-Duke
#     PYTHON=/path/to/env/bin/python   bash run_all.sh      # dùng môi trường có sẵn
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
REPO_DIR="$(pwd)"

# ----------------------------- CẤU HÌNH -------------------------------------
DATASETS="${DATASETS:-occ_duke p_duke}"            # dataset sẽ inference
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/data}"           # nơi chứa dataset
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO_DIR/weights}"    # nơi chứa checkpoint COPE
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/.cache_downloads}" # file zip tải về
ENV_NAME="${ENV_NAME:-cope}"                       # tên môi trường conda
CUDA_TAG="${CUDA_TAG:-cu121}"                      # cu118 / cu121 / cu124 tuỳ driver
GPU="${GPU:-0}"                                    # GPU dùng để chạy
DUKE_ZIP="${DUKE_ZIP:-}"                           # đường dẫn DukeMTMC-reID.zip (nếu có sẵn trên server)
DUKE_GDRIVE_ID="${DUKE_GDRIVE_ID:-}"               # hoặc ID file DukeMTMC-reID.zip trên Google Drive của bạn
CKPT_OCC_DUKE="${CKPT_OCC_DUKE:-}"                 # (tuỳ chọn) chỉ định thẳng checkpoint
CKPT_P_DUKE="${CKPT_P_DUKE:-}"
TEST_BATCH="${TEST_BATCH:-}"                       # (tuỳ chọn) TEST.IMS_PER_BATCH nếu thiếu VRAM
NUM_WORKERS="${NUM_WORKERS:-8}"

COPE_WEIGHTS_URL="https://drive.google.com/drive/folders/1ekVtlmAv_3mkgQUIM9db6wqCrr0--rNB"
PDUKE_REPO="https://github.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets"
PDUKE_URLS="https://raw.githubusercontent.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets/master/P-DukeMTMC-reid.zip $PDUKE_REPO/raw/master/P-DukeMTMC-reid.zip"
OCCDUKE_REPO="https://github.com/lightas/Occluded-DukeMTMC-Dataset.git"
# -----------------------------------------------------------------------------

mkdir -p logs "$DATA_ROOT" "$WEIGHTS_DIR" "$CACHE_DIR"
RUN_LOG="logs/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

step() { echo; echo "==================== $* ===================="; }
info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
die()  { echo "[ERROR] $*"; exit 1; }

# =============================== 1. MÔI TRƯỜNG ===============================
step "1/6  Môi trường Python"
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"; info "Dùng Python chỉ định: $PY"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
    if [ ! -x "$PY" ]; then
        info "Tạo conda env '$ENV_NAME' (python 3.10, kênh conda-forge)"
        conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.10
    else
        info "conda env '$ENV_NAME' đã có"
    fi
elif command -v python3.10 >/dev/null 2>&1; then
    PY="$REPO_DIR/.venv/bin/python"
    [ -x "$PY" ] || { info "Tạo venv .venv"; python3.10 -m venv "$REPO_DIR/.venv"; }
else
    die "Không có conda và không có python3.10. Cài Miniconda rồi chạy lại."
fi

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    info "SKIP_INSTALL=1 -> bỏ qua cài thư viện"
else
    info "Cài thư viện (requirements.txt, CUDA tag = $CUDA_TAG)"
    sed "s#/whl/cu[0-9]*#/whl/$CUDA_TAG#" requirements.txt > "$CACHE_DIR/requirements.$CUDA_TAG.txt"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q -r "$CACHE_DIR/requirements.$CUDA_TAG.txt" gdown
fi
"$PY" - <<'EOF'
import torch
print('[INFO] torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
EOF

# ================================ 2. CLIP =====================================
step "2/6  Trọng số CLIP ViT-B/16"
"$PY" -c "import clip.clip as c; print('[INFO] CLIP at', c._download(c._MODELS['ViT-B-16']))" \
    || die "Không tải được CLIP. Copy ViT-B-16.pt vào ~/.cache/clip/ rồi chạy lại."

# ============================ 3. CHECKPOINT COPE ==============================
step "3/6  Checkpoint COPE"
if [ -n "$(find "$WEIGHTS_DIR" -name '*.pth' -print -quit)" ]; then
    info "Đã có checkpoint trong $WEIGHTS_DIR, bỏ qua tải"
else
    info "Tải thư mục checkpoint của tác giả bằng gdown"
    "$PY" -m gdown --folder "$COPE_WEIGHTS_URL" -O "$WEIGHTS_DIR" \
        || warn "gdown lỗi (thường do Google Drive giới hạn lượt tải). Tải tay thư mục $COPE_WEIGHTS_URL vào $WEIGHTS_DIR rồi chạy lại."
fi
find "$WEIGHTS_DIR" -name '*.pth' -exec ls -lh {} \; || true

# ================================ 4. DATASET ==================================
count() { find "$1" -type f \( -name '*.jpg' -o -name '*.tif' -o -name '*.png' \) 2>/dev/null | wc -l; }

prepare_p_duke() {
    local dst="$DATA_ROOT/P-DukeMTMC-reid"
    if [ "$(count "$dst/train")" -ge 12927 ] && [ "$(count "$dst/test")" -ge 11216 ]; then
        info "P-DukeMTMC-reID đã sẵn sàng"; return 0
    fi
    local zip="$CACHE_DIR/P-DukeMTMC-reid.zip"
    if [ ! -s "$zip" ]; then
        info "Tải P-DukeMTMC-reID từ GitHub (~100MB)"
        for u in $PDUKE_URLS; do
            curl -L --fail --retry 3 -o "$zip.part" "$u" && mv "$zip.part" "$zip" && break
            warn "Không tải được từ $u"
        done
        if [ ! -s "$zip" ]; then
            info "Thử cách khác: git clone repo dataset (~215MB)"
            rm -rf "$CACHE_DIR/pduke_repo"
            git clone -q --depth 1 "$PDUKE_REPO.git" "$CACHE_DIR/pduke_repo" \
                && mv "$CACHE_DIR/pduke_repo/P-DukeMTMC-reid.zip" "$zip" && rm -rf "$CACHE_DIR/pduke_repo" \
                || { warn "Tải P-Duke thất bại"; return 1; }
        fi
    fi
    info "Giải nén vào $dst"
    mkdir -p "$dst" && unzip -q -o "$zip" -d "$dst" || { warn "Giải nén P-Duke lỗi (xoá $zip rồi chạy lại)"; return 1; }
    info "P-Duke: train $(count "$dst/train") ảnh, test $(count "$dst/test") ảnh (mong đợi 12927 / 11216)"
    [ "$(count "$dst/train")" -ge 12927 ] && [ "$(count "$dst/test")" -ge 11216 ]
}

prepare_occ_duke() {
    local dst="$DATA_ROOT/Occluded_Duke"
    if [ "$(count "$dst/bounding_box_train")" -ge 15618 ] && [ "$(count "$dst/query")" -ge 2210 ] \
       && [ "$(count "$dst/bounding_box_test")" -ge 17661 ]; then
        info "Occluded-Duke đã sẵn sàng"; return 0
    fi
    local zip=""
    if [ -n "$DUKE_ZIP" ]; then
        zip="$DUKE_ZIP"
    elif [ -s "$CACHE_DIR/DukeMTMC-reID.zip" ]; then
        zip="$CACHE_DIR/DukeMTMC-reID.zip"
    elif [ -n "$DUKE_GDRIVE_ID" ]; then
        info "Tải DukeMTMC-reID.zip từ Google Drive (id=$DUKE_GDRIVE_ID)"
        "$PY" -m gdown "https://drive.google.com/uc?id=$DUKE_GDRIVE_ID" -O "$CACHE_DIR/DukeMTMC-reID.zip" \
            || { warn "gdown không tải được file Duke (kiểm tra quyền chia sẻ 'Anyone with the link')"; return 1; }
        zip="$CACHE_DIR/DukeMTMC-reID.zip"
    else
        warn "Occluded-Duke KHÔNG có link tải ảnh trực tiếp: repo lightas chỉ cung cấp danh sách tên ảnh,"
        warn "ảnh phải sinh từ file gốc DukeMTMC-reID.zip (đã bị Duke gỡ khỏi trang chính thức)."
        warn "Hãy chạy lại với DUKE_ZIP=/duong/dan/DukeMTMC-reID.zip hoặc DUKE_GDRIVE_ID=<id>. Bỏ qua Occluded-Duke."
        return 1
    fi
    [ -s "$zip" ] || { warn "Không thấy file $zip"; return 1; }
    info "Sinh Occluded-Duke từ $zip bằng script của tác giả (lightas)"
    local work="$CACHE_DIR/occduke_work"
    rm -rf "$work"; git clone -q --depth 1 "$OCCDUKE_REPO" "$work" || { warn "Không clone được $OCCDUKE_REPO"; return 1; }
    ln -sf "$(readlink -f "$zip")" "$work/DukeMTMC-reID.zip"     # script yêu cầu đúng tên file này
    ( cd "$work" && "$PY" convert_duke_to_occduke.py DukeMTMC-reID.zip ) || { warn "Chuyển đổi Occluded-Duke lỗi"; return 1; }
    rm -rf "$dst"; mv "$work/Occluded_Duke" "$dst"; rm -f "$dst"/*.list; rm -rf "$work"
    info "Occ-Duke: train $(count "$dst/bounding_box_train"), query $(count "$dst/query"), gallery $(count "$dst/bounding_box_test") (mong đợi 15618 / 2210 / 17661)"
    [ "$(count "$dst/bounding_box_train")" -ge 15618 ] && [ "$(count "$dst/query")" -ge 2210 ] \
        && [ "$(count "$dst/bounding_box_test")" -ge 17661 ]
}

step "4/6  Dataset"
READY=""
for ds in $DATASETS; do
    case "$ds" in
        p_duke)   prepare_p_duke   && READY="$READY p_duke"   || warn "P-Duke chưa sẵn sàng" ;;
        occ_duke) prepare_occ_duke && READY="$READY occ_duke" || true ;;
        *) warn "Script này hỗ trợ occ_duke, p_duke. Bỏ qua '$ds'" ;;
    esac
done
[ -n "$READY" ] || die "Không có dataset nào sẵn sàng để inference."

# =============================== 5. INFERENCE =================================
step "5/6  Inference"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || die "PyTorch không thấy GPU (kiểm tra nvidia-smi / CUDA_TAG)."
export CUDA_VISIBLE_DEVICES="$GPU"

declare -A CFG=( [occ_duke]="configs/OCC_Duke/cope.yml" [p_duke]="configs/P_Duke/cope.yml" )
declare -A KEY=( [occ_duke]="occ" [p_duke]="p_duke" )
declare -A PAPER=( [occ_duke]="82.1 / 75.4" [p_duke]="93.3 / 88.4" )
declare -A USER_CKPT=( [occ_duke]="$CKPT_OCC_DUKE" [p_duke]="$CKPT_P_DUKE" )
DONE=""

for ds in $READY; do
    cfg="${CFG[$ds]}"
    ncls="$("$PY" - "$cfg" "$DATA_ROOT" <<'EOF' | tail -n1
import sys, io, contextlib
sys.path.insert(0, '.')
from config import cfg
from cope.dataloader import FACTORY
cfg.merge_from_file(sys.argv[1])
with contextlib.redirect_stdout(io.StringIO()):
    d = FACTORY[cfg.DATASETS.NAMES](root=sys.argv[2])
print(d.num_train_pids)
EOF
)"
    info "[$ds] số ID tập train = $ncls"

    stride=16
    if [ -n "${USER_CKPT[$ds]}" ]; then
        ckpt="${USER_CKPT[$ds]}"
    elif out="$("$PY" tools/find_checkpoint.py --dir "$WEIGHTS_DIR" --num_classes "$ncls" --keyword "${KEY[$ds]}")"; then
        ckpt="$(echo "$out" | sed -n 1p)"; stride="$(echo "$out" | sed -n 2p)"
    else
        warn "[$ds] Không tìm được checkpoint ($ncls lớp) trong $WEIGHTS_DIR. Đặt CKPT_$(echo "$ds" | tr a-z A-Z)=... rồi chạy lại."
        continue
    fi
    info "[$ds] checkpoint = $ckpt (stride $stride)"

    out_dir="logs/${ds}_inference"
    extra=()
    [ -n "$TEST_BATCH" ] && extra+=(TEST.IMS_PER_BATCH "$TEST_BATCH")
    "$PY" test_cope.py --config_file "$cfg" \
        TEST.WEIGHT "$ckpt" DATASETS.ROOT_DIR "$DATA_ROOT" OUTPUT_DIR "$out_dir" \
        MODEL.STRIDE_SIZE "[$stride, $stride]" DATALOADER.NUM_WORKERS "$NUM_WORKERS" ${extra[@]+"${extra[@]}"} \
        || { warn "[$ds] test_cope.py lỗi, xem $out_dir/test_log.txt"; continue; }
    DONE="$DONE $ds"
done

# ================================ 6. TỔNG HỢP =================================
step "6/6  Tổng hợp"
{
    echo "COPE inference summary  ($(date '+%Y-%m-%d %H:%M'))"
    printf "%-10s | %-8s | %-8s | %-8s | %-14s | %s\n" "Dataset" "Rank-1" "Rank-5" "mAP" "Paper R1/mAP" "Log"
    for ds in $DONE; do
        f="logs/${ds}_inference/test_log.txt"
        r1=$(grep -oP 'Rank-1\s*:\K[0-9.]+%' "$f" | tail -1)
        r5=$(grep -oP 'Rank-5\s*:\K[0-9.]+%' "$f" | tail -1)
        map=$(grep -oP 'mAP: \K[0-9.]+%' "$f" | tail -1)
        printf "%-10s | %-8s | %-8s | %-8s | %-14s | %s\n" "$ds" "$r1" "$r5" "$map" "${PAPER[$ds]}" "$f"
    done
} | tee logs/summary.txt
info "Log toàn bộ lần chạy: $RUN_LOG"
