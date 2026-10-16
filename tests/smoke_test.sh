#!/usr/bin/env bash
# End-to-end smoke test on synthetic data and tiny random SSL models (CPU, torchrun with 2 processes, a few minutes):
#   SLT22: main -> lm (initial_model = main last.ckpt) with tiny wav2vec2, resume
#   v4   : frozen -> ft with tiny WavLM
#   infer: exact val eval + public-test scoring with center normalisation
set -euo pipefail
cd "$(dirname "$0")/.."
R=${1:-/tmp/ssl_sv_smoke}
rm -rf "$R"; python tests/make_fake_data.py "$R"; python tests/make_tiny_ssl.py "$R/tiny"
K=$R/kaggle/input/datasets; T=$K/hongvitchugo/vi-sred26-train; P=$K/hongvitchugo/vi-sred26-publictest
DATA="--train_list $T/train_split.txt --train_path $T/Vi-SRED26-Train/data --eval_list $T/val_trials.txt --eval_path $T/Vi-SRED26-Train/data
      --musan_path $K/nhattruongdev/musan-noise --rir_path $K/nhattruongdev/rirs-noises
      --device cpu --amp false --n_cpu 2 --batch_size 4 --accum_steps 2 --crop_sec 1.0 --cohort_size 20 --log_interval 100
      --eval_batch 2 --eval_max_sec 3 --eval_sec 1 --num_eval 2 --head_nb 4 --compression_dim 8 --embed_dim 16"
RUN="python -m torch.distributed.run --standalone --nproc_per_node 2 train.py"
W2V=$R/tiny/tiny_w2v2; WLM=$R/tiny/tiny_wavlm

$RUN --config configs/slt22/xlsr300m.yaml    $DATA --ssl $W2V --save_path $R/slt22_main --max_epoch 1
$RUN --config configs/slt22/xlsr300m.yaml    $DATA --ssl $W2V --save_path $R/slt22_main --max_epoch 2      # resumes
$RUN --config configs/slt22/xlsr300m_lm.yaml $DATA --ssl $W2V --save_path $R/slt22_lm   --max_epoch 1 \
     --initial_model $R/slt22_main/model/last.ckpt
$RUN --config configs/v4/wavlm_large_frozen.yaml $DATA --ssl $WLM --save_path $R/v4_frozen --max_epoch 1
$RUN --config configs/v4/wavlm_large_ft.yaml     $DATA --ssl $WLM --save_path $R/v4_ft     --max_epoch 1 \
     --initial_model $R/v4_frozen/model/last.ckpt

python infer.py --mode eval  --model $R/slt22_lm/model/best.model --trials $T/val_trials.txt --data_path $T/Vi-SRED26-Train/data \
                --out $R/val/results.txt --device cpu --eval_mode both --eval_sec 1 --num_eval 2 --eval_max_sec 3 --eval_batch 1
python infer.py --mode score --model $R/v4_ft/model/best.model --trials $P/public-test.txt --data_path $P/data \
                --out $R/sub/results.txt --zip --device cpu --score_norm center --eval_batch 1
unzip -l $R/sub/results.zip
echo SMOKE TEST OK
