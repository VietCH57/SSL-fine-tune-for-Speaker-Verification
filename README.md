# SSL fine-tune for Speaker Verification (Vi-SRED26)

Repo này fine-tune **XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`) và **WavLM-Large** (`microsoft/wavlm-large`) với back-end **MHFA** (Multi-Head Factorized Attentive pooling) trên Vi-SRED26. Hai mô hình này làm **thầy** để distill sang ReDimNet-M (repo `redimnet-vi-sred`).

Công thức huấn luyện chép từ code chính thức của MHFA. Không có giá trị nào tự đặt ra: mỗi dòng trong `configs/` đều ghi nguồn. Những điểm khác bản gốc được liệt kê đầy đủ ở [mục 4](#4-khác-biệt-so-với-code-chính-thức).

## 1. Nguồn

| | Nguồn | Dùng cho |
|---|---|---|
| Bài báo | J. Peng, O. Plchot, T. Stafylakis, L. Mošner, L. Burget, J. Černocký, *An attention-based backend allowing efficient fine-tuning of transformer models for speaker verification*, IEEE SLT 2022 (tổ chức đầu năm 2023) | kiến trúc MHFA, các con số tham chiếu |
| Code SLT22 | https://github.com/JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling | recipe `slt22` (`trainSpeakerNet.py`, `SpeakerNet.py`, `models/Baseline/Spk_Encoder.py`, `loss/aamsoftmax.py`, `DatasetLoader.py`, `yaml/Baseline.yaml`) |
| Code wespeaker_ssl_public | https://github.com/BUTSpeechFIT/wespeaker_ssl_public, nhánh `hubert`, `examples/voxceleb/v4` | recipe `v4` (`conf/MHFA/MHFA_Large_{Frozen, Frozen_FT, Frozen_FT_LMFT}.yaml`, `wespeaker/models/ssl_backend.py`, `wespeaker/models/projections.py`, `wespeaker/utils/schedulers.py`, `run_MHFA_Large.sh`) |
| Checkpoint | Hugging Face `transformers` (`Wav2Vec2Model`, `WavLMModel`), dùng `config.json` và `preprocessor_config.json` của từng checkpoint | backbone |

Cả hai bộ code gốc chỉ có recipe cho WavLM / HuBERT. **XLS-R chưa có recipe MHFA nào được công bố**, nên ở đây áp nguyên công thức của bản Large cho XLS-R (cùng 24 layer × 1024).

## 2. Mô hình

- **Biểu diễn từng layer:** lấy giống `WavLM.extract_features(..., output_layer=L, ret_layer_results=True)` mà MHFA gọi:
  - chỉ số 0 = đầu vào transformer (sau positional conv và dropout);
  - chỉ số i = đầu ra thô của layer i, **không** qua LayerNorm cuối;
  - layer bị LayerDrop bỏ qua thì lặp lại biểu diễn trước đó.

  Các biểu diễn được lấy bằng forward hook, nên không phụ thuộc cách `hidden_states` được dựng ở từng phiên bản `transformers` (bản 4.x và 5.x khác nhau ở layer cuối).
- **Masking (SpecAugment của pre-training):** tắt, tương đương `mask=False` trong code MHFA. LayerDrop giữ giá trị của checkpoint (`slt22`) hoặc đặt 0,05 (`v4`, `layer_drop`).
- **MHFA:** công thức và tên tham số giống hệt code gốc: `weights_k`, `weights_v` (25 trọng số layer), `cmp_linear_k/v` 1024→128, `att_head` 128→64 head, `pooling_fc` 64·128→256.
- **Đầu vào:** `input_norm: utterance` chuẩn hoá mean 0 / var 1 cho từng câu (`do_normalize: true` trong `preprocessor_config.json` của cả hai checkpoint). Code MHFA đưa waveform thô vào; đặt `input_norm: none` nếu muốn theo code MHFA.
- **File `.model`:** chứa config và trọng số HF, MHFA, **lớp phân loại 5.130 lớp** (để distill logit) và vector mean dùng cho chấm `center`. Nạp lại không cần mạng (`models.load_encoder`).

## 3. Hai recipe

### `slt22`: code của bài báo (`configs/slt22/`)

| | `main` (`<ssl>.yaml`) | `lm` (`<ssl>_lm.yaml`) |
|---|---|---|
| Phần được train | `encoder.layers` + MHFA + head (CNN chạy `no_grad`) | như `main` |
| Optimizer | AdamW, weight decay 0,01 (mặc định của torch: `optimizer/adamw.py` chỉ truyền `lr`) | như `main` |
| LR | MHFA + head 5e-3; mỗi layer 2e-5 × `LLRD_factor`^i, với `LLRD_factor` = 1 | tiếp tục chuỗi giảm (`lr_epoch_offset: 15`) |
| Lịch LR | StepLR γ 0,95 mỗi epoch | như `main` |
| Điều chuẩn | `nloss/nloss.detach() + 0.01·l2/(l2.detach()+1e-5)`, với l2 = Σ‖θ−θ_pretrained‖₂ | như `main` |
| Loss | AAM-softmax, m 0,2, s 30 | m 0,5 |
| Crop | `max_frames` 300 → 48.240 mẫu (3,015 s) | 600 → 6,015 s |
| Batch | 120 | 40 |
| Số epoch | 15 | 3 |
| Augment | kiểu voxceleb_trainer, 6 loại đều nhau (clean / reverb / noise / speech / music / tv) | như `main` |
| Đánh giá | trung bình điểm cả câu (≤ 700.000 mẫu = 43,75 s) và 5 đoạn 4,015 s | như `main` |

Số epoch và margin của LMF lấy theo bài báo (thêm 3 epoch, m 0,5); độ dài crop 6,015 s lấy theo code (`max_frames` 600).

### `v4`: bản viết lại của cùng tác giả trong wespeaker (`configs/v4/`)

| | `frozen` | `ft` | `lmft` |
|---|---|---|---|
| Backbone | đóng băng (`feature_grad_mult` 0) | train toàn bộ, `feature_grad_mult` 0,1 | như `ft` |
| Optimizer | SGD, momentum 0,9, weight decay 1e-7 | AdamW, weight decay 1e-7 | AdamW |
| LR (ExponentialDecrease, × `scale_ratio`) | 0,1 → 5e-3, warm-up 3 epoch từ 0 | 5e-5 → 1e-6, warm-up 1 | 1e-5 → 1e-7, warm-up 1 |
| Head | ArcMargin inter-top-k sub-center (K 3, k_top 5, mp 0,06), s 32 | như `frozen` | như `frozen` |
| Margin | 0 → 0,2 (tăng từ epoch 2, cố định từ epoch 5) | 0,2 | 0,5 |
| Crop | 3,0 s | 3,0 s | 3,0 s (config gốc không đổi `num_frms`) |
| Batch hiệu dụng | 512 | 128 | 64 |
| Số epoch | 10 | 5 | 3 |
| Augment | `aug_prob` 0,6: reverb hoặc một file MUSAN; peak-normalise; lọc câu 1–8 s | như `frozen` | như `frozen` |

- **Batch:** bản gốc dùng 4 GPU với batch mỗi GPU lần lượt 128 / 32 / 16. Ở đây `batch_size` là batch hiệu dụng tương ứng, và LR được nhân với `scale_ratio` = batch hiệu dụng / 64, đúng như `train_hubert.py`.
- **Đánh giá:** cosine sau khi trừ mean của embedding tập train (`select_score: center`).

## 4. Khác biệt so với code chính thức

| Khác biệt | Lý do |
|---|---|
| **Speed perturb 0,9 / 1,1 → 5.130 lớp** (cả hai recipe gốc đều không dùng; v4 ghi rõ `speed_perturb: False`) | để thầy và trò ReDimNet-M có cùng bộ lớp khi distill logit |
| **AMP fp16** (SLT22 có `--mixedprec`; v4 đặt `enable_amp: False`) | T4 chạy fp32 chậm khoảng 2,6 lần (đã đo bằng benchmark notebook) |
| **Batch đạt được bằng cộng dồn gradient** trên 2 GPU (`batch_size` × `accum_steps`) thay vì nhiều GPU | phần cứng. Cộng dồn gradient cho cùng gradient trung bình. Với `slt22`, 120 là giá trị trong yaml (gốc là batch mỗi GPU; số GPU của bài báo không được ghi trong repo). |
| **`input_norm: utterance`** | theo `preprocessor_config.json` của checkpoint HF; có thể tắt |
| **Không có sampler "mỗi speaker tối đa một câu trong batch"** của SLT22 | dùng chung sampler của repo ReDimNet |
| **Không lấy trung bình mô hình (`num_avg`) và không có AS-norm** của v4 | chưa cài đặt; chọn `best.model` theo EER trên tập val |
| **XLS-R dùng công thức của WavLM/HuBERT** | không có recipe MHFA được công bố cho XLS-R |
| `sine` được kẹp trong [0, 1] trước khi lấy căn trong lớp ArcMargin | tránh NaN khi chạy fp16; không đổi kết quả khi tính chính xác |

## 5. Cách dùng

```bash
pip install -r requirements.txt
# 1 GPU
python train.py --config configs/slt22/xlsr300m.yaml
# 2 GPU
torchrun --standalone --nproc_per_node 2 train.py --config configs/slt22/xlsr300m.yaml --batch_size 8 --accum_steps 15
# stage sau khởi tạo từ last.ckpt của stage trước
torchrun --standalone --nproc_per_node 2 train.py --config configs/slt22/xlsr300m_lm.yaml \
         --initial_model /kaggle/working/exps/xlsr300m_slt22/model/last.ckpt
```

- **Chọn batch:** `python probe_memory.py --ssl xlsr300m --ssl_trainable layers --crop_sec 3.015 --target_batch 120` gợi ý `--batch_size` / `--accum_steps` để batch hiệu dụng đúng bằng config. Nếu không vừa bộ nhớ, thêm `--grad_checkpointing 1`.
- **Dừng và chạy tiếp:** `--max_hours` dừng sạch trước khi hết giờ; chạy lại cùng lệnh sẽ resume từ `last.ckpt`.
- **Chấm điểm:**

  ```bash
  python infer.py --mode eval  --model exps/.../best.model --trials val_trials.txt --data_path <train data> --eval_mode both --eval_sec 4.015 --eval_max_sec 43.75
  python infer.py --mode score --model exps/.../best.model --trials public-test.txt --data_path <test data> --score_norm center --zip
  ```
- **Smoke test:** `bash tests/smoke_test.sh` chạy trên CPU vài phút, dùng dữ liệu giả và mô hình HF nhỏ khởi tạo ngẫu nhiên. Nó chạy torchrun 2 tiến trình qua cả hai recipe, resume và infer.

Notebook Kaggle: xem [KAGGLE.md](KAGGLE.md).

## 6. Cấu trúc

| File | Nội dung |
|---|---|
| `models.py` | nạp backbone HF, hook lấy biểu diễn từng layer, MHFA, `SSLEncoder`, export / `load_encoder` |
| `loss.py` | `AAMsoftmax` (SLT22), `ArcMarginIntertopkSubcenter` (wespeaker) |
| `trainer.py` | optimizer và lịch LR của hai recipe, bộ điều chuẩn SLT22, margin scheduler, DDP, cộng dồn gradient, AMP, đánh giá phân tán |
| `train.py` | vòng lặp epoch, resume, `--max_hours`, `--initial_model` |
| `config.py` | toàn bộ tuỳ chọn (YAML + dòng lệnh) |
| `dataLoader.py`, `tools.py`, `inference.py`, `dist_utils.py` | dùng chung với repo `redimnet-vi-sred` |
| `infer.py` | đánh giá / chấm điểm / trích embedding |
| `probe_memory.py` | đo batch tối đa theo bộ nhớ GPU |
| `configs/slt22`, `configs/v4` | hai recipe × hai mô hình |
