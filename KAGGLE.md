# Chạy trên Kaggle (2× T4)

Có hai notebook, mỗi mô hình một notebook, chạy **lần lượt**:
1. `vi_sred26_teacher_xlsr300m.ipynb`
2. `vi_sred26_teacher_wavlm_large.ipynb`

## Chuẩn bị
- **Accelerator:** GPU T4 x2.
- **Internet:** bật, để clone repo và tải checkpoint từ Hugging Face (khoảng 1,2 GB mỗi mô hình).
- **Dataset:** add `vi-sred26-train`, `vi-sred26-publictest`, `musan-noise`, `rirs-noises`.
- **Chạy:** Save Version → Save & Run All.

## Notebook làm gì
1. Clone repo và kiểm tra dữ liệu (speaker của tập val không được có trong train), rồi tải checkpoint HF một lần.
2. Với mỗi stage của recipe (`RECIPE = 'slt22'`: main → lm; `'v4'`: frozen → ft → lmft):
   - chạy `probe_memory.py` để tìm batch mỗi GPU lớn nhất vừa bộ nhớ;
   - chọn `batch_size` × `accum_steps` bằng đúng batch hiệu dụng trong config;
   - nếu không vừa, tự bật gradient checkpointing.
3. Train từng stage bằng `torchrun` (2 tiến trình). Stage sau khởi tạo từ `last.ckpt` của stage trước. Log được ghi vào `exps/<ssl>_<recipe>_<stage>/train.log` và hiện trực tiếp trong notebook.
4. Vẽ đường cong loss, accuracy và EER trên val.
5. Chọn stage và epoch có EER val tốt nhất, rồi:
   - đánh giá chính xác trên val theo cách đánh giá của recipe;
   - chấm public test bằng cả `cos` và `center`.
6. Ghi kết quả:
   - `submission/results_<ssl>_<recipe>.zip`: file nộp;
   - `exps/teacher_<ssl>_<recipe>.model`: checkpoint thầy, gồm backbone, MHFA và lớp phân loại 5.130 lớp;
   - `exps/summary_<ssl>_<recipe>.json`.

## Thời gian
Notebook tự tính `--max_hours` cho từng stage và để dành `INFER_HOURS` (1,5 h) cho bước chấm điểm. Nếu hết 12 giờ:
1. Add output của version vừa chạy làm input.
2. Đặt `RESUME_DIR = '/kaggle/input/<tên>/exps'`.
3. Chạy lại. Notebook copy checkpoint sang và resume đúng stage, đúng epoch.

Thời gian mỗi epoch phụ thuộc batch và gradient checkpointing. Xem dòng `min/epoch` trong log của epoch đầu để ước lượng. Nếu không đủ thời gian, có thể giảm số epoch bằng `OVERRIDES = {'main': dict(max_epoch=...)}`; khi đó cần ghi rõ đây là thay đổi so với recipe gốc.

## Các biến cấu hình
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `SSL` | theo notebook | `xlsr300m` / `wavlm_large` |
| `RECIPE` | `slt22` | `slt22` / `v4` (xem README mục 3) |
| `INPUT_NORM` | `None` | `None` = theo config (`utterance`); `'none'` = waveform thô như code MHFA |
| `OVERRIDES` | `{}` | ghi đè tuỳ chọn cho từng stage |
| `BATCH_OVERRIDE` | `None` | đặt tay `(batch_size, accum_steps)` cho từng stage, bỏ qua bước probe |
| `GRAD_CHECKPOINTING` | `'auto'` | `True` / `False` / `'auto'` |
| `INFER_AMP` | `False` | fp16 khi chấm điểm |
| `RESUME_DIR` | `''` | thư mục `exps` của lần chạy trước |
