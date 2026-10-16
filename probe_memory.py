'''
Largest per-GPU training batch that fits (forward + backward of SSL + MHFA + head, fp16 autocast),
for the trainable setting of a stage, with or without activation checkpointing.

    python probe_memory.py --ssl xlsr300m --ssl_trainable layers --crop_sec 3.015 --n_gpu 2 --target_batch 120
'''

import argparse, math, torch
import models
from loss import AAMsoftmax


def fits(enc, head, b, n, amp, train_ssl):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randn(b, n, device='cuda') * 0.1
        y = torch.randint(0, head.weight.shape[0], (b,), device='cuda')
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            e = enc(x)
        loss, _, _ = head(e.float(), y)
        loss.backward()
        enc.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated() / 2 ** 30
    except torch.cuda.OutOfMemoryError:
        enc.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ssl', default='wavlm_large')
    p.add_argument('--hf_cache', default='')
    p.add_argument('--ssl_trainable', default='layers', choices=['none', 'layers', 'all'])
    p.add_argument('--grad_checkpointing', type=int, default=0)
    p.add_argument('--crop_sec', type=float, default=3.015)
    p.add_argument('--head_nb', type=int, default=64)
    p.add_argument('--n_class', type=int, default=5130)
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--n_gpu', type=int, default=0)
    p.add_argument('--target_batch', type=int, default=120)
    p.add_argument('--margin', type=float, default=0.80, help='Use at most this fraction of GPU memory (optimizer states not included: keep headroom)')
    p.add_argument('--candidates', type=str, default='4,6,8,10,12,16,20,24,32,40,48,64')
    a = p.parse_args()
    assert torch.cuda.is_available(), 'needs a GPU'
    ssl = models.load_hf_ssl(a.ssl, a.hf_cache or None)
    enc = models.SSLEncoder(ssl, head_nb=a.head_nb, ssl_trainable=a.ssl_trainable, grad_checkpointing=bool(a.grad_checkpointing),
                            meta={'ssl': a.ssl}).cuda().train()
    models._report(enc)
    head = AAMsoftmax(a.n_class, enc.embed_dim).cuda()
    # optimizer states (AdamW: 2 x fp32) + SLT22 reference copy of the trainable backbone weights
    n_tr = sum(p.numel() for p in enc.parameters() if p.requires_grad) + head.weight.numel()
    extra_gb = n_tr * 4 * (2 + (1 if a.ssl_trainable != 'none' else 0)) / 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    n = int(round(a.crop_sec * 16000))
    best = None
    for b in sorted({int(x) for x in a.candidates.split(',')}):
        gb = fits(enc, head, b, n, bool(a.amp), a.ssl_trainable != 'none')
        need = None if gb is None else gb + extra_gb
        print('per-GPU batch %3d x %.2f s: %s' % (b, a.crop_sec, 'OOM' if gb is None else
              '%.2f GB peak + %.2f GB optimiser/reference = %.2f GB' % (gb, extra_gb, need)), flush=True)
        if need is None or need > a.margin * total:
            break
        best = b
    n_gpu = a.n_gpu or torch.cuda.device_count()
    if best is None:
        print('Even the smallest batch does not fit: try --grad_checkpointing 1'); return
    per_step = best * n_gpu
    accum = max(1, math.ceil(a.target_batch / per_step))
    bs = min(per_step, int(math.ceil(a.target_batch / accum / n_gpu)) * n_gpu)
    print('\nGPU %.1f GB | largest per-GPU batch within %.0f%%: %d' % (total, 100 * a.margin, best))
    print('Suggested: --batch_size %d --accum_steps %d  (effective %d on %d GPU)' % (bs, accum, bs * accum, n_gpu))


if __name__ == '__main__':
    main()
