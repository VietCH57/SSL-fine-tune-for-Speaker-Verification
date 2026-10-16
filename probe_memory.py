'''
Largest per-GPU training batch that fits, measured with a FULL training step as train.py runs it:
forward (fp16 autocast) + head + SLT22 regulariser (reference weights on the GPU) + backward
+ optimizer step (AdamW / SGD states allocated) + DDP gradient buckets (one extra fp32 copy of the
trainable gradients when n_gpu > 1).

    python probe_memory.py --ssl xlsr300m --ssl_trainable layers --crop_sec 3.015 --n_gpu 2 --target_batch 120 \
                           --optimizer adamw --reg slt22
'''

import argparse, math, torch
import models
from loss import AAMsoftmax


def train_step(enc, head, params, opt, ref, b, n, amp, reg_weight=0.01):
    x = torch.randn(b, n, device='cuda') * 0.1
    y = torch.randint(0, head.weight.shape[0], (b,), device='cuda')
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        e = enc(x)
    loss, _, _ = head(e.float(), y)
    if ref:
        l2 = sum(torch.norm(p.float() - p0, 2) for p, p0 in ref)
        loss = loss / loss.detach() + reg_weight * l2 / (l2.detach() + 1e-5)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()
    opt.zero_grad(set_to_none=True)


def fits(*args):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        train_step(*args)
        return torch.cuda.max_memory_allocated() / 2 ** 30
    except torch.cuda.OutOfMemoryError:
        args[3].zero_grad(set_to_none=True)
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
    p.add_argument('--optimizer', default='adamw', choices=['adamw', 'sgd'])
    p.add_argument('--reg', default='none', choices=['none', 'slt22'])
    p.add_argument('--n_gpu', type=int, default=0)
    p.add_argument('--target_batch', type=int, default=120)
    p.add_argument('--margin', type=float, default=0.85, help='Use at most this fraction of GPU memory (peak of a full step)')
    p.add_argument('--candidates', type=str, default='4,6,8,10,12,16,20,24,32,40,48,64')
    a = p.parse_args()
    assert torch.cuda.is_available(), 'needs a GPU'
    n_gpu = a.n_gpu or torch.cuda.device_count()
    ssl = models.load_hf_ssl(a.ssl, a.hf_cache or None)
    enc = models.SSLEncoder(ssl, head_nb=a.head_nb, ssl_trainable=a.ssl_trainable, grad_checkpointing=bool(a.grad_checkpointing),
                            meta={'ssl': a.ssl}).cuda().train()
    models._report(enc)
    head = AAMsoftmax(a.n_class, enc.embed_dim).cuda()
    params = [p for p in list(enc.parameters()) + list(head.parameters()) if p.requires_grad]
    opt = (torch.optim.AdamW(params, lr=0.0) if a.optimizer == 'adamw'
           else torch.optim.SGD(params, lr=0.0, momentum=0.9, nesterov=True))
    for q in params:                           # allocate the optimizer states once (lr 0: weights unchanged)
        q.grad = torch.zeros_like(q)
    opt.step(); opt.zero_grad(set_to_none=True)
    ref = ([(q, q.detach().float().clone()) for n_, q in enc.ssl.named_parameters() if q.requires_grad]
           if a.reg == 'slt22' and a.ssl_trainable != 'none' else [])
    grad_gb = sum(q.numel() for q in params) * 4 / 2 ** 30
    buckets = torch.empty(int(grad_gb * 2 ** 30 / 4), device='cuda') if n_gpu > 1 else None   # DDP reducer buckets
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print('trainable %.1fM | optimizer %s | SLT22 reference %s | DDP buckets %.2f GB'
          % (sum(q.numel() for q in params) / 1e6, a.optimizer, 'yes' if ref else 'no', grad_gb if buckets is not None else 0))
    n = int(round(a.crop_sec * 16000))
    best = None
    for b in sorted({int(x) for x in a.candidates.split(',')}):
        gb = fits(enc, head, params, opt, ref, b, n, bool(a.amp))
        print('per-GPU batch %3d x %.2f s: %s' % (b, a.crop_sec, 'OOM' if gb is None else '%.2f GB peak (full step)' % gb), flush=True)
        if gb is None or gb > a.margin * total:
            break
        best = b
    if best is None:
        print('Even the smallest batch does not fit: try --grad_checkpointing 1'); return
    per_step = best * n_gpu
    accum = max(1, math.ceil(a.target_batch / per_step))
    bs = min(per_step, int(math.ceil(a.target_batch / accum / n_gpu)) * n_gpu)
    print('\nGPU %.1f GB | largest per-GPU batch within %.0f%%: %d' % (total, 100 * a.margin, best))
    print('Suggested: --batch_size %d --accum_steps %d  (effective %d on %d GPU)' % (bs, accum, bs * accum, n_gpu))


if __name__ == '__main__':
    main()
