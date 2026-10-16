'''
Training loop for SSL + MHFA, following the two official MHFA code bases:

SLT22 (github.com/JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling, SpeakerNet.py):
  * optimiser param groups: [MHFA + head: LR_MHFA] + [transformer layer i: LR_Transformer * LLRD_factor^i];
    nothing else of the backbone is optimised (CNN runs under no_grad)
  * torch.optim.AdamW(param_groups) with its default weight decay (0.01)
  * StepLR(step_size=test_interval, gamma=lr_decay), stepped once per epoch
  * loss = nloss/nloss.detach() + reg * l2/(l2.detach()+1e-5), l2 = sum_params ||theta - theta_pretrained||_2

wespeaker_ssl_public (branch hubert, wespeaker/bin/train_hubert.py + utils/executor.py + utils/schedulers.py):
  * one LR for every trainable parameter; ExponentialDecrease per iteration with warm-up,
    multiplied by scale_ratio = world_size * batch_size / 64
  * MarginScheduler (exponential margin increase), CrossEntropy on ArcMargin intertopk+subcenter logits

Common: DDP via torchrun (find_unused_parameters=True as both official codes), fp16 autocast optional.
'''

import contextlib, math, os, sys, time
import numpy, torch
import torch.nn as nn
import torch.nn.functional as F
import models
import dist_utils as du
from loss import build_head
from inference import parse_trials, extract_embeddings, score_trials, mean_embedding, subset_metrics
from tools import compute_metrics


class _EncHead(nn.Module):
    '''encoder + head (+ SLT22 regulariser) in ONE forward, as SpeakerNet.forward in the SLT22 code:
    with DDP every gradient (task loss and regulariser) must come from the wrapped forward.'''

    def __init__(self, enc, head, amp):
        super().__init__()
        self.enc, self.head, self.amp = enc, head, amp
        self.reference, self.reg_weight = None, 0.0

    def forward(self, x, y):
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            e = self.enc(x)
        nloss, acc, logits = self.head(e.float(), y)
        if self.reference is None:
            return nloss, nloss, acc, None
        l2 = sum(torch.norm(p.float() - p0, 2) for p, p0 in self.reference)
        total = nloss / nloss.detach() + self.reg_weight * l2 / (l2.detach() + 1e-5)
        return total, nloss, acc, l2.detach()


class Trainer(nn.Module):
    def __init__(self, args, n_class, init_path=None, steps_per_epoch=1):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device if (torch.cuda.is_available() or not args.device.startswith('cuda')) else 'cpu')
        if init_path:
            self.encoder, self.extra = models.load_encoder(init_path, args.ssl_trainable, args.ssl_grad_mult,
                                                           args.grad_checkpointing, args.layerdrop)
        else:
            self.encoder, self.extra = models.build_encoder(args), {}
        n = self.encoder.n_params()
        if args.max_params and n > args.max_params:
            raise SystemExit('Encoder has %d parameters > --max_params %d' % (n, args.max_params))
        self.encoder.to(self.device)
        self.head = build_head(args, n_class, self.encoder.embed_dim).to(self.device)
        self.n_class = n_class
        self.amp = bool(args.amp) and self.device.type == 'cuda'
        self.ddp = du.is_dist()
        self._combo = [_EncHead(self.encoder, self.head, self.amp)]
        self._ddp_model = None
        self.steps_per_epoch = max(1, steps_per_epoch)        # optimiser steps per epoch (after accumulation)
        self._build_optimizer()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.reference = None
        self.center = None
        if du.is_main():
            print(time.strftime('%m-%d %H:%M:%S') + ' head %s %d x %d | optimiser %s, scheduler %s'
                  % (args.head, n_class, self.encoder.embed_dim, args.optimizer, args.lr_scheduler))

    # ------------------------------------------------------------------ optimiser
    def _build_optimizer(self):
        a = self.args
        enc = self.encoder
        backend = list(enc.backend.parameters()) + list(self.head.parameters())
        if a.lr_scheduler == 'step':     # SLT22 param groups
            groups = [{'params': backend, 'lr': a.lr_mhfa, 'name': 'mhfa+head'}]
            if enc.ssl_trainable != 'none':
                in_groups = set()
                for i, layer in enumerate(enc.ssl.encoder.layers):
                    ps = [p for p in layer.parameters() if p.requires_grad]
                    in_groups.update(id(p) for p in ps)
                    groups.append({'params': ps, 'lr': a.lr_transformer * a.llrd_factor ** i, 'name': 'layer%d' % i})
                rest = [n for n, p in enc.ssl.named_parameters() if p.requires_grad and id(p) not in in_groups]
                if rest:
                    raise SystemExit('lr_scheduler step (SLT22) optimises only transformer layers; use --ssl_trainable layers '
                                     '(trainable outside the layers: %s ...)' % rest[:3])
        else:                            # wespeaker: one LR for all trainable parameters
            params = backend + [p for _, p in enc.ssl_trainable_parameters()]
            groups = [{'params': params, 'lr': a.initial_lr, 'name': 'all'}]
        for g in groups:
            g['base_lr'] = g['lr']
        if a.optimizer == 'adamw':
            self.optim = torch.optim.AdamW(groups, lr=groups[0]['lr'], weight_decay=a.weight_decay)
        else:
            self.optim = torch.optim.SGD(groups, lr=groups[0]['lr'], momentum=a.momentum, nesterov=True, weight_decay=a.weight_decay)
        eff = a.batch_size * max(1, a.accum_steps)
        self.scale_ratio = eff / a.lr_scale_base
        if du.is_main():
            if a.lr_scheduler == 'step':
                print('LR groups: ' + ', '.join('%s %.1e' % (g['name'], g['lr']) for g in groups[:3])
                      + (' ... %s %.1e' % (groups[-1]['name'], groups[-1]['lr']) if len(groups) > 3 else '')
                      + ' | StepLR gamma %.3f / epoch | AdamW wd %g' % (a.lr_decay, a.weight_decay))
            else:
                print('LR: ExponentialDecrease %.1e -> %.1e, warm-up %.1f epoch(s) from %s, scale_ratio %.3f (effective batch %d / %g)'
                      % (a.initial_lr, a.final_lr, a.warm_up_epoch, 'zero' if a.warm_from_zero else 'one', self.scale_ratio,
                         eff, a.lr_scale_base))

    def _set_lr(self, epoch, it_in_epoch):
        '''epoch: 1-based; it_in_epoch: optimiser steps done in this epoch.'''
        a = self.args
        if a.lr_scheduler == 'step':     # StepLR(step_size=test_interval) stepped at the end of each epoch
            f = a.lr_decay ** ((epoch - 1 + a.lr_epoch_offset) // max(1, a.test_step))
            for g in self.optim.param_groups:
                g['lr'] = g['base_lr'] * f
        else:                            # wespeaker BaseClass + ExponentialDecrease
            cur = (epoch - 1) * self.steps_per_epoch + it_in_epoch
            max_iter = a.max_epoch * self.steps_per_epoch
            warm = a.warm_up_epoch * self.steps_per_epoch
            coeff = self.scale_ratio
            if cur < warm:
                if a.warm_from_zero:
                    coeff = self.scale_ratio * cur / warm
                elif self.scale_ratio > 1:
                    coeff = (self.scale_ratio - 1) * cur / warm + 1.0
            lr = coeff * a.initial_lr * math.exp((cur / max_iter) * math.log(a.final_lr / a.initial_lr))
            for g in self.optim.param_groups:
                g['lr'] = lr
        return self.optim.param_groups[0]['lr'], self.optim.param_groups[-1]['lr']

    def _set_margin(self, epoch, it_in_epoch):
        a = self.args
        if not a.m_schedule:
            return
        cur = (epoch - 1) * self.steps_per_epoch + it_in_epoch                  # wespeaker MarginScheduler
        inc_start = (a.m_increase_start_epoch - 1) * self.steps_per_epoch
        fix_start = (a.m_fix_start_epoch - 1) * self.steps_per_epoch
        if cur >= fix_start:
            m = a.m_final
        elif cur >= inc_start:
            ratio = 1.0 - math.exp(((cur - inc_start) / max(fix_start - inc_start, 1)) * math.log(1e-3 / (1.0 + 1e-6))) * 1.0
            m = a.m_initial + (a.m_final - a.m_initial) * ratio
        else:
            m = a.m_initial
        self.head.update(m)

    # ------------------------------------------------------------------ SLT22 regulariser
    def set_reference(self, state=None):
        '''theta_0 = pre-trained weights of the trainable backbone parameters (SLT22 Learned_dict).'''
        if self.args.reg_type == 'none' or self.encoder.ssl_trainable == 'none':
            return
        named = dict(self.encoder.ssl.named_parameters())
        if state is None:
            ref = models.load_hf_ssl(self.args.ssl, self.args.hf_cache or None).state_dict()
        else:
            ref = state
        self.reference = [(p, ref[n].to(self.device, torch.float32)) for n, p in self.encoder.ssl_trainable_parameters()]
        self._combo[0].reference, self._combo[0].reg_weight = self.reference, self.args.reg_weight
        if du.is_main():
            print('SLT22 regulariser: %d tensors, weight %g' % (len(self.reference), self.args.reg_weight))

    # ------------------------------------------------------------------ training
    def _forward_fn(self):
        if not self.ddp:
            return self._combo[0], None
        if self._ddp_model is None:
            from torch.nn.parallel import DistributedDataParallel as DDP
            ids = [self.device.index] if self.device.type == 'cuda' else None
            self._ddp_model = DDP(self._combo[0], device_ids=ids, output_device=ids[0] if ids else None,
                                  find_unused_parameters=True)          # as both official codes
        return self._ddp_model, self._ddp_model

    def _mem(self):
        if self.device.type != 'cuda':
            return ''
        return ' | GPU%d peak %.1f GB' % (self.device.index or 0, torch.cuda.max_memory_allocated(self.device) / 2 ** 30)

    def train_network(self, epoch, loader):
        a = self.args
        self.encoder.train(); self.head.train()
        if hasattr(loader.batch_sampler, 'set_epoch'):
            loader.batch_sampler.set_epoch(epoch)
        n_batches = len(loader)
        accum = max(1, a.accum_steps)
        fwd, ddp = self._forward_fn()
        show = du.is_main()
        tot_loss = tot_acc = tot_reg = 0.0
        n_reg, step_i = 0, 0
        t0 = time.time()
        lr0, lr1 = self._set_lr(epoch, 0)
        self._set_margin(epoch, 0)
        self.optim.zero_grad(set_to_none=True)
        for num, (data, labels) in enumerate(loader, start=1):
            data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            step = num % accum == 0 or num == n_batches
            sync = ddp.no_sync() if (ddp is not None and not step) else contextlib.nullcontext()
            with sync:
                total, nloss, acc, l2 = fwd(data, labels)
                if l2 is not None:
                    tot_reg += l2.item(); n_reg += 1
                self.scaler.scale(total / accum).backward()
            if step:
                if a.grad_clip > 0:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_([p for g in self.optim.param_groups for p in g['params']], a.grad_clip)
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
                step_i += 1
                lr0, lr1 = self._set_lr(epoch, step_i)
                self._set_margin(epoch, step_i)
            tot_loss += nloss.item(); tot_acc += acc.item()
            if show and (num % a.log_interval == 0 or num == n_batches):
                el = time.time() - t0
                sys.stderr.write(time.strftime('%m-%d %H:%M:%S') +
                                 ' [%2d] %.1f%% | lr %.2e/%.2e m %.3f | loss %.4f acc %.2f%% |theta-theta0| %.3f | %.1f min/epoch%s\r'
                                 % (epoch, 100 * num / n_batches, lr0, lr1, self.head.margin, tot_loss / num, tot_acc / num,
                                    tot_reg / max(n_reg, 1), el / num * n_batches / 60, self._mem()))
                sys.stderr.flush()
        if show:
            sys.stderr.write('\n')
        st = torch.tensor([tot_loss / n_batches, tot_acc / n_batches], device=self.device)
        st = du.all_reduce_sum(st) / du.world()
        return {'loss': st[0].item(), 'acc': st[1].item(), 'reg_l2': tot_reg / max(n_reg, 1),
                'lr_first': lr0, 'lr_last': lr1, 'margin': self.head.margin}

    # ------------------------------------------------------------------ evaluation
    def workers(self):
        return max(1, self.args.n_cpu // du.world())

    def _embed(self, files, root, kw):
        if not self.ddp:
            return extract_embeddings(self.encoder, files, root, **kw)
        mine = files[du.rank()::du.world()]
        part = extract_embeddings(self.encoder, mine, root, **kw) if mine else {}
        out = {}
        for p in du.gather_objects(part):
            out.update(p)
        return out

    @torch.no_grad()
    def eval_network(self, eval_list, eval_path, cohort=None, p_targets=(0.05,), eval_meta=''):
        a = self.args
        enroll, test, labels = parse_trials(eval_list)
        if labels[0] is None:
            raise ValueError('%s has no labels' % eval_list)
        files = sorted(set(enroll) | set(test))
        if hasattr(self, 'optim'):              # free gradients of the last step before embedding long utterances
            self.optim.zero_grad(set_to_none=True)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        kw = dict(device=self.device, mode=a.eval_mode, eval_sec=a.eval_sec, num_eval=a.num_eval,
                  min_sec=a.eval_min_sec, max_sec=a.eval_max_sec, len_tol=a.eval_len_tol,
                  max_batch=a.eval_batch, max_batch_sec=a.eval_batch_sec, amp=a.eval_amp, workers=self.workers(), progress=False)
        emb = self._embed(files, eval_path, kw)
        cemb = self._embed(cohort[0], cohort[1], kw) if (cohort and cohort[0]) else None
        if not du.is_main():
            return None
        scores = score_trials(emb, enroll, test, a.eval_mode)
        res = compute_metrics(scores, labels, p_targets)
        if eval_meta:
            res.update(subset_metrics(scores, labels, enroll, test, eval_meta, p_targets[0]))
        if cemb is not None:
            self.center = mean_embedding(cemb)
            res['EER_center'] = compute_metrics(score_trials(emb, enroll, test, a.eval_mode, self.center), labels, [])['EER']
        return res

    # ------------------------------------------------------------------ checkpoints
    def save_model(self, path, **meta):
        extra = {'train_meta': meta, 'head_state': self.head.state_dict(), 'n_class': self.n_class, 'head_type': self.args.head}
        if self.center is not None:
            extra['center_mean'] = {'full': self.center[0], 'crops': self.center[1]}
        self.encoder.export(path, **extra)

    def save_checkpoint(self, path, epoch, extra=None):
        enc = self.encoder
        ck = {'encoder': {'ssl_config': enc.ssl.config.to_dict(), 'ssl_state': enc.ssl.state_dict(),
                          'backend_state': enc.backend.state_dict(), 'meta': enc.meta, 'format': 'ssl-mhfa-v1'},
              'head': self.head.state_dict(), 'n_class': self.n_class, 'head_type': self.args.head,
              'optim': self.optim.state_dict(), 'scaler': self.scaler.state_dict(), 'epoch': epoch, 'center': self.center}
        if extra:
            ck.update(extra)
        torch.save(ck, path + '.tmp')
        os.replace(path + '.tmp', path)

    def load_head(self, ck):
        st = ck.get('head') or ck.get('head_state')
        ok = st is not None and ck.get('n_class') == self.n_class and ck.get('head_type') == self.args.head
        if ok:
            self.head.load_state_dict(st)
        return ok

    def load_checkpoint(self, path, with_optim=True):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        enc = ck.get('encoder', ck)
        self.encoder.ssl.load_state_dict(enc['ssl_state'], strict=False)
        self.encoder.backend.load_state_dict(enc['backend_state'])
        head_ok = self.load_head(ck)
        if with_optim and head_ok and 'optim' in ck:
            self.optim.load_state_dict(ck['optim'])
            if 'scaler' in ck:
                self.scaler.load_state_dict(ck['scaler'])
        self.center = ck.get('center')
        return ck, head_ok
