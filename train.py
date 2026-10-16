'''
Fine-tune an SSL model (XLS-R / WavLM) + MHFA for speaker verification.

    torchrun --standalone --nproc_per_node 2 train.py --config configs/slt22/xlsr300m.yaml
    python train.py --config configs/slt22/wavlm_large.yaml --batch_size 32 --accum_steps 4
    python train.py --eval --initial_model exps/x/model/best.model --eval_list ... --eval_path ...

Outputs in --save_path: config.json, speakers.txt, score.txt,
    model/model_XXXX.model, model/best.model  (encoder + MHFA + head, rebuildable offline)
    model/last.ckpt                            (+ optimiser, exact resume)
'''

import glob, math, os, random, re, sys, time, warnings
import numpy, torch
import dist_utils as du
from config import parse_args, save_config
from dataLoader import TrainSet, CropBatchSampler, read_list, worker_init
from trainer import Trainer


def seed_everything(seed):
    random.seed(seed); numpy.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def epoch_of(path):
    m = re.search(r'model_(\d+)\.model$', path)
    return int(m.group(1)) if m else -1


def fmt(res):
    return ', '.join('%s %.4f' % (k, v) for k, v in res.items())


def cohort_files(args):
    '''A fixed random subset of training files (relative paths) for the center-scoring mean.'''
    if args.cohort_size <= 0:
        return None
    _, paths = read_list(args.train_list)
    rng = random.Random(1234)
    return (sorted(rng.sample(paths, min(args.cohort_size, len(paths)))), args.train_path)


def main():
    warnings.simplefilter('ignore')
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    rank, world, local = du.init()
    if rank > 0:
        sys.stdout = open(os.devnull, 'w')      # only rank 0 prints (errors still go to stderr)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA not available -> using CPU'); args.device = 'cpu'
    if world > 1:
        args.device = 'cuda:%d' % local if args.device.startswith('cuda') else 'cpu'
    if args.seed >= 0:
        seed_everything(args.seed + rank)       # different augmentation per GPU; weights are synchronised by DDP
    model_dir = os.path.join(args.save_path, 'model')
    p_targets = args.dcf_ptarget

    ## Only evaluate
    if args.eval:
        assert args.eval_list, '--eval needs --eval_list'
        args.reg_type = 'none'
        s = Trainer(args, n_class=1, init_path=args.initial_model)
        coh = cohort_files(args) if args.train_list else None
        res = s.eval_network(args.eval_list, args.eval_path, coh, p_targets, args.eval_meta)
        if du.is_main():
            print(fmt(res))
        du.cleanup()
        return

    ## Data
    os.makedirs(model_dir, exist_ok=True)
    trainset = TrainSet(verbose=du.is_main(), **{k: v for k, v in vars(args).items() if k != 'verbose'})
    args.n_class = trainset.n_class
    if du.is_main():
        save_config(args, os.path.join(args.save_path, 'config.json'))
        with open(os.path.join(args.save_path, 'speakers.txt'), 'w') as f:
            for k, v in trainset.spk2id.items():
                f.write('%s %d\n' % (k, v))
    n_workers = max(1, args.n_cpu // world) if args.n_cpu > 0 else 0      # --n_cpu = total over all GPUs
    accum = max(1, args.accum_steps)
    sampler = CropBatchSampler(len(trainset), args.batch_size, args.crop_sec, args.crop_probs, seed=max(args.seed, 0),
                               rank=rank, world=world)
    loader = torch.utils.data.DataLoader(trainset, batch_sampler=sampler, num_workers=n_workers,
                                         pin_memory=args.device.startswith('cuda'), worker_init_fn=worker_init,
                                         persistent_workers=n_workers > 0)
    steps_per_epoch = math.ceil(len(loader) / accum)
    print('Training: %d utterances, %d classes, %d batches/epoch of %d (%d per GPU x %d) x accum %d = effective %d, %d optimiser steps/epoch, crops %s s, %d loader workers per GPU'
          % (len(trainset), trainset.n_class, len(loader), args.batch_size, args.batch_size // world, world, accum,
             args.batch_size * accum, steps_per_epoch, args.crop_sec, n_workers))

    ## Model, possibly resumed
    last_ckpt = os.path.join(model_dir, 'last.ckpt')
    resume = args.resume
    if resume == 'auto':
        resume = last_ckpt if os.path.exists(last_ckpt) else ''
    elif resume == 'none':
        resume = ''
    epoch, EERs, best = 1, [], float('inf')
    if resume:
        s = Trainer(args, trainset.n_class, init_path=resume, steps_per_epoch=steps_per_epoch)
        ck, head_ok = s.load_checkpoint(resume)
        assert head_ok, '%s: head (%s, %s classes) does not match (%s, %d)' % (resume, ck.get('head_type'), ck.get('n_class'), args.head, trainset.n_class)
        epoch, EERs, best = ck['epoch'] + 1, ck.get('EERs', []), ck.get('best', best)
        print('Resumed from %s, starting at epoch %d' % (resume, epoch))
    else:
        s = Trainer(args, trainset.n_class, init_path=args.initial_model or None, steps_per_epoch=steps_per_epoch)
        if args.initial_model:
            ck = torch.load(args.initial_model, map_location='cpu', weights_only=False)
            head_ok = s.load_head(ck)
            del ck
            print('Initial model %s (encoder + MHFA%s)' % (args.initial_model, ' + head' if head_ok else '; head NOT restored: type/classes differ'))
    s.set_reference()            # SLT22 regulariser: pre-trained weights of the hub checkpoint
    du.barrier()

    score_file = open(os.path.join(args.save_path, 'score.txt'), 'a+') if du.is_main() else None
    do_eval = bool(args.eval_list)
    cohort = cohort_files(args)
    sel = 'EER' if args.select_score == 'cos' else 'EER_center'

    def evaluate(tag_epoch, msg):
        nonlocal best
        t0 = time.time()
        res = s.eval_network(args.eval_list, args.eval_path, cohort, p_targets, args.eval_meta)
        if not du.is_main():                    # every rank embeds its share; rank 0 scores and saves
            du.barrier(); return
        EERs.append(res[sel])
        if res[sel] < best:
            best = res[sel]
            if args.save_best:
                s.save_model(os.path.join(model_dir, 'best.model'), epoch=tag_epoch, **{k: float(v) for k, v in res.items()})
        msg += ', %s %.2f%%, best%s %.2f%%' % (sel, res[sel], sel, best)
        msg += ', ' + fmt({k: v for k, v in res.items() if k != sel})
        print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, '| eval %.1f min' % ((time.time() - t0) / 60))
        score_file.write(msg + '\n'); score_file.flush()
        du.barrier()

    if do_eval and args.eval_init and epoch == 1:
        evaluate(0, '0 epoch (start: %s)' % (args.initial_model or args.pretrained))

    t_start, ep_times = time.time(), []
    while epoch <= args.max_epoch:
        stop = False
        if args.max_hours > 0 and ep_times:
            need = max(ep_times[-args.test_step:]) * 1.1
            stop = time.time() - t_start + need > args.max_hours * 3600
        if du.broadcast_object(stop):          # every rank follows rank 0's decision
            print('Stopping before epoch %d: --max_hours %.2f would be exceeded (%.2f h used). Rerun to resume.'
                  % (epoch, args.max_hours, (time.time() - t_start) / 3600))
            break
        t0 = time.time()
        st = s.train_network(epoch, loader)
        train_min = (time.time() - t0) / 60
        if not math.isfinite(st['loss']):
            raise RuntimeError('epoch %d: training loss is %s (diverged). Last checkpoint left untouched: %s'
                               % (epoch, st['loss'], last_ckpt))
        if epoch % args.test_step == 0 or epoch == args.max_epoch:
            msg = '%d epoch, LR %.2e/%.2e, m %.3f, LOSS %.4f, ACC %.2f%%, DIST %.3f' % (
                epoch, st['lr_first'], st['lr_last'], st['margin'], st['loss'], st['acc'], st['reg_l2'])
            if do_eval:
                evaluate(epoch, msg)
            elif du.is_main():
                print(time.strftime('%Y-%m-%d %H:%M:%S'), msg); score_file.write(msg + '\n'); score_file.flush()
            if du.is_main():
                s.save_model(os.path.join(model_dir, 'model_%04d.model' % epoch), epoch=epoch)
                if args.keep_last > 0:
                    for f in sorted(glob.glob(os.path.join(model_dir, 'model_0*.model')), key=epoch_of)[:-args.keep_last]:
                        os.remove(f)
        print('epoch %d: train %.1f min' % (epoch, train_min))
        if du.is_main():
            s.save_checkpoint(last_ckpt, epoch, {'EERs': [float(x) for x in EERs], 'best': float(best)})
        du.barrier()
        ep_times.append(time.time() - t0)
        epoch += 1
    if score_file:
        score_file.close()
    du.cleanup()


if __name__ == '__main__':
    main()
