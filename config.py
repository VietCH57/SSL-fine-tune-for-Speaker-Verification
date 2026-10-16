'''
Command-line / config-file arguments for train.py.  Priority: command line > --config YAML > defaults.
The YAML files in configs/ reproduce the two official MHFA recipes (see README); defaults below are the
SLT22 code defaults (trainSpeakerNet.py + yaml/Baseline.yaml) where they exist.
'''

import argparse, json


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('1', 'true', 'yes', 'y', 't'):
        return True
    if v.lower() in ('0', 'false', 'no', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected, got %r' % v)


def _floats(v):
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(x) for x in str(v).replace(' ', '').split(',') if x != '']


def _ints(v):
    return [int(round(x)) for x in _floats(v)]


def _strs(v):
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [x for x in str(v).replace(' ', '').split(',') if x != '']


def _float_or_none(v):
    return None if v in (None, '', 'none', 'None') else float(v)


def build_parser():
    p = argparse.ArgumentParser(description='Fine-tune an SSL model (XLS-R / WavLM) + MHFA for speaker verification',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, default='', help='YAML/JSON with any of the options below (command line overrides it)')

    g = p.add_argument_group('Data')
    g.add_argument('--train_list', type=str, default='', help='"<speaker> <relative/path.wav>" per line')
    g.add_argument('--train_path', type=str, default='', help='Root of the training wavs')
    g.add_argument('--crop_sec', type=_floats, default=[3.015], help='Training crop (s). SLT22: max_frames 300 -> 300*160+240 samples = 3.015 s')
    g.add_argument('--crop_probs', type=_floats, default=[], help='Probabilities of --crop_sec (empty = uniform)')
    g.add_argument('--min_dur', type=float, default=0.0, help='Drop training files shorter than this (s); wespeaker filter min_num_frames/100')
    g.add_argument('--max_dur', type=float, default=0.0, help='Drop training files longer than this (s, 0 = keep); wespeaker filter max_num_frames/100')
    g.add_argument('--batch_size', type=int, default=120, help='GLOBAL batch per forward pass, split over the GPUs (see --accum_steps)')
    g.add_argument('--accum_steps', type=int, default=1, help='Optimiser step every N batches (effective batch = batch_size * N)')
    g.add_argument('--n_cpu', type=int, default=4, help='DataLoader workers in total (shared by the torchrun processes)')
    g.add_argument('--seed', type=int, default=20211202, help='Random seed (SLT22 default)')

    g = p.add_argument_group('Augmentation')
    g.add_argument('--aug_mode', type=str, default='kaldi', choices=['kaldi', 'wespeaker'],
                   help='kaldi: voxceleb_trainer/SLT22 (one of clean/reverb/music/speech/noise/speech+music, uniform); '
                        'wespeaker: with prob --aug_prob either RIR or ONE MUSAN file from the pooled set, then peak-normalised')
    g.add_argument('--musan_path', type=str, default='', help='MUSAN root (folder with music/ noise/ speech/ or a parent)')
    g.add_argument('--rir_path', type=str, default='', help='RIR root (simulated_rirs is used if found)')
    g.add_argument('--aug_probs', type=_floats, default=[1, 1, 1, 1, 1, 1], help='[kaldi] weights of [clean, reverb, babble, music, noise, tv]; SLT22: uniform')
    g.add_argument('--snr_noise', type=_floats, default=[0, 15], help='[kaldi] SNR range of MUSAN noise')
    g.add_argument('--snr_speech', type=_floats, default=[13, 20], help='[kaldi] SNR range of babble')
    g.add_argument('--snr_music', type=_floats, default=[5, 15], help='[kaldi] SNR range of music')
    g.add_argument('--num_noise', type=_ints, default=[1, 1], help='[kaldi] noise files mixed')
    g.add_argument('--num_speech', type=_ints, default=[3, 8], help='[kaldi] babble speakers mixed')
    g.add_argument('--num_music', type=_ints, default=[1, 1], help='[kaldi] music files mixed')
    g.add_argument('--aug_prob', type=float, default=0.6, help='[wespeaker] probability of reverb-or-noise')
    g.add_argument('--speed_perturb', type=_floats, default=[0.9, 1.1], help='Extra speeds; each adds a copy of every speaker as new classes (wespeaker speed_perturb). Empty = off')
    g.add_argument('--speed_probs', type=_floats, default=[], help='Probabilities of [1.0, *speed_perturb] (empty = uniform, as wespeaker)')
    g.add_argument('--narrowband_prob', type=float, default=0.0, help='Narrow-band channel simulation (not in the official recipes; 0 = off)')
    g.add_argument('--narrowband_types', type=_strs, default=['8k', 'tel'])

    g = p.add_argument_group('Model')
    g.add_argument('--ssl', type=str, default='wavlm_large', help='xlsr300m | wavlm_large | any HF wav2vec2/WavLM model id')
    g.add_argument('--hf_cache', type=str, default='', help='HF download cache (default: HF_HOME)')
    g.add_argument('--head_nb', type=int, default=64, help='MHFA heads (paper: 64 best for Large; wespeaker_ssl_public Large: 64)')
    g.add_argument('--compression_dim', type=int, default=128, help='MHFA compression dim (official code: 128)')
    g.add_argument('--embed_dim', type=int, default=256, help='Embedding size (official: nOut 256 / embed_dim 256)')
    g.add_argument('--input_norm', type=str, default='utterance', choices=['utterance', 'none'],
                   help='utterance = do_normalize of the checkpoints\' preprocessor_config.json; none = raw waveform (as the MHFA code feeds WavLM)')
    g.add_argument('--layerdrop', type=_float_or_none, default=None, help='Transformer LayerDrop (None = checkpoint value; wespeaker_ssl_public: 0.05)')
    g.add_argument('--ssl_trainable', type=str, default='layers', choices=['none', 'layers', 'all'],
                   help='none: frozen backbone; layers: only transformer layers (SLT22); all: whole backbone incl. CNN (wespeaker_ssl_public FT)')
    g.add_argument('--ssl_grad_mult', type=float, default=1.0, help='Gradient multiplier on the layer outputs (wespeaker_ssl_public feature_grad_mult, FT: 0.1)')
    g.add_argument('--grad_checkpointing', type=_str2bool, default=False, help='Activation checkpointing in the transformer (memory, not maths)')
    g.add_argument('--max_params', type=float, default=0, help='Refuse an encoder with more parameters (0 = no check; teachers have no limit)')

    g = p.add_argument_group('Head / loss')
    g.add_argument('--head', type=str, default='aam', choices=['aam', 'aam_intertopk_subcenter'])
    g.add_argument('--m', type=float, default=0.2, help='Margin (constant unless --m_schedule)')
    g.add_argument('--s', type=float, default=30.0, help='Scale (SLT22: 30, wespeaker_ssl_public: 32)')
    g.add_argument('--subcenter_k', type=int, default=3, help='[intertopk_subcenter] K')
    g.add_argument('--intertopk_k', type=int, default=5, help='[intertopk_subcenter] k_top')
    g.add_argument('--intertopk_mp', type=float, default=0.06, help='[intertopk_subcenter] mp')
    g.add_argument('--do_lm', type=_str2bool, default=False, help='[intertopk_subcenter] remove the hard-sample penalty (wespeaker do_lm)')
    g.add_argument('--m_schedule', type=_str2bool, default=False, help='wespeaker MarginScheduler (exp increase)')
    g.add_argument('--m_initial', type=float, default=0.0)
    g.add_argument('--m_final', type=float, default=0.2)
    g.add_argument('--m_increase_start_epoch', type=int, default=1)
    g.add_argument('--m_fix_start_epoch', type=int, default=1)

    g = p.add_argument_group('Optimisation')
    g.add_argument('--max_epoch', type=int, default=15)
    g.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    g.add_argument('--weight_decay', type=float, default=0.01, help='SLT22: torch.optim.AdamW default (0.01); wespeaker_ssl_public: 1e-7')
    g.add_argument('--momentum', type=float, default=0.9, help='[sgd] (nesterov)')
    g.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'exp_wespeaker'],
                   help='step: SLT22 param groups + StepLR per epoch; exp_wespeaker: one LR for all, ExponentialDecrease per iteration')
    g.add_argument('--lr_mhfa', type=float, default=5e-3, help='[step] LR of MHFA + head (SLT22 LR_MHFA)')
    g.add_argument('--lr_transformer', type=float, default=2e-5, help='[step] LR of transformer layer 0 (SLT22 LR_Transformer)')
    g.add_argument('--llrd_factor', type=float, default=1.0, help='[step] LR of layer i = lr_transformer * llrd_factor^i (SLT22 LLRD_factor)')
    g.add_argument('--lr_decay', type=float, default=0.95, help='[step] StepLR gamma per epoch (SLT22 yaml: 0.95)')
    g.add_argument('--lr_epoch_offset', type=int, default=0, help='[step] epochs already done before this stage (SLT22 LM-FT continues the StepLR decay of stage 1)')
    g.add_argument('--initial_lr', type=float, default=5e-5, help='[exp_wespeaker] initial_lr')
    g.add_argument('--final_lr', type=float, default=1e-6, help='[exp_wespeaker] final_lr')
    g.add_argument('--warm_up_epoch', type=float, default=1, help='[exp_wespeaker] warm_up_epoch')
    g.add_argument('--warm_from_zero', type=_str2bool, default=True, help='[exp_wespeaker] warm_from_zero')
    g.add_argument('--lr_scale_base', type=float, default=64, help='[exp_wespeaker] scale_ratio = effective batch / this (wespeaker: world_size*batch/64)')
    g.add_argument('--reg_type', type=str, default='slt22', choices=['none', 'slt22'],
                   help='slt22: loss/loss.detach() + w*||theta-theta0||/||theta-theta0||.detach() (SpeakerNet.py)')
    g.add_argument('--reg_weight', type=float, default=0.01, help='SLT22 weight_finetuning_reg (yaml: 0.01)')
    g.add_argument('--grad_clip', type=float, default=0.0, help='Clip grad-norm (0 = off). SLT22: off; wespeaker_ssl_public executor_hubert.py: 5.0')
    g.add_argument('--amp', type=_str2bool, default=True, help='fp16 autocast (T4 has no bf16; SLT22 has --mixedprec, wespeaker_ssl_public enable_amp False)')

    g = p.add_argument_group('Evaluation')
    g.add_argument('--eval_list', type=str, default='')
    g.add_argument('--eval_path', type=str, default='')
    g.add_argument('--eval_meta', type=str, default='')
    g.add_argument('--test_step', type=int, default=1, help='Evaluate/save every N epochs (SLT22 test_interval 1)')
    g.add_argument('--eval_init', type=_str2bool, default=False, help='Evaluate before training (epoch 0); random MHFA -> only useful when resuming a stage')
    g.add_argument('--eval_mode', type=str, default='full', choices=['full', 'crops', 'both'])
    g.add_argument('--eval_sec', type=float, default=4.015, help='[crops] SLT22 eval_frames 400')
    g.add_argument('--num_eval', type=int, default=5)
    g.add_argument('--eval_min_sec', type=float, default=1.0)
    g.add_argument('--eval_max_sec', type=float, default=20.0, help='wespeaker_ssl_public cuts inputs to 20 s')
    g.add_argument('--eval_len_tol', type=float, default=0.02)
    g.add_argument('--eval_batch', type=int, default=32)
    g.add_argument('--eval_batch_sec', type=float, default=120.0, help='Max total audio (s) per eval batch: bounds GPU memory with long utterances')
    g.add_argument('--eval_amp', type=_str2bool, default=True)
    g.add_argument('--cohort_size', type=int, default=1000, help='Training files for the center (mean) of "center" scoring; 0 = off')
    g.add_argument('--select_score', type=str, default='cos', choices=['cos', 'center'])
    g.add_argument('--dcf_ptarget', type=_floats, default=[0.05])

    g = p.add_argument_group('Checkpoints & logging')
    g.add_argument('--save_path', type=str, default='exps/exp1')
    g.add_argument('--initial_model', type=str, default='', help='last.ckpt / model of a previous stage (encoder + MHFA; head too if classes match)')
    g.add_argument('--resume', type=str, default='auto', help='auto | none | path to last.ckpt')
    g.add_argument('--max_hours', type=float, default=0)
    g.add_argument('--keep_last', type=int, default=1, help='model_XXXX.model kept (each ~1.3 GB)')
    g.add_argument('--save_best', type=_str2bool, default=True)
    g.add_argument('--log_interval', type=int, default=20)
    g.add_argument('--device', type=str, default='cuda')

    g = p.add_argument_group('Command')
    g.add_argument('--eval', dest='eval', action='store_true', help='Only evaluate --initial_model on --eval_list')
    return p


def load_config_file(path):
    with open(path) as f:
        if path.endswith(('.yaml', '.yml')):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f)


def parse_args(argv=None):
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        cfg = load_config_file(pre.config)
        known = {a.dest for a in parser._actions}
        unknown = [k for k in cfg if k not in known]
        if unknown:
            raise ValueError('Unknown keys in %s: %s' % (pre.config, unknown))
        for a in parser._actions:
            if a.dest in cfg and a.type is not None and cfg[a.dest] is not None:
                cfg[a.dest] = a.type(cfg[a.dest])
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)
    if len(args.aug_probs) != 6:
        parser.error('--aug_probs needs 6 values')
    if not args.eval and not (args.train_list and args.train_path):
        parser.error('--train_list and --train_path are required for training')
    if args.eval and not args.initial_model:
        parser.error('--eval needs --initial_model')
    return args


def save_config(args, path):
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
