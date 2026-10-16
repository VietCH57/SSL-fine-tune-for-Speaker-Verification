'''
SSL backbone (Hugging Face transformers) + MHFA back-end for speaker verification.

MHFA follows the official code of Peng et al., "An attention-based backend allowing efficient
fine-tuning of transformer models for speaker verification", SLT 2023:
  https://github.com/JunyiPeng00/SLT22_MultiHead-Factorized-Attentive-Pooling  (models/Baseline/Spk_Encoder.py)
  https://github.com/BUTSpeechFIT/wespeaker_ssl_public  branch hubert  (wespeaker/models/ssl_backend.py)

Layer representations follow the official WavLM code used by MHFA (WavLM.extract_features with
output_layer = all layers): index 0 = transformer input after the positional convolution and its
dropout, index i = raw output of transformer layer i (the final encoder LayerNorm is NOT applied),
and a layer skipped by LayerDrop repeats the previous representation. They are captured with forward
hooks, so the code does not depend on how a transformers version builds `hidden_states`.
'''

import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

SSL_ALIASES = {
    'xlsr300m': 'facebook/wav2vec2-xls-r-300m',
    'wavlm_large': 'microsoft/wavlm-large',
}


class GradMultiply(torch.autograd.Function):
    '''fairseq GradMultiply: identity forward, gradient scaled by `scale` (used by the wespeaker_ssl_public recipe).'''

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.new(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class MHFA(nn.Module):
    '''Multi-head factorized attentive pooling (identical maths and parameter names to the official code).
    Input: layer representations [B, T, D, L] (official code uses [B, D, T, L]; the result is the same).'''

    def __init__(self, head_nb=64, inputs_dim=1024, compression_dim=128, outputs_dim=256, nb_layer=25):
        super().__init__()
        self.weights_k = nn.Parameter(torch.ones(nb_layer))
        self.weights_v = nn.Parameter(torch.ones(nb_layer))
        self.head_nb, self.ins_dim, self.cmp_dim, self.ous_dim = head_nb, inputs_dim, compression_dim, outputs_dim
        self.cmp_linear_k = nn.Linear(inputs_dim, compression_dim)
        self.cmp_linear_v = nn.Linear(inputs_dim, compression_dim)
        self.att_head = nn.Linear(compression_dim, head_nb)
        self.pooling_fc = nn.Linear(head_nb * compression_dim, outputs_dim)

    def forward(self, x):                                   # x: [B, T, D, L]
        k = torch.sum(x * F.softmax(self.weights_k, dim=-1), dim=-1)     # [B, T, D]
        v = torch.sum(x * F.softmax(self.weights_v, dim=-1), dim=-1)
        k = self.cmp_linear_k(k)                                         # [B, T, C]
        v = self.cmp_linear_v(v)
        att_k = self.att_head(k)                                         # [B, T, H]
        v = v.unsqueeze(-2)                                              # [B, T, 1, C]
        pooling_outs = torch.sum(v * F.softmax(att_k, dim=1).unsqueeze(-1), dim=1)   # [B, H, C]
        return self.pooling_fc(pooling_outs.reshape(pooling_outs.shape[0], -1))


def _hf_classes(model_type):
    from transformers import Wav2Vec2Config, Wav2Vec2Model, WavLMConfig, WavLMModel
    if model_type == 'wavlm':
        return WavLMConfig, WavLMModel
    if model_type == 'wav2vec2':
        return Wav2Vec2Config, Wav2Vec2Model
    raise ValueError('unsupported SSL model_type %r (wavlm / wav2vec2)' % model_type)


def load_hf_ssl(name, cache_dir=None, config_dict=None, state_dict=None, layerdrop=None):
    '''From the hub (name) or rebuilt offline from a saved config + state_dict.
    SpecAugment-style masking of the pre-training task is disabled (MHFA calls extract_features with
    mask=False); LayerDrop keeps the checkpoint value unless `layerdrop` is given.'''
    if config_dict is not None:
        Cfg, Model = _hf_classes(config_dict['model_type'])
        cfg = Cfg.from_dict(dict(config_dict))
        cfg.apply_spec_augment = False
        if layerdrop is not None:
            cfg.layerdrop = layerdrop
        model = Model(cfg)
        if state_dict is not None:
            r = model.load_state_dict(state_dict, strict=False)
            bad = [k for k in r.missing_keys if 'masked_spec_embed' not in k]
            assert not bad and not r.unexpected_keys, (bad[:5], r.unexpected_keys[:5])
        return model
    from transformers import AutoConfig
    name = SSL_ALIASES.get(name, name)
    cfg = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
    Cfg, Model = _hf_classes(cfg.model_type)
    kw = dict(apply_spec_augment=False)
    if layerdrop is not None:
        kw['layerdrop'] = layerdrop
    return Model.from_pretrained(name, cache_dir=cache_dir, **kw)


class SSLEncoder(nn.Module):
    '''waveform [B, T] at 16 kHz -> speaker embedding [B, embed_dim].

    ssl_trainable : 'none'   -> backbone run under no_grad (frozen stage of the wespeaker_ssl_public recipe)
                    'layers' -> only the transformer layers are trained, the CNN runs without gradients
                                (SLT22 code: only encoder.layers are in the optimiser, CNN under no_grad)
                    'all'    -> CNN, projection, positional conv and layers are trained
                                (wespeaker_ssl_public fine-tuning stage, cnn_scale 1.0)
    ssl_grad_mult : gradient multiplier on the layer representations (wespeaker_ssl_public feature_grad_mult)
    input_norm    : 'utterance' = zero mean / unit variance per utterance, as do_normalize=True in the
                    preprocessor_config.json of both checkpoints; 'none' = raw waveform.'''

    def __init__(self, ssl, head_nb=64, compression_dim=128, embed_dim=256, input_norm='utterance',
                 ssl_trainable='layers', ssl_grad_mult=1.0, grad_checkpointing=False, meta=None):
        super().__init__()
        self.ssl = ssl
        cfg = ssl.config
        self.n_layers = cfg.num_hidden_layers
        self.backend = MHFA(head_nb, cfg.hidden_size, compression_dim, embed_dim, self.n_layers + 1)
        self.input_norm = input_norm
        self.ssl_grad_mult = ssl_grad_mult
        self.meta = dict(meta or {})
        self.meta.update(head_nb=head_nb, compression_dim=compression_dim, embed_dim=embed_dim, input_norm=input_norm)
        self.embed_dim = embed_dim
        self._reps = {}
        enc = ssl.encoder
        enc.dropout.register_forward_hook(self._hook(0))
        for i, layer in enumerate(enc.layers):
            layer.register_forward_hook(self._hook(i + 1))
        if grad_checkpointing:
            ssl.gradient_checkpointing_enable()
        self.set_trainable(ssl_trainable)

    def _hook(self, idx):
        def fn(module, inputs, output):
            self._reps[idx] = output[0] if isinstance(output, (tuple, list)) else output
        return fn

    def set_trainable(self, mode):
        assert mode in ('none', 'layers', 'all'), mode
        self.ssl_trainable = mode
        for n, p in self.ssl.named_parameters():
            if mode == 'none' or 'masked_spec_embed' in n:
                p.requires_grad_(False)
            elif mode == 'layers':
                p.requires_grad_(n.startswith('encoder.layers.'))
            else:
                p.requires_grad_(True)

    def ssl_trainable_parameters(self):
        return [(n, p) for n, p in self.ssl.named_parameters() if p.requires_grad]

    def layer_representations(self, wav):
        if self.input_norm == 'utterance':        # Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm (eps 1e-7)
            wav = wav.float()
            wav = (wav - wav.mean(dim=1, keepdim=True)) / torch.sqrt(wav.var(dim=1, keepdim=True, unbiased=False) + 1e-7)
        self._reps = {}
        if self.ssl_trainable == 'layers':
            with torch.no_grad():                 # SLT22: feature extractor under no_grad
                feats = self.ssl.feature_extractor(wav)
            self._run_after_cnn(feats)
        else:
            self.ssl(wav)
        reps = [self._reps[0]]
        for i in range(1, self.n_layers + 1):     # LayerDrop: a skipped layer repeats the previous output
            reps.append(self._reps.get(i, reps[-1]))
        self._reps = {}
        return torch.stack(reps, dim=-1)          # [B, T, D, L+1]

    def _run_after_cnn(self, feats):
        m = self.ssl
        feats = feats.transpose(1, 2)
        out = m.feature_projection(feats)
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        m.encoder(hidden)

    def forward(self, wav):
        if self.ssl_trainable == 'none':
            with torch.no_grad():
                x = self.layer_representations(wav)
        else:
            x = self.layer_representations(wav)
            if self.ssl_grad_mult != 1.0:
                x = GradMultiply.apply(x, self.ssl_grad_mult)
        # back-end always in fp32: the official recipes train in fp32, and with the large SGD LR of the
        # wespeaker frozen stage (0.1 x scale_ratio) fp16 activations of MHFA can overflow
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.backend(x.float())

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    # ---- checkpoints (rebuildable offline: HF config + weights) ----
    def export(self, path, **extra):
        ck = {'ssl_config': self.ssl.config.to_dict(), 'ssl_state': self.ssl.state_dict(),
              'backend_state': self.backend.state_dict(), 'meta': self.meta, 'format': 'ssl-mhfa-v1'}
        ck.update(extra)
        torch.save(ck, path + '.tmp')
        os.replace(path + '.tmp', path)


def build_encoder(args, verbose=True):
    '''New encoder from the hub (args.ssl) with the options of args.'''
    ssl = load_hf_ssl(args.ssl, args.hf_cache or None, layerdrop=args.layerdrop)
    enc = SSLEncoder(ssl, args.head_nb, args.compression_dim, args.embed_dim, args.input_norm,
                     args.ssl_trainable, args.ssl_grad_mult, args.grad_checkpointing,
                     meta={'ssl': SSL_ALIASES.get(args.ssl, args.ssl)})
    if verbose:
        _report(enc)
    return enc


def load_encoder(path, ssl_trainable='layers', ssl_grad_mult=1.0, grad_checkpointing=False, layerdrop=None, verbose=True):
    '''Encoder from a file written by this repo (best.model / model_XXXX.model / last.ckpt), no hub access needed.'''
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if 'encoder' in ck and isinstance(ck['encoder'], dict):
        ck = ck['encoder']
    assert ck.get('format') == 'ssl-mhfa-v1', '%s is not an SSL+MHFA checkpoint of this repo' % path
    m = ck['meta']
    ssl = load_hf_ssl(None, config_dict=ck['ssl_config'], state_dict=ck['ssl_state'], layerdrop=layerdrop)
    enc = SSLEncoder(ssl, m['head_nb'], m['compression_dim'], m['embed_dim'], m['input_norm'],
                     ssl_trainable, ssl_grad_mult, grad_checkpointing, meta=m)
    enc.backend.load_state_dict(ck['backend_state'])
    if verbose:
        _report(enc, path)
    extra = {k: v for k, v in ck.items() if k not in ('ssl_config', 'ssl_state', 'backend_state', 'meta', 'format')}
    return enc, extra


def _report(enc, src=''):
    n_ssl = sum(p.numel() for p in enc.ssl.parameters())
    n_be = sum(p.numel() for p in enc.backend.parameters())
    n_tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    c = enc.ssl.config
    print('Encoder %s%s: %s (%s, %d layers x %d) %.1fM + MHFA %.2fM (%d heads) | trainable %.1fM | layerdrop %.2f | input_norm %s'
          % (enc.meta.get('ssl', ''), (' from ' + src) if src else '', c.model_type, type(enc.ssl).__name__,
             c.num_hidden_layers, c.hidden_size, n_ssl / 1e6, n_be / 1e6, enc.backend.head_nb, n_tr / 1e6,
             c.layerdrop, enc.input_norm))
