'''Tiny randomly initialised WavLM and wav2vec2 (same classes / layer layout as WavLM-Large and XLS-R)
for CPU smoke tests without hub access:  python tests/make_tiny_ssl.py <out_dir>'''
import sys, os, torch
from transformers import WavLMConfig, WavLMModel, Wav2Vec2Config, Wav2Vec2Model
out = sys.argv[1]
torch.manual_seed(0)
common = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64, conv_dim=(32,) * 7,
              do_stable_layer_norm=True, feat_extract_norm='layer', num_conv_pos_embeddings=16,
              num_conv_pos_embedding_groups=2, layerdrop=0.1, apply_spec_augment=True, mask_time_prob=0.075)
WavLMModel(WavLMConfig(num_buckets=32, max_bucket_distance=64, **common)).save_pretrained(os.path.join(out, 'tiny_wavlm'))
Wav2Vec2Model(Wav2Vec2Config(**common)).save_pretrained(os.path.join(out, 'tiny_w2v2'))
print('tiny models in', out)
