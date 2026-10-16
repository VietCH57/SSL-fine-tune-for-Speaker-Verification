'''
Classification heads, copied from the official MHFA code bases:

  AAMsoftmax                : SLT22 code, loss/aamsoftmax.py (voxceleb_trainer), margin 0.2, scale 30
  ArcMarginIntertopkSubcenter: wespeaker_ssl_public (branch hubert) wespeaker/models/projections.py,
                              ArcMarginProduct_intertopk_subcenter (K=3 sub-centres, k_top=5, mp=0.06)
                              + CrossEntropyLoss, margin changed by MarginScheduler via update()

Both return (loss, accuracy %, logits); the logits over the training classes are what a student
needs for label-level knowledge distillation.
'''

import math, torch
import torch.nn as nn
import torch.nn.functional as F


class AAMsoftmax(nn.Module):
    def __init__(self, n_class, embed_dim, m=0.2, s=30.0, easy_margin=False):
        super().__init__()
        self.s, self.easy_margin = s, easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(n_class, embed_dim))
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce = nn.CrossEntropyLoss()
        self.update(m)

    def update(self, margin):
        self.m = self.margin = float(margin)
        self.cos_m, self.sin_m = math.cos(self.m), math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def logits(self, x, label):
        cosine = F.linear(F.normalize(x.float()), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1), 1)
        return self.s * ((one_hot * phi) + ((1.0 - one_hot) * cosine))

    def forward(self, x, label):
        out = self.logits(x, label)
        loss = self.ce(out, label)
        acc = (out.detach().argmax(1) == label).float().mean() * 100
        return loss, acc, out


class ArcMarginIntertopkSubcenter(nn.Module):
    def __init__(self, n_class, embed_dim, m=0.2, s=32.0, easy_margin=False, K=3, mp=0.06, k_top=5, do_lm=False):
        super().__init__()
        self.in_features, self.out_features, self.scale = embed_dim, n_class, s
        self.do_lm, self.K, self.easy_margin = do_lm, K, easy_margin
        self.mp, self.k_top = (0.0, 0) if do_lm else (mp, k_top)
        self.weight = nn.Parameter(torch.FloatTensor(K * n_class, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()
        self.update(m)

    def update(self, margin=0.2):
        self.margin = self.m = float(margin)
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
        mp = self.mp * (margin / 0.2) if margin > 0.001 else 0.0     # hard-sample margin grows with the margin
        self.cos_mp, self.sin_mp = math.cos(mp), math.sin(mp)

    def logits(self, x, label):
        cosine = F.linear(F.normalize(x.float()), F.normalize(self.weight))
        cosine = torch.reshape(cosine, (-1, self.out_features, self.K)).max(2)[0]
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
        one_hot = torch.zeros_like(cosine).scatter_(1, label.view(-1, 1).long(), 1)
        if self.k_top > 0:
            _, top_k_index = torch.topk(cosine - 2 * one_hot, self.k_top)
            top_k_one_hot = torch.zeros_like(cosine).scatter_(1, top_k_index, 1)
            out = (one_hot * phi) + (top_k_one_hot * phi_mp) + ((1.0 - one_hot - top_k_one_hot) * cosine)
        else:
            out = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return out * self.scale

    def forward(self, x, label):
        out = self.logits(x, label)
        loss = self.ce(out, label)
        acc = (out.detach().argmax(1) == label).float().mean() * 100
        return loss, acc, out


def build_head(args, n_class, embed_dim):
    if args.head == 'aam':
        return AAMsoftmax(n_class, embed_dim, m=args.m, s=args.s)
    if args.head == 'aam_intertopk_subcenter':
        return ArcMarginIntertopkSubcenter(n_class, embed_dim, m=args.m, s=args.s, K=args.subcenter_k,
                                           mp=args.intertopk_mp, k_top=args.intertopk_k, do_lm=args.do_lm)
    raise ValueError(args.head)
