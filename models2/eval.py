import torch
import torch.nn as nn
import numpy as np
from model import Batch, GRUEncoderDecoder, EncoderDecoder
import matplotlib.pyplot as plt

from decode import greedy_decode, beam_decode

PAD = 0
SOS = 1
EOS = 2

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None, scheduler=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        self.scheduler = scheduler

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.data.item() * norm

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def get_predictions(model, batch: Batch, reverse_mapping: dict, maxi=None, pr=False, beam=None):
    """Greedy decode predictions from a batch."""
    length = len(batch.src_lengths)
    res = []
    if maxi is not None:
        length = min(maxi, length)
    for i in range(length):
        src = [reverse_mapping[x.item()] for x in batch.src[i] if x.item() != PAD]
        trg = [reverse_mapping[x.item()] for x in batch.trg[i] if x.item() != PAD]

        # beam decode
        if beam is not None:
            pred = beam_decode(
                model, batch.src[i].reshape(1, -1), batch.src_mask[i].reshape(1, -1),
                [batch.src_lengths[i]], beam_size=beam
            )
            beam_res = []
            print(i)
            print(' '.join(src))
            print(' '.join(trg))
            for i, (out, prob) in enumerate(pred):
                output = [reverse_mapping[x.item()] for x in out if x.item() != PAD]
                if i == 0: res.append([src, trg, output])
                print(' '.join(output), f"({prob.exp().detach().item():.6%})")
            print()
        # greedy decode
        else:
            pred, attns, probs = greedy_decode(
                model, batch.src[i].reshape(1, -1), batch.src_mask[i].reshape(1, -1),
                [batch.src_lengths[i]]
            )
            pred = [reverse_mapping[x.item()] for x in pred if x.item() != PAD]
            res.append([src, trg, pred])

            # print
            if pr:
                print(i)
                print(' '.join(src))
                print(' '.join(trg))
                print(' '.join(pred), f"({probs.exp().detach().item():.6%})")
                print()
    
    return res