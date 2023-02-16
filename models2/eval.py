import torch
import torch.nn as nn
import numpy as np
from model import Batch, GRUEncoderDecoder, EncoderDecoder

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

def greedy_decode(model, src, src_mask, src_lengths, max_len=100):
    """Greedily decode a sentence."""
    output = []
    attention_scores = []
    prod = 0

    if isinstance(model, GRUEncoderDecoder):
        with torch.no_grad():
            encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
            prev_y = torch.ones(1, 1).fill_(SOS).type_as(src)
            trg_mask = torch.ones_like(prev_y)

        hidden = None

        for i in range(max_len):
            with torch.no_grad():
                out, hidden, pre_output = model.decode(
                encoder_hidden, encoder_final, src_mask,
                prev_y, trg_mask, hidden)

                # we predict from the pre-output layer, which is
                # a combination of Decoder state, prev emb, and context
                prob = model.generator(pre_output[:, -1])

            _, next_word = torch.max(prob, dim=1)
            prod += _
            next_word = next_word.data.item()
            output.append(next_word)

            prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
            attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
            if next_word == EOS:
                break
        
        output = np.array(output)
        
    elif isinstance(model, EncoderDecoder):
        memory = model.encode(src, src_mask, src_lengths)
        ys = torch.zeros(1, 1).fill_(SOS).type_as(src.data)

        for i in range(max_len):
            out = model.decode(
                memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
            )
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            prod += _
            next_word = next_word.data[0]
            output.append(next_word)

            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
            if next_word == EOS:
                break
     
    # cut off everything starting from </s> 
    # (only when EOS provided)
    if EOS is not None:
        first_eos = np.where(output==EOS)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1) if attention_scores else [], prod

def get_predictions(model, batch: Batch, reverse_mapping: dict, maxi=None, pr=False):
    """Greedy decode predictions from a batch."""
    length = len(batch.src_lengths)
    res = []
    if maxi is not None:
        length = min(maxi, length)
    for i in range(length):
        # greedy decode
        pred, attns, probs = greedy_decode(
            model, batch.src[i].reshape(1, -1), batch.src_mask[i].reshape(1, -1), [batch.src_lengths[i]]
        )
        src = [reverse_mapping[x.item()] for x in batch.src[i] if x.item() != PAD]
        trg = [reverse_mapping[x.item()] for x in batch.trg[i] if x.item() != PAD]
        pred = [reverse_mapping[x.item()] for x in pred if x.item() != PAD]
        res.append([src, trg, pred])

        # print
        if pr:
            print(src, trg, pred, probs.exp())
    
    return res