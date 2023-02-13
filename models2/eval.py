import torch
import numpy as np
from model import Batch

PAD = 0
SOS = 1
EOS = 2

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm

def greedy_decode(model, src, src_mask, src_lengths, max_len=100):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(SOS).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    prod = 0
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
        prod -= _
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
        if next_word == EOS:
            break
    
    output = np.array(output)
     
    # cut off everything starting from </s> 
    # (only when EOS provided)
    if EOS is not None:
        first_eos = np.where(output==EOS)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1), prod

def get_predictions(model, batch: Batch, reverse_mapping: dict):
    print(batch.src, batch.src_lengths)
    for i in range(len(batch.src_lengths)):
        pred, attns, probs = greedy_decode(
            model, batch.src[i].reshape(1, -1), batch.src_mask[i].reshape(1, -1), [batch.src_lengths[i]]
        )
        src = [reverse_mapping[x.item()] for x in batch.src[i] if x.item() != PAD]
        trg = [reverse_mapping[x.item()] for x in batch.trg[i] if x.item() != PAD]
        pred = [reverse_mapping[x.item()] for x in pred if x.item() != PAD]
        print(src, trg, pred)