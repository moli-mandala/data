import torch
import numpy as np
from heapq import heappop, heappush
from collections import defaultdict

def beam_decode(model, src, src_mask, src_lengths, beam_size=10, max_len=30, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    # attention_scores = []
    hidden = None

    beam = defaultdict(list)
    complete = []
    heappush(beam[0], (0, 0, [], encoder_hidden, encoder_final, src_mask, prev_y, trg_mask, hidden))

    for i in range(max_len):
        # print(f'Depth: {i} ({len(beam[i])}, {len(complete)})')
        # input()
        # for x in complete:
        #     print(x)
        # input()
        l = len(beam[i])
        for j in range(min(l, beam_size)):
            d, prob, output, encoder_hidden, encoder_final, src_mask, prev_y, trg_mask, hidden = heappop(beam[i])

            with torch.no_grad():
                out, hidden, pre_output = model.decode(
                encoder_hidden, encoder_final, src_mask,
                prev_y, trg_mask, hidden)

                # we predict from the pre-output layer, which is
                # a combination of Decoder state, prev emb, and context
                pr = model.generator(pre_output[:, -1])

            # print(torch.max(pr, dim=1))
            for n in range(len(pr[0])):
                step, next_word = pr[0][n], n
                # next_word = next_word.data.item()
                prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
                # print('Depth:', d + 1)
                # print('Prob:', prob + step)
                # print('Output:', output + [next_word])
                if next_word == eos_index:
                    heappush(complete, [prob - step, output + [next_word]])
                    continue
                heappush(beam[d + 1], (d + 1, prob - step, output + [next_word], encoder_hidden,
                    encoder_final, src_mask, prev_y, trg_mask, hidden))

            # _, next_word = torch.max(prob, dim=1)
            # prod *= torch.exp(_)
            # next_word = next_word.data.item()
            # output.append(next_word)
            # prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
            # attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    res = []    
    l = len(complete)
    for _ in range(min(l, beam_size)):
        i = heappop(complete)
        output = np.array(i[1])
        if eos_index is not None:
            first_eos = np.where(output==eos_index)[0]
            if len(first_eos) > 0:
                output = output[:first_eos[0]]
        res.append([output, i[0]])  
    
    return res

def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
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
        if next_word == eos_index:
            break
    
    output = np.array(output)
        
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1), prod