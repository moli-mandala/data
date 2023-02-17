import torch
import numpy as np
from heapq import heappop, heappush
from collections import defaultdict

from model import GRUEncoderDecoder, EncoderDecoder

PAD = 0
SOS = 1
EOS = 2

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
            output.append(next_word.data.item())
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
            if next_word == EOS:
                break
        
        for i in range(len(model.encoder.layers)):
            attention_scores.append([model.encoder.layers[i].self_attn.attn.cpu().numpy()])
        for i in range(len(model.decoder.layers)):
            attention_scores.append([model.decoder.layers[i].self_attn.attn.cpu().numpy()])
            attention_scores.append([model.decoder.layers[i].src_attn.attn.cpu().numpy()])
    
    output = np.array(output)

    # cut off everything starting from </s> 
    # (only when EOS provided)
    if EOS is not None:
        first_eos = np.where(output==EOS)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, attention_scores, prod

def beam_decode(model, src, src_mask, src_lengths, beam_size=10, max_len=30):
    """Greedily decode a sentence."""

    res = []    

    if isinstance(model, GRUEncoderDecoder):
        with torch.no_grad():
            encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
            prev_y = torch.ones(1, 1).fill_(SOS).type_as(src)
            trg_mask = torch.ones_like(prev_y)

        output = []
        # attention_scores = []
        hidden = None

        beam = defaultdict(list)
        complete = []
        heappush(beam[0], (0, 0, [], prev_y, hidden))

        for i in range(max_len):
            l = len(beam[i])
            for j in range(min(l, beam_size)):
                d, prob, output, prev_y, hidden = heappop(beam[i])

                with torch.no_grad():
                    out, hidden, pre_output = model.decode(
                    encoder_hidden, encoder_final, src_mask,
                    prev_y, trg_mask, hidden)

                    # we predict from the pre-output layer, which is
                    # a combination of Decoder state, prev emb, and context
                    pr = model.generator(pre_output[:, -1])

                for n in range(len(pr[0])):
                    step, next_word = pr[0][n], n
                    prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
                    if next_word == EOS:
                        heappush(complete, [prob - step, output + [next_word]])
                        continue
                    heappush(beam[d + 1], (d + 1, prob - step, output + [next_word], prev_y, hidden))
        
        l = len(complete)
        for _ in range(min(l, beam_size)):
            i = heappop(complete)
            output = np.array(i[1])
            if EOS is not None:
                first_eos = np.where(output==EOS)[0]
                if len(first_eos) > 0:
                    output = output[:first_eos[0]]
            res.append([output, -i[0]]) 
        
    elif isinstance(model, EncoderDecoder): 
        with torch.no_grad():
            memory = model.encode(src, src_mask, src_lengths)
            ys = torch.zeros(1, 1).fill_(SOS).type_as(src.data)

        output = []
        # attention_scores = []
        hidden = None

        beam = defaultdict(list)
        complete = []
        heappush(beam[0], (0, 0, [], ys))

        for i in range(max_len):
            l = len(beam[i])
            for j in range(min(l, beam_size)):
                d, prob, output, ys = heappop(beam[i])

                with torch.no_grad():
                    out = model.decode(
                        memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
                    )
                    pr = model.generator(out[:, -1])

                for n in range(len(pr[0])):
                    step, next_word = pr[0][n], n
                    ys = torch.cat(
                        [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
                    )
                    if next_word == EOS:
                        heappush(complete, [prob - step, output + [next_word]])
                        continue
                    heappush(beam[d + 1], (d + 1, prob - step, output + [next_word], ys))
        
        l = len(complete)
        for _ in range(min(l, beam_size)):
            i = heappop(complete)
            output = np.array(i[1])
            if EOS is not None:
                first_eos = np.where(output==EOS)[0]
                if len(first_eos) > 0:
                    output = output[:first_eos[0]]
            res.append([output, -i[0]]) 
    
    return res
