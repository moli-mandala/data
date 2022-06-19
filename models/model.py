import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import csv
from tqdm import tqdm
import random
from decode import *
from sparsemax import Sparsemax
import argparse
from sklearn.decomposition import PCA
import unicodedata
import os
from collections import Counter
import nltk
from torch.utils.tensorboard import SummaryWriter

checkpoint = str(time.time())
writer = SummaryWriter()

global args
sparsemax = Sparsemax(dim=-1)

# we will use CUDA if it is available
USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0') # or set to 'cpu'
print("CUDA:", USE_CUDA)
print(DEVICE)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0., bidirectional=False):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=self.bidirectional, dropout=dropout)
        
    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        # print('GRU output:', output.size(), final.size())
        # torch.Size([10, 21, 128]) torch.Size([2, 10, 64])
        # torch.Size([10, 21, 64]) torch.Size([1, 10, 64])

        # we need to manually concatenate the final states for both directions
        if self.bidirectional:
            fwd_final = final[0:final.size(0):2]
            bwd_final = final[1:final.size(0):2]
            final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]
        # print('GRU output:', output.size(), final.size())
        # input()

        return output, final

class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True, bidirectional=False):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.bidirectional = bidirectional
        bridge = bidirectional
                 
        self.rnn = nn.GRU(emb_size + (2*hidden_size if bidirectional else hidden_size), hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.bridge = nn.Linear((2*hidden_size if bidirectional else hidden_size), hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + (2*hidden_size if bidirectional else hidden_size) + emb_size,
                                          hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # print("Encoder final:", encoder_final.size())
        # print("Encoder hidden:", encoder_hidden.size())

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        if self.bridge:
            return torch.tanh(self.bridge(encoder_final))
        else:
            return encoder_final

class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None, bidirectional=False):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = (2 * hidden_size if bidirectional else hidden_size) if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        if args.sparsemax:
            # Mask out invalid positions.
            # The mask marks valid positions so we invert it using `mask & 0`.
            scores.data.masked_fill_(mask == 0, -1000000000000)
            
            # Turn scores to probabilities.
            # print(scores.size())
            alphas = sparsemax(scores)
        else:
            # SOFTMAX VERSION
            scores.data.masked_fill_(mask == 0, -float('inf'))
            alphas = F.softmax(scores, dim=-1)
        
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1, bidirectional=True):
    "Helper: Construct a model from hyperparameters."
    attention = BahdanauAttention(hidden_size, bidirectional=bidirectional)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab))
    print(model)

    return model.cuda() if USE_CUDA else model

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()
        
        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()

def run_epoch(data_iter, model, loss_compute, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    # print(len(data_iter))
    for i, batch in enumerate(tqdm(data_iter), 1):
        # print(i, batch.ntokens, total_tokens)
        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        
        # if model.training and i % print_every == 0:
        #     elapsed = time.time() - start
        #     print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
        #             (i, loss / batch.nseqs, print_tokens / elapsed))
        #     start = time.time()
        #     print_tokens = 0

    # print(total_tokens)
    return math.exp(total_loss / float(total_tokens))

def get_buru_data(batch_size=16, length=20, pad_index=0, sos_index=1, eos_index=2):
    data = []
    chars = set()

    with open('../data/burushaski.tsv', 'r') as fin:
        reader = csv.reader(fin, delimiter='\t')
        next(reader)
        for row in reader:
            row = [unicodedata.normalize('NFC', x) for x in row]
            data.append([list(row[0]) + ['+'] + list(row[5].replace('nc', 'uc')), list(row[1])])
            for i in data[-1][0] + data[-1][1]:
                chars.add(i)
    
    # make encoding to numbers
    to_num = {}
    to_char = {}
    for i, j in enumerate(chars):
        to_num[j] = i + 3
        to_char[i + 3] = j
    to_char[2] = 'EOS'
    to_char[1] = 'SOS'
    to_char[0] = 'PAD'

    dat = []
    for i, j in enumerate(data):
        dat.append([[sos_index] + [to_num[x] for x in j[0]] + [eos_index] + [pad_index] * (length - len(j[0])),
            [sos_index] + [to_num[x] for x in j[1]] + [eos_index] + [pad_index] * (length - len(j[1]))])
    random.shuffle(dat)

    tot, eval_data = [], []
    sp = len(dat) // 5
    for i in range(0, sp, batch_size):
        if i >= len(dat): continue
        sz = min(batch_size, len(dat) - i)
        batch = torch.LongTensor(dat[i:i + sz])
        # print(batch.size())
        batch = batch.cuda() if USE_CUDA else batch
        src, trg = batch[:, 0, 1:], batch[:, 1]
        src_lengths = [length + 1] * sz
        trg_lengths = [length + 2] * sz
        ret = Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)
        eval_data.append(ret)
    for i in range(sp, len(dat), batch_size):
        if i >= len(dat): continue
        sz = min(batch_size, len(dat) - i)
        batch = torch.LongTensor(dat[i:i + sz])
        # print(batch.size())
        batch = batch.cuda() if USE_CUDA else batch
        src, trg = batch[:, 0, 1:], batch[:, 1]
        src_lengths = [length + 1] * sz
        trg_lengths = [length + 2] * sz
        ret = Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)
        tot.append(ret)
    return tot, eval_data, to_char
    


def get_data(batch_size=16, length=20, pad_index=0, sos_index=1, eos_index=2):
    sanskrit = {}
    chars = []

    # collect Sanskrit etyma headwords
    with open('../cldf/cognates.csv', 'r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            if row[1] == 'Indo-Aryan':
                sanskrit[row[0]] = row[2]
    
    # get reflexes from Hindi only, and pair with first word span in Sanskrit etymon
    data = []
    print('Reading data')
    with open('../cldf/forms.csv', 'r') as fin:
        reader = csv.reader(fin)
        for row in tqdm(reader):
            if 'dedr' in row[0] or row[2] not in sanskrit or not row[3]:
                continue
            if args.langs:
                if row[1] not in args.langs:
                    continue
            if row[1] == 'Indo-Aryan':
                continue
            lang = f'[{row[1]}]'
            if lang not in chars: chars.append(lang)
            for i in sanskrit[row[2]]:
                if i not in chars: chars.append(i)
            for i in row[3]:
                if i not in chars: chars.append(i)
            if len(row[3]) > 1:
                src = list(sanskrit[row[2]].split(',')[0].strip(',- 1234567890')) + [lang]
                trg = list(row[3])
                data.append([src, trg])
    print(data[:10])
    
    # make encoding to numbers
    to_num = {}
    to_char = {}
    for i, j in enumerate(chars):
        to_num[j] = i + 3
        to_char[i + 3] = j
    to_char[2] = 'EOS'
    to_char[1] = 'SOS'
    to_char[0] = 'PAD'
    
    # encoding sounds to numbers for embeddings, add padding
    # basically we want the pattern SOS [word] EOS PAD...
    # encoder input doesn't need SOS (start of string) tag
    dat = []
    for i, j in enumerate(data):
        if len(j[0]) > length or len(j[1]) > (length - 1) or len(j[0]) == 0 or len(j[1]) == 0:
            # print(len(j[0]), len(j[1]))
            continue
        dat.append([[sos_index] + [to_num[x] for x in j[0]] + [eos_index] + [pad_index] * (length - len(j[0])),
            [sos_index] + [to_num[x] for x in j[1]] + [eos_index] + [pad_index] * (length - len(j[1]))])
    random.shuffle(dat)
    # print(data[:5])
    # input()

    tot, eval_data = [], []
    sp = len(dat) // 5
    for i in range(0, sp, batch_size):
        if i >= len(dat): continue
        sz = min(batch_size, len(dat) - i)
        batch = torch.LongTensor(dat[i:i + sz])
        # print(batch.size())
        batch = batch.cuda() if USE_CUDA else batch
        src, trg = batch[:, 0, 1:], batch[:, 1]
        src_lengths = [length + 1] * sz
        trg_lengths = [length + 2] * sz
        ret = Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)
        eval_data.append(ret)
    for i in range(sp, len(dat), batch_size):
        if i >= len(dat): continue
        sz = min(batch_size, len(dat) - i)
        batch = torch.LongTensor(dat[i:i + sz])
        # print(batch.size())
        batch = batch.cuda() if USE_CUDA else batch
        src, trg = batch[:, 0, 1:], batch[:, 1]
        src_lengths = [length + 1] * sz
        trg_lengths = [length + 2] * sz
        ret = Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)
        tot.append(ret)
    return tot, eval_data, to_char

def data_gen(num_words=11, batch_size=16, num_batches=100, length=10, pad_index=0, sos_index=1):
    """Generate random data for a src-tgt copy task."""
    for i in range(num_batches):
        data = torch.from_numpy(
          np.random.randint(1, num_words, size=(batch_size, length)))
        # print(data)
        # input()
        data[:, 0] = sos_index
        data = data.cuda() if USE_CUDA else data
        src = data[:, 1:]
        trg = data
        print(src, trg)
        src_lengths = [length-1] * batch_size
        trg_lengths = [length] * batch_size
        ret = Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)
        # print((src, src_lengths), (trg, trg_lengths))
        # input()
        yield ret

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
  

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab[i] for i in x]

    return [str(t) for t in x]

def print_probs(example_iter, model, fout, n=2, max_len=100, 
                   sos_index=1, 
                   src_eos_index=2, 
                   trg_eos_index=2, mapping=None):
    """Print probabilities"""

    model.eval()
    count = 0
    print()
    fout.write('\n')
    res = []
        
    for _, batch in enumerate(tqdm(example_iter)):    

        for i in range(len(batch.src)):
            src = batch.src.cpu().numpy()[i, :]
            trg = batch.trg_y.cpu().numpy()[i, :]
            # print(src, trg)

            # remove </s> (if it is there)
            src = src[:-1] if src[-1] == src_eos_index else src
            trg = trg[:-1] if trg[-1] == trg_eos_index else trg  

            _, _, true_prob = force_decode(
                model, torch.reshape(batch.src[i], (1, -1)), torch.reshape(batch.src_mask[i], (1, -1)),
                [batch.src_lengths[i]], torch.reshape(batch.trg_y[i], (1, -1)),
                torch.reshape(batch.trg_mask[i], (1, -1)), [batch.trg_lengths[i]],
                max_len=max_len, sos_index=sos_index, eos_index=trg_eos_index)
            src_text = "".join([x for x in lookup_words(src, vocab=mapping) if x not in ['EOS', 'PAD']])
            trg_text = "".join([x for x in lookup_words(trg, vocab=mapping) if x not in ['EOS', 'PAD']])
            res.append([src_text, trg_text, true_prob.item(), true_prob.item() / len(src_text)])
    
    for x in sorted(res, key=lambda x: x[3]):
        fout.write(f'{x[0]}\t{x[1]}\t{x[2]}\t{x[3]}\n')

def print_examples(example_iter, model, fout, n=2, max_len=100, 
                   sos_index=1, 
                   src_eos_index=2, 
                   trg_eos_index=2, mapping=None, plot=0):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()
    fout.write('\n')
    plotted = 0
    results = []

    for i, batch in enumerate(example_iter):

        for i in range(len(batch.src)):
            src = batch.src.cpu().numpy()[i, :]
            trg = batch.trg_y.cpu().numpy()[i, :]
            # print(src, trg)

            # remove </s> (if it is there)
            src = src[:-1] if src[-1] == src_eos_index else src
            trg = trg[:-1] if trg[-1] == trg_eos_index else trg

            if args.beam:
                beam = beam_decode(
                    model, torch.reshape(batch.src[i], (1, -1)), torch.reshape(batch.src_mask[i], (1, -1)),
                    [batch.src_lengths[i]],
                    max_len=max_len, sos_index=sos_index, eos_index=trg_eos_index, beam_size=10)
            result, attentions, prob = greedy_decode(
                model, torch.reshape(batch.src[i], (1, -1)), torch.reshape(batch.src_mask[i], (1, -1)),
                [batch.src_lengths[i]],
                max_len=max_len, sos_index=sos_index, eos_index=trg_eos_index)
            src_clean = [x for x in lookup_words(src, vocab=mapping) if x not in ['EOS', 'PAD']]
            true_result, true_attentions, true_prob = force_decode(
                model, torch.reshape(batch.src[i], (1, -1)), torch.reshape(batch.src_mask[i], (1, -1)),
                [batch.src_lengths[i]], torch.reshape(batch.trg_y[i], (1, -1)),
                torch.reshape(batch.trg_mask[i], (1, -1)), [batch.trg_lengths[i]],
                max_len=max_len, sos_index=sos_index, eos_index=trg_eos_index)
            if plotted < plot:
                plot_heatmap(src_clean, lookup_words(true_result, vocab=mapping), true_attentions[0].T[:len(src_clean), :len(result)])
                plotted += 1
            src_text = "".join([x for x in lookup_words(src, vocab=mapping) if x not in ['EOS', 'PAD']])
            trg_text = "".join([x for x in lookup_words(trg, vocab=mapping) if x not in ['EOS', 'PAD']])
            result_text = "".join(lookup_words(result, vocab=mapping))
            fout.write(f'{src_text}\t{trg_text}\t({true_prob.item()})\t{result_text}\t({prob.item()})\n')
            if args.beam:
                print("".join(lookup_words(beam[0][0], vocab=mapping)), result_text)
                results.append([trg_text, "".join(lookup_words(beam[0][0], vocab=mapping))])
            else:
                results.append([trg_text, result_text])
                # for x, pr in beam:
                #     fout.write(f'\t\t{"".join(lookup_words(x, vocab=mapping))}\t({pr})\n') 
            count += 1
            if count % 10 == 0:
                print(count)
            if count == n:
                break 

        if count == n:
            break
    return results

def eval_model():
    batch_size = args.bs
    data, eval_data, to_char = get_data(batch_size=batch_size)
    num_words = len(to_char)
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(num_words, num_words, emb_size=args.emb, hidden_size=args.emb, num_layers=args.layers, bidirectional=True if args.bidirectional else False)

    data = []
    fout = open('results.txt', 'w')
    for i in tqdm(range(30, 30)):
        model.load_state_dict(torch.load(f"checkpoints/1649910095.3520331/model_all_{i}.pt"))
        model.eval()
        per, wer = 0, 0
        with torch.no_grad(): 
            # perplexity = run_epoch(eval_data, model,
            #                         SimpleLossCompute(model.generator, criterion, None))
            # print("Evaluation perplexity: %f" % perplexity)
            result = print_examples(eval_data, model, fout, n=100, max_len=20, mapping=to_char, plot=0)
            for x, y in result:
                per += nltk.edit_distance(x, y) / max(len(x), len(y))
                wer += (x != y)
            per /= len(result)
            wer /= len(result)
        data.append([i, 64, per, wer])
    model = make_model(num_words, num_words, emb_size=32, hidden_size=32, bidirectional=True if args.bidirectional else False)
    for i in tqdm(range(13, 13)):
        model.load_state_dict(torch.load(f"checkpoints/1649955551.0526102/model_all_{i}.pt"))
        model.eval()
        per, wer = 0, 0
        with torch.no_grad(): 
            # perplexity = run_epoch(eval_data, model,
            #                         SimpleLossCompute(model.generator, criterion, None))
            # print("Evaluation perplexity: %f" % perplexity)
            result = print_examples(eval_data, model, fout, n=1000, max_len=20, mapping=to_char, plot=0)
            for x, y in result:
                per += nltk.edit_distance(x, y) / max(len(x), len(y))
                wer += (x != y)
            per /= len(result)
            wer /= len(result)
        data.append([i, 32, per, wer])
    model = make_model(num_words, num_words, emb_size=128, hidden_size=128, bidirectional=True if args.bidirectional else False)
    for i in tqdm(range(0, 7)):
        model.load_state_dict(torch.load(f"checkpoints/1649957496.844018/model_all_{i}.pt"))
        model.eval()
        per, wer = 0, 0
        with torch.no_grad(): 
            # perplexity = run_epoch(eval_data, model,
            #                         SimpleLossCompute(model.generator, criterion, None))
            # print("Evaluation perplexity: %f" % perplexity)
            result = print_examples(eval_data, model, fout, n=100, max_len=20, mapping=to_char, plot=0)
            for x, y in result:
                per += nltk.edit_distance(x, y) / max(len(x), len(y))
                wer += (x != y)
            per /= len(result)
            wer /= len(result)
        data.append([i, 128, per, wer])
    fout.close()
    print(data)
    # df = pd.DataFrame(data, columns=['Epoch', 'Emb', 'PER', 'WER'])
    # a = (
    #     ggplot(df, aes(x='Epoch', y='PER', color='Emb'))
    #     + geom_line() # add an arrow to the end of the line
    #     + labs(x='epoch', y='PER')
    # )
    # a.draw()
    # plt.show()

def train_copy_task():
    """Train the simple copy task."""
    batch_size = args.bs
    data, eval_data, to_char = get_data(batch_size=batch_size)
    # data = data[:1]
    # eval_data = eval_data[:1]
    num_words = len(to_char)
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(num_words, num_words, emb_size=args.emb, hidden_size=args.emb, num_layers=args.layers, bidirectional=True if args.bidirectional else False)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(to_char)
    if not args.no_save:
        with open(f'checkpoints/{checkpoint}/README.txt', 'w') as fout:
            fout.write(f'Bidirectional: {args.bidirectional}\n')
            fout.write(f'Embedding size: {args.emb}\n')
            fout.write(f'Hidden size: {args.emb}\n')
            fout.write(f'Batch size: {batch_size}\n')
            fout.write(f'Learning rate: {args.lr}\n')
            fout.write(f'Langs: {str(args.langs)}\n')
            fout.write(f'Sparsemax: {args.sparsemax}\n')
    dev_perplexities = []

    if not args.no_save:
        fout = open(f'checkpoints/{checkpoint}/result.txt', 'w')
    
    if USE_CUDA:
        model.cuda()

    for epoch in range(args.epochs):
        
        print("Epoch %d" % epoch)

        if not args.no_save:
            fout.write("Epoch %d\n" % epoch)

        # train
        model.train()
        # data = data_gen(num_words=num_words, batch_size=32, num_batches=100)
        train_loss = run_epoch(data, model,
                  SimpleLossCompute(model.generator, criterion, optim))
        writer.add_scalar("train / loss", train_loss, epoch)

        # evaluate
        model.eval()
        # X = model.src_embed.weight.detach().numpy()
        # pca = PCA(n_components=2)
        # X_r = pca.fit(X).transform(X).tolist()
        # for i in to_char:
        #     y = to_char[i]
        #     X_r[i].append(y)
        #     if '[' in y:
        #         X_r[i].append('special')
        #     else:
        #         X_r[i].append('sound')
        # print(X_r[:5])
        # print(
        #     "explained variance ratio (first two components): %s"
        #     % str(pca.explained_variance_ratio_)
        # )
        # df = pd.DataFrame(X_r, columns=['1', '2', 'Label', 'Type'])
        # p = ggplot(df, aes(x='1', y='2', label='Label', color='Type')) + geom_text()
        # p.draw()
        # plt.show()
        
        # input()
        with torch.no_grad(): 
            perplexity = run_epoch(eval_data, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            writer.add_scalar("dev / loss", perplexity, epoch)
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            if not args.no_save:
                if args.check_probs:
                    print_probs(eval_data, model, fout, n=50, max_len=20, mapping=to_char)
                else:
                    per, wer = 0, 0
                    result = print_examples(eval_data, model, fout, n=100, max_len=20, mapping=to_char, plot=0)
                    for x, y in result:
                        per += nltk.edit_distance(x, y) / max(len(x), len(y))
                        wer += (x != y)
                    per /= len(result)
                    wer /= len(result)
            
        with open('stats.tsv', 'a') as stats:
            if args.no_save:
                stats.write('\t'.join(list(map(str, ([f"{checkpoint}/model_all_{epoch}.pt", epoch, args.bidirectional, args.emb, batch_size, args.lr,
                    str(args.langs), args.sparsemax, epoch, perplexity])))) + '\n')
            else:
                stats.write('\t'.join(list(map(str, ([f"{checkpoint}/model_all_{epoch}.pt", epoch, args.bidirectional, args.emb, batch_size, args.lr,
                    str(args.langs), args.sparsemax, epoch, perplexity, per, wer])))) + '\n')
                writer.add_scalar("dev / per", per, epoch)
                writer.add_scalar("dev / wer", wer, epoch)

        # save model
        if not args.no_save:
            torch.save(model.state_dict(), f"checkpoints/{checkpoint}/model_all_{epoch}.pt")

    if not args.no_save:
        fout.close()
        writer.add_hparams({
            "bidirectional": args.bidirectional, "embedding_size": args.emb, "hidden_size": args.emb,
            "batch_size": batch_size, "lr": args.lr, "langs": str(args.langs), "sparsemax": args.sparsemax,
            "layers": args.layers, "epochs": args.epochs
        }, {"loss": perplexity, "per": per, "wer": wer})
    
    return dev_perplexities

def main():
    if not args.no_save:
        os.mkdir(f'checkpoints/{checkpoint}')

    # train the copy task
    dev_perplexities = train_copy_task()

    # def plot_perplexity(perplexities):
    #     """plot perplexities"""
    #     plt.title("Perplexity per Epoch")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Perplexity")
    #     plt.plot(perplexities)
        
    # plot_perplexity(dev_perplexities)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train encoder-decoder model.')
    parser.add_argument('-l', '--langs', dest='langs', nargs='+', help='Languages')
    parser.add_argument('--sparsemax', dest='sparsemax', action='store_true',
                   help='Use sparsemax in attention probability calculation instead of softmax.')
    parser.add_argument('--beam', dest='beam', action='store_true',
                   help='Show beam search outputs on eval set as well.')
    parser.add_argument('--bidirectional', dest='bidirectional', action='store_true',
                   help='Bidirectional RNN.')
    parser.add_argument('--check-probs', dest='check_probs', action='store_true',
                   help='Check probabilities of input data to find anomalies.')
    parser.add_argument('--no-save', dest='no_save', action='store_true',
                   help='Don\'t save models.')
    parser.add_argument('-lr', '--learning-rate', dest='lr', type=float, default=0.01)
    parser.add_argument('-emb', '--embedding-hidden-size', dest='emb', type=int, default=64)
    parser.add_argument('-bs', '--batch-size', dest='bs', type=int, default=1024)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('-la', '--layers', dest='layers', type=int, default=1)
    parser.add_argument('-eval', '--evaluate', dest='eval', action='store_true',
                   help='Evaluate.')
    # parser.add_argument('-eval', '--evaluate', dest='eval', type=str)
    args = parser.parse_args()
    print(args)
    
    if args.eval:
        eval_model()
        # eval_model("checkpoints/1649957496.844018/model_all_1.pt")
        # eval_model("checkpoints/1649955551.0526102/model_all_13.pt")
    else:
        main()
    writer.close()