import time
import math
from pycldf import Dataset
from segments.tokenizer import Tokenizer, Profile
from tqdm import tqdm
import random
import pickle

from model import *

import wandb
import pandas as pd
from plotnine import ggplot, aes, geom_line
import matplotlib.pyplot as plt

PAD = 0
SOS = 1
EOS = 2

def load_data(batch_size=16, length=20, file=None, saveto="", filter=None):
    # load cldf dataset
    if file is None:
        dataset = Dataset.from_metadata("../cldf/Wordlist-metadata.json")
        forms, cognates, langs = dataset.objects('FormTable'), dataset.objects('CognateTable'), dataset.objects('LanguageTable')

        tokenizer = Tokenizer("../conversion/cdial.txt")

        # create mapping from char to num
        mapping = {'<pad>': PAD, '<sos>': SOS, '<eos>': EOS}

        def normalise(form, lang=None):
            # tokenize
            form = list(tokenizer(form).split())

            # convert chars to ints, lang to int
            for i, c in enumerate(form):
                if c not in mapping: mapping[c] = len(mapping)
                form[i] = mapping[c]
            if lang is not None:
                if lang not in mapping: mapping[lang] = len(mapping)
                lang = mapping[lang]

            # padding and stuff
            if len(form) > length: return None
            form = ([lang if lang is not None else SOS] + form +
                [EOS] + [PAD for _ in range(length - len(form))])
            return form
        
        # for each form, get source form
        data: list[list[list[int]]] = []
        for form in tqdm(forms):
            try:
                # get cognateset
                source = form.parameter
                lang = form.data["Language_ID"]

                # append only Indo-Aryan data
                if source and source.id[0].isdigit() and (lang in filter if filter is not None else True):
                    # get input (OIA form) and output (MIA/NIA form)
                    X = normalise(source.data["Name"], lang)
                    Y = normalise(form.data["Form"])
                    
                    # append
                    if X and Y: data.append([X, Y])

            except Exception as e:
                continue
        
        print(f"{len(data)} data items loaded.")
        print(data[0])
        random.shuffle(data)

        # make batches
        batched: list[Batch] = []
        for i in range(0, len(data), batch_size):
            # partition batch
            sz = min(batch_size, len(data) - i)
            batch = torch.LongTensor(data[i : i + batch_size])
            if USE_CUDA: batch.cuda()

            # get out src and trg tensors
            src, trg = batch[:, 0], batch[:, 1]
            ret = Batch((src, [length + 2] * sz), (trg, [length + 2] * sz), pad_index=PAD)
            batched.append(ret)
        
        # save data
        with open(saveto, "wb") as fout:
            pickle.dump((mapping, batched), fout)
    else:
        # load data
        with open(file, "rb") as fin:
            mapping, batched = pickle.load(fin)
    
    # split into train and test
    train, test = batched[len(batched) // 5:], batched[:len(batched) // 5]

    return train, test, mapping

def run_epoch(data_iter, model, loss_compute, print_every=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        
        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))

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

def train(
    file: str,
    architecture: str,
    batch_size: int,
    length: int,
    emb_size: int,
    hidden_size: int,
    lr: float,
    num_layers: int,
    lang_labelling: str
):
    """Train the simple copy task."""
    # get data
    train, test, mapping = load_data(batch_size=batch_size, length=length, file=file)

    # make model and optimiser
    num_words = len(mapping)
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(num_words, num_words, emb_size=emb_size, num_layers=num_layers, hidden_size=hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
 
    dev_perplexities = []
    
    if USE_CUDA:
        model.cuda()

    for epoch in range(10):
        
        print("Epoch %d" % epoch)

        # train
        model.train()
        run_epoch(train, model, SimpleLossCompute(model.generator, criterion, optim))

        # evaluate
        model.eval()
        with torch.no_grad(): 
            perplexity = run_epoch(test, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            wandb.log({"dev_perplexity": perplexity})
    
    return dev_perplexities

def main():
    length = 20
    batch_size = 32
    # load_data(length=length, batch_size=batch_size, saveto="pickles/all.pickle")
    hyperparams = {
        "file": "pickles/all.pickle",
        "architecture": "GRU",
        "batch_size": batch_size,
        "length": length,
        "emb_size": 32,
        "hidden_size": 32,
        "lr": 0.0003,
        "num_layers": 1,
        "lang_labelling": "left"
    }

    wandb.init(
        project="Jambu",
        config=hyperparams
    )

    train(**hyperparams)

if __name__ == "__main__":
    main()