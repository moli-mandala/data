import time
import math
from pycldf import Dataset
from segments.tokenizer import Tokenizer, Profile
from tqdm import tqdm
import random
import pickle

from sacrebleu.metrics import BLEU, CHRF, TER

from model import *
from eval import *

import wandb

PAD = 0
SOS = 1
EOS = 2

bleu = BLEU()
chrf = CHRF()
ter = TER()

def load_data(batch_size=16, file="pickles/all.pickle"):
    """Load training data from a pickle."""
    # load data
    with open(file, "rb") as fin:
        mapping, length, data = pickle.load(fin)
    reverse_mapping = {}
    for i in mapping:
        reverse_mapping[mapping[i]] = i
    
    # shuffle data
    random.seed(42)
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
    
    # split into train and test
    train, test = batched[len(batched) // 5:], batched[:len(batched) // 5]

    return train, test, mapping, reverse_mapping

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

def train(
    file: str,
    architecture: str,
    batch_size: int,
    length: int,
    emb_size: int,
    hidden_size: int,
    lr: float,
    num_layers: int,
    lang_labelling: str,
    epochs: int
):
    """Train the simple copy task."""
    # get data
    train, test, mapping, reverse_mapping = load_data(batch_size=batch_size, file=file)

    # make model and optimiser
    num_words = len(mapping)
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(num_words, num_words, emb_size=emb_size, num_layers=num_layers, hidden_size=hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
 
    dev_perplexities = []
    
    if USE_CUDA:
        model.cuda()

    for epoch in range(epochs):
        
        print("Epoch %d" % epoch)

        # train
        model.train()
        run_epoch(train, model, SimpleLossCompute(model.generator, criterion, optim))

        # evaluate
        model.eval()
        with torch.no_grad(): 

            # dev perplexity
            perplexity = run_epoch(test, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            dev_perplexities.append(perplexity)

            # BLEU
            res = get_predictions(model, test[0], reverse_mapping, maxi=100)
            gold, pred = [[' '.join(x[1][1:-1])] for x in res], [' '.join(x[2]) for x in res]
            print(gold[0], pred[0])
            b, c, t = bleu.corpus_score(pred, gold), chrf.corpus_score(pred, gold), ter.corpus_score(pred, gold)

            # log
            print(f"Evaluation perplexity: {perplexity} ({b} / {c} / {t})")
            wandb.log({
                "dev_perplexity": perplexity,
                "eval/bleu": b.score,
                "eval/chr": c.score,
                "eval/ter": t.score
            })

    
    return dev_perplexities

def main():
    # set hyperparameters for training
    hyperparams = {
        "file": "pickles/hindi.pickle",
        "architecture": "GRU",
        "batch_size": 32,
        "length": 20, # do not change this, it does nothing (is defined when preprocessing dataset)
        "emb_size": 32,
        "hidden_size": 32,
        "lr": 0.0003,
        "num_layers": 1,
        "epochs": 20,
        "lang_labelling": "none"
    }

    # logging
    wandb.init(
        project="Jambu",
        config=hyperparams
    )

    # train
    train(**hyperparams)

if __name__ == "__main__":
    main()