import time
import math
from pycldf import Dataset
from segments.tokenizer import Tokenizer, Profile
from tqdm import tqdm
import random
import pickle
import argparse
from torch.optim.lr_scheduler import LambdaLR
import os

from sacrebleu.metrics import BLEU, CHRF, TER
import wandb
from typing import Optional, List

from model import *
from eval import *

PAD = 0
SOS = 1
EOS = 2

bleu = BLEU()
chrf = CHRF()
ter = TER()

def load_data(batch_size=16, file="pickles/all.pickle", unsq=False):
    """Load training data from a pickle."""
    # load data
    with open(file, "rb") as fin:
        mapping, length, data = pickle.load(fin)
    reverse_mapping = {}
    for i in mapping:
        reverse_mapping[mapping[i]] = i
    
    print(len(data[0][0]), len(data[0][1]), data[0])
    print("vocab size:", len(mapping))
    
    # shuffle data
    random.seed(42)
    random.shuffle(data)
    
    # make batches
    batched: List[Batch] = []
    for i in range(0, len(data), batch_size):
        # partition batch
        sz = min(batch_size, len(data) - i)
        batch = torch.LongTensor(data[i : i + batch_size])
        if USE_CUDA: batch.cuda()

        # get out src and trg tensors
        src, trg = batch[:, 0], batch[:, 1]
        ret = Batch((src, [length + 2] * sz), (trg, [length + 2] * sz), pad_index=PAD, unsq=unsq)
        batched.append(ret)
    
    # split into train and test
    train, test = batched[len(batched) // 5:], batched[:len(batched) // 5]

    return train, test, mapping, reverse_mapping

def run_epoch(data_iter: List[Batch], model, loss_compute: SimpleLossCompute, print_every: int=50):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)
        if isinstance(model, GRUEncoderDecoder):
            out, _, pre_output = pre_output
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

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

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
    heads: int,
    epochs: int,
    beam: Optional[int],
    save: bool
):
    # save
    if save:
        checkpoint = str(time.time())
        if not os.path.exists("checkpoints/"): os.mkdir("checkpoints/")
        os.mkdir(f"checkpoints/{checkpoint}/")

    # get data
    train, test, mapping, reverse_mapping = load_data(batch_size=batch_size, file=file, unsq=(architecture=="Transformer"))
    scheduler = None

    # make model and optimiser
    num_words = len(mapping)
    if architecture == "GRU":
        model = make_gru_model(num_words, num_words, emb_size=emb_size, num_layers=num_layers, hidden_size=hidden_size)
        criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        model = make_model(num_words, num_words, num_layers, emb_size, hidden_size, heads)
        # criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=PAD)
        criterion = LabelSmoothing(size=num_words, padding_idx=PAD, smoothing=0.0)
        optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = LambdaLR(
            optimizer=optim, lr_lambda=lambda step: rate(step, emb_size, 1, 1000)
        )
 
    dev_perplexities = []
    
    if USE_CUDA:
        model.cuda()

    for epoch in range(epochs):
        
        print("Epoch %d" % epoch)

        # train
        model.train()
        print("Learning rate:", optim.param_groups[0]["lr"])
        train_perplexity = run_epoch(train, model, SimpleLossCompute(model.generator, criterion, optim, scheduler))

        # evaluate
        model.eval()
        with torch.no_grad(): 

            # dev perplexity
            perplexity = run_epoch(test, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            dev_perplexities.append(perplexity)

            # BLEU
            res = get_predictions(model, test[0], reverse_mapping, maxi=100, pr=True, beam=beam)
            gold, pred = [[' '.join(x[1][1:-1]) for x in res]], [' '.join(x[2]) for x in res]
            print(f"[{gold[0][0]}] [{pred[0]}]")
            b, c, t = bleu.corpus_score(pred, gold), chrf.corpus_score(pred, gold), ter.corpus_score(pred, gold)

            # log
            print(f"Evaluation perplexity: {perplexity} ({b} / {c} / {t})")
            wandb.log({
                "dev_perplexity": perplexity,
                "train_perplexity": train_perplexity,
                "eval/bleu": b.score,
                "eval/chr": c.score,
                "eval/ter": t.score
            })
        
        if save:
            torch.save(model.state_dict(), f"checkpoints/{checkpoint}/model_all_{epoch}.pt")

    
    return dev_perplexities

def label_type(file: str):
    if "-both" in file: return "both"
    if "-left" in file: return "left"
    if "-right" in file: return "right"
    return "none"

def main():
    parser = argparse.ArgumentParser(description='Train models on reflex prediction.')
    parser.add_argument('-a', '--architecture', dest='arch', type=int, default=0)
    parser.add_argument('-f', '--file', dest='file', type=str, default="hindi.pickle")
    parser.add_argument('-lr', '--learning-rate', dest='lr', type=float, default=0.0003)
    parser.add_argument('-emb', '--embedding-size', dest='emb', type=int, default=32)
    parser.add_argument('-hid', '--hidden-size', dest='hid', type=int, default=None)
    parser.add_argument('-bs', '--batch-size', dest='bs', type=int, default=32)
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('-la', '--layers', dest='layers', type=int, default=1)
    parser.add_argument('-he', '--heads', dest='heads', type=int, default=8)
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('-be', '--beam', dest='beam', type=int, default=None)
    args = parser.parse_args()

    # check only pickles dir
    args.file = "pickles/" + args.file
    print(args)

    # set hyperparameters for training
    hyperparams = {
        "file": args.file,
        "architecture": "GRU" if args.arch == 0 else "Transformer",
        "batch_size": args.bs,
        "length": 20, # do not change this, it does nothing (is defined when preprocessing dataset)
        "emb_size": args.emb,
        "hidden_size": args.hid if args.hid is not None else args.emb,
        "lr": args.lr,
        "num_layers": args.layers,
        "heads": args.heads,
        "epochs": args.epochs,
        "beam": args.beam,
        "lang_labelling": label_type(args.file)
    }

    # logging
    wandb.init(
        project="Jambu",
        config=hyperparams
    )

    # train
    train(**hyperparams, save=args.save)

if __name__ == "__main__":
    main()