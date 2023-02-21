from pycldf import Dataset
from segments.tokenizer import Tokenizer
import pickle
from tqdm import tqdm
from collections import defaultdict
import random

PAD = 0
SOS = 1
EOS = 2

def normalise(form, tokenizer, mapping, asterisk, length, pad, lang=None, lang2=None):
    # tokenize
    form = list(tokenizer(form).split())

    # convert chars to ints, lang to int
    for i, c in enumerate(form):
        if c not in mapping: mapping[c] = len(mapping)
        form[i] = mapping[c]
    if lang is not None:
        lang = "[" + lang + "]"
        if lang not in mapping: mapping[lang] = len(mapping)
        lang = mapping[lang]
    if lang2 is not None:
        lang2 = "[" + lang2 + "]"
        if lang2 not in mapping: mapping[lang2] = len(mapping)
        lang2 = mapping[lang2]
    
    # reconstruction label (yes or no)
    if not asterisk:
        if form[0] == '*': form = form[1:]

    # padding and stuff
    if len(form) > length: return None, mapping
    res = [SOS]
    offset = 2
    if lang2 is not None:
        res += [lang, lang2]
        offset += 2
    elif lang is not None:
        res += [lang]
        offset += 1

    res += form + [EOS]
    if pad: res += [PAD for _ in range(length + offset - len(res))]

    return res, mapping

def make_cognate_data(forms, length=20, number=1000000, saveto="", asterisk=True, pad=True):
    tokenizer = Tokenizer("../conversion/cdial-post.txt")

    # create mapping from char to num
    mapping = {'<pad>': PAD, '<sos>': SOS, '<eos>': EOS}

    # group by source
    by_source = defaultdict(list)
    lens = defaultdict(int)
    sources = {}
    for form in tqdm(forms):
        try:
            # get cognateset
            source = form.parameter
            id = source.id
            lang = form.data["Language_ID"]

            # append form
            sources[id] = source
            Y = form.data["Form"]
            by_source[id].append((Y, lang))
            lens[id] += 1

        except Exception as e:
            continue
    
    # probability of sampling from a group is # of pairs
    for id in lens:
        lens[id] = (lens[id] * (lens[id] - 1)) // 2
    keys = list(lens.keys())
    print("Keys:", len(keys))

    # sample groups
    sample = random.sample(keys, number, counts=list(lens.values()))

    # sample forms from groups
    data = []
    for key in sample:
        # get left and right
        left = random.randint(0, len(by_source[key]) - 1)
        right = left
        while left == right:
            right = random.randint(0, len(by_source[key]) - 1)

        # get data
        left, lang1 = by_source[key][left]
        right, lang2 = by_source[key][right]
        X, mapping = normalise(left, tokenizer, mapping, asterisk, length, pad, lang=lang1, lang2=lang2)
        Y, mapping = normalise(right, tokenizer, mapping, asterisk, length, pad)

        # append
        if X and Y: data.append([X, Y])
    
    print(f"{len(data)} data items loaded.")
    print(data[0])
    
    # save data
    with open(saveto, "wb") as fout:
        pickle.dump((mapping, length, data), fout)


def make_reflex_data(forms, length=20, saveto="", filter=None, asterisk=True, pad=True):
    """Save a dataset to a pickle."""
    tokenizer = Tokenizer("../conversion/cdial-post.txt")

    # create mapping from char to num
    mapping = {'<pad>': PAD, '<sos>': SOS, '<eos>': EOS}
    
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
                X, mapping = normalise(source.data["Name"], tokenizer, mapping, asterisk, length, pad, lang)
                Y, mapping = normalise(form.data["Form"], tokenizer, mapping, asterisk, length, pad)
                
                # append
                if X and Y: data.append([X, Y])

        except Exception as e:
            continue
    
    print(f"{len(data)} data items loaded.")
    print(data[0])
    
    # save data
    with open(saveto, "wb") as fout:
        pickle.dump((mapping, length, data), fout)

def make_datas(length: int=20):
    dataset = Dataset.from_metadata("../cldf/Wordlist-metadata.json")
    forms = dataset.objects('FormTable')

    # all Indo-Aryan data
    make_reflex_data(forms, length, saveto="pickles/all.pickle")

    # unpadded
    make_reflex_data(forms, length, saveto="pickles/all-unpadded.pickle", pad=False)

    # all Indo-Aryan data
    make_reflex_data(forms, length, saveto="pickles/all-nostar.pickle", asterisk=False)

    # only Hindi data
    make_reflex_data(forms, length, saveto="pickles/hindi.pickle", filter=["H"])

    # cognates
    make_cognate_data(forms, length, 1000000, saveto="pickles/cognates.pickle")
    make_cognate_data(forms, length, 1000000, saveto="pickles/cognates-nostar.pickle", asterisk=False)

def main():
    make_datas()

if __name__ == "__main__":
    main()