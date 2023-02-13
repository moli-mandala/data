from pycldf import Dataset
from segments.tokenizer import Tokenizer
import pickle
from tqdm import tqdm

PAD = 0
SOS = 1
EOS = 2

def make_data(length=20, saveto="", filter=None, lang_label="none"):
    """Save a dataset to a pickle."""
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
        start, end = SOS, EOS
        if lang is not None:
            if lang_label in ["left", "both"]: start = lang
            if lang_label in ["right", "both"]: end = lang

        form = ([start] + form + [end] + [PAD for _ in range(length - len(form))])
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
    
    # save data
    with open(saveto, "wb") as fout:
        pickle.dump((mapping, length, data), fout)

def make_datas(length: int=20):
    # all Indo-Aryan data
    make_data(length, saveto="pickles/all-left.pickle", lang_label="left")
    make_data(length, saveto="pickles/all-right.pickle", lang_label="right")
    make_data(length, saveto="pickles/all-both.pickle", lang_label="both")

    # only Hindi data
    make_data(length, saveto="pickles/hindi.pickle", filter=["H"], lang_label="none")

def main():
    make_datas()

if __name__ == "__main__":
    main()