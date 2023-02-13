import pickle
from eval import PAD
import random

file = "pickles/all-both.pickle"
with open(file, "rb") as fin:
    mapping, length, data = pickle.load(fin)
reverse_mapping = {}
for i in mapping:
    reverse_mapping[mapping[i]] = i

OHNO = mapping['�']

random.shuffle(data)
ct = 0
for i in range(len(data)):
    src, trg = data[i]
    if OHNO in src or OHNO in trg:
        ct += 1
        # print([reverse_mapping[x] for x in src if x != PAD], [reverse_mapping[x] for x in trg if x != PAD])
        # input()

print(ct, ct / len(data))