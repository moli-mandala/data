from tqdm import tqdm
import re
from collections import Counter, defaultdict

# abbrevs = {
#     'Ta': 'Tamil',
#     'Ma': 'Malayalam',
#     'Ko': 'Kota',
#     'To': 'Toda',
#     'Ka': 'Kannada',
#     'Kod': 'Kodagu',
#     'Tu': 'Tulu',
#     'Te': 'Telugu',
#     'Kol': 'Kolami',
#     'Nk': 'Naiki',
#     'Pa': 'Pali',
#     'Pkt': 'Prakrit',
#     'Mar': 'Marathi'
# }
entries = []

ct = Counter()
n = defaultdict(Counter)
last = ''
with open('dbia.txt', 'r') as fin, open('dbia_cleaned.txt', 'w') as fout:
    for line in tqdm(fin):
        m = re.findall(r'([A-Z][\S]*?).[\s\n]', line)
        # get first match group
        if m:
            ct[m[0]] += 1
            n[last][m[0]] += 1
            last = m[0]
        line = re.sub(r'([A-Z][\S]*?).[\s\n]', r'\n>\1\n\t', line)
        fout.write(line)
    
print(ct.most_common())
print(n)