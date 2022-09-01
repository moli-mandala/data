import pickle
import os
import urllib.request
import re
import json
import copy
import csv
from bs4 import BeautifulSoup
from collections import defaultdict
from enum import Enum
from tqdm import tqdm

from abbrevs import abbrevs

TOTAL_PAGES = 514
ERR = False

def remove_text_between_parens(text): # lazy: https://stackoverflow.com/questions/37528373/how-to-remove-all-text-between-the-outer-parentheses-in-a-string
    n = 1  # run at least once
    while n:
        text, n = re.subn(r'[\(\[][^()]*[\)\]]', '', text)  # remove non-nested/flat balanced parts
    return text

# useful regexes
langs = '(' + "|".join(sorted(list(abbrevs.keys()), key=lambda x: -len(x))) + r')\.?'
regex = re.compile(r'(<i>|<b>|^)*' + langs + r'(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=((<i>|<b>)*' + langs + r'|DED|$))')
lemmata = re.compile(r'(<b>|^)(.*?)</b>(.*?)((?=<b>)|$)')
formatter = re.compile(r'<.*?>')

# response caching logic
soups = []
cached = False
if os.path.exists('dedr.pickle'):
    with open('dedr.pickle', 'rb') as fin:
        soups = pickle.load(fin)
    cached = True
print('Caching?', cached)

# file
fout = open('dedr2.csv', 'w')
writer = csv.writer(fout)

count = 1

# go through each entire digitised page
for page in tqdm(range(1, TOTAL_PAGES + 1)):
    if ERR: print(page)
    
    # get content
    link = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + str(page)
    resp = None
    if not cached: resp = urllib.request.urlopen(link)
    if ERR: print('fetched page')

    # html parse, split into entries
    soup = None
    if cached: soup = BeautifulSoup(soups[page - 1], 'html5lib')
    else:
        soup = BeautifulSoup(resp, 'html5lib').find(class_='hw_result')
        soups.append(str(soup))
    soup = str(soup).split('<number>')
    if ERR: print('made soup')

    # for each entry on the page, parse
    for entry in soup:

        # prettify
        entry = BeautifulSoup('<number>' + entry, 'html5lib')

        # only if this is an actual entry
        if entry.find('number'):

            # store and get rid of number
            number = entry.find('number').text
            entry.find('number').decompose()
            entry = str(entry)
            if ERR: print(entry)

            # go through each span: one span has only one language tag at the start
            for x in regex.findall(entry):
                span = x[2]
                lang = abbrevs[x[1].replace('.', '\.')]
                if ERR: print('lang', x)
                
                # get every forms + gloss pairing (delineated by bold tags)
                for y in lemmata.finditer(span):
                    if ERR: print('    lemma', y)
                    forms = [form.strip().replace(' /', '/').replace('/ ', '/') for form in y.group(2).split(',')]
                    gloss = y.group(3).strip(' ;,')

                    for form in forms:
                        if ERR: print('        form', form)
                        form = formatter.sub('', form).strip()
                        if lang == 'OIA' and (form == '' or 'no.' in form):
                            continue
                        writer.writerow([lang, number, form, gloss, '', '', form, '', '', 'dedr'])
                        count += 1

                    if ERR: print('        done with forms')
                if ERR: print('    done with spans')
            if ERR: print('done with lang')
    
    if ERR: print('deleting')
    if not cached: del resp
    if ERR: print('deleted')

# close file
fout.close()

if not cached:
    with open('dedr.pickle', 'wb') as fout:
        pickle.dump(soups, fout)