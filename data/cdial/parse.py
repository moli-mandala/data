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

TOTAL_PAGES = 836

def remove_text_between_parens(text): # lazy: https://stackoverflow.com/questions/37528373/how-to-remove-all-text-between-the-outer-parentheses-in-a-string
    n = 1  # run at least once
    while n:
        text, n = re.subn(r'[\(\[][^()]*[\)\]]', '', text)  # remove non-nested/flat balanced parts
    return text

reflexes = defaultdict(list)

# this is such a big brain regex
regex = r'(?<=[\W\.])(' + f'{"|".join(list(abbrevs.keys()))}' + r')\.'

at_map = {
    'l': 'ɬ',
    'e_': 'ɛ̄',
    'e': 'ɛ',
    'e/': 'ɛ́',
    'e\\': 'ɛ̀',
    'g': 'ɣ',
    '*l': 'ʌ',
    '*l/': 'ʌ́',
    '*l\\': 'ʌ̀',
    '*l_': 'ʌ̄',
    'b': 'β',
    'd': 'δ',
    'x': 'x'
}
al = str(at_map.values())

fout = open('cdial.csv', 'w')
writer = csv.writer(fout)

# response caching logic
soups = []
cached = False
if os.path.exists('cdial.pickle'):
    with open('cdial.pickle', 'rb') as fin:
        soups = pickle.load(fin)
    cached = True

# go through each entire digitised page
for page in tqdm(range(1, TOTAL_PAGES + 1)):
    
    # get content
    link = "https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page=" + str(page)
    resp = None
    if not cached: resp = urllib.request.urlopen(link)

    # html parse, split into entries
    soup = None
    if cached: soup = BeautifulSoup(soups[page - 1], 'html.parser')
    else:
        soup = BeautifulSoup(resp, 'html.parser')
        soups.append(str(soup))
    soup = str(soup).split('<number>')

    # for each entry on the page, parse
    for entry in soup:

        # rectify artifacts of the transcription process that hurt parsing
        # e.g. punctuation marks that break italics
        entry = str(entry).replace('\n', ' ')
        entry = entry.replace('</i><at>', '').replace('</at><i>', '')
        entry = entry.replace('<at>', '<i>').replace('</at>', '</i>')
        entry = entry.replace('</i>(<i>', '').replace('</i>)<i>', '')
        entry = entry.replace('*<b>', '<b>*')
        entry = entry.replace(':</b>', '</b><br>')
        entry = entry.replace('*<i>', '<i>*')
        entry = entry.replace('<i>\'</i>', '\'')
        for i in at_map:
            entry = entry.replace(f'<at>{i}</at>', f'<at>{at_map[i]}</at>')
        entry = BeautifulSoup('<number>' + entry)

        # add entry only if it has a bold member (the headword[s])
        if entry.find('b'):
            lemmas = entry.find_all('b')
            number = entry.find('number').text
            if 'A Comparative Dictionary of Indo-Aryan Languages' in number:
                continue

            # reflexes are grouped into paragraphs or marked by Ext. when they share
            # a common origin that is a derived form from the headword (e.g. -kk- extensions)
            data = re.split(r'(<br/>|Ext.)', str(entry))

            # store headwords
            for lemma in lemmas:
                reflexes[number].append({'lang': 'Indo-Aryan', 'words': [lemma.text], 'ref': data[0], 'cognateset': f'{number}.0'})

            # ignore headword from rest of parsing; if no other reflexes ignore this entry
            if (len(data) == 1): continue
            data = [x for x in data[1:] if x]

            # a subentry is a block of descendants; these are separated by newlines in CDIAL
            langs = []
            subnum = 0
            for subentry in data[1:]:

                # ignore text inside parentheses for now
                # TODO: recover information from here while still assigning to right lang
                subentry = remove_text_between_parens(subentry)

                # find lemmas in current subgroup
                matches = list(re.finditer(regex, subentry))
                if len(matches) != 0:
                    subnum += 1
                
                for i in range(len(matches)):

                    # generate row template
                    lang = matches[i].group(1)
                    lang_entry = {'lang': lang, 'words': [], 'cognateset': f'{number}.{subnum}'}

                    # word is actually the data of the current lang, not just the word
                    word = None
                    if i == len(matches) - 1:
                        word = subentry[matches[i].start():]
                    else:
                        word = subentry[matches[i].start():matches[i + 1].start()]
                    word = word.split('&lt;')[0]
                    
                    # formatting
                    word = word.replace('ˊ', '́')
                    word = word.replace(' -- ', '–')
                    word = word.replace('--', '–')
                    
                    # forms are the actual words (italicised)
                    forms = list(re.finditer(r'(<i>(.*?)</i>|\'(.*?)\')', word))
                    
                    # handling Kutchi data getting duplicated to Sindhi
                    # TODO: West Pahari data might be similarly flawed
                    if lang == 'kcch':
                        if langs:
                            if langs[-1] == 'S':
                                langs.pop()
                    if lang == 'mald':
                        lang = 'Md'

                    # langs is a stack of langs, if there are no forms
                    # we just add to the stack and continue (means later
                    # lang has relevant data)
                    langs.append(lang)
                    if len(forms) == 0:
                        continue

                    # extract definitions
                    # TODO: get morphological labels, notes
                    cur = None
                    defs = []
                    for form in forms:
                        if form.group(0).startswith('<i>'):
                            if cur:
                                for each in cur.split(','):
                                    each = each.replace('\*l', 'ʌ')
                                    each = each.replace('<smallcaps>i</smallcaps>', 'ɪ')
                                    definition = '; '.join(defs) if defs != [] else ''
                                    lang_entry['words'].append([each.strip(), definition])
                            defs = []
                            cur = form.group(2)
                        else:
                            defs.append(form.group(3).strip())
                    if cur:
                        for each in cur.split(','):
                            each = each.replace('\*l', 'ʌ')
                            each = each.replace('<smallcaps>i</smallcaps>', 'ɪ')
                            definition = '; '.join(defs) if defs != [] else ''
                            lang_entry['words'].append([each.strip(), definition])

                    # for each language on the stack, add this entry
                    for l in langs:
                        lang_entry['lang'] = l
                        writer.writerow([l, number, each.strip(), definition, '', '', '', f'{number}.{subnum}', '', 'CDIAL'])
                        reflexes[number].append(copy.deepcopy(lang_entry))
                    langs = []
    
    if not cached: del resp

fout.close()

with open(f'all.json', 'w') as fout:
    json.dump(reflexes, fout, indent=4)

if not cached:
    with open('cdial.pickle', 'wb') as fout:
        pickle.dump(soups, fout)