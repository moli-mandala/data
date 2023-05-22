import pickle
import os
import urllib.request
import re
import json
import copy
import csv
import unicodedata
from bs4 import BeautifulSoup
from collections import defaultdict
from enum import Enum
from tqdm import tqdm

from abbrevs import abbrevs

TOTAL_PAGES = 836

# this is such a big brain regex
langs = r'([OM]?(' + "|".join(sorted(list(abbrevs.keys()), key=lambda x: -len(x))) + r'))\.'
langs = unicodedata.normalize('NFC', langs)
regex = re.compile(r'(?<!\w)' + langs + r'(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=([^\(]?' + langs + r'|</div>|$))')
formatter = re.compile(r'(<i>([^\(\)]*?)</i>|\'([^\(\)]*?)\')(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=$|<i>([^\(\)]*?)</i>|\'([^\(\)]*?)\'|\.)')

rows = []
params = []
done = set()

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
        entry = re.sub(r'</i>\(<i>([\w]*?)</i>\)<i>', r'{\1}', entry)
        entry = re.sub(r'</i>\(<i>([\w]*?)</i>\)', r'{\1}</i>', entry)
        entry = re.sub(r'\(<i>([\w]*?)</i>\)<i>', r'<i>{\1}', entry)
        entry = entry.replace('</i><i>', '')
        entry = entry.replace('*<b>', '<b>*')
        entry = entry.replace(':</b>', '</b><br>')
        entry = entry.replace('*<i>', '<i>*')
        entry = entry.replace('<i>\'</i>', '\'')

        entry = unicodedata.normalize('NFC', entry)
        entry = BeautifulSoup('<number>' + entry)

        # add entry only if it has a bold member (the headword[s])
        if entry.find('b'):
            lemmas = entry.find_all('b')
            number = entry.find('number').text
            if 'A Comparative Dictionary of Indo-Aryan Languages' in number:
                continue

            # reflexes are grouped into paragraphs or marked by Ext. when they share
            # a common origin that is a derived form from the headword (e.g. -kk- extensions)
            data = re.split(r'(<br/>|Ext.|[;\.,:\?] — )', str(entry))

            # store headwords
            for lemma in lemmas:
                rows.append(['Indo-Aryan', number, lemma.text, '', '', '', '', 'CDIAL', ''])
            if number not in done:
                params.append([number, lemmas[0].text, '', data[0], ''])
            done.add(number)

            # ignore headword from rest of parsing; if no other reflexes ignore this entry
            if (len(data) == 1): continue
            data = [x for x in data[1:] if x]

            # a subentry is a block of descendants; these are separated by newlines in CDIAL
            subnum = 0
            info = None
            for subentry in data[1:]:
                langs = []

                # find lemmas in current subgroup
                matches = list(regex.finditer(subentry))
                if len(matches) != 0:
                    subnum += 1
                    info = subentry[:matches[0].span()[0]].strip()
                    info = info.strip(':.;')
                
                for i in range(len(matches)):

                    # grab lang and rest of span
                    lang = matches[i].group(1)
                    span = matches[i].group(3)
                    
                    # formatting
                    span = span.replace('ˊ', '́')
                    span = span.replace(' -- ', '–')
                    span = span.replace('--', '–')
                    
                    # forms are the actual words (italicised)
                    forms = list(formatter.finditer(span))

                    # if number == '22':
                    #     print(lang, matches[i].groups())
                    #     for i in forms:
                    #         print('    ', i.groups())
                    
                    # handling Kutchi data getting duplicated to Sindhi
                    # TODO: West Pahari data might be similarly flawed
                    if lang == 'kcch':
                        if langs:
                            if langs[-1] == 'S':
                                langs.pop()
                    if lang == 'mald':
                        lang = 'Md'
                    if lang[0].islower():
                        if langs:
                            if langs[-1] == 'WPah':
                                langs.pop()

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
                    words = []

                    def append_to_words(cur, defs):
                        if cur:
                            for each in cur[0].split(','):
                                definition = '; '.join([d[0] for d in defs]) if defs != [] else ''
                                notes = '; '.join([d[1] for d in defs if d[1] != '']) if defs != [] else ''
                                notes = cur[1] + ('; ' if (cur[1] and notes) else '') + notes
                                words.append([each.strip(), definition, notes])

                    for form in forms:
                        if form.group(0).startswith('<i>'):
                            append_to_words(cur, defs)
                            defs = []
                            cur = [form.group(2), form.group(4).strip(' -,;.')]
                        else:
                            defs.append([form.group(3).strip(), form.group(4).strip(' -,;.')])
                    if cur:
                        for each in cur[0].split(','):
                            append_to_words(cur, defs)

                    # for each language on the stack, add this entry
                    for l in langs:
                        for word, defn, notes in words:

                            if '°' in word and word != '°':
                                old = word[:]
                                reference = rows[-1][2]
                                if word[-1] == '°':
                                    word = re.sub(r'^.*?' + word[-2], word[:-1], reference)
                                elif word[0] == '°':
                                    word = re.sub(word[1] + r'[^' + word[1] + r']*?$', word[1:], reference)
                                if reference == word:
                                    word = old[:]

                            # normalisation
                            word = word.replace('λ', 'ɬ')
                            word = word.replace('Λ', 'ʌ')
                            word = word.strip('.,;-: ')
                            word = word.replace('<? >', '')
                            word = word.lower()
                            word = word.replace('˜', '̃')
                            word = word.replace(f'<smallcaps>i</smallcaps>', 'ɪ')

                            # handle macron/breve combo, which we store as two forms (long vowel, short vowel)
                            oldest = unicodedata.normalize('NFD', word)
                            oldest = oldest.replace('̄˘', '̄̆')
                            oldest = oldest.replace('̆̄', '̄̆')
                            oldest = oldest.replace('̄̆', '̄̆')
                            if '̄̆' in oldest:
                                words.append([oldest.replace('̄̆', '̄'), defn, notes])
                                oldest = oldest.replace('̄̆', '')
                                word = oldest
                            if '{' in oldest:
                                words.append([re.sub(r'{.*?}', '', oldest), defn, notes])
                                oldest = oldest.replace('{', '').replace('}', '')
                                word = oldest
                            word = unicodedata.normalize('NFC', word)
                                    
                            rows.append([l, number, word, defn, '', '', notes, 'CDIAL', f'{number}.{subnum}' if info is None else f'{info}'])

                    langs = []
    
    if not cached: del resp

with open(f'cdial.csv', 'w') as fout:
    writer = csv.writer(fout)
    writer.writerows(rows)

with open(f'params.csv', 'w') as fout:
    writer = csv.writer(fout)
    writer.writerows(params)

if not cached:
    with open('cdial.pickle', 'wb') as fout:
        pickle.dump(soups, fout)