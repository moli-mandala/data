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

from abbrevs import abbrevs, refs, dialects

TOTAL_PAGES = 514
APPENDIX = 509
ERR = False

# useful regexes
l = '(' + "|".join(sorted(list(abbrevs.keys()), key=lambda x: -len(x))) + r')'
langs_regex = re.compile(r'(' + l + r')(\.|$)')
l += r'\.?'
regex = re.compile(r'(<i>|<b>|^)*' + l + r'(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=((<i>|<b>)*' + l + r'|DED|DEN|</div>|$))')
lemmata = re.compile(r'(<b>|^)(.*?)</b>(.*?)((?=<b>)|$)')
formatter = re.compile(r'<.*?>')
comma_split = re.compile(r',(?![^\(]*?\))')

def is_bold_or_italic(tag):
    return tag.name in ('b', 'i') and tag.parent.name in ('b', 'i')

# response caching logic
soups = []
cached = False
if os.path.exists('dedr.pickle'):
    with open('dedr.pickle', 'rb') as fin:
        soups = pickle.load(fin)
    cached = True
print('Caching?', cached)

# file
fout = open('dedr_new.csv', 'w')
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
        entry = entry.replace('\n', '')
        entry = BeautifulSoup('<number>' + entry, 'html.parser')

        # only if this is an actual entry
        if entry.find('number'):

            # store and get rid of number
            number: str = entry.find('number').text
            entry.find('number').decompose()
            if page >= APPENDIX:
                number = 'a' + number

            if ERR: print(entry)
            entry = BeautifulSoup(str(entry), 'html.parser')
            entry_str = str(entry)

            # find all bold+italic tags (includes languages)
            langs = entry.find_all(is_bold_or_italic)
            spans = []
            start = 0

            for lang in langs:

                # append everything up to this tag to the previous tag
                if spans:
                    spans[-1][1] += entry_str[start:lang.sourcepos]
                
                # append this tag as a new span
                # but it may not be a real language tag, in which case just expand previous
                m = langs_regex.search(lang.text)
                if m:
                    spans.append([m.group(1), ""])
                else:
                    if spans:
                        spans[-1][1] += str(lang)

                # update start
                start = lang.sourcepos + len(str(lang))

            # tail of entry
            if spans:
                spans[-1][1] += entry_str[start:]

            for span in spans:

                lang = abbrevs[span[0].strip('.')]

                # get every forms + gloss pairing (delineated by bold tags)
                rows = []
                last_paren = False
                for y in lemmata.finditer(span[1]):
                    if ERR: print('    lemma', y)
                    gloss = y.group(3).strip(' ')

                    if last_paren:
                        rows[-1][3] += y.group(2) + gloss
                        last_paren = rows[-1][3].count('(') > rows[-1][3].count(')')
                        continue

                    if y.group(2) in ['Voc.', 'n.', 'adj.', 'adv.', 'v.']:
                        rows[-1][3] += y.group(2) + gloss
                        last_paren = rows[-1][3].count('(') > rows[-1][3].count(')')
                        continue
                    
                    row = [lang, 'd' + str(number), y.group(2).strip(), gloss, '', '', '', 'dedr']
                    row[2] = row[2].replace('</i>', '').replace('</b>', '').replace('<i>', '').replace('<b>', '')
                    row[2] = row[2].strip()

                    # extract parentheticals from previous row--they are sources or notes about this one
                    if rows:
                        if rows[-1][3].endswith(')'):
                            paren = rows[-1][3].rfind('(')
                            row[6] = rows[-1][3][paren:][1:-1]
                            rows[-1][3] = rows[-1][3][:paren]
                    
                    # extract parentheticals from this row
                    if row[2].startswith('('):
                        paren = row[2].find(')')
                        row[6] += (' ' if row[6] else '') + row[2][:paren].strip(' ()')
                        row[2] = row[2][paren + 1:].strip()

                    rows.append(row)

                    if gloss.count('(') > gloss.count(')'):
                        last_paren = True

                    if ERR: print('        done with forms')
                
                for pos, row in enumerate(rows):
                    # fix Tamil (-pp-, -tt-)
                    if row[0] == 'Tam' and row[2] == '' and row[6] == '-pp-, -tt-':
                        row[2] = rows[pos - 1][2].split(' (')[0] + ' (-pp-, -tt-)'
                        row[6] = ""

                    forms = [form.strip() for form in comma_split.split(row[2])]
                    row[3] = row[3].strip(';,./ ')

                    # refs
                    for ref in row[-2].split():
                        if (ref, row[0]) in refs:
                            row[-1] += ';' + refs[(ref, row[0])]

                    for form in forms:
                        new_row = row[::]

                        if ERR: print('        form', form)
                        form = formatter.sub('', form).strip()

                        # extract parentheticals from this row
                        if form.startswith('('):
                            paren = form.find(')')
                            new_row[6] += (' ' if new_row[6] else '') + form[:paren].strip(' ()')
                            form = form[paren + 1:].strip()

                        # handle parse fails for Turner cognates
                        if lang == 'OIA' and (form == '' or 'no.' in form):
                            continue

                        for altform in form.split('/'):
                            new_row[2] = altform.strip(" ;.,/")
                            writer.writerow(new_row)
                            count += 1

                if ERR: print('    done with spans')
            
    
    if ERR: print('deleting')
    if not cached: del resp
    if ERR: print('deleted')

# close file
fout.close()

if not cached:
    with open('dedr.pickle', 'wb') as fout:
        pickle.dump(soups, fout)