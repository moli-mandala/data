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

from abbrevs import abbrevs, dialects, replacements, fixes

TOTAL_PAGES = 514
APPENDIX = 509
ERR = False

# useful regexes
l = '(' + "|".join(sorted([x.replace('(', '\(').replace(')', '\)') for x in abbrevs.keys()], key=lambda x: -len(x))) + r')'
langs_regex = re.compile(r'(' + l + r')(\.|$)')
l += r'\.?'
regex = re.compile(r'(<i>|<b>|^)*' + l + r'(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=((<i>|<b>)*' + l + r'|DED|DEN|</div>|$))')
lemmata = re.compile(r'(<b>|^)(.*?)(</b>|$)(.*?)((?=<b>)|$)')
formatter = re.compile(r'<.*?>')
comma_split = re.compile(r',(?![^\(]*?\))')

def is_bold_or_italic(tag):
    return tag.name in ('b', 'i') and not (any([x.name in ('b', 'i') for x in tag.children]))

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

ref_ct = defaultdict(int)

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
            entry_str = str(entry).split(' / ')
            if len(entry_str) == 1:
                entry_str = entry_str[0]
            else:
                entry_str[1] = ' / '.join(entry_str[1:])
                for f in sorted(fixes, key=lambda x: -len(x)):
                    entry_str[1] = entry_str[1].replace(f, f'<b><i>{f}</i></b>')
                entry_str = ' / '.join(entry_str)
            entry = BeautifulSoup(entry_str, 'html.parser')

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
                span[1] = span[1].strip()

                # get every forms + gloss pairing (delineated by bold tags)
                rows = []
                last_paren = False
                for y in lemmata.finditer(span[1]):
                    if ERR: print('    lemma', y)
                    gloss = y.group(4).strip(' ').split('\t')[0]

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

                    for replacement in replacements:
                        row[-2] = row[-2].replace(replacement, replacements[replacement])

                    # refs and dialects
                    dial_forms = []
                    for ref in row[-2].split():
                        ref = ref.strip(' ,;')
                        if (ref, row[0]) in dialects:
                            ref, dial = dialects[(ref, row[0])]
                            if dial:
                                dial_forms.append(dial)
                            if ref:
                                row[-1] += ';' + ref
                        else:
                            ref_ct[(ref, row[0])] += 1

                    if not dial_forms:
                        dial_forms.append(row[0])

                    # add forms for each dialect
                    for dial in dial_forms:
                        for form in forms:
                            new_row = row[::]
                            new_row[0] = dial

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
                                if new_row[2]:
                                    writer.writerow(new_row)
                                count += 1

                if ERR: print('    done with spans')
    
    if ERR: print('deleting')
    if not cached: del resp
    if ERR: print('deleted')

# print top values in ref_ct
for key in sorted(ref_ct, key=lambda x: ref_ct[x], reverse=True)[:100]:
    print(key, ref_ct[key])

# close file
fout.close()

if not cached:
    with open('dedr.pickle', 'wb') as fout:
        pickle.dump(soups, fout)