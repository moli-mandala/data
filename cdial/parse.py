from bs4 import BeautifulSoup
import urllib.request
import re
from collections import defaultdict
import json
from enum import Enum
import copy
from tqdm import tqdm
from abbrevs import abbrevs

def remove_text_between_parens(text): # lazy: https://stackoverflow.com/questions/37528373/how-to-remove-all-text-between-the-outer-parentheses-in-a-string
    n = 1  # run at least once
    while n:
        text, n = re.subn(r'[\(\[][^()]*[\)\]]', '', text)  # remove non-nested/flat balanced parts
    return text

reflexes = defaultdict(list)

regex = r'(?<=[\W\.])(' + f'{"|".join(list(abbrevs.keys()))}' + r')\.'


# this is such a big brain regex

total_pages = 836
for page in tqdm(range(1, total_pages + 1)):
    link = "https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page=" + str(page)
    with urllib.request.urlopen(link) as resp:
        soup = BeautifulSoup(resp, 'html.parser')
        soup = str(soup).split('<number>')
        for entry in soup:
            entry = BeautifulSoup('<number>' + entry)
            if entry.find('b'):
                lemmas = entry.find_all('b')
                number = entry.find('number').text
                data = str(entry).replace('\n', '')
                data = data.replace('</i><at>', '').replace('</at><i>', '').replace('<at>', '<i>').replace('</at>', '</i>')
                data = data.replace('</i>(<i>', '').replace('</i>)<i>', '')
                data = re.split(r'(<br/>|Ext.)', data)

                for lemma in lemmas:
                    reflexes[number].append({'lang': 'Indo-Aryan', 'words': [lemma.text], 'ref': data[0], 'cognateset': f'{number}.0'})
                if (len(data) == 1): continue
                data = [x for x in data[1:] if x]

                # a subentry is a block of descendants; these are separated by newlines in CDIAL
                langs = []
                subnum = 0
                for subentry in data[1:]:
                    # ignore text under parentheses for now
                    # TODO: recover information from here while still assigning to right lang
                    subentry = remove_text_between_parens(subentry)
                    # find lemmas
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
                        
                        # formatting
                        word = word.replace('ˊ', '́')
                        word = word.replace(' -- ', '–')
                        word = word.replace('--', '–')
                        
                        # forms are the actual words (italicised)
                        forms = list(re.finditer(r'(<i>(.*?)</i>|ʻ(.*?)ʼ)', word))
                        # handling Kutchi data getting duplicated to Sindhi
                        # TODO: West Pahari data might be similarly flawed
                        if lang == 'kcch':
                            if langs:
                                if langs[-1] == 'S':
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
                        for form in forms:
                            if form.group(0).startswith('<i>'):
                                if cur:
                                    for each in cur.split(','):
                                        definition = '; '.join(defs) if defs != [] else ''
                                        lang_entry['words'].append([each.strip(), definition])
                                defs = []
                                cur = form.group(2)
                            else:
                                defs.append(form.group(3).strip())
                        
                        if cur:
                            for each in cur.split(','):
                                definition = '; '.join(defs) if defs != [] else ''
                                lang_entry['words'].append([each.strip(), definition])

                        for l in langs:
                            lang_entry['lang'] = l
                            reflexes[number].append(copy.deepcopy(lang_entry))
                        langs = []

with open(f'all.json', 'w') as fout:
    json.dump(reflexes, fout, indent=4)