import json
import csv
import unidecode
import re
import glob
from segments.tokenizer import Tokenizer, Profile
import unicodedata
from tqdm import tqdm
from collections import Counter

superscript = {
    'a': 'ᵃ', 'e': 'ᵉ', 'i': 'ᶦ',
    'o': 'ᵒ', 'u': 'ᵘ', 'ü': 'ᵘ̈',
    'y': 'ʸ', 'ə': 'ᵊ', 'ŭ': 'ᵘ̆',
    'z': 'ᶻ', 'gy': 'ᵍʸ', 'h': 'ʰ',
    'ŕ': 'ʳ́', 'ĕ': 'ᵉ̆', 'n': 'ⁿ'
}

tokenizers = {}
convertors = {}

for file in glob.glob("data/cdial/ipa/cdial/*.txt"):
    lang = file.split('/')[-1].split('.')[0]
    tokenizers[lang] = Tokenizer(file)

for file in glob.glob("conversion/*.txt"):
    lang = file.split('/')[-1].split('.')[0]
    convertors[lang] = Tokenizer(file)

with open('data/cdial/all.json', 'r') as fin:
    data = json.load(fin)

change = {
    'khaś': 'khash',
    'Māl': 'Malw',
    'Brah': 'Brahui',
    'Drav': 'Pdr.',
    'Ga': 'Gadaba',
    'Kan': 'Kannada',
    'Kol': 'Kolami',
    'Kur': 'Kurux',
    'Mal': 'Malayalam',
    'Nk': 'Naikri',
    'Prj': 'Parji',
    'Tam': 'Tamil',
    'Tel': 'Telugu',
    'Tu': 'Tulu'
}

a = set()
with open('cldf/forms.csv', 'w') as fout, open('errors.txt', 'w') as errors:
    num = 0
    write = csv.writer(fout)
    result = []
    write.writerow(['ID', 'Language_ID', 'Parameter_ID', 'Form', 'Gloss', 'Native', 'Phonemic', 'Original', 'Cognateset', 'Description', 'Source'])
    for entry in tqdm(data):
        headword = data[entry][0]
        for form in data[entry]:
            lang = form['lang'].replace('.', '')
            cognateset = form['cognateset']

            lang = unidecode.unidecode(lang)
            a.add(lang)
            reference = ''
            for i, word in enumerate(form['words']):
                if (len(word[0]) <= 1 or (len(word[0]) == 2 and word[0][0] == word[0][1])) and (len(word) > 1 and word[1] == ''): continue
                num += 1
                if lang == 'Indo-Aryan':
                    diffs = re.search(r'ʻ(.*?)ʼ', form['ref'])
                    desc = ''
                    if diffs != None:
                        desc = diffs.groups(2)[0].strip()
                    if isinstance(word, str):
                        word = [word, desc]

                word[0] = word[0].strip('.,;-: ')
                word[0] = word[0].replace('<? >', '')
                word[0] = word[0].lower()
                word[0] = word[0].replace('˜', '̃')

                word[0] = word[0].replace(f'<smallcaps>i</smallcaps>', 'ɪ')
                word[0] = word[0].replace('*l', 'ʌ')
                for i in superscript:
                    word[0] = word[0].replace('ˊ', '́').replace('ˋ', '̀').replace(' -- ', '-')
                    word[0] = word[0].replace(f'<superscript>{i}</superscript>', superscript[i])

                oldest = unicodedata.normalize('NFD', word[0])
                oldest = oldest.replace('̄˘', '̄̆')
                oldest = oldest.replace('̆̄', '̄̆')
                oldest = oldest.replace('̄̆', '̄̆')
                if '̄̆' in oldest:
                    form['words'].append([oldest.replace('̄̆', '̄'), word[1]])
                    oldest = oldest.replace('̄̆', '')
                    word[0] = oldest
                word[0] = unicodedata.normalize('NFC', word[0])

                if '˚' not in word[0]: reference = word[0]
                else:
                    old = word[0]
                    if word[0] != '˚':
                        if word[0][-1] == '˚':
                            word[0] = re.sub(r'^.*?' + word[0][-2], word[0][:-1], reference)
                        elif word[0][0] == '˚':
                            word[0] = re.sub(word[0][1] + r'[^' + word[0][1] + r']*?$', word[0][1:], reference)
                        if reference == word[0]:
                            word[0] = old

                ipa = ''
                if lang in tokenizers and '˚' not in word[0]:
                    ipa = tokenizers[lang](word[0], column='IPA').replace(' ', '').replace('#', ' ')
                    if '�' in ipa:
                        # if lang in ['A']: errors.write(f'{lang} {oldest} {word[0]} {ipa}\n')
                        ipa = ''
                
                reformed = ''
                if '˚' not in word[0]:
                    reformed = convertors['cdial'](word[0].strip('-123456,;'), column='IPA').replace(' ', '').replace('#', ' ')
                    if '�' in reformed:
                        errors.write(f'{lang} {oldest} {word[0]} {ipa} {reformed}\n')
                        reformed = ''

                if word[0] or reformed:
                    result.append([num, lang, entry, reformed if reformed else word[0], word[1], '', ipa, word[0], cognateset, '', 'CDIAL'])
    
    i = 0
    mapping = {
        'patyal': 'cdial', 'thari': 'cdial', 'kvari': 'cdial', 'dhivehi': 'none', 'kholosi': 'none',
        'konkani': 'none', 'khetrani': 'none'
    }
    for file in glob.glob("data/other/ia/*.csv"):
        # get filename
        name = file.split('/')[-1].split('.')[0]
        print(name)
        convert = name in convertors or name in mapping
        name = mapping.get(name, name)
        with open(file, 'r') as fin:
            read = csv.reader(fin)
            for row in tqdm(read):
                if row[1]:
                    # handle subentries
                    if '.' in row[1]:
                        row[6] = row[1]
                        row[1] = row[1].split('.')[0]
                    reformed = row[2]
                    if name != 'none':
                        reformed = re.sub(r'ʹ(.)', r'\1ʹ', row[2])
                        reformed = re.sub(r'`(.)', r'\1`', reformed)
                        reformed = re.sub(r'´(.)', r'\1´', reformed)
                        if '˚' not in row[2] and convert:
                            reformed = convertors[name](reformed.strip('-123456,;'), column='IPA').replace(' ', '').replace('#', ' ')
                        if '�' in reformed:
                            errors.write(f'{row[0]} {row[2]} {row[2]} {row[5]} {reformed}\n')
                            reformed = ''
                    result.append([f'e{i}', row[0], row[1], reformed if reformed else row[2], row[3], row[4], row[5], row[2], row[1], row[6], row[7]])
                    i += 1

    dravidian_entries = {}
    for file in glob.glob("data/dedr/dedr.csv"):
        with open(file, 'r') as fin:
            read = csv.reader(fin, delimiter=',', quotechar="'", skipinitialspace=True)
            for row in read:
                row = [x.strip(' "') for x in row]
                num = 'd' + row[1]
                if num not in dravidian_entries:
                    dravidian_entries[num] = ''
                if row[3] == 'PDr.':
                    row[3] = 'PDr'
                    dravidian_entries[num] = row[4]
                row[3] = row[3].replace(' ', '')
                result.append([f'dedr{row[0]}', row[3], num, row[4], row[5], '', '', row[4], 'd' + row[1], row[6] if row[6] != 'NULL' else '', 'dedr'])

    done = set()
    for row in result:
        key = tuple(row[1:])
        if key not in done:
            write.writerow(row)
        done.add(key)

with open('cldf/cognates.csv', 'w') as fout, open('cldf/parameters.csv', 'w') as fout2:
    write = csv.writer(fout)
    write2 = csv.writer(fout2)
    write.writerow(['Cognateset_ID', 'Language_ID', 'Form', 'Description', 'Source'])
    write2.writerow(['ID', 'Name', 'Concepticon_ID', 'Description'])
    for entry in data:
        headword = data[entry][0]['words'][0].replace('ˊ', '́').replace('`', '̀').replace(' --', '-').replace('-- ', '-')
        headword = headword.strip('.,;-: ')
        headword = headword.replace('<? >', '')
        headword = headword.lower()
        headword = headword.replace('˜', '̃')
        headword = headword.split()[0]
        reformed = ''
        if '˚' not in headword:
            reformed = convertors['cdial'](headword.strip('-123456,;'), column='IPA').replace(' ', '').replace('#', ' ')
            # if '�' in reformed:
            #     errors.write(f'{lang} {oldest} {word[0]} {ipa} {reformed}\n')
            #     reformed = ''
        write.writerow([entry, 'Indo-Aryan', reformed if reformed else headword, data[entry][0]['ref'], 'cdial'])
        write2.writerow([entry, headword, '', data[entry][0]['ref']])
    with open('data/other/extensions_ia.csv', 'r') as fin:
        read = csv.reader(fin)
        for row in read:
            write.writerow(row)
            write2.writerow([row[0], row[2], '', row[3]])
    for entry in dravidian_entries:
        write.writerow([entry, 'PDr', dravidian_entries[entry], '', 'dedr'])
        write2.writerow([entry,  dravidian_entries[entry], '', '?'])

# print(sorted(list(a)))
b = set()
with open('cldf/languages.csv', 'r') as fin:
    for row in fin.readlines():
        x = row.split(',')[0]
        b.add(x)

for i in a:
    if i not in b:
        print(i)