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

change = {
    'khaś': 'khash',
    'Māl': 'Malw',
    'Brah': 'Brahui',
    'Drav': 'PDr',
    'Ga': 'Gadaba',
    'Kan': 'Kannada',
    'Kol': 'Kolami',
    'Kur': 'Kurux',
    'Mal': 'Malayalam',
    'Nk': 'Naikri',
    'Prj': 'Parji',
    'Tam': 'Tamil',
    'Tel': 'Telugu',
    'Tu': 'Tulu',
    'Go': 'Gondi',
    'OIA': 'Sk',
    'J': 'kiũth',
    'mald': 'Md',
    'kua': 'kvar',
    'Sant': 'sa',
    'Arb': 'Ar',
    'Prs': 'Pers',
    'Arb-Prs': 'Ar',
    'gamb': 'Gmb',
    'Kmd': 'Kamd',
    'Kan': 'Kannada',
    'Tam': 'Tamil',
    'Tel': 'Telugu',
    'Mal': 'Malayalam'
}

# read in tokenizer/convertors for IPA and form normalisation
tokenizers = {}
convertors = {}
for file in glob.glob("data/cdial/ipa/cdial/*.txt"):
    lang = file.split('/')[-1].split('.')[0]
    tokenizers[lang] = Tokenizer(file)
for file in glob.glob("conversion/*.txt"):
    lang = file.split('/')[-1].split('.')[0]
    convertors[lang] = Tokenizer(file)

# a set to track what languages are included
lang_set = set()

# write out forms.csv
with open('errors.txt', 'w') as errors:
    form_count = 0
    result = []

    # now do the same thing for non-CDIAL IA languages
    i = 0
    mapping = {
        'patyal': 'cdial', 'thari': 'cdial', 'kvari': 'cdial', 'dhivehi': None, 'kholosi': None,
        'konkani': None, 'khetrani': None, 'vaagri': 'cdial', 'cdial': 'cdial', 'palula': 'liljegren',
        'strand': 'strand', 'strand2': 'strand', 'strand3': 'strand'
    }
    for file in ['data/cdial/cdial.csv', 'data/munda/forms.csv'] + glob.glob("data/other/forms/*.csv"):
        # get filename
        name = file.split('/')[-1].split('.')[0]
        print(name)
        convert = name in convertors or name in mapping
        name = mapping.get(name, None)
        with open(file, 'r') as fin:
            read = csv.reader(fin)
            for row in tqdm(read):
                if row[0] == 'Drav': continue
                if row[0] == 'Indo-Aryan': row[2] = row[2].lower()
                if row[1]:
                    # handle subentries
                    if '.' in row[1]:
                        row[6] = row[1]
                        row[1] = row[1].split('.')[0]

                    for form in row[2].split(','):
                        reformed = form

                        # convert to Samopriyan system
                        if name is not None:
                            if '˚' not in form and convert:
                                # fix accentuation from Strand
                                if name == "strand":
                                    reformed = reformed.replace("′", "´")
                                    reformed = re.sub(r"([`´])(.)", r"\2\1", reformed)
                                reformed = convertors[name](reformed.strip('-1234⁴5⁵67⁷,;.'), column='IPA').replace(' ', '').replace('#', ' ')
                            if '�' in reformed:
                                errors.write(f'{row[0]} {form} {form} {row[5]} {reformed}\n')
                                reformed = ''

                        result.append([f'{i}', row[0], row[1], reformed if reformed else form, row[3], row[4], row[5], form, row[8 if 'cdial' in file else 1], row[6], row[7]])
                        i += 1

    # dravidian languages
    dravidian_entries = {}
    for file in glob.glob("data/dedr/dedr.csv"):
        with open(file, 'r') as fin:
            read = csv.reader(fin, delimiter=',', quotechar="'", skipinitialspace=True)
            for row in tqdm(read):
                row = [x.strip(' "') for x in row]
                form_count = 'd' + row[1]
                if form_count not in dravidian_entries:
                    dravidian_entries[form_count] = ''
                if row[3] == 'PDr.':
                    row[3] = 'PDr'
                    dravidian_entries[form_count] = row[4]
                row[3] = row[3].replace(' ', '')
                result.append([f'dedr{row[0]}', row[3], form_count, row[4], row[5], '', '', row[4], 'd' + row[1], row[6] if row[6] != 'NULL' else '', 'dedr'])

    # write out all the forms
    with open('cldf/forms.csv', 'w') as fout:
        forms = csv.writer(fout)
        forms.writerow(['ID', 'Language_ID', 'Parameter_ID', 'Form', 'Gloss', 'Native', 'Phonemic', 'Original', 'Cognateset', 'Description', 'Source'])

        done = set()
        for row in result:
            if not row[3]: continue
            if row[2] == '?' or not row[2]: continue
            if row[1] in change:
                row[1] = change[row[1]]
            row[1] = unidecode.unidecode(row[1])
            row[1] = row[1].replace('.', '')
            row[3] = unicodedata.normalize('NFC', row[3])
            lang_set.add(row[1])

            key = tuple(row[1:])
            if key not in done:
                forms.writerow(row)
            done.add(key)

    etyma = {}
    with open('data/etymologies.csv', 'r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            etyma[row[0]] = row[1]

    # finally, cognates (unused so far) and parameters
    with open('cldf/cognates.csv', 'w') as f, open('cldf/parameters.csv', 'w') as g:
        
        mapping = {'cdial': 'cdial', 'extensions_ia': 'cdial', 'strand3': 'strand'}

        cognates = csv.writer(f)
        params = csv.writer(g)

        cognates.writerow(['Cognateset_ID', 'Language_ID', 'Form', 'Description', 'Source'])
        params.writerow(['ID', 'Name', 'Concepticon_ID', 'Description', 'Etyma'])

        with open('data/cdial/params.csv', 'r') as fin:
            read = csv.reader(fin)
            for row in read:
                headword = row[1].replace('ˊ', '́').replace('`', '̀').replace(' --', '-').replace('-- ', '-')
                headword = headword.strip('.,;-: ')
                headword = headword.replace('<? >', '')
                headword = headword.lower()
                headword = headword.replace('˜', '̃')
                headword = headword.split()[0]
                reformed = ''
                if '˚' not in headword:
                    reformed = convertors['cdial'](headword.strip('-123456,;'), column='IPA').replace(' ', '').replace('#', ' ')
                    if '�' in reformed:
                        errors.write(f'{name} {headword} {"?"} {"?"} {reformed}\n')
                        reformed = ''
                
                cognates.writerow([row[0], 'Indo-Aryan', reformed if reformed else headword, row[3], 'cdial'])
                params.writerow([row[0], reformed if reformed else headword, '', row[3], etyma.get(row[0], '')])

        for file in glob.glob("data/other/params/*.csv"):
            # get filename
            name = file.split('/')[-1].split('.')[0]
            convert = name in convertors or name in mapping
            print(name)
            name = mapping.get(name, name)
            with open(file, 'r') as f:
                read = csv.reader(f)
                for row in tqdm(read):
                    if name == 'strand':
                        if row[1] in ['PNur', 'PA']: row[2] = '*' + row[2]
                        row[2] = row[2].replace('′', 'ʹ').replace('-', '')
                    if convert:
                        reformed = convertors[name](row[2].strip('-123456,;'), column='IPA').replace(' ', '').replace('#', ' ')
                        if '�' in reformed:
                            errors.write(f'{name} {row[2]} {"?"} {"?"} {reformed}\n')
                            reformed = ''
                        else: row[2] = reformed
                    cognates.writerow(row)
                    params.writerow([row[0], row[2], '', row[3], etyma.get(row[0], '')])

        with open('data/munda/params.csv', 'r') as f:
            read = csv.reader(f)
            for row in read:
                cognates.writerow([row[0], 'PMu', row[1], row[3], 'rau'])
                params.writerow(row)

        for entry in dravidian_entries:
            cognates.writerow([entry, 'PDr', dravidian_entries[entry], '', 'dedr'])
            params.writerow([entry,  dravidian_entries[entry], '', '', etyma.get(entry, '')])


# ensure that all languages in forms.csv are also in languages.csv
cldf_langs = set()
with open('cldf/languages.csv', 'r') as fin:
    for row in fin.readlines():
        x = row.split(',')[0]
        cldf_langs.add(x)

for i in sorted(lang_set):
    if i not in cldf_langs:
        print(i)