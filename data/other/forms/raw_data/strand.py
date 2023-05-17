from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import csv
import re
from segments.tokenizer import Tokenizer, Profile

t = Tokenizer('ipa/strand.txt')

chars = ['p', 'b', 'bAsp', 'f', 'v', 'w', 'm', 'uFrn', 'u', 'o', 'oFrn', 'uTns', 'oTns',
         'cDen', 'zDen', 't', 'd', 'dAsp', 's', 'z', 'l', 'lVls', 'n', 'cRet', 'jRet',
         'tRet', 'dRet', 'dRetAsp', 'sRet', 'zRet', 'r', 'rFlp', 'lBak', 'rApx', 'nApx',
         'nRet', 'rVoc', 'cLam', 'jLam', 'jLamAsp', 'sLam', 'zLam', 'y', 'i', 'e', 'aLam',
         'iTns', 'eTns', 'kPal', 'gPal', 'gPalAsp', 'k', 'g', 'gAsp', 'x', 'gSpi', 'nasVel',
         'iBak', 'a', 'aOpn', 'aRnd', 'kLab', 'gLab', 'gLabAsp', 'nas', 'q', 'hPhyr', 'Ayn',
         'AgltStp', 'h', 'hPglt', 'hPgltRnd']

languages = ['IndoAryan/Pashai/Degan/DeganLanguage',
             'Nuristani/Kamkata/Kom/KomLanguage',
             'Nuristani/Kamkata/Kata/KataLanguage',
             'Nuristani/AshkunEtc/SaNu/SaNuLanguage',
             'Nuristani/Kalasha/Nishei/NisheiLanguage',
             'IndoAryan/Chitral/Khow/KhowLanguage',
             'IndoAryan/Indus/Atsaret/AtsaretLanguage']

codes = ['deg', 'Kam', 'Kata', 'Ash', 'Wg', 'Kho', 'Phal']
CHECK = '�'

lang_mapping = {
    'Vâsi.u': 'usut', 'Vâsi.z': 'zumu', 'Vâsi.üć': 'ucu', 'Vâsi.s': 'sec',
    'Vâsi.ṣu': 'supu', 'Kmkt.ktv': 'ktivi', 'Kmkt.km': 'Kam', 'Aṣk.s': 'sanu',
    'Kal.n': 'nis', 'Kal.v': 'vagal', 'Kal.a': 'ames', 'Treg.g': 'gamb', 'Vâsi.?': 'Pr'
}
src_mapping = {
    'S': 'strand', 'M': 'morgenstierne', 'B': 'buddruss', 'L': 'lentz', 'LSI': 'LSI'
}

def strand3():
    with open('strand3.csv', 'w') as f, open('../params/strand3.csv', 'w') as p:
        forms = csv.writer(f)
        params = csv.writer(p)
        ct = 0
        done = False
        stack = []
        
        for char in chars:
            link = f'https://nuristan.info/Nuristani/Nuristani/Nuristani/NuristaniLanguage/Lexicon/alph-{char}.html'
            print(link)
            try:
                with urlopen(Request(link, headers={'User-Agent': 'Mozilla/5.0'})) as resp:
                    soup = BeautifulSoup(resp, 'html5lib')
                    last_head = {}
                    for table in soup.find_all('table'):
                        for row in table.find_all('tr'):
                            tds = row.find_all('td')

                            # store headwords
                            if row.find(class_='lng1') or row.find(class_='lng2'):
                                comment = tds[-1].find(class_='mid')
                                text = tds[-1].text.replace('\n', '')
                                defns = re.findall(r'‘(.*?)’', text)
                                level = int(tds[0].get('colspan', 1) or 1 if row.find(class_='lng2') else 0)
                                while stack and level <= stack[-1]['level']:
                                    stack.pop()
                                
                                turner = None
                                if comment:
                                    turner = re.findall(r'T\. (\d+(\.\d+)?)', comment.text)
                                    if turner: turner = turner[0][0]
                                if not turner:
                                    for s in stack:
                                        if not s['id'].startswith('n'):
                                            turner = s['id']
                                            break
                                if not turner:
                                    ct += 1

                                l = {
                                    'lang': tds[-2].find('em').text,
                                    'level': int(level),
                                    'form': tds[-1].find('em').text,
                                    'defn': defns[0] if defns else '',
                                    'id': turner if turner else f'n{ct}',
                                    'comment': comment.text if comment else ''}
                                last_head = l
                                stack.append(l)
                                done = False

                            # forms    
                            elif row.find(class_='lng'):
                                comment = tds[-1].find(class_='sm')
                                text = tds[-1].text.replace('\n', '')
                                defns = re.findall(r'‘(.*?)’', text)
                                lang, dial, src = tds[-2].find('em').text.split('.')
                                r = [lang_mapping[lang + '.' + dial], last_head['id'],
                                    tds[-1].find('em').text, defns[0] if defns else '', '', '', comment.text if comment else '',
                                    src_mapping[src]]
                                # print(stack)
                                # print(r)
                                # input()

                                forms.writerow(r)
                                if not done and last_head['id'].startswith('n'):
                                    params.writerow([last_head['id'], last_head['lang'], last_head['form'], last_head['defn'], 'strand'])
                                    done = True


            except HTTPError as e:
                pass


def strand2():
    with open('strand2.csv', 'w') as fout:
        writer = csv.writer(fout)
        link = f'http://nuristan.info/IndoAryan/SwatIndus/Bhatera/BhateraLanguage/Lexicon/lex.html'
        try:
            with urlopen(Request(link, headers={'User-Agent': 'Mozilla/5.0'})) as resp:
                soup = BeautifulSoup(resp, 'html.parser')
                for data in soup.find_all(class_='dic'):
                    word = data.find(class_='l')
                    if word:
                        print(word)
                        word = word.find(text=True, recursive=False)
                        word2 = re.sub(r'ʹ(.)', r'\1ʹ', word)
                        word2 = re.sub(r'`(.)', r'\1`', word2)
                        word2 = re.sub(r'´(.)', r'\1´', word2)
                        data = str(data).replace('\n', ' ')
                        l = re.search(r'<b>]</b>\xa0 (.*?)\.\xa0 (.*?)\.', data)
                        if not l:
                            l = re.search(r'</span>[\xa0 ]+(.*?)\.\xa0\xa0([^\.]+)\.', data)
                        print(l)
                        if l:
                            pos = l.group(1).lower()
                            definition = l.group(2).lower()
                            turner = re.search(r'T\..(\d+)', data)
                            if turner:
                                turner = turner.group(1)
                                ipa = t(word2, column='IPA').replace(' ', '').replace('#', ' ')
                                writer.writerow(['bhatr', turner, word, definition, '', ipa, '', 'strand'])

        except HTTPError as e:
            pass

def strand():
    with open('strand.csv', 'w') as fout:
        writer = csv.writer(fout)
        for i, language in enumerate(languages):
            for char in chars:
                link = f'http://nuristan.info/{language}/Lexicon/alph-{char}.html'
                print(link)
                try:
                    with urlopen(Request(link, headers={'User-Agent': 'Mozilla/5.0'})) as resp:
                        soup = BeautifulSoup(resp, 'html.parser')
                        for data in soup.find_all(class_='dic'):
                            word = data.find(class_='l')
                            if word:
                                word = word.find(text=True, recursive=False)
                                word2 = re.sub(r'ʹ(.)', r'\1ʹ', word)
                                l = re.search(r'<b>]</b>\xa0 (.*?)\.\xa0 (.*?)\.', str(data))
                                if not l:
                                    l = re.search(r'</span>\xa0 (.*?)\.\xa0 (.*?)\.', str(data))
                                if l:
                                    pos = l.group(1).lower()
                                    definition = l.group(2).lower()
                                    turner = re.search(r'T\. (\d+)', str(data))
                                    if turner:
                                        turner = turner.group(1)
                                        ipa = t(word2, column='IPA').replace(' ', '').replace('#', ' ')
                                        writer.writerow([codes[i], turner, word, definition, '', ipa, '', 'strand'])

                except HTTPError as e:
                    pass

def main():
    strand3()

if __name__ == "__main__":
    main()