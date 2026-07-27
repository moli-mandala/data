from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import quote
import csv
import re
from pathlib import Path
from segments.tokenizer import Tokenizer, Profile

DATA_ROOT = Path(__file__).resolve().parents[4]
t = Tokenizer(str(Path(__file__).resolve().parents[1] / 'ipa/strand.txt'))


def cached_soup(url, cache_path, parser):
    """Read a cached Strand page, downloading it once when the cache is absent."""
    cache_path = DATA_ROOT / cache_path
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'})) as resp:
            cache_path.write_bytes(resp.read())
    return BeautifulSoup(cache_path.read_bytes(), parser)

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
DIALECT_NAMES = {
    'Vâsi.u': 'Prasun: Usut', 'Vâsi.z': 'Prasun: Zumu', 'Vâsi.üć': 'Prasun: Ucu',
    'Vâsi.s': 'Prasun: Sec', 'Vâsi.ṣu': 'Prasun: Supu', 'Vâsi.?': 'Prasun',
    'Kmkt.ktv': 'Katavari: Ktivi', 'Kmkt.km': 'Kamviri', 'Aṣk.s': 'Ashkun: Sanu',
    'Kal.n': 'Nuristani Kalasha: Nisheigram', 'Kal.v': 'Nuristani Kalasha: Vagal',
    'Kal.a': 'Nuristani Kalasha: Amesdes', 'Treg.g': 'Tregami: Gambir',
    'deg': 'Pashai: Gorayk (Degano)', 'Kam': 'Kamviri', 'Kata': 'Katavari',
    'Ash': 'Ashkun', 'Wg': 'Nuristani Kalasha', 'Kho': 'Khowar', 'Phal': 'Palula',
    'bhatr': 'Bhateri', 'ktivi': 'Katavari: Ktivi', 'sanu': 'Ashkun: Sanu',
    'nis': 'Nuristani Kalasha: Nisheigram', 'vagal': 'Nuristani Kalasha: Vagal',
    'ames': 'Nuristani Kalasha: Amesdes', 'gamb': 'Tregami: Gambir',
    'sec': 'Prasun: Sec', 'usut': 'Prasun: Usut', 'supu': 'Prasun: Supu',
    'zumu': 'Prasun: Zumu', 'ucu': 'Prasun: Ucu', 'Pr': 'Prasun',
}
src_mapping = {
    'S': 'strand', 'M': 'morgenstierne', 'B': 'buddruss', 'L': 'lentz', 'LSI': 'LSI'
}

# Strand uses the same compact grammatical notation throughout these lexica. Convert the labels
# to Jambu's shared structured vocabulary instead of discarding them with the surrounding HTML.
_POS_TOKEN = re.compile(
    r"(?<![A-Za-z])(?:VT|VI|V|N[A-Za-z0-9?+<*`]*|Aj[A-Za-z0-9?+<*`]*|AJ|"
    r"Av[A-Za-z0-9?+<*`]*|Pn[A-Za-z0-9?+<*`]*|Cj|Id|I|Neg|Emp|Mod|St|"
    r"D[`*]?|L[A-Za-z0-9?+<*`]*|M|En)(?![A-Za-z])",
    re.IGNORECASE,
)


def strand_pos_tags(label):
    """Translate a Strand grammatical code into canonical, searchable Jambu tags."""
    match = _POS_TOKEN.search(label or "")
    if not match:
        return ""
    raw_code = match.group(0)
    code = {
        "vt": "VT", "vi": "VI", "v": "V", "aj": "Aj", "cj": "Cj",
        "id": "Id", "i": "I", "neg": "Neg", "emp": "Emp", "mod": "Mod",
        "st": "St", "m": "M", "en": "En",
    }.get(raw_code.casefold(), raw_code)
    tags = []
    if code == "VT":
        tags = ["verb", "tr"]
    elif code == "VI":
        tags = ["verb", "intr"]
    elif code == "V":
        tags = ["verb"]
    elif code.startswith("N"):
        tags = ["num"] if "Qt" in code else (["adj"] if "Ql" in code else ["noun"])
        if "F" in code:
            tags.append("f")
        if "Pl" in code:
            tags.append("pl")
    elif code.startswith(("Aj", "AJ")):
        tags = ["adj"]
        if "Qt" in code:
            tags.append("num")
    elif code.startswith("Av"):
        tags = ["adv"]
    elif code.startswith("Pn"):
        tags = ["pron"]
        if "Pl" in code:
            tags.append("pl")
        if "?" in code:
            tags.append("interr")
    elif code == "Cj":
        tags = ["conj"]
    elif code in {"I", "Id"}:
        tags = ["interj"]
    elif code == "Neg":
        tags = ["part"]
    elif code == "Emp":
        tags = ["part", "emph"]
    elif code == "Mod":
        tags = ["part"]
    elif code == "St":
        tags = ["part", "interr"]
    elif code.startswith("D"):
        tags = ["dir"]
    elif code in {"M", "En"}:
        tags = ["suffix"]
    # L... distinguishes many spatial/temporal locatives and place-name classes. Its prefix alone
    # does not safely imply adverb, postposition, or noun, so those codes remain untagged.
    return " ".join(dict.fromkeys(tags))


def strand_row(language, parameter, form, definition, ipa, notes, source, pos, location):
    """Return the 15-column manual-import shape, with structured tags in column 15."""
    tags = strand_pos_tags(pos).split()
    dialect = DIALECT_NAMES.get(location, location)
    if dialect:
        tags.append("dialect:" + quote(dialect, safe=""))
    return [
        language, parameter, form, definition, "", ipa, notes, source,
        "", "", "", "", "", "", " ".join(dict.fromkeys(tags)),
    ]

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
                soup = cached_soup(link, Path('.cache/strand') / f'alph-{char}.html', 'html5lib')
                if soup:
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
                                location = lang + '.' + dial
                                form = tds[-1].find('em').text
                                # Broken legacy table markup occasionally makes html.parser omit
                                # the repeated form from ``.text``. Do not abort the page; in that
                                # rare case the POS cannot be recovered safely from this row.
                                before_gloss = (
                                    text.split(form, 1)[1].split('‘', 1)[0]
                                    if form in text else ""
                                )
                                r = strand_row(
                                    lang_mapping[location], last_head['id'], form,
                                    defns[0] if defns else '', '', comment.text if comment else '',
                                    src_mapping[src], before_gloss, location,
                                )
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
            soup = cached_soup(link, '.cache/strand-legacy/bhatera.html', 'html.parser')
            if soup:
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
                            pos = l.group(1)
                            definition = l.group(2).lower()
                            turner = re.search(r'T\..(\d+)', data)
                            if turner:
                                turner = turner.group(1)
                                ipa = t(word2, column='IPA').replace(' ', '').replace('#', ' ')
                                writer.writerow(strand_row(
                                    'bhatr', turner, word, definition, ipa, '', 'strand', pos, 'bhatr'
                                ))

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
                    cache = Path('.cache/strand-legacy') / f'{codes[i]}-{char}.html'
                    soup = cached_soup(link, cache, 'html.parser')
                    if soup:
                        for data in soup.find_all(class_='dic'):
                            word = data.find(class_='l')
                            if word:
                                word = word.find(text=True, recursive=False)
                                word2 = re.sub(r'ʹ(.)', r'\1ʹ', word)
                                l = re.search(r'<b>]</b>\xa0 (.*?)\.\xa0 (.*?)\.', str(data))
                                if not l:
                                    l = re.search(r'</span>\xa0 (.*?)\.\xa0 (.*?)\.', str(data))
                                if l:
                                    pos = l.group(1)
                                    definition = l.group(2).lower()
                                    turner = re.search(r'T\. (\d+)', str(data))
                                    if turner:
                                        turner = turner.group(1)
                                        ipa = t(word2, column='IPA').replace(' ', '').replace('#', ' ')
                                        writer.writerow(strand_row(
                                            codes[i], turner, word, definition, ipa, '', 'strand', pos, codes[i]
                                        ))

                except HTTPError as e:
                    pass

def main():
    strand()
    strand2()
    strand3()

if __name__ == "__main__":
    main()
