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
legacy_output_languages = {
    'deg': 'deg', 'Kam': 'Kam', 'Kata': 'ktivi', 'Ash': 'sanu',
    'Wg': 'nis', 'Kho': 'Kho', 'Phal': 'Phal',
}
legacy_locations = {
    'Kata': 'Kmkt.ktv', 'Ash': 'Aṣk.s', 'Wg': 'Kal.n',
}
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
    r"(?<![A-Za-z])(?:VT[A-Za-z0-9?+<*`>~ḍṭ&;!:-]*|VI[A-Za-z0-9?+<*`>~ḍṭ&;!:-]*|"
    r"V[A-Za-z0-9?+<*`>~ḍṭ&;-]*|NP[A-Za-z0-9?+<*`>~ḍṭ&;-]*|"
    r"N[A-Za-z0-9?+<*`>~ḍṭ&;-]*|Aj[A-Za-z0-9?+<*`>~ḍṭ&;-]*|"
    r"AJ[A-Za-z0-9?+<*`>~ḍṭ&;-]*|Av[A-Za-z0-9?+<*`>~ḍṭ&;-]*|"
    r"Pn[A-Za-z0-9?+<*`>~ḍṭ&;-]*|Cj[A-Za-z0-9?+<*`>~ḍṭ&;-]*|C|"
    r"Id[A-Za-z0-9?+<*`>~ḍṭ&;-]*|I|Neg|Emp|Mode?[A-Za-z0-9?+<*`>~ḍṭ&;-]*|"
    r"Stat|St|D[A-Za-z0-9?+<*`>~ḍṭ&;-]*|L[A-Za-z0-9?+<*`>~ḍṭ&;!:-]*|"
    r"M[A-Za-z0-9?+<*`>~ḍṭ&;-]*|En[A-Za-z0-9?+<*`>~ḍṭ&;-]*|"
    r"Qt[A-Za-z0-9?+<*`>~ḍṭ&;-]*)"
    r"(?![A-Za-z])",
)


def clean_strand_text(value):
    """Collapse source line wrapping and non-breaking whitespace without changing spelling."""
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_legacy_stress(value):
    """Move Strand's pre-vowel stress symbols after the vowel for combining output."""
    return re.sub(r"([ʹ`´ˊ'])(.)", r"\2\1", value)


_GENDER_WORD = re.compile(
    r"(?<![A-Za-z])(?:m\.?|masc\.?|masculine|male|f\.?|fem\.?|feminine|female|"
    r"n\.?|neut\.?|neuter)(?![A-Za-z])",
    re.IGNORECASE,
)


def strand_definition_tags(definition):
    """Extract bracketed Strand gender labels from a definition.

    A bare marker such as ``[m.]`` is structured metadata and is removed from the gloss. More
    descriptive brackets such as ``[fem. form only]`` still contribute a gender tag, but remain in
    the gloss so the non-gender information is not lost.
    """
    tags = []

    def replace(match):
        content = clean_strand_text(match.group(1))
        found = _GENDER_WORD.findall(content)
        for word in found:
            folded = word.rstrip(".").casefold()
            if folded in {"m", "masc", "masculine", "male"}:
                tags.append("m")
            elif folded in {"f", "fem", "feminine", "female"}:
                tags.append("f")
            elif folded in {"n", "neut", "neuter"}:
                tags.append("n")
        if not found:
            return match.group(0)

        remainder = _GENDER_WORD.sub("", content)
        remainder = re.sub(r"(?<![A-Za-z])(?:s|sg|singular|pl|plural)\.?(?![A-Za-z])", "", remainder, flags=re.IGNORECASE)
        remainder = re.sub(r"[\s./,;|]+", "", remainder)
        return "" if not remainder else match.group(0)

    cleaned = re.sub(r"\[([^]]+)\]", replace, definition or "")
    cleaned = re.sub(r"\s+([,;:.!?])", r"\1", clean_strand_text(cleaned))
    return cleaned, list(dict.fromkeys(tags))


def _strand_code_tags(code):
    folded = code.casefold()
    tags = []
    if folded.startswith("vt"):
        tags = ["verb", "tr"]
    elif folded.startswith("vi"):
        tags = ["verb", "intr"]
    elif folded.startswith("v"):
        tags = ["verb"]
    elif folded.startswith("pn") or re.match(r"np(?:n|d|[1234]|$)", folded):
        tags = ["pron"]
        if "pl" in folded:
            tags.append("pl")
        if "?" in code:
            tags.append("interr")
    elif folded.startswith("n"):
        tags = ["num"] if "qt" in folded else (["adj"] if "ql" in folded else ["noun"])
        if "F" in code:
            tags.append("f")
        if "pl" in folded:
            tags.append("pl")
    elif folded.startswith("aj"):
        tags = ["adj"]
        if "qt" in folded:
            tags.append("num")
        if "F" in code:
            tags.append("f")
        if "pl" in folded:
            tags.append("pl")
    elif folded.startswith("av"):
        tags = ["adv"]
    elif folded.startswith("cj") or folded == "c":
        tags = ["conj"]
    elif folded == "i" or folded.startswith("id"):
        tags = ["interj"]
    elif folded == "neg" or folded.endswith("neg"):
        tags = ["part", "neg"]
    elif folded == "emp":
        tags = ["part", "emph"]
    elif folded.startswith("mod"):
        tags = ["part"]
    elif folded in {"st", "stat"}:
        tags = ["part", "interr"]
    elif folded.startswith("d"):
        tags = ["dir"]
        if "neg" in folded:
            tags.append("neg")
    elif folded.startswith("man"):
        tags = ["adv", "manner"]
    elif folded.startswith(("m", "en")):
        tags = ["suffix"]
    elif folded.startswith("qt"):
        tags = ["num"]
    elif folded.startswith("l"):
        # L... spans locative adverbs, suffixes, and place-name classes. ``spatial`` is the safe
        # shared grammatical feature; only the explicitly temporal LTm subtype gets more detail.
        tags = ["spatial"]
        if "tm" in folded:
            tags.append("temporal")
    return tags


def strand_pos_tags(label):
    """Translate Strand grammatical codes into canonical, searchable Jambu tags."""
    tags = []
    label = re.sub(r"\(\s*via\b.*?\)", "", label or "", flags=re.IGNORECASE)
    for match in _POS_TOKEN.finditer(label):
        tags.extend(_strand_code_tags(match.group(0)))
    return " ".join(dict.fromkeys(tags))


_LEGACY_POS = re.compile(
    rf"(({_POS_TOKEN.pattern})(?:\|({_POS_TOKEN.pattern}))*)"
    rf"(?:\s+(?:Z|\d+))?\.(?:\s|&nbsp;|\xa0)*"
)
_LEGACY_ANALYSIS = re.compile(
    r"\.\s*(?:\xa0|&nbsp;| )*\[<span[^>]*class=[\"']dic[\"'][^>]*>", re.IGNORECASE
)


def parse_legacy_entry(data):
    """Extract one old-style Strand dictionary paragraph.

    Definitions end at the bracketed morphological analysis, not at the first full stop: the
    latter truncates common source text such as ``[m.]`` and leaks inline ``span`` markup.
    """
    word_node = data.find(class_="l")
    if not word_node:
        return None
    word = word_node.find(string=True, recursive=False) or word_node.get_text()
    html = str(data).replace("\n", " ")
    pos_match = _LEGACY_POS.search(html)
    if not pos_match:
        return None
    definition_start = pos_match.end()
    analysis_match = _LEGACY_ANALYSIS.search(html, definition_start)
    if not analysis_match:
        return None
    definition_html = html[definition_start:analysis_match.start()]
    definition = BeautifulSoup(definition_html, "html.parser").get_text()
    turner = re.search(r"T\.\s*(\d+)", html)
    return {
        "word": clean_strand_text(word),
        "pos": pos_match.group(1),
        "definition": clean_strand_text(definition).lower(),
        "turner": turner.group(1) if turner else "",
    }


def strand_row(language, parameter, form, definition, ipa, notes, source, pos, location):
    """Return the 15-column manual-import shape, with structured tags in column 15."""
    tags = strand_pos_tags(pos).split()
    definition, definition_tags = strand_definition_tags(definition)
    tags.extend(definition_tags)
    dialect = DIALECT_NAMES.get(location, location)
    if dialect:
        tags.append("dialect:" + quote(dialect, safe=""))
    return [
        language, parameter, form, definition, "", ipa, notes, source,
        "", "", "", "", "", "", " ".join(dict.fromkeys(tags)),
    ]

def strand3():
    with open('strand3.csv', 'w') as f, open('../params/strand3.csv', 'w') as p:
        forms = csv.writer(f, lineterminator="\r\n")
        params = csv.writer(p, lineterminator="\r\n")
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
                                text = clean_strand_text(tds[-1].get_text())
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
                                    'lang': clean_strand_text(tds[-2].find('em').text),
                                    'level': int(level),
                                    'form': clean_strand_text(tds[-1].find('em').text),
                                    'defn': clean_strand_text(defns[0]) if defns else '',
                                    'id': turner if turner else f'n{ct}',
                                    'comment': clean_strand_text(comment.text) if comment else ''}
                                last_head = l
                                stack.append(l)
                                done = False

                            # forms    
                            elif row.find(class_='lng'):
                                comment = tds[-1].find(class_='sm')
                                text = clean_strand_text(tds[-1].get_text())
                                defns = re.findall(r'‘(.*?)’', text)
                                location, src = clean_strand_text(
                                    tds[-2].find('em').text
                                ).rsplit('.', 1)
                                form = clean_strand_text(tds[-1].find('em').text)
                                # Broken legacy table markup occasionally makes html.parser omit
                                # the repeated form from ``.text``. Do not abort the page; in that
                                # rare case the POS cannot be recovered safely from this row.
                                before_gloss = (
                                    text.split(form, 1)[1].split('‘', 1)[0]
                                    if form in text else ""
                                )
                                r = strand_row(
                                    lang_mapping[location], last_head['id'], form,
                                    clean_strand_text(defns[0]) if defns else '', '',
                                    clean_strand_text(comment.text) if comment else '',
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
        writer = csv.writer(fout, lineterminator="\r\n")
        link = f'http://nuristan.info/IndoAryan/SwatIndus/Bhatera/BhateraLanguage/Lexicon/lex.html'
        try:
            soup = cached_soup(link, '.cache/strand-legacy/bhatera.html', 'html.parser')
            if soup:
                for data in soup.find_all(class_='dic'):
                    parsed = parse_legacy_entry(data)
                    if parsed:
                        word = parsed["word"]
                        word2 = normalize_legacy_stress(word)
                        if parsed["turner"]:
                            ipa = t(word2, column='IPA').replace(' ', '').replace('#', ' ')
                            writer.writerow(strand_row(
                                'bhatr', parsed["turner"], word, parsed["definition"], ipa, '',
                                'strand', parsed["pos"], 'bhatr'
                            ))

        except HTTPError as e:
            pass

def strand():
    with open('strand.csv', 'w') as fout:
        writer = csv.writer(fout, lineterminator="\r\n")
        for i, language in enumerate(languages):
            for char in chars:
                link = f'http://nuristan.info/{language}/Lexicon/alph-{char}.html'
                print(link)
                try:
                    cache = Path('.cache/strand-legacy') / f'{codes[i]}-{char}.html'
                    soup = cached_soup(link, cache, 'html.parser')
                    if soup:
                        for data in soup.find_all(class_='dic'):
                            parsed = parse_legacy_entry(data)
                            if parsed:
                                word = parsed["word"]
                                word2 = normalize_legacy_stress(word)
                                if parsed["turner"]:
                                    ipa = t(word2, column='IPA').replace(' ', '').replace('#', ' ')
                                    writer.writerow(strand_row(
                                        legacy_output_languages[codes[i]], parsed["turner"], word,
                                        parsed["definition"], ipa,
                                        '', 'strand', parsed["pos"],
                                        legacy_locations.get(codes[i], codes[i])
                                    ))

                except HTTPError as e:
                    pass

def main():
    strand()
    strand2()
    strand3()

if __name__ == "__main__":
    main()
