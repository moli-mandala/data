import re
import csv
import unicodedata
from pathlib import Path


HERE = Path(__file__).parent
PROJECT = Path(__file__).parents[5]
OUTPUT = Path(__file__).parents[2] / "20230521-rajasthani.csv"
AUDIT = PROJECT / "source_checklists/audits/20230521-rajasthani-exclusions.csv"


def lexical_key(language, form, gloss, source):
    """Identify a survey response independently of its curated etymology.

    Parameter_ID was assigned by hand after the original extraction.  The raw survey files do
    not contain those CDIAL IDs, so a re-extraction must carry them forward from the installed
    table rather than silently replacing them with blanks.
    """
    return (
        language,
        unicodedata.normalize("NFC", form).strip(),
        unicodedata.normalize("NFC", gloss).strip(),
        source.split("[", 1)[0],
    )


def load_curated_parameters(path):
    parameters = {}
    if not path.exists():
        return parameters
    with path.open(encoding="utf-8", newline="") as stream:
        for row_number, row in enumerate(csv.reader(stream), 1):
            if len(row) < 8 or not row[1]:
                continue
            key = lexical_key(row[0], row[2], row[3], row[7])
            previous = parameters.setdefault(key, row[1])
            if previous != row[1]:
                raise ValueError(
                    f"conflicting curated Parameter_ID values at {path}:{row_number}: "
                    f"{previous!r} versus {row[1]!r} for {key!r}"
                )
    return parameters

lects = {
    'hadothi': {
        'F': 'patera',
        'B': 'gothda',
        'C': 'kawai',
        'E': 'pakkarana',
        'H': 'shyampura',
        'A': 'silori',
        'I': 'devpura',
        'G': 'piplia',
        'D': 'kelwada',
        'h': None
    },
    'mewati': {
        'G': 'goyla',
        'T': 'akera',
        'H': 'hathiya',
        'U': 'udaka',
        'S': 'gulpeda',
        'J': 'jakhopur',
        'Q': 'chirkana',
        'A': 'jhambaus',
        'h': None
    },
    'dhundari': {
        's': 'Pathalvas',
        'D': 'Deladi',
        'N': 'Bamore',
        'J': 'Jorpura',
        'B': 'Bhagatpur',
        'A': 'Badagaon',
        'P': 'Chalkoi',
        'h': None
    },
    'marwari': {
        'w': 'Mukheri',
        'g': 'Gomat',
        'D': 'Fatehgarh',
        'E': 'Husangsar',
        'd': 'Degana',
        'k': 'Kherwa',
        'N': 'Bagra',
        'F': 'Falna',
        'B': 'Bhagatpur',
        'A': 'Badagaon',
        'P': 'Chalkoi',
        'h': None
    },
    'mewari': {
        'Y': 'Gorana',
        'w': 'Kannouj',
        'y': 'Sangad',
        'k': 'Padarada',
        'l': 'Dindoli',
        'z': 'Kalnsas',
        'd': 'Dholpura',
        'b': 'Eklingpura',
        # The registry's stable source lect ID is ``mewari_kishanji``; the full locality name is
        # retained as its display name in cldf/dialects.csv.
        'c': 'Kishanji',
        'i': 'Bannoda',
        'j': 'Hurda',
        'G': 'Khor',
        'M': 'Ajmer',
        'f': 'Kalgav',
        'm': 'Basad',
        'J': 'Jesingpura',
        'n': 'Bhunyakhedi',
        'e': 'Pathera',
        'X': 'Godra',
        'h': None,
        'g': None
    },
    'bagri': {
        'A': 'Pallu',
        'B': 'Loonkansar',
        'C': 'Sardarsahar',
        'D': 'Old_Abaddi',
        'E': 'Makkasar',
        'F': 'Mirzawala',
        'J': 'Jamal',
        'K': 'Karnigedda',
        'L': 'Lakjikirani',
        'P': 'Panjkosi',
        'T': 'Fatehabad',
        'V': 'Mannaksar',
        'h': None
    }
}

# Most prompts use ``12. gloss``; the OCR has one comma and one omitted stop.  A leading number
# followed by punctuation or an ASCII gloss is a prompt header, while page-number-only lines are
# discarded separately above.
match_str = r'^(\*)?\d+(?:[\.,]|(?=\s+[A-Za-z]))\s*'

rows = []
exclusions = []
curated_parameters = load_curated_parameters(OUTPUT)
for file in lects:
    lines = []

    with (HERE / file).open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip('\n').lstrip()
            if line.strip().isdigit():
                continue
            elif re.match(match_str, line):
                if line[-1] in ['h', 'ɦ']:
                    toks = line.split(' ')
                    line = ' '.join(toks[:-1])
                    lines.append(re.sub(match_str, '', line) + ' | ')
                    lines.append(toks[-1])
                else:
                    lines.append(re.sub(match_str, '', line) + ' | ')
                print(line)
            else:
                lines[-1] = lines[-1] + line

    cur_gloss = None
    for line in lines:
        line = line.strip()
        print(f'line: "{line}"')
        if not line: continue
        gloss, text = line.split('|')
        gloss = gloss.strip()
        words = text.split(']')
        print(words)
        for word in words:
            word = word.strip()
            if not word: continue
            lemma, dialects = word.split('[')
            lemma = lemma.strip()
            dialects = dialects.strip()

            # Parentheses following a real response are editorial qualifiers, not part of its
            # pronunciation.  Pure placeholders and explicit "no entry" responses are absences,
            # so preserve them in the audit rather than installing English as lexical data.
            note = ""
            qualified = re.fullmatch(r"(.+?)\s+(\([^()]+\))", lemma)
            if qualified:
                lemma, note = qualified.groups()

            for dialect in dialects:
                if lects[file][dialect] == None: continue
                lang = file + '_' + lects[file][dialect].lower()
                if file == 'marwari' and dialect in ['A', 'B', 'P']:
                    lang = 'dhundari_' + lects['dhundari'][dialect].lower()
                reason = ""
                if not lemma:
                    reason = "blank source response"
                elif lemma.casefold() == "no entry":
                    reason = "source explicitly says no entry"
                elif lemma.startswith("(") and lemma.endswith(")"):
                    reason = "metalinguistic placeholder, not a lexical form"
                if reason:
                    exclusions.append((
                        "skipped", reason, str(OUTPUT.relative_to(PROJECT)), "", lang,
                        gloss, file, lemma,
                    ))
                    continue
                parameter = curated_parameters.get(
                    lexical_key(lang, lemma, gloss, file), ""
                )
                rows.append((lang, parameter, lemma, gloss, '', lemma, note, file))

rows = sorted(set(rows), key=lambda row: (row[3], row[2], row[0], *row[1:]))
emitted_keys = {lexical_key(row[0], row[2], row[3], row[7]) for row in rows}
missing_curated = set(curated_parameters) - emitted_keys
if missing_curated:
    samples = sorted(missing_curated)[:5]
    raise ValueError(
        f"re-extraction dropped {len(missing_curated)} curated Rajasthani responses; "
        f"examples: {samples!r}"
    )
with OUTPUT.open('w', encoding="utf-8", newline="") as fout:
    writer = csv.writer(fout, lineterminator="\n")
    writer.writerows(rows)

exclusions.sort(key=lambda row: (row[5], row[4], row[7]))
with AUDIT.open('w', encoding="utf-8", newline="") as fout:
    writer = csv.writer(fout, lineterminator="\n")
    writer.writerow([
        "Status", "Reason", "Installed_File", "Former_Row", "Language_ID", "Gloss",
        "Source", "Raw_Form",
    ])
    writer.writerows(exclusions)
