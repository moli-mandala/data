import csv
import unidecode
import re
import glob
from segments.tokenizer import Tokenizer, Profile
import unicodedata
from tqdm import tqdm
import os
from copy import deepcopy

from utils import mapping, superscript, change
from dialects import load_dialect_aliases, normalize_dialect
from tags import extract_tags
from form_grammar import extract_gloss_tags
from tamil_morphology import append_note, extract_tamil_verb_morphology
from dedr_variants import (
    expand_attached_sound_variants,
    expand_length_variants,
    normalize_dedr_marks,
)
from data.dedr.cleanup import is_footer_misparse

# read in tokenizer/convertors for IPA and form normalisation
tokenizers = {}
convertors = {}
for file in glob.glob("data/cdial/ipa/cdial/*.txt"):
    lang = file.split("/")[-1].split(".")[0]
    tokenizers[lang] = Tokenizer(file)
for file in glob.glob("conversion/*.txt"):
    lang = file.split("/")[-1].split(".")[0]
    convertors[lang] = Tokenizer(file)

# a set to track what languages and params are included
lang_set = set()
param_set = set()
included_params = set()

CROSS_FAMILY_COLUMNS = [
    "ID",
    "Entry_ID",
    "Compared_Entry_ID",
    "Relation",
    "Direction",
    "Confidence",
    "Source",
    "Evidence",
]
CROSS_FAMILY_RELATIONS = {"loan", "influence", "related"}
CROSS_FAMILY_DIRECTIONS = {
    "entry-from-compared",
    "compared-from-entry",
    "undetermined",
}
CROSS_FAMILY_CONFIDENCES = {"high", "medium", "low"}


def write_cross_family_comparisons(parameter_ids):
    """Validate and install article-level DEDR/CDIAL comparisons as a CLDF sidecar."""
    source_paths = [
        "data/cross-family-comparisons.csv",
        "data/manual-cross-family-comparisons.csv",
        "data/dbia/comparisons.csv",
    ]
    target_path = "cldf/comparisons.csv"
    rows = []
    for source_path in source_paths:
        with open(source_path, encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin)
            if reader.fieldnames != CROSS_FAMILY_COLUMNS:
                raise ValueError(
                    f"{source_path} columns are {reader.fieldnames!r}, "
                    f"expected {CROSS_FAMILY_COLUMNS!r}"
                )
            rows.extend(reader)

    seen = set()
    for row in rows:
        comparison_id = row["ID"]
        if not comparison_id or comparison_id in seen:
            raise ValueError(f"Missing or duplicate cross-family comparison ID: {comparison_id!r}")
        seen.add(comparison_id)
        endpoints = (row["Entry_ID"], row["Compared_Entry_ID"])
        missing = [entry_id for entry_id in endpoints if entry_id not in parameter_ids]
        if missing:
            raise ValueError(f"Comparison {comparison_id} references missing entries: {missing}")
        if endpoints[0] == endpoints[1]:
            raise ValueError(f"Comparison {comparison_id} links an entry to itself")
        if row["Relation"] not in CROSS_FAMILY_RELATIONS:
            raise ValueError(f"Comparison {comparison_id} has invalid relation {row['Relation']!r}")
        if row["Direction"] not in CROSS_FAMILY_DIRECTIONS:
            raise ValueError(f"Comparison {comparison_id} has invalid direction {row['Direction']!r}")
        if row["Confidence"] not in CROSS_FAMILY_CONFIDENCES:
            raise ValueError(f"Comparison {comparison_id} has invalid confidence {row['Confidence']!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+\[[^\]]+\]", row["Source"]):
            raise ValueError(f"Comparison {comparison_id} has invalid source locator {row['Source']!r}")
        if not row["Evidence"].strip():
            raise ValueError(f"Comparison {comparison_id} lacks printed evidence")

    with open(target_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=CROSS_FAMILY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class Row:
    def __init__(self, row, id):
        self.id = id
        self.lang = row[0]
        self.param = row[1].split('.')[0]
        self.form = row[2]
        self.old_form = self.form
        self.gloss = row[3]
        self.native = row[4]
        self.ipa = row[5]
        self.notes = row[6]
        # Structured tokens may be supplied by richer manual importers.
        # ``inherited`` used to duplicate the graph's reflex relationship.  Drop the
        # retired token while reading legacy importer output; unresolved rows remain
        # unlinked rather than gaining an implied etymology.
        self.tags = "" if len(row) < 15 else " ".join(
            tag for tag in row[14].split() if tag != "inherited"
        )
        self.source = row[7]
        self.variant_of = ""  # for a comma-listed alternate: the id of the first (main) form
        self.cognateset = '' if len(row) < 9 else row[8]
        # Optional tenth manual-import column: source-specific etymological analysis.
        self.etymology = '' if len(row) < 10 else row[9]
        # Stable source-local graph keys, resolved to generated IDs by unify_cldf.py.
        self.entry_key = '' if len(row) < 11 else row[10]
        self.variant_of_key = '' if len(row) < 12 else row[11]
        self.borrowed_from_key = '' if len(row) < 13 else row[12]
        self.derivation_parent_keys = '' if len(row) < 14 else row[13]
        if '.' in self.cognateset:
            parts = list(self.cognateset.split("."))
            if parts[1] == '0':
                self.cognateset = parts[0]

    @property
    def formatted(self):
        rows = [
            self.id,
            self.lang,
            self.param,
            self.form,
            self.gloss,
            self.native,
            self.ipa,
            self.old_form,
            self.cognateset,
            self.notes,
            self.tags,
            self.source,
            self.variant_of,
            self.etymology,
            self.entry_key,
            self.variant_of_key,
            self.borrowed_from_key,
            self.derivation_parent_keys,
        ]
        return rows

    def __repr__(self):
        return f"<Row {self.lang} {self.param} {self.form} {self.gloss}>"


STRAND3_FILE = "20221003-strand3.csv"
LEGACY_STRAND_FILES = {"20220913-strand.csv", "20220913-strand2.csv"}
MERRIAM_DRAVIDIAN_DB_FILE = "data/other/forms/20260718-merriam-dravidian-db.csv"


def _append_distinct(primary, secondary, separator="; "):
    """Keep the preferred value first and append genuinely new fallback metadata."""
    values = []
    for value in (primary, secondary):
        for part in value.split(separator) if value else ():
            part = part.strip()
            if part and part not in values:
                values.append(part)
    return separator.join(values)


def _strand_gloss(value):
    """A deliberately strict gloss key used only to disambiguate exact-form collisions."""
    return re.sub(r"[^a-z]+", " ", value.casefold()).strip()


def format_munda_parameter(row):
    """Map Rau's Munda parameter row onto the compiled parameter schema.

    The legacy five-column file stores the lexical gloss in column four and the
    Pinnow/MKCD comparison prose in column five.  The latter is entry-level
    etymology, not the generic ``Etyma``/description field.
    """
    if len(row) != 5:
        raise ValueError(f"Munda parameter row has {len(row)} columns, expected 5: {row!r}")
    entry_id, headword, _legacy_language, gloss, etymology = row
    return [
        entry_id,
        headword.split(",")[0].strip(),
        "PMu",
        gloss,
        "",
        etymology,
    ]


SCHMIDT_VOWELS = "aeiouəæãẽõũ"
SCHMIDT_PROFILE_LANGUAGES = {"K", "kash", "pog", "sir"}

# These profiles operate on source forms whose boundary marks, homonym numbers, and internal
# punctuation are meaningful source data.  The legacy generic converter strips such characters
# before tokenization because many older wordlists used them as disposable list notation.
PRESERVE_SOURCE_PROFILE_INPUT = {
    "house", "vaagri", "drasi", "yoshioka", "gandhari", "kullui", "toda", "rabha", "lsi",
    "western-tamang",
    "humla",
    "gurung",
    "dotyali",
    "kudiya",
    "majhi-bote",
    "kochila-tharu",
    "pyangaun-newar",
    "maikoti-kham",
    "thakali",
    "mustang-loke",
    "kurux-nepal",
    "north-gorkha",
    "weinreich-domaaki",
    "brahui",
    "emeneau-brahui",
    "kannauji",
    "pahari",
    "naaba",
    "magar-2024",
    "nihali",
    "dewas-rai",
    "hajong-survey",
    "santali-cluster",
    "torwali-student",
    "sampang",
    "mewahang",
    "chhulung",
    "magahi-survey",
    "badaga-hockings",
    "nured",
    "merriam-reconstruction",
}


def normalize_schmidt_stress(value):
    """Move Schmidt & Kaul's pre-syllable apostrophe onto its vowel.

    Table 3 prints stress before the onset of a non-initial stressed syllable.
    Profiles substitute graphemes locally, so make the mark combining first;
    the profile can then retain it while converting colon length to macrons.
    """
    pattern = rf"'([^{SCHMIDT_VOWELS}\s]*)([{SCHMIDT_VOWELS}])(:?)"
    return re.sub(pattern, rf"\1\2\3́", value)


def merge_redundant_strand_rows(results):
    """Fold legacy Strand duplicates into Strand3 while keeping Strand3's etymology.

    A duplicate must have the same language and post-conversion form. If Strand3 contains
    homophones, select a target only when the old Parameter_ID agrees, the normalized gloss
    uniquely identifies one Parameter_ID, or every Strand3 candidate has the same Parameter_ID.
    Ambiguous homophones remain separate rather than acquiring an arbitrary etymology.
    """
    strand3 = {}
    for row in results:
        if getattr(row, "input_file", "") != STRAND3_FILE:
            continue
        key = (row.lang, unicodedata.normalize("NFC", row.form).strip())
        strand3.setdefault(key, []).append(row)

    removed = set()
    merged = 0
    for row in results:
        if getattr(row, "input_file", "") not in LEGACY_STRAND_FILES:
            continue
        key = (row.lang, unicodedata.normalize("NFC", row.form).strip())
        candidates = strand3.get(key, ())
        if not candidates:
            continue

        candidate_params = {candidate.param for candidate in candidates}
        same_param = [candidate for candidate in candidates if candidate.param == row.param]
        gloss = _strand_gloss(row.gloss)
        same_gloss = [
            candidate
            for candidate in candidates
            if gloss and _strand_gloss(candidate.gloss) == gloss
        ]

        if same_param:
            eligible = same_param
        elif same_gloss and len({candidate.param for candidate in same_gloss}) == 1:
            eligible = same_gloss
        elif len(candidate_params) == 1:
            eligible = list(candidates)
        else:
            continue

        # Multiple Strand3 rows can repeat the same analysis. Prefer its exact gloss when possible;
        # the ordinary (language, parameter, form) deduper below will fold the remaining copies.
        target = next(
            (candidate for candidate in eligible if gloss and _strand_gloss(candidate.gloss) == gloss),
            eligible[0],
        )
        target.gloss = _append_distinct(target.gloss, row.gloss)
        target.native = _append_distinct(target.native, row.native)
        target.notes = _append_distinct(target.notes, row.notes)
        target.source = _append_distinct(target.source, row.source, separator=";")
        target.ipa = _append_distinct(target.ipa, row.ipa)
        target.old_form = _append_distinct(target.old_form, row.old_form)
        removed.add(id(row))
        merged += 1

    return [row for row in results if id(row) not in removed], merged


def parse_file(file: str, errors, name=None, file_num=0, param_counter=None):
    stats = {
        "converted": 0,
        "for_conversion": 0
    }
    is_cdial = "cdial" in file
    is_manual = "other/forms" in file  # hand-curated source CSVs may carry unetymologised forms
    if param_counter is None:
        param_counter = {}
    # get filename
    if name is None:
        name = os.path.splitext(os.path.basename(file))[0]
        if "-" in name:
            name = name.split("-")[1]

    # check if convertible
    convert = name in convertors or name in mapping
    ipa = mapping.get(name, None)

    fin = open(file, "r")
    lines = fin.readlines()
    read = csv.reader(lines)
    result = []

    i = 0
    for row in tqdm(read, total=len(lines)):
        row = Row(row, id=f"{file_num}-{i}")
        row.input_file = os.path.basename(file)
        source_key = row.source.split(";", 1)[0].split("[", 1)[0]
        row.gloss, gloss_tags = extract_gloss_tags(
            row.gloss,
            input_file=row.input_file,
            source_key=source_key,
            full_input_path=file,
        )
        row.tags = " ".join(
            dict.fromkeys(filter(None, [*row.tags.split(), *gloss_tags]))
        )
        # Merriam's reconstruction column omits the conventional leading asterisk. Preserve that
        # exact upstream spelling in Original, but mark the display Form as reconstructed. Some
        # cited Krishnamurti records already include ``*`` and are left unchanged.
        is_merriam_reconstruction = row.source.split("[", 1)[0] == "merriam2026dravidiandb"
        if is_merriam_reconstruction and not row.form.startswith("*"):
            row.form = "*" + row.form
        # Both hand-entered and OCR-derived Shackle rows use the same CDIAL-style
        # romanisation. The auto filename does not reduce to ``old_punjabi`` via
        # the legacy filename heuristic, so select its phonetic parser by source.
        row_ipa = "cdial" if row.source in {"shackle", "shackle-auto"} else ipa
        row_convert = row_ipa is not None and (row.source in {"shackle", "shackle-auto"} or convert)
        # Hindu Kush Areal Typology supplies canonical IPA, unlike Liljegren's Palula dictionary
        # (practical orthography). The dated filename heuristic reduces both to ``liljegren``;
        # route this source explicitly through its IPA-to-house-transcription profile.
        if row.source.split("[", 1)[0] == "liljegren-hindukush":
            row_ipa = "liljegren-hindukush"
            row_convert = True
        # The dictionary supplies Unicode IPA. This source-key route keeps the
        # transcription contract stable if the dated snapshot filename changes.
        if row.source.split("[", 1)[0] == "torwali2023student":
            row_ipa = "torwali-student"
            row_convert = True
        # Abraham & Sako's sixteen Arunachal Pradesh wordlists supply Unicode IPA.
        # Convert only the display Form to Jambu transcription and retain the
        # source transcription unchanged in Phonemic.
        if row.source.split("[", 1)[0] == "abraham-sako2021":
            row_ipa = "tagin-puroik"
            row_convert = True

        if row.source.split("[", 1)[0] == "kondakov2013rabha":
            row_ipa = "rabha"
            row_convert = True
        # Hilty & Mitchell's nine comparative wordlists are Unicode IPA.
        # Keep the source IPA in Phonemic and normalize only the display Form.
        if row.source.split("[", 1)[0] == "hilty-mitchell2014":
            row_ipa = "yamphu"
            row_convert = True
        if row.source.split("[", 1)[0] == "hilty2013eastern-magar":
            row_ipa = "eastern-magar"
            row_convert = True
        # Lipp's three Western Tamang survey wordlists use Unicode IPA. Keep
        # that source value in Phonemic and convert only the display Form.
        if row.source.split("[", 1)[0] == "lipp2014western-tamang":
            row_ipa = "western-tamang"
            row_convert = True
        if row.source.split("[", 1)[0] == "devries2020humla":
            row_ipa = "humla"
            row_convert = True
        if row.source.split("[", 1)[0] == "swenson2019gurung":
            row_ipa = "gurung"
            row_convert = True
        if row.source.split("[", 1)[0] == "eichentopf-tupper2019dotyali":
            row_ipa = "dotyali"
            row_convert = True
        if row.source.split("[", 1)[0] == "joseph2024kudiya":
            row_ipa = "kudiya"
            row_convert = True
        if row.source.split("[", 1)[0] == "page2024majhi-bote":
            row_ipa = "majhi-bote"
            row_convert = True
        if row.source.split("[", 1)[0] == "eichentopf-mitchell2020kochila":
            row_ipa = "kochila-tharu"
            row_convert = True
        if row.source.split("[", 1)[0] == "smith2021pyangaun":
            row_ipa = "pyangaun-newar"
            row_convert = True
        if row.source.split("[", 1)[0] == "leman2020maikoti":
            row_ipa = "maikoti-kham"
            row_convert = True
        if row.source.split("[", 1)[0] == "webster2021thakali":
            row_ipa = "thakali"
            row_convert = True
        if row.source.split("[", 1)[0] == "khadgi-marcuson-marcuson2021mustang":
            row_ipa = "mustang-loke"
            row_convert = True
        if row.source.split("[", 1)[0] == "shackelford-swenson-chaudhary-maggard2022kurux":
            row_ipa = "kurux-nepal"
            row_convert = True
        if row.source.split("[", 1)[0] == "webster2022north-gorkha":
            row_ipa = "north-gorkha"
            row_convert = True
        if row.source.split("[", 1)[0] == "weinreich2008":
            row_ipa = "weinreich-domaaki"
            row_convert = True
        if row.source.split("[", 1)[0] == "ali-kobayashi2024":
            row_ipa = "brahui"
            row_convert = True
        # Emeneau underlines the digraph ``gh`` for the Brahui voiced velar fricative. Preserve
        # the article transcription in Original while rendering that digraph as Jambu ɣ.
        if row.source.split("[", 1)[0] == "emeneau1997brahui":
            row_ipa = "emeneau-brahui"
            row_convert = True
        # Burrow & Emeneau's 1972 DEN supplement follows the DED transcription conventions.
        # Route by bibliographic source ID because this is a manual, article-level import rather
        # than a file inside data/dedr; Original remains the exact printed form.
        if row.source.split("[", 1)[0] in {
            "burrow-emeneau1972den1", "burrow-emeneau1972den2",
        }:
            row_ipa = "dedr"
            row_convert = True
        # John & Varghese's thirteen target wordlists use Unicode IPA. The
        # source value remains in Phonemic while the display form is converted.
        if row.source.split("[", 1)[0] == "kannauji":
            row_ipa = "kannauji"
            row_convert = True
        # Smith's five Pahari field-site wordlists use Unicode IPA. Preserve
        # that source value in Phonemic and convert only the display Form.
        if row.source.split("[", 1)[0] == "smith2022pahari":
            row_ipa = "pahari"
            row_convert = True
        if row.source.split("[", 1)[0] == "swenson2025naaba":
            row_ipa = "naaba"
            row_convert = True
        if row.source.split("[", 1)[0] == "swenson2024magar":
            row_ipa = "magar-2024"
            row_convert = True
        if row.source.split("[", 1)[0] == "shackelford2019dewas-rai":
            row_ipa = "dewas-rai"
            row_convert = True
        if row.source.split("[", 1)[0] == "kim-ahmad-kim-sangma2011hajong":
            row_ipa = "hajong-survey"
            row_convert = True
        if row.source.split("[", 1)[0] == "kim-kim-ahmad-sangma2010santali-cluster":
            row_ipa = "santali-cluster"
            row_convert = True
        if row.source.split("[", 1)[0] == "rai-rai-thokar2015sampang":
            row_ipa = "sampang"
            row_convert = True
        if row.source.split("[", 1)[0] == "rai-rai-thokar2014mewahang":
            row_ipa = "mewahang"
            row_convert = True
        if row.source.split("[", 1)[0] == "rai-rai-thokar2014chhulung":
            row_ipa = "chhulung"
            row_convert = True
        if row.source.split("[", 1)[0] == "thakur-thakur2016magahi":
            row_ipa = "magahi-survey"
            row_convert = True
        # Hockings and Pilot-Raichoor mark vowel length with a colon and use
        # Dravidianist underdots. Preserve the source transcription in
        # Original while normalising the display form through its own profile.
        if row.source.split("[", 1)[0] == "hockings-pilotraichoor1992":
            row_ipa = "badaga-hockings"
            row_convert = True
        # Schmidt & Kaul use one transcription system for the four Table 3
        # Table 3 varieties. Route by provenance and language rather than the
        # dated filename, whose legacy parser reduces both Schmidt imports to
        # the ambiguous name ``schmidt``.
        if row.source == "schmidt" and row.lang in SCHMIDT_PROFILE_LANGUAGES:
            row_ipa = "schmidt-kashmiri"
            row_convert = True
        # Synthetic donor-language category nodes emitted by the Kalasha
        # importer are English labels, not Trail orthography.
        if name == "kalasha" and row.lang not in {"Kal", "bumb", "rumb", "bir", "urt"}:
            row_convert = False
        # Bashir donor nodes retain the source language's cited spelling; only
        # Khowar and its contributor/regional dialect rows use the Khowar profile.
        if name == "bashir" and not row.lang.startswith("Kho"):
            row_convert = False
        # DEDR forms (incl. the PDr reconstructions in pdr.csv, whose source is "krishnamurti")
        # are already in a Dravidianist transcription; the shared profile only normalises house
        # conventions (ழ r̤ -> ṛ̆, ṅ -> ŋ, aspirates -> superscript, anusvara -> ṁ, marked vowels
        # -> IPA). Length-ambiguous vowels are split into variants below, not here.
        if "dedr" in file:
            row_ipa = "dedr"
            row_convert = True
        # Backstrom's Urdu and Pashto lists are survey controls rather than the
        # northern locality varieties we want to publish in Jambu.
        if os.path.basename(file) == "20230416-northern.csv" and row.lang in {"Urdu", "Pashto"}:
            continue
        if "dedr" in file and is_footer_misparse(row.form):
            continue
        if row.lang == "Drav":
            continue
        # cdial/dedr/munda rows without an etymon are parse junk and dropped; a blank Param_ID in a
        # manual import instead means "attested but unetymologised" — keep it as a lone node.
        if not row.param and not is_manual:
            continue
        row.is_lone = not row.param  # a surviving blank Param_ID ⇒ manual-import lone node
        if row.lang == "Indo-Aryan":
            row.form = row.form.lower()

        # param fix if .
        if "." in row.param:
            row.param = row.param.split(".")[0]

        # split multiple forms into separate rows; comma-listed alternates share one definition, so
        # the first is the main reflex and the rest are variants of it (same etymon, own alignment).
        source_form = row.form
        uses_dedr_transcription = (
            "dedr" in file
            or row.source.split("[", 1)[0] in {
                "burrow-emeneau1972den1", "burrow-emeneau1972den2",
            }
        )
        forms = (
            [
                normalize_dedr_marks(length_variant)
                for base in expand_attached_sound_variants(row.form)
                for length_variant in expand_length_variants(base)
            ]
            if uses_dedr_transcription
            # Commas and slashes inside a reconstruction are source notation, not the legacy
            # manual-import convention for expanded attested alternates. One upstream record must
            # remain one stable reconstruction node.
            else [row.form] if is_merriam_reconstruction
            else list(row.form.split(","))
        )
        main_id = None
        for fj, form in enumerate(forms):
            reformed = form
            if not is_merriam_reconstruction:
                row.old_form = source_form if uses_dedr_transcription and len(forms) > 1 else form
            row.form = form
            # Forms on a CDIAL-style numeric etymon (CDIAL itself, plus other-source additions that
            # hang reflexes on a CDIAL entry by its number) keep <file>-<row> ids, so the <etymon>-<n>
            # space stays free for promoted section forms. Every other source (Munda m1, Dravidian d1,
            # …) namespaces its reflexes under their etymon, e.g. m1-1, d1-2.
            epid = row.param.lstrip(">~")
            # Schmidt rows are one-form manual records and historically keep
            # their stable <file>-<row> IDs.  Preserve those IDs when a later
            # audit attaches them to a nonnumeric Proto-II/Dravidian root;
            # otherwise an ID such as pii-4147-2 can collide with the promoted
            # Proto-II reflex occupying that same namespace.
            stable_manual_id = is_manual and row.source == "schmidt"
            if is_cdial or stable_manual_id or not epid or re.fullmatch(r"\d+[a-z]?", epid):
                row.id = f"{file_num}-{i}"
            else:
                param_counter[epid] = param_counter.get(epid, 0) + 1
                row.id = f"{epid}-{param_counter[epid]}"
            if fj == 0:
                main_id = row.id
                row.variant_of = ""
            else:
                row.variant_of = main_id

            # convert IPA
            if row_ipa in PRESERVE_SOURCE_PROFILE_INPUT and row_convert:
                stats["for_conversion"] += 1
                # LSI's Form is Grierson's historical transcription, while its
                # Phonemic column is upstream's canonical CLTS segmentation.
                # Drive the display conversion from that analysis and leave
                # old_form untouched so it is emitted as Original.
                source_value = row.ipa if row_ipa == "lsi" else reformed
                src = unicodedata.normalize("NFC", source_value)
                form_out = unicodedata.normalize(
                    "NFC",
                    convertors[row_ipa](src, column="IPA")
                    .replace(" ", "")
                    .replace("#", " "),
                )
                if "�" in form_out:
                    errors.write(str(row) + " " + form_out + "\n")
                else:
                    row.form = form_out
                    stats["converted"] += 1
            elif row_ipa == "zoller" and row_convert:
                # Zoller's tonal pseudo-IPA. NFD so precomposed vowels split into base + marks the
                # profile matches (acute->caron rising, grave->circumflex falling); the "IPA" column
                # is the normalised transcription (a-allophones merged), the "Phon" column the IPA
                # pronunciation which is kept in the Phonemic field.
                stats["for_conversion"] += 1
                src = unicodedata.normalize("NFD", reformed.strip("-1234,;."))
                form_out = unicodedata.normalize(
                    "NFC", convertors["zoller"](src, column="IPA").replace(" ", "").replace("#", " ")
                )
                phon_out = unicodedata.normalize(
                    "NFC", convertors["zoller"](src, column="Phon").replace(" ", "").replace("#", " ")
                )
                if "�" in form_out:
                    errors.write(str(row) + " " + form_out + "\n")
                else:
                    row.form = form_out
                    row.ipa = phon_out
                    stats["converted"] += 1
            elif row_ipa == "khowar" and row_convert:
                # Bashir marks stress on the vowel and low tone with a doubled
                # vowel whose second member is stressed (aá). NFD lets the
                # profile match these marks consistently. Keep bound-form
                # hyphens and homonym superscripts in the display form.
                stats["for_conversion"] += 1
                src = unicodedata.normalize("NFD", reformed.strip(",;."))
                form_out = unicodedata.normalize(
                    "NFC",
                    convertors["khowar"](src, column="IPA")
                    .replace(" ", "")
                    .replace("#", " "),
                )
                phon_out = unicodedata.normalize(
                    "NFC",
                    convertors["khowar"](src, column="Phon")
                    .replace(" ", "")
                    .replace("#", " "),
                )
                if "�" in form_out or "�" in phon_out:
                    errors.write(str(row) + " " + form_out + " " + phon_out + "\n")
                else:
                    row.form = form_out
                    row.ipa = phon_out
                    stats["converted"] += 1
            elif row_ipa == "ssnp" and row_convert:
                # The SSNP extractor supplies Unicode IPA in both source columns. Convert only
                # the display Form to Jambu transcription and retain the decoded IPA in Phonemic.
                stats["for_conversion"] += 1
                src = unicodedata.normalize("NFD", reformed.strip(","))
                form_out = unicodedata.normalize(
                    "NFC",
                    convertors["ssnp"](src, column="IPA")
                    .replace(" ", "")
                    .replace("#", " "),
                )
                if "�" in form_out:
                    errors.write(str(row) + " " + form_out + "\n")
                else:
                    row.form = form_out
                    stats["converted"] += 1
            elif row_ipa == "liljegren-hindukush" and row_convert:
                # Keep upstream canonical IPA in Phonemic and convert only the display form.
                # NFC matches the source's CLDF grapheme inventory; underscores are word spaces.
                stats["for_conversion"] += 1
                src = unicodedata.normalize("NFC", reformed.strip(",;."))
                form_out = unicodedata.normalize(
                    "NFC",
                    convertors["liljegren-hindukush"](src, column="IPA")
                    .replace(" ", "")
                    .replace("#", " "),
                )
                if "�" in form_out:
                    errors.write(str(row) + " " + form_out + "\n")
                else:
                    row.form = form_out
                    stats["converted"] += 1
            elif row_ipa is not None and "˚" not in form and row_convert:
                stats["for_conversion"] += 1
                # fix accentuation from Strand
                if row_ipa == "strand":
                    reformed = reformed.replace("′", "´")
                    reformed = re.sub(r"([`´])(.)", r"\2\1", reformed)
                    # Strand prints stress before the syllable, while this legacy normalization
                    # moves it after the vowel and deliberately drops it in the profile. Keep a
                    # following length sign adjacent to that vowel so the existing long-vowel
                    # graphemes (iː -> ī, âː -> āā, etc.) can still match.
                    reformed = re.sub(r"([`´])([ː:])", r"\2\1", reformed)
                elif row_ipa == "schmidt-kashmiri":
                    reformed = normalize_schmidt_stress(reformed)

                # do the conversion
                reformed = reformed.strip("-1234⁴5⁵67⁷,;.")
                reformed = convertors[row_ipa](reformed, column="IPA")
                reformed = reformed.replace(" ", "").replace("#", " ")

                # if conversion error then log it
                if "�" in reformed:
                    errors.write(str(row) + " " + reformed + "\n")
                else:
                    row.form = reformed
                    stats["converted"] += 1

            # add the result
            result.append(deepcopy(row))
            i += 1

    fin.close()
    return result, stats


def main():
    # write out forms.csv
    errors = open("errors.txt", "w")
    dialect_aliases = load_dialect_aliases()

    form_count = 0
    results: list[Row] = []
    files = [
        "data/cdial/cdial.csv",
        "data/munda/forms.csv",
        "data/dedr/dedr_new.csv",
        "data/dedr/pdr.csv",
    ] + [
        path for path in glob.glob("data/other/forms/*.csv")
        if path != MERRIAM_DRAVIDIAN_DB_FILE
    ]
    files.sort()
    # Append new imports after sorting so they cannot renumber every existing source's legacy
    # <file>-<row> IDs. Persistent IDs normally absorb ordering changes, but curated graph overlays
    # must also remain valid during the pre-ID build. Merriam has immutable Entry_Key values, so
    # its own identity does not depend on this append position.
    files.append(MERRIAM_DRAVIDIAN_DB_FILE)
    files.append("data/dbia/forms.csv")

    # now do the same thing for non-CDIAL languages
    tot_stats = {
        "converted": 0,
        "for_conversion": 0
    }
    param_counter: dict = {}  # shared <etymon>-<n> reflex counter across all non-CDIAL source files
    for file_num, file in enumerate(files):
        print(file)
        result, stats = parse_file(file, errors=errors, file_num=file_num, param_counter=param_counter)
        tot_stats["converted"] += stats["converted"]
        tot_stats["for_conversion"] += stats["for_conversion"]
        results.extend(result)
    
    print(tot_stats)

    # The older Strand scrape and Strand3 overlap heavily, but often disagree on the etymon.
    # Strand3 preserves the source hierarchy, so keep its row/Parameter_ID and use the older row
    # only to fill metadata. This must precede the generic deduper, whose key includes Parameter_ID.
    results, strand_merged = merge_redundant_strand_rows(results)
    print(f"merged {strand_merged} legacy Strand rows into Strand3")

    # clean up duplicates in results
    cleaned = {}
    for i, row in enumerate(tqdm(results)):
        # SSNP survey lists are intentionally unetymologised. The same short
        # form can answer several different prompts, so collapsing blank-param
        # rows on form alone silently merges distinct lexical entries. The
        # Andersen concordance likewise has true homographs (e.g. sa- 'six'
        # and pronominal sa-), which must remain separate dictionary senses.
        key = (
            row.lang,
            row.param,
            row.form,
            # Rich source-keyed imports can contain genuine homographs or the same form under
            # several elicitation prompts. Their immutable record keys keep those entries distinct
            # while retaining the legacy dedupe behaviour for other sources.
            row.entry_key
            if row.source.split("[", 1)[0] in {
                "gandhari", "grierson-lsi1928", "kullui-org", "liljegren-hindukush", "tulpule1999",
                "wolf-kota", "bhaskararao-toda2025", "weinreich2008", "yoshioka2012",
                "kannauji",
                "smith2022pahari",
                "swenson2025naaba",
                "swenson2024magar",
                "shackelford2019dewas-rai",
                "kim-ahmad-kim-sangma2011hajong",
                "kim-kim-ahmad-sangma2010santali-cluster",
                "rai-rai-thokar2015sampang",
                "rai-rai-thokar2014mewahang",
                "rai-rai-thokar2014chhulung",
                "thakur-thakur2016magahi",
                "mundlay1996",
                "nagaraja2014",
                "bhattacharya1957",
                "konow1906",
                "wiktionary-nihali",
                "hockings-pilotraichoor1992",
                "nured",
                "emeneau1997brahui",
                "buddruss-grangali1979",
                "torwali2023student",
                "merriam2026dravidiandb",
            }
            else "",
            row.gloss
            if not row.param and (
                row.lang.startswith("SSNP-")
                or row.notes.startswith("SSNP ")
                or row.source == "andersen1990"
                # These survey lists received hand-curated etymologies after import.  A form
                # repeated under different elicitation prompts is a distinct lexical sense;
                # merging it here loses the ability to retain both ancestry assignments (e.g.
                # Chhattisgarhi nā̃v 'name' versus nā̃v 'nine').
                or row.source.split("[", 1)[0] in {
                    "chattisgarhi", "bagri", "dhundari", "hadothi", "marwari", "mewari",
                    "mewati",
                }
            )
            else "",
        )
        if key not in cleaned:
            cleaned[key] = (row, i)
        else:
            orig_row = cleaned[key][0]
            if row.cognateset is None or row.cognateset == "":
                # dict.fromkeys, NOT set: set iteration order is hash-randomised per run, and
                # these joined fields feed the durable-ID fingerprint — a nondeterministic order
                # here silently re-minted ~650 f_ ids on every rebuild
                def _merge(*values, sep='; '):
                    return sep.join(dict.fromkeys(x for x in values if x))
                orig_row.gloss = _merge(orig_row.gloss, row.gloss)
                orig_row.native = _merge(orig_row.native, row.native)
                orig_row.notes = _merge(orig_row.notes, row.notes)
                orig_row.source = _merge(orig_row.source, row.source, sep=';')
                orig_row.ipa = _merge(orig_row.ipa, row.ipa)
                orig_row.old_form = _merge(orig_row.old_form, row.old_form)
                # Long source analyses are already deduplicated by their
                # extractor; do not recursively concatenate a growing block
                # when normalised duplicate forms collapse here.
                orig_row.etymology = orig_row.etymology or row.etymology

                cleaned[key] = (orig_row, cleaned[key][1])
                results[cleaned[key][1]] = orig_row
                results[i] = None

    tamil_morphology_review = []

    # write out all the forms
    with open("cldf/forms.csv", "w") as fout:
        forms = csv.writer(fout)
        forms.writerow(
            [
                "ID",
                "Language_ID",
                "Parameter_ID",
                "Form",
                "Gloss",
                "Native",
                "Phonemic",
                "Original",
                "Cognateset",
                "Description",
                "Tags",
                "Source",
                "Variant_Of",
                "Etymology",
                "Entry_Key",
                "Variant_Of_Key",
                "Borrowed_From_Key",
                "Derivation_Parent_Keys",
            ]
        )

        done = set()
        for row in results:
            if row is None or not row.form:
                continue
            # drop "?" and blank-param junk, but keep manual-import lone nodes (blank param, flagged)
            if row.param == "?" or (not row.param and not getattr(row, "is_lone", False)):
                continue
            if row.lang in change:
                row.lang = change[row.lang]
            row.lang = unidecode.unidecode(row.lang)
            row.lang = row.lang.replace(".", "")
            row.form = unicodedata.normalize("NFC", row.form)
            if row.param:
                param_set.add(row.param.lstrip(">~"))

            # Normalize source lects before creating language-qualified regional dialect tags.
            row.lang, row.tags = normalize_dialect(row.lang, row.tags, dialect_aliases)

            # Regional labels are a CDIAL convention. Other sources can contain the same place
            # names in bibliographic prose, so only CDIAL rows receive regional dialect tags.
            regional_language_id = (
                row.lang if row.source.split(";", 1)[0].split("[", 1)[0] == "CDIAL" else None
            )
            parsed_tags, row.notes = extract_tags(
                row.notes, language_id=regional_language_id
            )
            row.tags = " ".join(
                dict.fromkeys(filter(None, row.tags.split() + parsed_tags.split()))
            )
            lang_set.add(row.lang)

            if row.lang == "Tamil" and row.source == "dedr":
                morphology = extract_tamil_verb_morphology(row.form)
                if morphology:
                    row.form = morphology.citation_form
                    row.notes = append_note(row.notes, morphology.note)
                    row.tags = " ".join(
                        dict.fromkeys(filter(None, row.tags.split() + list(morphology.tags)))
                    )
                    if morphology.review_reason:
                        tamil_morphology_review.append(
                            [
                                row.id,
                                row.param,
                                row.form,
                                morphology.note,
                                row.gloss,
                                morphology.review_reason,
                            ]
                        )

            key = tuple(row.formatted[1:])
            if key not in done:
                forms.writerow(row.formatted)
            done.add(key)

    with open("data/tamil_verb_morphology_review.csv", "w") as fout:
        review = csv.writer(fout, lineterminator="\n")
        review.writerow(["ID", "Parameter_ID", "Form", "Morphology", "Gloss", "Reason"])
        review.writerows(tamil_morphology_review)

    etyma = {}
    with open("data/etymologies.csv", "r") as fin:
        reader = csv.reader(fin)
        for row in reader:
            etyma[row[0]] = row[1]

    # finally, cognates (unused so far) and parameters
    with open("cldf/parameters.csv", "w") as g:
        mapping = {"cdial": "cdial", "extensions_ia": "cdial", "strand3": "strand"}

        params = csv.writer(g)
        params.writerow([
            "ID", "Name", "Language_ID", "Description", "Etyma", "Etymology", "Source"
        ])

        with open("data/cdial/params.csv", "r") as fin:
            read = csv.reader(fin)
            for row in read:
                headword = (
                    row[1]
                    .replace("ˊ", "́")
                    .replace("`", "̀")
                    .replace(" --", "-")
                    .replace("-- ", "-")
                )
                headword = headword.strip(".,;-: ")
                headword = headword.replace("<? >", "")
                headword = headword.lower()
                headword = headword.replace("˜", "̃")
                # a comma lists alternate forms — the head-word is the first of them; a space
                # WITHOUT a comma is a genuine multi-word head-word (e.g. "kaḥ punar"), kept whole.
                headword = headword.split(",")[0].strip()
                reformed = ""
                if " " in headword:
                    reformed = headword
                elif "˚" not in headword:
                    reformed = (
                        convertors["cdial"](headword.strip("-123456,;"), column="IPA")
                        .replace(" ", "")
                        .replace("#", " ")
                    )
                    if "�" in reformed:
                        errors.write(f'{row[2]} {headword} {"?"} {"?"} {reformed}\n')
                        reformed = ""

                params.writerow(
                    [
                        row[0],
                        reformed if reformed else headword,
                        "Indo-Aryan",
                        row[3],
                        etyma.get(row[0], ""),
                        "",
                        row[4] if len(row) > 4 else "",
                    ]
                )
                included_params.add(row[0])

        for file in tqdm(sorted(glob.glob("data/other/params/*.csv"))):
            # get filename
            name = file.split("/")[-1].split(".")[0]
            convert = name in convertors or name in mapping
            name = mapping.get(name, name)
            with open(file, "r") as f:
                lines = f.readlines()
                read = csv.reader(lines)
                for row in read:
                    if name == "strand":
                        if row[1] in ["PNur", "PA"]:
                            row[2] = "*" + row[2]
                        row[2] = row[2].replace("′", "ʹ").replace("-", "")
                    if convert:
                        reformed = (
                            convertors[name](row[2].strip("-123456,;"), column="IPA")
                            .replace(" ", "")
                            .replace("#", " ")
                        )
                        if "�" in reformed:
                            errors.write(f'{name} {row[2]} {"?"} {"?"} {reformed}\n')
                            reformed = ""
                        else:
                            row[2] = reformed
                    params.writerow(
                        [
                            row[0], row[2].split(",")[0].strip(), row[1], row[3],
                            etyma.get(row[0], ""), "", "",
                        ]
                    )
                    included_params.add(row[0])

        munda_entry_texts = []
        with open("data/munda/params.csv", "r") as f:
            read = csv.reader(f)
            for row in read:
                compiled = format_munda_parameter(row)
                params.writerow(compiled + [""])
                munda_entry_texts.append(
                    [compiled[0], 0, "etymology", "markdown", compiled[5], "rau"]
                )
                included_params.add(compiled[0])

        # DBIA articles are form-less PDr loan-set entries.  Unmatched printed IA
        # terms are retained as source-local IA comparison entries in the same file.
        # Their HTML Description contains the full OCR transcription or a typed stub.
        with open("data/dbia/params.csv", "r") as f:
            read = csv.reader(f)
            for row in read:
                params.writerow(row[:5] + ["", ""])
                included_params.add(row[0])

        with open("data/dedr/params.csv", "r") as f:
            read = csv.reader(f)
            for row in read:
                row[2] = "PDr"
                row[1] = row[1].split(",")[0].strip()  # main head-word = first of the listed forms
                # Residual DEDR prose and old-edition locators now have typed, independently
                # sourced sidecars; do not retain parser fragments in the legacy scalar field.
                params.writerow(row[:5] + ["", ""])
                included_params.add(row[0])

        with open("data/nuristani_cognates.csv", encoding="utf-8") as f:
            ancestor_ids = sorted({row["Ancestor_ID"] for row in csv.DictReader(f)})
        collisions = sorted(set(ancestor_ids) & included_params)
        if collisions:
            raise ValueError(f"Proto-Indo-Iranian ancestor ID collisions: {collisions}")
        for ancestor_id in ancestor_ids:
            params.writerow([ancestor_id, "", "Indo-ir", "", "", "", ""])
            included_params.add(ancestor_id)

    write_cross_family_comparisons(included_params)

    # Preserve source-level prose as explicitly typed blocks. ``assign_form_ids.py`` rewrites
    # source-local entry IDs (including Strand PNur IDs) to their durable public Form_IDs.
    for file in sorted(glob.glob("data/other/entry_texts/*.csv")):
        with open(file, encoding="utf-8", newline="") as fin:
            for row in csv.DictReader(fin):
                munda_entry_texts.append([
                    row["Form_ID"], row["Position"], row["Kind"], row["Format"],
                    row["Content"], row["Source"],
                ])
    with open("cldf/entry-texts.csv", "w", encoding="utf-8", newline="") as fout:
        texts = csv.writer(fout)
        texts.writerow(["Form_ID", "Position", "Kind", "Format", "Content", "Source"])
        texts.writerows(munda_entry_texts)

    # A dangling Language_ID makes the CLDF invalid and used to pass as an easily missed print.
    with open("cldf/languages.csv", encoding="utf-8", newline="") as fin:
        cldf_langs = {row["ID"] for row in csv.DictReader(fin)}
    missing_languages = sorted(lang_set - cldf_langs)
    if missing_languages:
        raise ValueError(f"Forms reference missing languages: {missing_languages}")

    # check params
    for i in sorted(param_set):
        if i not in included_params:
            print(i)

    errors.close()


if __name__ == "__main__":
    main()
