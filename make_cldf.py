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
from tags import extract_tags
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
        self.tags = "" if len(row) < 15 else row[14]
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


SCHMIDT_VOWELS = "aeiouəæãẽõũ"
SCHMIDT_PROFILE_LANGUAGES = {"K", "kash", "pog", "sir"}


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
        # Both hand-entered and OCR-derived Shackle rows use the same CDIAL-style
        # romanisation. The auto filename does not reduce to ``old_punjabi`` via
        # the legacy filename heuristic, so select its phonetic parser by source.
        row_ipa = "cdial" if row.source in {"shackle", "shackle-auto"} else ipa
        row_convert = row_ipa is not None and (row.source in {"shackle", "shackle-auto"} or convert)
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
        forms = (
            [
                normalize_dedr_marks(length_variant)
                for base in expand_attached_sound_variants(row.form)
                for length_variant in expand_length_variants(base)
            ]
            if "dedr" in file
            else list(row.form.split(","))
        )
        main_id = None
        for fj, form in enumerate(forms):
            reformed = form
            row.old_form = source_form if "dedr" in file and len(forms) > 1 else form
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
            if row_ipa == "zoller" and row_convert:
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
            elif row_ipa is not None and "˚" not in form and row_convert:
                stats["for_conversion"] += 1
                # fix accentuation from Strand
                if row_ipa == "strand":
                    reformed = reformed.replace("′", "´")
                    reformed = re.sub(r"([`´])(.)", r"\2\1", reformed)
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

    form_count = 0
    results: list[Row] = []
    files = [
        "data/cdial/cdial.csv",
        "data/munda/forms.csv",
        "data/dedr/dedr_new.csv",
        "data/dedr/pdr.csv",
    ] + glob.glob("data/other/forms/*.csv")
    files.sort()

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
            # Gandhari.org has genuine homographic articles, including senses with the same
            # Sanskrit etymon and English gloss.  Its stable article key keeps those dictionary
            # entries distinct while retaining the legacy dedupe behaviour for other sources.
            row.entry_key if row.source == "gandhari" else "",
            row.gloss
            if not row.param and (
                row.lang.startswith("SSNP-")
                or row.notes.startswith("SSNP ")
                or row.source == "andersen1990"
            )
            else "",
        )
        if key not in cleaned:
            cleaned[key] = (row, i)
        else:
            orig_row = cleaned[key][0]
            if row.cognateset is None or row.cognateset == "":
                orig_row.gloss = '; '.join([x for x in set([orig_row.gloss, row.gloss]) if x])
                orig_row.native = '; '.join([x for x in set([orig_row.native, row.native]) if x])
                orig_row.notes = '; '.join([x for x in set([orig_row.notes, row.notes]) if x])
                orig_row.source = ';'.join([x for x in set([orig_row.source, row.source]) if x])
                orig_row.ipa = '; '.join([x for x in set([orig_row.ipa, row.ipa]) if x])
                orig_row.old_form = '; '.join([x for x in set([orig_row.old_form, row.old_form]) if x])
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
            lang_set.add(row.lang)

            # lift structured tokens (gender, grammatical category) out of notes into Tags
            parsed_tags, row.notes = extract_tags(row.notes)
            row.tags = " ".join(
                dict.fromkeys(filter(None, row.tags.split() + parsed_tags.split()))
            )

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
        params.writerow(["ID", "Name", "Language_ID", "Description", "Etyma", "Etymology"])

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
                    ]
                )
                included_params.add(row[0])

        for file in tqdm(glob.glob("data/other/params/*.csv")):
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
                        [row[0], row[2].split(",")[0].strip(), row[1], row[3], etyma.get(row[0], ""), ""]
                    )
                    included_params.add(row[0])

        with open("data/munda/params.csv", "r") as f:
            read = csv.reader(f)
            for row in read:
                row[2] = "PMu"
                row[1] = row[1].split(",")[0].strip()  # main head-word = first of the listed forms
                params.writerow(row + [""])
                included_params.add(row[0])

        with open("data/dedr/footer_notes.csv", "r") as f:
            dedr_footer_notes = dict(csv.reader(f))

        with open("data/dedr/params.csv", "r") as f:
            read = csv.reader(f)
            for row in read:
                row[2] = "PDr"
                row[1] = row[1].split(",")[0].strip()  # main head-word = first of the listed forms
                params.writerow(row[:5] + [dedr_footer_notes.get(row[0], "")])
                included_params.add(row[0])

        with open("data/nuristani_cognates.csv", encoding="utf-8") as f:
            ancestor_ids = sorted({row["Ancestor_ID"] for row in csv.DictReader(f)})
        collisions = sorted(set(ancestor_ids) & included_params)
        if collisions:
            raise ValueError(f"Proto-Indo-Iranian ancestor ID collisions: {collisions}")
        for ancestor_id in ancestor_ids:
            params.writerow([ancestor_id, "", "Indo-ir", "", "", ""])
            included_params.add(ancestor_id)

    # ensure that all languages in forms.csv are also in languages.csv
    cldf_langs = set()
    with open("cldf/languages.csv", "r") as fin:
        for row in fin.readlines():
            x = row.split(",")[0]
            cldf_langs.add(x)

    for i in sorted(lang_set):
        if i not in cldf_langs:
            print(i)

    # check params
    for i in sorted(param_set):
        if i not in included_params:
            print(i)

    errors.close()


if __name__ == "__main__":
    main()
