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
        self.tags = ""  # structured tokens lifted from notes at write time (see extract_tags)
        self.source = row[7]
        self.variant_of = ""  # for a comma-listed alternate: the id of the first (main) form
        self.cognateset = '' if len(row) < 9 else row[8]
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
        ]
        return rows

    def __repr__(self):
        return f"<Row {self.lang} {self.param} {self.form} {self.gloss}>"


def parse_file(file: str, errors, name=None, file_num=0, param_counter=None):
    stats = {
        "converted": 0,
        "for_conversion": 0
    }
    is_cdial = "cdial" in file
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
        if row.lang == "Drav" or not row.param:
            continue
        if row.lang == "Indo-Aryan":
            row.form = row.form.lower()

        # param fix if .
        if "." in row.param:
            row.param = row.param.split(".")[0]

        # split multiple forms into separate rows; comma-listed alternates share one definition, so
        # the first is the main reflex and the rest are variants of it (same etymon, own alignment).
        forms = list(row.form.split(",")) if "dedr" not in file else [row.form]
        main_id = None
        for fj, form in enumerate(forms):
            reformed = form
            row.old_form = form
            row.form = form
            # Forms on a CDIAL-style numeric etymon (CDIAL itself, plus other-source additions that
            # hang reflexes on a CDIAL entry by its number) keep <file>-<row> ids, so the <etymon>-<n>
            # space stays free for promoted section forms. Every other source (Munda m1, Dravidian d1,
            # …) namespaces its reflexes under their etymon, e.g. m1-1, d1-2.
            epid = row.param.lstrip(">~")
            if is_cdial or re.fullmatch(r"\d+[a-z]?", epid):
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
            if ipa is not None and "˚" not in form and convert:
                stats["for_conversion"] += 1
                # fix accentuation from Strand
                if ipa == "strand":
                    reformed = reformed.replace("′", "´")
                    reformed = re.sub(r"([`´])(.)", r"\2\1", reformed)

                # do the conversion
                reformed = reformed.strip("-1234⁴5⁵67⁷,;.")
                reformed = convertors[ipa](reformed, column="IPA")
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

    # clean up duplicates in results
    cleaned = {}
    for i, row in enumerate(tqdm(results)):
        key = (row.lang, row.param, row.form)
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
            ]
        )

        done = set()
        for row in results:
            if row is None or not row.form:
                continue
            if row.param == "?" or not row.param:
                continue
            if row.lang in change:
                row.lang = change[row.lang]
            row.lang = unidecode.unidecode(row.lang)
            row.lang = row.lang.replace(".", "")
            row.form = unicodedata.normalize("NFC", row.form)
            param_set.add(row.param.lstrip(">~"))
            lang_set.add(row.lang)

            # lift structured tokens (gender, grammatical category) out of notes into Tags
            row.tags, row.notes = extract_tags(row.notes)

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
        params.writerow(["ID", "Name", "Language_ID", "Description", "Etyma"])

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
                        [row[0], row[2].split(",")[0].strip(), row[1], row[3], etyma.get(row[0], "")]
                    )
                    included_params.add(row[0])

        with open("data/munda/params.csv", "r") as f:
            read = csv.reader(f)
            for row in read:
                row[2] = "PMu"
                row[1] = row[1].split(",")[0].strip()  # main head-word = first of the listed forms
                params.writerow(row)
                included_params.add(row[0])

        with open("data/dedr/params.csv", "r") as f:
            read = csv.reader(f)
            for row in read:
                row[2] = "PDr"
                row[1] = row[1].split(",")[0].strip()  # main head-word = first of the listed forms
                params.writerow(row)
                included_params.add(row[0])

        with open("data/nuristani_cognates.csv", encoding="utf-8") as f:
            ancestor_ids = sorted({row["Ancestor_ID"] for row in csv.DictReader(f)})
        collisions = sorted(set(ancestor_ids) & included_params)
        if collisions:
            raise ValueError(f"Proto-Indo-Iranian ancestor ID collisions: {collisions}")
        for ancestor_id in ancestor_ids:
            params.writerow([ancestor_id, "", "Indo-ir", "", ""])
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
