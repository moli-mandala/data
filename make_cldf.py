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
        self.source = row[7]
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
            self.source,
        ]
        return rows

    def __repr__(self):
        return f"<Row {self.lang} {self.param} {self.form} {self.gloss}>"


def parse_file(file: str, errors, name=None, file_num=0):
    stats = {
        "converted": 0,
        "for_conversion": 0
    }
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

        # split multiple forms into separate rows
        forms = list(row.form.split(",")) if "dedr" not in file else [row.form]
        for form in forms:
            reformed = form
            row.old_form = form
            row.form = form
            row.id = f"{file_num}-{i}"

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
    for file_num, file in enumerate(files):
        print(file)
        result, stats = parse_file(file, errors=errors, file_num=file_num)
        tot_stats["converted"] += stats["converted"]
        tot_stats["for_conversion"] += stats["for_conversion"]
        results.extend(result)
    
    print(tot_stats)

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
                "Source",
            ]
        )

        done = set()
        for row in results:
            if not row.form:
                continue
            if row.param == "?" or not row.param:
                continue
            if row.lang in change:
                row.lang = change[row.lang]
            row.lang = unidecode.unidecode(row.lang)
            row.lang = row.lang.replace(".", "")
            row.form = unicodedata.normalize("NFC", row.form)
            param_set.add(row.param)
            lang_set.add(row.lang)

            key = tuple(row.formatted[1:])
            if key not in done:
                forms.writerow(row.formatted)
            done.add(key)

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
                headword = headword.split()[0]
                reformed = ""
                if "˚" not in headword:
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
                        [row[0], row[2], row[1], row[3], etyma.get(row[0], "")]
                    )
                    included_params.add(row[0])

        with open("data/munda/params.csv", "r") as f:
            read = csv.reader(f)
            for row in read:
                row[2] = "PMu"
                params.writerow(row)
                included_params.add(row[0])

        with open("data/dedr/params.csv", "r") as f:
            read = csv.reader(f)
            for row in read:
                row[2] = "PDr"
                params.writerow(row)
                included_params.add(row[0])

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
