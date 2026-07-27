"""Add Strand grammatical and dialect tags without changing stable row/etymon identities.

The live Strand3 head sequence is not a stable identifier source: regenerating it can renumber the
curated ``nN`` etyma.  This pass therefore matches cached source entries back onto the three checked-
in CSVs by language, source spelling, and gloss, and changes only the structured Tags column.
"""

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

from strand import DATA_ROOT, DIALECT_NAMES, lang_mapping, strand_pos_tags


FORMS = DATA_ROOT / "data/other/forms"
CACHE = DATA_ROOT / ".cache"


def key(language, form, gloss):
    clean = lambda value: " ".join(unicodedata.normalize("NFC", value).casefold().split())
    return language, clean(form), clean(gloss)


def add(index, language, form, gloss, pos, location):
    tags = strand_pos_tags(pos).split()
    dialect = DIALECT_NAMES.get(location, location)
    if dialect:
        from urllib.parse import quote
        tags.append("dialect:" + quote(dialect, safe=""))
    index[key(language, form, gloss)].add(" ".join(dict.fromkeys(tags)))


def source_index():
    index = defaultdict(set)

    # Etymological lexicon (Strand3): table rows carry a dialect code, form, POS, and quoted gloss.
    for path in (CACHE / "strand").glob("alph-*.html"):
        soup = BeautifulSoup(path.read_bytes(), "html5lib")
        for row in soup.find_all("tr"):
            marker = row.find(class_="lng")
            if not marker:
                continue
            cells = row.find_all("td")
            entry = cells[-1]
            form_node = entry.find("em")
            if not form_node:
                continue
            try:
                language, dialect, _source = marker.find("em").get_text().split(".")
            except (AttributeError, ValueError):
                continue
            location = f"{language}.{dialect}"
            if location not in lang_mapping:
                continue
            form = form_node.get_text()
            text = entry.get_text().replace("\n", "")
            glosses = re.findall(r"‘(.*?)’", text)
            before_gloss = text.split(form, 1)[1].split("‘", 1)[0] if form in text else ""
            add(index, lang_mapping[location], form, glosses[0] if glosses else "", before_gloss, location)

    # The older regional lexica use one paragraph per entry: form, grammatical code, definition.
    for path in (CACHE / "strand-legacy").glob("*.html"):
        source_code = path.stem.split("-", 1)[0]
        if source_code == "bhatera":
            language = location = "bhatr"
        elif source_code in DIALECT_NAMES:
            language = location = source_code
        else:
            continue
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        for entry in soup.find_all(class_="dic"):
            form_node = entry.find(class_="l")
            if not form_node:
                continue
            form = form_node.find(string=True, recursive=False) or form_node.get_text()
            html = str(entry).replace("\n", " ")
            match = re.search(r"<b>]</b>\xa0 (.*?)\.\xa0 (.*?)\.", html, re.I)
            if not match:
                match = re.search(r"</span>[\xa0 ]+(.*?)\.\xa0\xa0([^\.]+)\.", html, re.I)
            if match:
                add(index, language, form, match.group(2).lower(), match.group(1), location)
    return index


def apply(path, index):
    with open(path, encoding="utf-8") as file:
        rows = list(csv.reader(file))

    by_form = defaultdict(set)
    for (language, form, _gloss), values in index.items():
        by_form[(language, form)].update(values)

    tagged = grammatical = 0
    for row in rows:
        while len(row) < 15:
            row.append("")
        candidates = index.get(key(row[0], row[2], row[3]), set())
        if len(candidates) != 1:
            candidates = by_form.get(key(row[0], row[2], "")[:2], set())
        if len(candidates) != 1:
            # Location is still certain even when a homophonous source form prevents POS matching.
            dialect = DIALECT_NAMES.get(row[0], row[0])
            from urllib.parse import quote
            candidates = {"dialect:" + quote(dialect, safe="")}
        parsed = next(iter(candidates)).split()
        # Dialect labels are derived here, so replace stale/older spellings instead of accumulating
        # two location tags when this annotation pass is rerun.
        existing = [tag for tag in row[14].split() if not tag.startswith("dialect:")]
        row[14] = " ".join(dict.fromkeys(existing + parsed))
        tagged += any(tag.startswith("dialect:") for tag in parsed)
        grammatical += any(not tag.startswith("dialect:") for tag in parsed)

    with open(path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)
    print(f"{path.name}: {len(rows)} rows, {grammatical} grammatical, {tagged} dialect-tagged")


def main():
    index = source_index()
    for name in ("20220913-strand.csv", "20220913-strand2.csv", "20221003-strand3.csv"):
        apply(FORMS / name, index)


if __name__ == "__main__":
    main()
