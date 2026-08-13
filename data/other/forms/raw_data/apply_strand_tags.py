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

from strand import (
    DATA_ROOT,
    DIALECT_NAMES,
    clean_strand_text,
    legacy_locations,
    lang_mapping,
    parse_legacy_entry,
    strand_definition_tags,
    strand_pos_tags,
)


FORMS = DATA_ROOT / "data/other/forms"
CACHE = DATA_ROOT / ".cache"


def key(language, form, gloss):
    clean = lambda value: " ".join(unicodedata.normalize("NFC", value).casefold().split())
    return language, clean(form), clean(gloss)


def add(index, language, form, gloss, pos, location):
    gloss, definition_tags = strand_definition_tags(gloss)
    tags = strand_pos_tags(pos).split()
    tags.extend(definition_tags)
    dialect = DIALECT_NAMES.get(location, location)
    if dialect:
        from urllib.parse import quote
        tags.append("dialect:" + quote(dialect, safe=""))
    index[key(language, form, gloss)].add(" ".join(dict.fromkeys(tags)))


def source_index(include_strand3=True, include_legacy=True):
    index = defaultdict(set)

    # Etymological lexicon (Strand3): table rows carry a dialect code, form, POS, and quoted gloss.
    strand3_paths = (CACHE / "strand").glob("alph-*.html") if include_strand3 else ()
    for path in strand3_paths:
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
            form = clean_strand_text(form_node.get_text())
            text = clean_strand_text(entry.get_text())
            glosses = re.findall(r"‘(.*?)’", text)
            before_gloss = text.split(form, 1)[1].split("‘", 1)[0] if form in text else ""
            add(index, lang_mapping[location], form, glosses[0] if glosses else "", before_gloss, location)

    # The older regional lexica use one paragraph per entry: form, grammatical code, definition.
    legacy_paths = (CACHE / "strand-legacy").glob("*.html") if include_legacy else ()
    for path in legacy_paths:
        source_code = path.stem.split("-", 1)[0]
        if source_code == "bhatera":
            language = location = "bhatr"
        elif source_code in DIALECT_NAMES:
            language = {
                "Kata": "ktivi", "Ash": "sanu", "Wg": "nis",
            }.get(source_code, source_code)
            location = legacy_locations.get(source_code, source_code)
        else:
            continue
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        for entry in soup.find_all(class_="dic"):
            parsed = parse_legacy_entry(entry)
            if parsed:
                add(
                    index, language, parsed["word"], parsed["definition"],
                    parsed["pos"], location,
                )
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
        row[2] = clean_strand_text(row[2])
        row[3], definition_tags = strand_definition_tags(clean_strand_text(row[3]))
        row[6] = clean_strand_text(row[6])
        candidates = index.get(key(row[0], row[2], row[3]), set())
        exact_match = bool(candidates)
        if not candidates:
            candidates = by_form.get(key(row[0], row[2], "")[:2], set())
        if candidates:
            ordered = [candidate.split() for candidate in sorted(candidates)]
            shared = set(ordered[0]).intersection(*ordered[1:])
            parsed = [tag for tag in ordered[0] if tag in shared]
            if exact_match and not any(not tag.startswith("dialect:") for tag in parsed):
                # Some source rows are otherwise identical but carry complementary codes. Since
                # the checked-in CSV cannot distinguish those duplicates, retain the union rather
                # than dropping all grammatical information.
                parsed = list(dict.fromkeys(tag for tags in ordered for tag in tags))
        else:
            # Location is still certain even when a homophonous source form prevents POS matching.
            dialect = DIALECT_NAMES.get(row[0], row[0])
            from urllib.parse import quote
            parsed = ["dialect:" + quote(dialect, safe="")]
        # Dialect labels are derived here, so replace stale/older spellings instead of accumulating
        # two location tags when this annotation pass is rerun.
        parsed = list(dict.fromkeys(parsed + definition_tags))
        parsed_grammar = [tag for tag in parsed if not tag.startswith("dialect:")]
        existing = [] if parsed_grammar else [
            tag for tag in row[14].split() if not tag.startswith("dialect:")
        ]
        row[14] = " ".join(dict.fromkeys(existing + parsed))
        tagged += any(tag.startswith("dialect:") for tag in parsed)
        grammatical += any(not tag.startswith("dialect:") for tag in parsed)

    with open(path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file, lineterminator="\r\n").writerows(rows)
    print(f"{path.name}: {len(rows)} rows, {grammatical} grammatical, {tagged} dialect-tagged")


def main():
    legacy_index = source_index(include_strand3=False)
    strand3_index = source_index(include_legacy=False)
    for name in ("20220913-strand.csv", "20220913-strand2.csv"):
        apply(FORMS / name, legacy_index)
    apply(FORMS / "20221003-strand3.csv", strand3_index)


if __name__ == "__main__":
    main()
