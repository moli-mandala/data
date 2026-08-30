"""Install Beine's (1994) 46-site Gondi survey word lists from the Rama et al. digitization.

Upstream is https://github.com/PhyloStar/Gondi-Dialect-Analysis, release ``v1.0``
(commit ``f24fc743a1c72f1b69774450b790cfc4e346f800``, DOI ``10.5281/zenodo.1220088``), the
supplementary repository of Rama, Çöltekin & Sofroniev (2017).  Its ``data/`` folder holds a
complete 46 sites x 210 concepts matrix of IPA word lists that the authors digitized from
David K. Beine's 1994 San Diego State University master's thesis, plus Taraka Rama's manual
cognate-class judgments; ``maps/gondi.kml`` geolocates the 46 survey sites.

What is installed
-----------------
Only the lexical matrix.  Every attested cell becomes one unetymologised (``unlinked``) Jambu
form beneath the existing ``Gondi`` base language, tagged with a registered dialect for its
survey site.  The 158 cells printed as ``-----`` carry no elicited word and are recorded in the
audit as ``missing`` rather than installed.  A cell listing two or three comma-separated
responses becomes one row per response: Beine's alternates are frequently distinct lexemes
(``kʰarab`` beside ``beshile`` for 'bad'), not spelling variants of one another, so they are not
linked as variants.

What is deliberately *not* installed
------------------------------------
* Rama's ``Cognate Class`` column.  These are Gondi-internal cognate judgments scoped to a single
  concept, and representing them would mean minting ~700 headword-less Proto-Gondi grouping
  nodes.  Every class label is preserved per record in the audit so that decision can be taken
  separately; nothing here asserts an etymological edge the source does not print.
* The ASJP and SCA columns, which are LingPy-derived recodings of the same IPA rather than source
  transcriptions.  They are kept in the audit for reproduction.
* The Nexus matrices, MrBayes trees and analysis scripts, which are results rather than lexemes.

Transcription
-------------
``conversion/gondi-beine.txt`` maps Beine's IPA onto Jambu's Dravidianist house transcription.
The one linguistically consequential decision is that the source marks dentality only
sporadically -- ``t̪``/``d̪``/``n̪``/``s̪`` alternate freely with plain ``t``/``d``/``n``/``s``
within a single site and even within one cognate set (``wort̪itor`` beside ``wortitur`` for
'eat'), and Gondi has no third coronal series -- so both spellings render as the house dental
while ``Original`` and ``Phonemic`` keep the source's own diacritics.  ``w``/``ʋ``/``v`` likewise
collapse to ``v`` and ``j``/``y`` to ``y``; vowel qualities are not reinterpreted.

Usage
-----
    uv run python data/other/forms/raw_data/gondi_beine.py            # preview + audit only
    uv run python data/other/forms/raw_data/gondi_beine.py --install  # write the installed CSV
    uv run python data/other/forms/raw_data/gondi_beine.py --dialects # print cldf/dialects.csv rows
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from urllib.parse import quote

from segments.tokenizer import Tokenizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SNAPSHOT = ROOT / "tmp" / "gondi-dialect-analysis"

WORDLIST = "data/gondi_combined_cognates.csv"
PLAIN_WORDLIST = "data/gondi_combined.tsv"
KML = "maps/gondi.kml"

PROFILE = ROOT / "conversion" / "gondi-beine.txt"
INSTALLED = HERE.parent / "20260825-gondi-beine.csv"
PREVIEW = ROOT / "tmp" / "20260825-gondi-beine-preview.csv"
AUDIT = HERE / "20260825-gondi-beine-audit.csv"
MANIFEST = HERE / "20260825-gondi-beine-manifest.json"
SAMPLE = HERE / "20260825-gondi-beine-sample.csv"

SOURCE = "rama-coltekin-sofroniev2017gondi"
FIELDWORK = "beine1994gondi"
LANGUAGE = "Gondi"
CLADE = "S. Dravidian II"
DIALECT_PREFIX = "beine_"

UPSTREAM = {
    "repository": "https://github.com/PhyloStar/Gondi-Dialect-Analysis",
    "release": "v1.0",
    "commit": "f24fc743a1c72f1b69774450b790cfc4e346f800",
    "doi": "10.5281/zenodo.1220088",
    "concept_doi": "10.5281/zenodo.1220087",
    "license": "Other (Open) as declared on Zenodo; the repository ships no LICENSE file",
    "retrieved": "2026-08-25",
}

# sha256 of the three upstream files this importer reads.  They are byte-identical at the v1.0
# tag and at master (642425c, 2018-04-18); only README.md changed after the release.
CHECKSUMS = {
    WORDLIST: "738f3a0bb4045d9c256542176aa18e7a96ba4320faed0061f738827d73162ef7",
    PLAIN_WORDLIST: "db68fc7e878c3abe31806ab57a5c909ebcae501ece06d8370e0bd3a20a99edf1",
    KML: "8fac9b91a69c30e3f97148a6ca3e70a1941c44c4a66915e656107a64b362cbbb",
}

EXPECTED_SITES = 46
EXPECTED_CONCEPTS = 210
EXPECTED_CELLS = EXPECTED_SITES * EXPECTED_CONCEPTS
MISSING_MARKER = "-"

# Glottolog subgrouping of the 46 sites reproduced from Rama et al. (2017), table 1.
SUBGROUPS = {
    "Northwest Gondi > Northern Gondi": (
        "gdh gam gar gse glb gtd gkt gch prg gka gwa grp khu ggg gcj bhe pmd psh pkh ght"
    ),
    "Northwest Gondi > Southern Gondi": "rui gki gni dog gut gra lxg",
    "Southeast Gondi > General Southeast Gondi > Hill Maria-Koya > Hill Maria": (
        "met get mad gba goa mal gja gbh mbh"
    ),
    "Southeast Gondi > General Southeast Gondi > Muria": "mku mdh ktg mud mso mlj gok",
    "Southeast Gondi > General Southeast Gondi > Bison Horn Maria": "bhm bhb bhs",
}
SITE_SUBGROUP = {
    code: subgroup for subgroup, codes in SUBGROUPS.items() for code in codes.split()
}

# The upstream concept labels are ASCII identifiers.  Underscores become spaces; these entries
# spell out the elicitation prompts whose identifier is not readable English on its own.
CONCEPT_GLOSSES = {
    "evening_afternoon": "evening, afternoon",
    "he_was_hungry": "he was hungry",
    "he_was_thirsty": "he was thirsty",
    "we_excl": "we (exclusive)",
    "we_incl": "we (inclusive)",
    "what_kind": "what kind",
    "you_pl": "you (plural)",
    "you_sg_for": "you (singular, formal)",
    "you_sg_inf": "you (singular, informal)",
}

# Leading editorial qualifiers Rama et al. print before a site description in the KML.
QUALIFIER = re.compile(r"^\s*(\((?:\?|shifted)\))\s*-?\s*")
TRAILING_UNIT = re.compile(r"\s+(?:Tehsil|Taluk|Talku|city)$", re.IGNORECASE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_snapshot() -> None:
    """Fail loudly when the pinned upstream snapshot is absent or has drifted."""
    if not SNAPSHOT.is_dir():
        raise SystemExit(
            f"missing upstream snapshot {SNAPSHOT}; clone it with\n"
            f"  git clone {UPSTREAM['repository']} {SNAPSHOT}\n"
            f"  git -C {SNAPSHOT} checkout {UPSTREAM['release']}"
        )
    for name, expected in CHECKSUMS.items():
        path = SNAPSHOT / name
        if not path.is_file():
            raise SystemExit(f"missing upstream file {path}")
        actual = digest(path)
        if actual != expected:
            raise SystemExit(
                f"{path} does not match the pinned {UPSTREAM['release']} snapshot:\n"
                f"  expected {expected}\n  found    {actual}"
            )


def clean_text(value: str) -> str:
    """Collapse the KML description's markup and whitespace into one readable line."""
    return re.sub(r"\s+", " ", value.replace("<br>", " ")).strip()


def read_sites() -> dict[str, dict[str, str]]:
    """Return per-site metadata from the upstream KML placemarks."""
    kml = "{http://www.opengis.net/kml/2.2}"
    tree = ElementTree.parse(SNAPSHOT / KML)
    sites: dict[str, dict[str, str]] = {}
    for placemark in tree.iter(kml + "Placemark"):
        code = (placemark.findtext(kml + "name") or "").strip()
        description = clean_text(placemark.findtext(kml + "description") or "")
        coordinates = (placemark.findtext(f".//{kml}coordinates") or "").strip()
        parts = coordinates.split(",")
        if len(parts) != 3:
            # the polygon outlining the Gondi area, not a survey site
            continue
        longitude, latitude, _ = parts
        qualifier = ""
        match = QUALIFIER.match(description)
        if match:
            qualifier = match.group(1)
            description = description[match.end():].strip()
        variety, _, place = description.partition(" from ")
        locality = TRAILING_UNIT.sub("", place.split(",")[0].strip())
        label = locality if locality else code
        name = (
            f"{label} ({code})"
            if variety.strip() in {"", LANGUAGE}
            else f"{label} ({variety.strip()}, {code})"
        )
        location = f"{description}; Beine (1994) survey site {code}"
        if qualifier:
            location += f", printed {qualifier} by Rama et al."
        location += f"; Glottolog subgroup per Rama et al. (2017): {SITE_SUBGROUP[code]}"
        sites[code] = {
            "code": code,
            "name": name,
            "variety": variety.strip(),
            "locality": locality,
            "description": description,
            "qualifier": qualifier,
            "latitude": f"{float(latitude):.6f}",
            "longitude": f"{float(longitude):.6f}",
            "location": location,
        }
    if len(sites) != EXPECTED_SITES:
        raise ValueError(f"expected {EXPECTED_SITES} KML survey sites, found {len(sites)}")
    return sites


def read_cells() -> list[dict[str, str]]:
    """Return the upstream word-list matrix, checked against the copy without cognate classes."""
    with (SNAPSHOT / WORDLIST).open(encoding="utf-8", newline="") as stream:
        cells = list(csv.DictReader(stream, delimiter="\t"))
    with (SNAPSHOT / PLAIN_WORDLIST).open(encoding="utf-8", newline="") as stream:
        plain = {(r["Language"], r["Concept"]): r for r in csv.DictReader(stream, delimiter="\t")}

    if len(cells) != EXPECTED_CELLS:
        raise ValueError(f"expected {EXPECTED_CELLS} source cells, found {len(cells)}")
    if len({c["Language"] for c in cells}) != EXPECTED_SITES:
        raise ValueError("the source matrix does not cover exactly 46 sites")
    if len({c["Concept"] for c in cells}) != EXPECTED_CONCEPTS:
        raise ValueError("the source matrix does not cover exactly 210 concepts")
    for cell in cells:
        twin = plain.get((cell["Language"], cell["Concept"]))
        if twin is None or any(twin[column] != cell[column] for column in ("IPA", "ASJP", "SCA")):
            raise ValueError(f"{WORDLIST} and {PLAIN_WORDLIST} disagree on {cell}")
    return cells


def gloss_for(concept: str) -> str:
    return CONCEPT_GLOSSES.get(concept, concept.replace("_", " "))


def build() -> tuple[list[list[str]], list[dict[str, str]]]:
    """Return the installed rows and the per-record audit rows."""
    sites = read_sites()
    convert = Tokenizer(str(PROFILE))
    forms: list[list[str]] = []
    audit: list[dict[str, str]] = []

    for index, cell in enumerate(read_cells(), start=2):  # +2: 1-based, past the header row
        code = cell["Language"]
        concept = cell["Concept"]
        raw = cell["IPA"]
        site = sites[code]
        gloss = gloss_for(concept)
        citation = (
            f"{SOURCE}[{WORDLIST}, row {index}, site {code}, concept {concept}]"
            f"; {FIELDWORK}[site {code}, item {concept}]"
        )
        shared = {
            "Site_Code": code,
            "Language_ID": DIALECT_PREFIX + code,
            "Site_Name": site["name"],
            "Subgroup": SITE_SUBGROUP[code],
            "Concept": concept,
            "Gloss": gloss,
            "Source_Row": str(index),
            "Raw_IPA": raw,
            "Cognate_Class": cell["Cognate Class"],
            "ASJP": cell["ASJP"],
            "SCA": cell["SCA"],
            "Source": citation,
        }

        if raw.startswith(MISSING_MARKER):
            audit.append({
                **shared,
                "Status": "missing",
                "Reason": "the source prints '-----': no word was elicited at this site",
                "Entry_Key": "",
                "Part_Index": "",
                "Part_Count": "0",
                "Part_IPA": "",
                "House_Form": "",
            })
            continue

        parts = [unicodedata.normalize("NFC", p.strip()) for p in raw.split(",")]
        if not all(parts):
            raise ValueError(f"empty comma-separated response in row {index}: {raw!r}")
        for position, part in enumerate(parts, start=1):
            house = unicodedata.normalize(
                "NFC",
                convert(part, column="IPA").replace(" ", "").replace("#", " "),
            )
            if "�" in house:
                raise ValueError(f"conversion/gondi-beine.txt does not cover {part!r} (row {index})")
            key = f"beine:{code}:{concept}:{position}"
            forms.append([
                DIALECT_PREFIX + code,  # Language_ID (normalized to Gondi + a dialect tag)
                "",                     # Parameter_ID: the source asserts no etymology
                part,                   # Form (converted to house transcription by the profile)
                gloss,                  # Gloss
                "",                     # Native
                part,                   # Phonemic: Beine's own IPA
                "",                     # Notes
                citation,               # Source
                "",                     # Cognateset
                "",                     # Etymology
                key,                    # Entry_Key
                "",                     # Variant_Of_Key
                "",                     # Borrowed_From_Key
                "",                     # Derivation_Parent_Keys
                "",                     # Tags
            ])
            audit.append({
                **shared,
                "Status": "ingested",
                "Reason": (
                    "single response"
                    if len(parts) == 1
                    else f"response {position} of {len(parts)} listed in one cell"
                ),
                "Entry_Key": key,
                "Part_Index": str(position),
                "Part_Count": str(len(parts)),
                "Part_IPA": part,
                "House_Form": house,
            })

    return forms, audit


AUDIT_COLUMNS = [
    "Status", "Reason", "Entry_Key", "Site_Code", "Site_Name", "Language_ID", "Subgroup",
    "Concept", "Gloss", "Source_Row", "Part_Index", "Part_Count", "Raw_IPA", "Part_IPA",
    "House_Form", "Cognate_Class", "ASJP", "SCA", "Source",
]


def write_audit(audit: list[dict[str, str]]) -> None:
    with AUDIT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(audit)


def write_sample(audit: list[dict[str, str]]) -> None:
    """Write the seeded raw-versus-parsed sample used for the manual 20-record audit."""
    picks = random.Random(20260825).sample(audit, 20)
    picks.sort(key=lambda row: (int(row["Source_Row"]), row["Part_Index"] or "0"))
    with SAMPLE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(picks)


def write_manifest(forms: list[list[str]], audit: list[dict[str, str]]) -> None:
    statuses: dict[str, int] = {}
    for row in audit:
        statuses[row["Status"]] = statuses.get(row["Status"], 0) + 1
    classes = {row["Cognate_Class"] for row in audit if row["Status"] == "ingested"}
    manifest = {
        "source_key": SOURCE,
        "fieldwork_key": FIELDWORK,
        "upstream": UPSTREAM,
        "files": {name: {"sha256": CHECKSUMS[name]} for name in CHECKSUMS},
        "coverage": {
            "sites": EXPECTED_SITES,
            "concepts": EXPECTED_CONCEPTS,
            "source_cells": EXPECTED_CELLS,
            "audit_rows": len(audit),
            "installed_rows": len(forms),
            "statuses": statuses,
            "distinct_cognate_class_labels": len(classes),
        },
        "excluded": [
            "data/combined_data.tsv (the same IPA matrix pre-segmented for the PMI scorer)",
            "ASJP and SCA columns (LingPy recodings; retained in the audit)",
            "Cognate Class column (Gondi-internal judgments; retained in the audit)",
            "*.nex, mrbayes/, paper/, and the analysis scripts (results, not lexemes)",
        ],
        "installed_file": str(INSTALLED.relative_to(ROOT)),
        "audit_file": str(AUDIT.relative_to(ROOT)),
        "profile": str(PROFILE.relative_to(ROOT)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dialect_rows() -> list[list[str]]:
    """Return cldf/dialects.csv rows for the 46 survey sites."""
    rows = []
    for code, site in sorted(read_sites().items()):
        source_id = DIALECT_PREFIX + code
        name = site["name"]
        rows.append([
            source_id,
            f"dialect:{quote(LANGUAGE, safe='')}:{quote(source_id, safe='')}:{quote(name, safe='')}",
            LANGUAGE,
            source_id,
            name,
            "",  # Glottocode: Beine's sites are not Glottolog languoids
            site["latitude"],
            site["longitude"],
            CLADE,
            site["location"],
            "B",
        ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="write the installed source CSV")
    parser.add_argument("--dialects", action="store_true", help="print cldf/dialects.csv rows")
    args = parser.parse_args()

    require_snapshot()
    if args.dialects:
        writer = csv.writer(sys.stdout)
        writer.writerows(dialect_rows())
        return

    forms, audit = build()
    target = INSTALLED if args.install else PREVIEW
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(forms)
    write_audit(audit)
    write_sample(audit)
    write_manifest(forms, audit)

    ingested = sum(1 for row in audit if row["Status"] == "ingested")
    missing = sum(1 for row in audit if row["Status"] == "missing")
    print(
        f"Wrote {len(forms)} forms from {EXPECTED_CELLS} source cells "
        f"({ingested} ingested responses, {missing} cells with no elicited word) "
        f"across {EXPECTED_SITES} survey sites to {target}"
    )


if __name__ == "__main__":
    main()
