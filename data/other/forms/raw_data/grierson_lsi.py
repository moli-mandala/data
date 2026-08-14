"""Import Grierson's LSI comparative vocabulary CLDF dataset into Jambu.

Source: https://github.com/lexibank/lsi, release v1.0.
Archived release: https://doi.org/10.5281/zenodo.8361936.

The upstream Wordlist contains 60,533 forms for 168 concepts and 363 source
varieties.  It makes no etymological claims, so every form is emitted as an
unlinked, source-keyed Jambu node.  Source variety IDs are prefixed with
``LSI-``: the printed comparative tables distinguish historical survey lects
which may now share a Glottocode, and collapsing those rows would lose data.

Run from the data repository, for example::

    uv run python data/other/forms/raw_data/grierson_lsi.py /path/to/lsi/cldf

The importer reads the released CLDF rather than the project corpus XML.  It
retains Grierson's normalized transcription in ``Form`` and the upstream CLTS
segmentation (without separator spaces) in ``Phonemic``.  Citation locators
carry the printed page range and immutable upstream form and concept IDs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path


SOURCE_ID = "grierson-lsi1928"
LANGUAGE_PREFIX = "LSI-"
DATA_ROOT = Path(__file__).resolve().parents[4]

# Glottocode matches inherit Jambu's existing, more specific display clade.
# These are the five Dravidian varieties not otherwise represented in the
# current language registry.  Everything else unmatched is a comparative
# control or belongs to a family for which Jambu uses the catch-all "Other".
DRAVIDIAN_CLADES = {
    "KORAVA": "S. Dravidian I",
    "KAIKADI": "S. Dravidian I",
    "KURUXORORAO": "N. Dravidian",
    "MALTOORMALER": "N. Dravidian",
    "GONDI": "C. Dravidian",
}


def language_id(upstream_id: str) -> str:
    return f"{LANGUAGE_PREFIX}{upstream_id}"


def source_label(language: dict[str, str]) -> str:
    """Return a readable, explicitly source-specific variety label."""
    label = language["NameInSource"].strip()
    glottolog_name = language["Glottolog_Name"].strip()
    if not label:
        label = glottolog_name or language["Name"].title()
    elif label.casefold().startswith("of ") and glottolog_name:
        label = f"{glottolog_name} {label}"
    return f"LSI — {label}"


def existing_clades(path: Path) -> dict[str, str]:
    """Load only unambiguous Glottocode-to-clade mappings."""
    by_glottocode: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Glottocode"] and row["Clade"]:
                by_glottocode[row["Glottocode"]].add(row["Clade"])
    return {
        glottocode: next(iter(clades))
        for glottocode, clades in by_glottocode.items()
        if len(clades) == 1
    }


def clade(language: dict[str, str], by_glottocode: dict[str, str]) -> str:
    """Map an LSI variety into Jambu's intentionally coarse display taxonomy."""
    if language["Glottocode"] in by_glottocode:
        return by_glottocode[language["Glottocode"]]
    if language["SubGroup"] == "Munda":
        return "Munda"
    if language["Family"] == "Burushaski":
        return "Burushaski"
    if language["ID"] in DRAVIDIAN_CLADES:
        return DRAVIDIAN_CLADES[language["ID"]]
    return "Other"


def import_languages(cldf_dir: Path, registry: Path) -> Iterable[list[str]]:
    """Yield Jambu language rows for all distinct historical LSI varieties."""
    clades = existing_clades(registry)
    with (cldf_dir / "languages.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            yield [
                language_id(row["ID"]),
                source_label(row),
                row["Glottocode"],
                row["Latitude"],
                row["Longitude"],
                clade(row, clades),
                (
                    "Lexibank LSI v1.0 coordinate inherited from its Glottolog mapping; "
                    "not asserted as Grierson's historical survey locality"
                    if row["Latitude"] and row["Longitude"]
                    else ""
                ),
                "B",  # curated retro-digitization of a printed historical survey
            ]


def phonemic(segments: str) -> str:
    """Turn CLDF's space-separated segment list into a display transcription."""
    return segments.replace(" ", "")


def import_rows(cldf_dir: Path) -> Iterable[list[str]]:
    """Yield complete rich-schema Jambu rows from the upstream Wordlist."""
    with (cldf_dir / "parameters.csv").open(encoding="utf-8", newline="") as stream:
        parameters = {row["ID"]: row for row in csv.DictReader(stream)}

    with (cldf_dir / "forms.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            parameter = parameters[row["Parameter_ID"]]
            locator = (
                f"p. {parameter['PageNumber']}, form {row['ID']}, "
                f"concept {row['Parameter_ID']}"
            )
            yield [
                language_id(row["Language_ID"]),
                "",  # LSI's comparative table supplies no etymological analysis
                row["Form"],
                parameter["Name"],
                "",
                phonemic(row["Segments"]),
                row["Comment"],
                f"{SOURCE_ID}[{locator}]",
                "",
                "",
                f"{SOURCE_ID}:{row['ID']}",
                "",
                "",
                "",
                "",
            ]


def write_csv(path: Path, rows: Iterable[Sequence[object]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(materialized)
    return len(materialized)


def update_language_registry(registry: Path, rows: Iterable[Sequence[object]]) -> int:
    """Replace this import's registry slice while preserving all unrelated rows."""
    with registry.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        retained = [row for row in reader if not row[0].startswith(LANGUAGE_PREFIX)]
    imported = list(rows)
    with registry.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(retained)
        writer.writerows(imported)
    return len(imported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cldf_dir", type=Path, help="Lexibank LSI CLDF directory")
    parser.add_argument(
        "--forms-output",
        type=Path,
        default=DATA_ROOT / "data/other/forms/20260813-grierson-lsi.csv",
    )
    parser.add_argument(
        "--language-registry",
        type=Path,
        default=DATA_ROOT / "cldf/languages.csv",
    )
    parser.add_argument(
        "--languages-output",
        type=Path,
        help="write generated language rows separately instead of updating the registry",
    )
    args = parser.parse_args()

    form_count = write_csv(args.forms_output, import_rows(args.cldf_dir))
    languages = import_languages(args.cldf_dir, args.language_registry)
    if args.languages_output:
        language_count = write_csv(args.languages_output, languages)
        destination = args.languages_output
    else:
        language_count = update_language_registry(args.language_registry, languages)
        destination = args.language_registry
    print(f"Wrote {form_count:,} forms to {args.forms_output}")
    print(f"Wrote {language_count:,} language rows to {destination}")


if __name__ == "__main__":
    main()
