"""Import Grierson's LSI comparative vocabulary CLDF dataset into Jambu.

Source: https://github.com/lexibank/lsi, release v1.0.
Archived release: https://doi.org/10.5281/zenodo.8361936.

LSI varieties are source lects, not new Jambu languages.  This importer maps
each defensibly identifiable variety to an existing Jambu parent language,
registers its ``LSI-*`` source ID in ``cldf/dialects.csv``, and emits forms
under that alias.  ``make_cldf.py`` then normalizes the alias to the parent and
adds the qualified dialect tag.  Comparative controls and varieties without a
defensible existing parent are recorded in an audit and are not imported.

The released CLDF supplies Glottolog-derived coordinates, not historical field
sites.  They are retained on dialect rows with an explicit provenance warning.
Grierson's normalized transcription is retained in ``Form`` and the upstream
CLTS segmentation (without separator spaces) in ``Phonemic``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path


SOURCE_ID = "grierson-lsi1928"
LANGUAGE_PREFIX = "LSI-"
DATA_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(DATA_ROOT))
from dialects import dialect_tag


# Overrides are limited to historical names, known dialect-to-language
# relations, and cases where modern Glottocodes split or merge differently
# from Jambu's existing language registry.  Distinct languages are not forced
# under a merely related or generic parent.
PARENT_OVERRIDES = {
    # Iranian and Nuristani
    "PASHTOOFPESHAWAR": "Psht",
    "PASHTOOFKANDAHAR": "Psht",
    "EASTERNBALOCHI": "Bal",
    "ISHKASHMIZEBAKI": "Ishk",
    "BASHGALI": "Bashg",
    # Pashai, Shina, and Kashmir-area lects
    "PASHAIEASTERN": "Pas",
    "PASHAIWESTERN": "Pas",
    "GAWARBATI": "Gaw",
    "SHINAGILGITI": "Sh",
    "CHILASI": "Sh",
    "CHILASIOFDRAS": "Sh",
    "CHILASIOFDAHHANU": "bro",
    "KASHTAWARI": "K",
    "DODASIRAJI": "dod",
    "KOHISTANIGARWI": "Bshk",
    "MAIYA": "Mai",
    # Lahnda, Punjabi, Sindhi, and adjacent varieties
    "MULTANI": "srk",
    "HINDKI": "awan",
    "THALI": "srk",
    "DHANNI": "L",
    "TINAULI": "L",
    "TINAULIOFSALTRANGE": "L",
    "POTHWARI": "poth",
    "CHIBHALI": "poth",
    "PUNCHHI": "poth",
    "LARI": "S",
    "PANJABIWRITTEN": "P",
    "PANJABISPOKEN": "P",
    "POWADHI": "P",
    # Central and eastern Indo-Aryan
    "NAGPURI": "M",
    "MAGAHI": "Bi",
    "BHOJPURINORTHERN": "Bhoj",
    "BHOJPURISOUTHERN": "Bhoj",
    "NAGPURIA": "Bi",
    "SIRIPURIA": "Bi",
    "EASTERNBENGALI": "B",
    "EASTERNBENGALIOFCACHAR": "B",
    "EASTERNBENGALIOFCHITTAGONG": "B",
    "WESTERNHINDIHINDOSTANI": "H",
    "VERNACULARHINDOSTANI": "H",
    "DAKHINI": "H",
    "BANAPHARI": "H",
    # Western Indo-Aryan
    "KATHIYAWADI": "G",
    "KHARAWA": "G",
    "GHISADI": "ghis",
    "RAJASTHANIMARWARI": "Marw",
    "NIMADI": "mewari_basad",
    "BAGHATI": "ba",
    "SHODOCHI": "sod",
    "GADI": "ga",
    "PANGWALI": "pan",
    "PADARI": "Pah",
    # Dravidian names whose current Glottocodes differ from LSI's mapping
    "KURUXORORAO": "Kurux",
    "MALTOORMALER": "Malto",
    "GONDI": "Gondi",
}


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


def language_id(upstream_id: str) -> str:
    return f"{LANGUAGE_PREFIX}{upstream_id}"


def source_label(language: dict[str, str]) -> str:
    """Return a readable historical variety label."""
    label = language["NameInSource"].strip()
    glottolog_name = language["Glottolog_Name"].strip()
    if not label:
        label = glottolog_name or language["Name"].title()
    elif label.casefold().startswith("of ") and glottolog_name:
        label = f"{glottolog_name} {label}"
    return label


def load_base_languages(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            row for row in csv.DictReader(stream)
            if not row["ID"].startswith(LANGUAGE_PREFIX)
        ]


def resolve_parents(
    cldf_dir: Path, registry: Path
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Resolve upstream IDs to existing parents and describe each decision."""
    bases = load_base_languages(registry)
    by_id = {row["ID"]: row for row in bases}
    by_name: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    by_glottocode: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in bases:
        for value in (row["ID"], row["Name"]):
            if normalized(value):
                by_name[normalized(value)][row["ID"]] = row
        if row["Glottocode"]:
            by_glottocode[row["Glottocode"]][row["ID"]] = row

    resolved: dict[str, dict[str, str]] = {}
    decisions: dict[str, str] = {}
    with (cldf_dir / "languages.csv").open(encoding="utf-8", newline="") as stream:
        for language in csv.DictReader(stream):
            upstream_id = language["ID"]
            override = PARENT_OVERRIDES.get(upstream_id)
            if override:
                if override not in by_id:
                    raise ValueError(f"Unknown LSI parent override {upstream_id} -> {override}")
                resolved[upstream_id] = by_id[override]
                decisions[upstream_id] = "manual historical/dialect mapping"
                continue

            # Exact historical or Glottolog names outrank Glottocode matches:
            # this avoids known upstream mergers such as Mewati -> Mewari.
            matched_by_name = False
            for value in (language["NameInSource"], language["Glottolog_Name"], language["Name"]):
                if value and not value.casefold().startswith("of "):
                    candidates = by_name.get(normalized(value), {})
                    if len(candidates) == 1:
                        resolved[upstream_id] = next(iter(candidates.values()))
                        decisions[upstream_id] = "unique existing-language name"
                        matched_by_name = True
                        break
            if matched_by_name:
                continue

            candidates = by_glottocode.get(language["Glottocode"], {})
            if language["Glottocode"] and len(candidates) == 1:
                resolved[upstream_id] = next(iter(candidates.values()))
                decisions[upstream_id] = "unique existing-language Glottocode"
                continue

            decisions[upstream_id] = (
                "ambiguous existing parent" if candidates else "no existing parent"
            )
    return resolved, decisions


def import_dialects(
    cldf_dir: Path, parents: dict[str, dict[str, str]]
) -> Iterable[list[str]]:
    """Yield explicit source-lect aliases beneath existing Jambu languages."""
    with (cldf_dir / "languages.csv").open(encoding="utf-8", newline="") as stream:
        for language in csv.DictReader(stream):
            parent = parents.get(language["ID"])
            if not parent:
                continue
            source_id = language_id(language["ID"])
            label = f"{source_label(language)} (LSI 1928)"
            yield [
                f"lsi_{language['ID'].lower()}",
                dialect_tag(parent["ID"], source_id, label),
                parent["ID"],
                source_id,
                label,
                language["Glottocode"],
                language["Latitude"],
                language["Longitude"],
                parent["Clade"],
                (
                    "Lexibank LSI v1.0 coordinate inherited from its Glottolog mapping; "
                    "not asserted as Grierson's historical survey locality"
                    if language["Latitude"] and language["Longitude"]
                    else "LSI v1.0 supplies no coordinate for this source variety"
                ),
                "B",
            ]


def phonemic(segments: str) -> str:
    return segments.replace(" ", "")


def import_rows(
    cldf_dir: Path, parents: dict[str, dict[str, str]]
) -> Iterable[list[str]]:
    """Yield rich-schema rows only for varieties with existing parents."""
    with (cldf_dir / "parameters.csv").open(encoding="utf-8", newline="") as stream:
        parameters = {row["ID"]: row for row in csv.DictReader(stream)}
    with (cldf_dir / "forms.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Language_ID"] not in parents:
                continue
            parameter = parameters[row["Parameter_ID"]]
            locator = (
                f"p. {parameter['PageNumber']}, form {row['ID']}, "
                f"concept {row['Parameter_ID']}"
            )
            yield [
                language_id(row["Language_ID"]),
                "",
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


def audit_rows(
    cldf_dir: Path,
    parents: dict[str, dict[str, str]],
    decisions: dict[str, str],
) -> Iterable[list[str]]:
    counts: Counter[str] = Counter()
    with (cldf_dir / "forms.csv").open(encoding="utf-8", newline="") as stream:
        counts.update(row["Language_ID"] for row in csv.DictReader(stream))
    yield [
        "Upstream_ID", "NameInSource", "Glottocode", "Form_Count",
        "Imported", "Parent_ID", "Parent_Name", "Decision",
    ]
    with (cldf_dir / "languages.csv").open(encoding="utf-8", newline="") as stream:
        for language in csv.DictReader(stream):
            parent = parents.get(language["ID"])
            yield [
                language["ID"],
                source_label(language),
                language["Glottocode"],
                counts[language["ID"]],
                "Yes" if parent else "No",
                parent["ID"] if parent else "",
                parent["Name"] if parent else "",
                decisions[language["ID"]],
            ]


def write_csv(path: Path, rows: Iterable[Sequence[object]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(materialized)
    return len(materialized)


def remove_legacy_languages(registry: Path) -> int:
    """Remove the obsolete first-pass LSI top-level language rows."""
    with registry.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    retained = [row for row in rows if not row[0].startswith(LANGUAGE_PREFIX)]
    with registry.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(retained)
    return len(rows) - len(retained)


def update_dialect_registry(registry: Path, rows: Iterable[Sequence[object]]) -> int:
    """Replace this import's dialect slice while preserving unrelated rows."""
    with registry.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        retained = [row for row in reader if not row[3].startswith(LANGUAGE_PREFIX)]
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
        "--forms-output", type=Path,
        default=DATA_ROOT / "data/other/forms/20260813-grierson-lsi.csv",
    )
    parser.add_argument(
        "--language-registry", type=Path,
        default=DATA_ROOT / "cldf/languages.csv",
    )
    parser.add_argument(
        "--dialect-registry", type=Path,
        default=DATA_ROOT / "cldf/dialects.csv",
    )
    parser.add_argument(
        "--audit-output", type=Path,
        default=DATA_ROOT / "data/other/forms/raw_data/20260813-grierson-lsi-audit.csv",
    )
    args = parser.parse_args()

    # Parent resolution must happen before deleting the obsolete first-pass rows.
    parents, decisions = resolve_parents(args.cldf_dir, args.language_registry)
    removed = remove_legacy_languages(args.language_registry)
    form_count = write_csv(args.forms_output, import_rows(args.cldf_dir, parents))
    dialect_count = update_dialect_registry(
        args.dialect_registry, import_dialects(args.cldf_dir, parents)
    )
    write_csv(args.audit_output, audit_rows(args.cldf_dir, parents, decisions))
    print(f"Removed {removed:,} obsolete LSI top-level language rows")
    print(f"Wrote {form_count:,} forms to {args.forms_output}")
    print(f"Wrote {dialect_count:,} dialect aliases to {args.dialect_registry}")
    print(f"Wrote mapping audit to {args.audit_output}")


if __name__ == "__main__":
    main()
