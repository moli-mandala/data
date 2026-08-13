"""Import the Hindu Kush Areal Typology CLDF wordlist into Jambu.

Source: https://github.com/cldf-datasets/liljegrenhindukush, release v1.1.0.
The release is archived at https://doi.org/10.5281/zenodo.21406260.

The upstream Wordlist has 11,600 forms for 194 concepts and 59 geographically
identified lects.  It contains no etymological assignments, so every imported
form is emitted as a source-keyed, unlinked Jambu node.  Upstream language IDs
are prefixed with ``HKAT-`` to retain separate elicitation lects (including
lects which share a Glottocode) without colliding with Jambu's existing IDs.

Run from the data repository, for example::

    uv run python data/other/forms/raw_data/liljegren_hindukush.py \
        /path/to/liljegrenhindukush/cldf

The generated form file uses Jambu's 15-column rich-import schema.  It imports
the canonical CLDF ``Form`` rather than ``Value``: some ``Value`` cells contain
comma-separated elicitation alternatives which the upstream dataset resolves
to one normalized lexeme in ``Form``.  Citations include the immutable upstream
FormTable and ParameterTable IDs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path


SOURCE_ID = "liljegren-hindukush"
LANGUAGE_PREFIX = "HKAT-"
DATA_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(DATA_ROOT))
from dialects import dialect_tag

# Jambu's display clades are more specific than the source's three-way
# Indo-Aryan/Iranian/Nuristani subgroup field.  These assignments follow the
# existing Jambu taxonomy for the same languages and lect groups.
INDO_ARYAN_CLADES = {
    "btv": "Kohistani",
    "bkk": "Shinaic",
    "kls": "Chitrali",
    "gwt_a": "Kunar",
    "gwt_p": "Kunar",
    "gwc": "Kohistani",
    "gju_a": "Rajasthanic",
    "gju_p": "Rajasthanic",
    "hno": "Lahndic",
    "mvy": "Kohistani",
    "xka": "Shinaic",
    "kas_i": "Kashmiric",
    "kas_p": "Kashmiric",
    "khw": "Chitrali",
    "plk": "Shinaic",
    "shd": "Shinaic",
    "phl": "Shinaic",
    "psh_ai": "Pashai",
    "psi_ar": "Pashai",
    "glh_ag": "Pashai",
    "psi_am": "Pashai",
    "aee_at": "Pashai",
    "aee_ch": "Pashai",
    "aee_kg": "Pashai",
    "glh_sn": "Pashai",
    "aee_sh": "Pashai",
    "phr": "Lahndic",
    "sdg": "Shinaic",
    "scl_p": "Shinaic",
    "scl_i": "Shinaic",
    "trw": "Kohistani",
    "ush": "Shinaic",
}

BASIC_POS = {
    # The source names these two items explicitly; the remaining assignments
    # are the ordinary lexical categories of the elicitation prompts.
    "full": "adj",
    "new": "adj",
    "come": "verb",
    "hear": "verb",
    "die": "verb",
    "drink (verb)": "verb",
    "see": "verb",
    "I": "pron",
    "we": "pron",
    "you": "pron",
    "one": "num",
    "two": "num",
}

# The browser DB treats ``Language: Dialect`` rows as located lects beneath a
# canonical language.  Use Jambu's established base-language spelling where a
# language is already present, and use the source elicitation locality (rather
# than a vague country abbreviation) as the dialect label.
LECT_NAMES = {
    "ask": "Ashkun: Titin",
    "bft": "Balti: Khaplu",
    "btv": "Bhateri: Palas",
    "bkk": "Brokskat: Batalik",
    "bsk_h": "Burushaski: Hunza",
    "bsk_n": "Burushaski: Nagar",
    "kls": "Indo-Aryan Kalasha: Bumburet (HKAT)",
    "dml": "Dameli: Damel",
    "prs_d": "Darwazi: Darwaz",
    "gwt_a": "Gawarbati: Naray (Afghanistan)",
    "gwt_p": "Gawarbati: Arandu (Pakistan)",
    "gwc": "Gawri: Kalam (HKAT)",
    "gju_a": "Gujari: Naray (Afghanistan)",
    "gju_p": "Gujari: Dawanan (Pakistan)",
    "hno": "Hindko: Mansehra (HKAT)",
    "mvy": "Indus Kohistani: Seo (HKAT)",
    "isk": "Ishkashimi: Ishkashim",
    "xka": "Kalkoti: Kalkot (HKAT)",
    "xvi": "Kamviri: Kamdesh",
    "kas_i": "Kashmiri: Srinagar (India)",
    "kas_p": "Kashmiri: Sharda (Pakistan)",
    "bsh_e": "Kati: Eastern (Bargi Matal)",
    "bsh_w": "Kati: Western (Du Ab)",
    "khw": "Khowar: Mastuj (HKAT)",
    "plk": "Shina: Palas (Kohistani)",
    "shd": "Kundal Shahi: Athmuqam",
    "kir": "Kyrgyz: Pamirkalan",
    "lbj": "Ladakhi: Leh",
    "mnj": "Munji: Sharan",
    "wbk": "Nuristani Kalasha: Muldesh",
    "phl": "Palula: Ashret (HKAT)",
    "prc": "Parachi: Ghochulan",
    "psh_ai": "Pashai: Alasai",
    "psi_ar": "Pashai: Alingar",
    "glh_ag": "Pashai: Alishang",
    "psi_am": "Pashai: Amla",
    "aee_at": "Pashai: Aret",
    "aee_ch": "Pashai: Chalas",
    "aee_kg": "Pashai: Korangal",
    "glh_sn": "Pashai: Sanjan",
    "aee_sh": "Pashai: Shemal",
    "pbu_a": "Pashto: Tagab (Afghanistan)",
    "pbu_i": "Pashto: Ganderbal (India)",
    "pbu_p": "Pashto: Swabi (Pakistan)",
    "phr": "Pahari-Pothwari: Rawalpindi",
    "prn": "Prasun: Pashki",
    "prx": "Purik: Kargil",
    "sgh_r": "Rushani: Roshan",
    "sgy": "Sanglechi: Sanglech",
    "sdg": "Sauji: Sau",
    "scl_p": "Shina: Gilgit (HKAT)",
    "scl_i": "Shina: Gurez (HKAT)",
    "sgh_a": "Shughni: Shughnan (Afghanistan)",
    "trw": "Torwali: Bahrain (HKAT)",
    "ush": "Ushojo: Chail (HKAT)",
    "uzs": "Uzbek: Argo",
    "wbl_a": "Wakhi: Abgach (Afghanistan)",
    "wbl_p": "Wakhi: Gojal (Pakistan)",
    "ydg": "Yidgha: Garam Chashma",
}


def language_id(upstream_id: str) -> str:
    return f"{LANGUAGE_PREFIX}{upstream_id}"


def lect_name(upstream_id: str) -> str:
    """Return the canonical ``Language: Dialect`` browser grouping name."""
    return LECT_NAMES[upstream_id]


def grammatical_tags(parameter: dict[str, str]) -> list[str]:
    """Derive canonical Jambu POS tags from the source concept list."""
    if parameter["domain"] == "Kinship":
        return ["noun"]
    if parameter["domain"] == "Numerals":
        return ["num"]
    return [BASIC_POS.get(parameter["Name"], "noun")]


def clade(language: dict[str, str]) -> str:
    """Map the upstream family/subgroup to Jambu's display taxonomy."""
    if language["Family"] == "Burushaski":
        return "Burushaski"
    if language["SubGroup"] == "Iranian" or language["Family"] in {"Sino-Tibetan", "Turkic"}:
        return "Other"
    if language["SubGroup"] == "Nuristani":
        # Jambu follows the established regional classification for Dameli.
        return "Kunar" if language["ID"] == "dml" else "Nuristani"
    if language["SubGroup"] == "Indo-Aryan":
        return INDO_ARYAN_CLADES[language["ID"]]
    raise ValueError(f"No Jambu clade for {language['ID']}: {language}")


def import_dialects(cldf_dir: Path) -> Iterable[list[str]]:
    """Yield rows for Jambu's explicit dialect registry."""
    with (DATA_ROOT / "cldf/languages.csv").open(encoding="utf-8", newline="") as stream:
        parent_ids = {row["Name"]: row["ID"] for row in csv.DictReader(stream)}
    with (cldf_dir / "languages.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source_id = language_id(row["ID"])
            base, dialect = lect_name(row["ID"]).split(": ", 1)
            parent_id = parent_ids[base]
            yield [
                source_id,
                dialect_tag(parent_id, source_id, dialect),
                parent_id,
                source_id,
                dialect,
                row["Glottocode"],
                row["Latitude"],
                row["Longitude"],
                clade(row),
                row["Location"],
                "A",  # directly elicited and geographically identified upstream
            ]


def import_rows(cldf_dir: Path) -> Iterable[list[str]]:
    """Yield complete rich-schema Jambu rows from the upstream Wordlist."""
    with (cldf_dir / "parameters.csv").open(encoding="utf-8", newline="") as stream:
        parameters = {row["ID"]: row for row in csv.DictReader(stream)}

    with (cldf_dir / "forms.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            parameter = parameters[row["Parameter_ID"]]
            source = (
                f"{SOURCE_ID}[form {row['ID']}, concept {row['Parameter_ID']}]"
            )
            yield [
                language_id(row["Language_ID"]),
                "",  # the source makes no etymological claim
                row["Form"],
                parameter["Name"],
                "",
                row["Form"],
                "",
                source,
                "",
                "",
                f"{SOURCE_ID}:{row['ID']}",
                "",
                "",
                "",
                " ".join(grammatical_tags(parameter)),
            ]


def write_csv(path: Path, rows: Iterable[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cldf_dir", type=Path, help="liljegrenhindukush CLDF directory")
    parser.add_argument(
        "--forms-output",
        type=Path,
        default=DATA_ROOT / "data/other/forms/20260810-liljegren-hindukush.csv",
    )
    parser.add_argument(
        "--dialects-output",
        type=Path,
        help="optional standalone dialect-registry output (without a header)",
    )
    args = parser.parse_args()

    rows = list(import_rows(args.cldf_dir))
    write_csv(args.forms_output, rows)
    print(f"Wrote {len(rows):,} forms to {args.forms_output}")
    if args.dialects_output:
        dialects = list(import_dialects(args.cldf_dir))
        write_csv(args.dialects_output, dialects)
        print(f"Wrote {len(dialects):,} dialect rows to {args.dialects_output}")


if __name__ == "__main__":
    main()
