"""Ordered list of lexical source files consumed by ``make_cldf.py``.

The legacy pre-ID form identifier is ``<file index>-<row index>``, where the file index is the
position in :func:`ordered_source_files`.  Keeping the ordering rules here, separate from the
heavy ``make_cldf`` module, lets the etymology sidecar tooling resolve a form's source file
without importing the converter (which builds every tokenizer at import time).

Rules, preserved exactly from the converter:

* the four dictionary inputs come first, then every ``data/other/forms/*.csv`` except the
  Merriam Dravidian database and the appended survey files, all sorted together;
* Merriam and DBIA are appended after the sort so their addition could not renumber older
  sources;
* the appended survey files follow in declaration order and receive their file *stem* rather
  than a numeric index as their legacy prefix.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

STRAND3_FILE = "20221003-strand3.csv"
LEGACY_STRAND_FILES = {"20220913-strand.csv", "20220913-strand2.csv"}
MERRIAM_DRAVIDIAN_DB_FILE = "data/other/forms/20260718-merriam-dravidian-db.csv"
WESTERN_SURVEY_FILES = (
    "data/other/forms/20260911-dadra-varli.csv",
    "data/other/forms/20260911-ghatage-konkani.csv",
    "data/other/forms/20260911-ghatage-kudali.csv",
    "data/other/forms/20260911-sdml.csv",
    "data/other/forms/20260911-bajjika.csv",
    "data/other/forms/20260911-lindgren.csv",
    "data/other/forms/20260911-dravlex.csv",
    "data/other/forms/20260911-census-tamil-nadu.csv",
    "data/other/forms/20260911-census-uttar-pradesh.csv",
    "data/other/forms/20260911-census-bihar.csv",
    "data/other/forms/20260911-census-sikkim2.csv",
    "data/other/forms/20260911-census-danuwar.csv",
    "data/other/forms/20260911-census-tharu.csv",
    "data/other/forms/20260911-more-jharkhand.csv",
    "data/other/forms/20260911-more-himachal.csv",
    "data/other/forms/20260911-more-rajasthan.csv",
    "data/other/forms/20260911-more-west-bengal.csv",
    "data/other/forms/20260911-more-kisan.csv",
    "data/other/forms/20260911-selected-angika.csv",
    "data/other/forms/20260911-selected-majhi.csv",
    "data/other/forms/20260911-selected-koraga.csv",
    "data/other/forms/20260911-selected-orissa.csv",
    "data/other/forms/20260912-keed.csv",
    "data/other/forms/20260912-muduga.csv",
    "data/other/forms/20260913-zoller-linguistic-data.csv",
)
# Append new keyed sources after every existing input to preserve legacy aliases.
MANUAL_SURVEY_FILES = (
    "data/other/forms/20260914-sil-ho.csv",
    "data/other/forms/20260914-sil-bhumij.csv",
    "data/other/forms/20260914-sil-dhurwa.csv",
)
SHETH_FILE = "data/other/forms/20260914-sheth.csv"
SHETH_SANSKRIT_FILE = "data/other/forms/20260914-sheth-sanskrit.csv"
# The tuples above are historical groupings kept for tests and audits. Which sources are appended
# (and in what order) is declared per source in its YAML: ``defaults.identity.legacy_ids: stem``
# with ``append_order``; see source_meta.py and appended_source_files().


def appended_source_files() -> list[str]:
    """Sources whose legacy ids are ``<stem>-<row>``, in their declared processing order."""
    import source_meta  # lazy: needs PyYAML, which the plain-python sidecar CLI may lack

    items = []
    for stem, block in source_meta.load().files.items():
        identity = block.get("identity", {})
        if identity.get("legacy_ids") == "stem":
            items.append((identity.get("append_order", 1_000_000), stem, f"{FORMS_DIR}/{stem}.csv"))
    return [file for _, _, file in sorted(items)]


def __getattr__(name: str):
    if name == "APPENDED_SURVEY_FILES":
        return tuple(appended_source_files())
    raise AttributeError(name)

DICTIONARY_FILES = [
    "data/cdial/cdial.csv",
    "data/munda/forms.csv",
    "data/dedr/dedr_new.csv",
    "data/dedr/pdr.csv",
]

FORMS_DIR = "data/other/forms"


def ordered_source_files(root: str | os.PathLike[str] | None = None) -> list[str]:
    """Return the converter's input files in processing order, as repo-relative paths."""
    base = Path(root) if root is not None else Path.cwd()
    appended = appended_source_files()
    pattern = str(base / FORMS_DIR / "*.csv")
    files = DICTIONARY_FILES + [
        os.path.relpath(path, base) for path in glob.glob(pattern)
        if os.path.relpath(path, base) != MERRIAM_DRAVIDIAN_DB_FILE
        and os.path.relpath(path, base) not in appended
    ]
    files.sort()
    files.append(MERRIAM_DRAVIDIAN_DB_FILE)
    files.append("data/dbia/forms.csv")
    files.extend(path for path in appended if (base / path).exists())
    return files


def legacy_prefix(file: str, file_num: int) -> str:
    """The legacy ID prefix the converter gives rows of ``file`` at position ``file_num``."""
    if file in appended_source_files():
        return os.path.splitext(os.path.basename(file))[0]
    return str(file_num)


def legacy_prefix_files(root: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Map every legacy ID prefix to the repo-relative source file it numbers."""
    return {
        legacy_prefix(file, file_num): file
        for file_num, file in enumerate(ordered_source_files(root))
    }
