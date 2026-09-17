#!/usr/bin/env python3
"""Per-source etymology assignment sidecars.

Manual etymology decisions are keyed by persistent form ID and stored *next to the source they
annotate*, one CSV per source file, all sharing the overlay schema
``Form_ID,Etymon_ID,Kind,Rank,Status,Source,Notes,Pos``:

* ``data/other/forms/etymologies/<source stem>.csv`` — decisions about attested forms from
  ``data/other/forms/<source stem>.csv``.  The sidecars live in a sub-directory because every
  ``*.csv`` directly in ``data/other/forms/`` is ingested as a lexical source.
* ``data/cdial/etymologies.csv`` and ``data/dedr/etymologies.csv`` — decisions whose child is a
  dictionary entry (CDIAL numbers, DEDR ``d…`` ids), e.g. the Proto-Indo-Iranian layer and
  restored CDIAL section-form derivations.
* ``data/other/forms/etymologies/_pending.csv`` — an inbox for tools that cannot resolve a form's
  source (the browser dev tool).  ``assign_form_ids.py`` files these rows into the right sidecar
  on every build; ``uv run python etymology_assignments.py sort`` does the same offline.

A row belongs to the sidecar of the source that owns its *child* (``Form_ID``).  The build reads
every sidecar and applies them together, so the split is purely organisational: the graph is
identical to the former single ``data/etymology-assignments.csv`` overlay.

Resolution of a form to its source file uses the identity registry
(``data/form-identities.csv``): the legacy ``<file>-<row>`` id gives the file through
``source_files.legacy_prefix_files``; keyed sources fall back to their ``Entry_Key``.

Run the CLI with ``uv run python etymology_assignments.py …``: resolving legacy ids reads the
per-source YAML settings, which need PyYAML from the project environment.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from source_files import FORMS_DIR, legacy_prefix_files

ROOT = Path(__file__).resolve().parent
FIELDS = ["Form_ID", "Etymon_ID", "Kind", "Rank", "Status", "Source", "Notes", "Pos"]

FORMS_SIDECAR_DIR = ROOT / FORMS_DIR / "etymologies"
PENDING = FORMS_SIDECAR_DIR / "_pending.csv"
CDIAL_SIDECAR = ROOT / "data/cdial/etymologies.csv"
DEDR_SIDECAR = ROOT / "data/dedr/etymologies.csv"
MUNDA_SIDECAR = ROOT / "data/munda/etymologies.csv"
DBIA_SIDECAR = ROOT / "data/dbia/etymologies.csv"
DICTIONARY_SIDECARS = (CDIAL_SIDECAR, DEDR_SIDECAR, MUNDA_SIDECAR, DBIA_SIDECAR)
REGISTRY = ROOT / "data/form-identities.csv"
LEGACY_OVERLAY = ROOT / "data/etymology-assignments.csv"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


class AssignmentRow(dict):
    """An overlay row that remembers which sidecar it was read from (``None`` = unfiled)."""

    __slots__ = ("path",)

    def __init__(self, *args, path: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path


# --------------------------------------------------------------------------- locating files


def sidecar_for_source(source: str | Path) -> Path:
    """Sidecar path for a lexical source given as a repo-relative path, file name or stem."""
    stem = Path(source).name
    if stem.endswith(".csv"):
        stem = stem[:-4]
    return FORMS_SIDECAR_DIR / f"{stem}.csv"


def source_for_sidecar(path: Path) -> Path | None:
    """The lexical source CSV a forms sidecar annotates (``None`` for dictionary/pending files)."""
    path = Path(path)
    if path.parent != FORMS_SIDECAR_DIR or path == PENDING:
        return None
    return ROOT / FORMS_DIR / path.name


def assignment_files() -> list[Path]:
    """Every existing sidecar, in a deterministic order (forms sidecars, pending, dictionaries)."""
    files = sorted(FORMS_SIDECAR_DIR.glob("*.csv")) if FORMS_SIDECAR_DIR.is_dir() else []
    files = [p for p in files if p != PENDING] + ([PENDING] if PENDING.exists() else [])
    files.extend(p for p in DICTIONARY_SIDECARS if p.exists())
    return files


# ------------------------------------------------------------------------------- read/write


def _read(path: Path) -> Iterator[AssignmentRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield AssignmentRow(row, path=path)


def read_assignments(paths: Iterable[Path] | None = None) -> list[AssignmentRow]:
    """All overlay rows across the sidecars (or the given files), file order preserved."""
    rows: list[AssignmentRow] = []
    for path in (assignment_files() if paths is None else paths):
        path = Path(path)
        if path.exists():
            rows.extend(_read(path))
    return rows


def write_file(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, extrasaction="ignore", restval="", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_assignments(
    rows: Iterable[dict], resolver: "SidecarResolver | None" = None
) -> dict[Path, int]:
    """Write the *complete* set of overlay rows back to their sidecars.

    Rows that remember a sidecar return to it.  Unfiled rows (plain dicts, or rows read from the
    pending inbox) are routed with ``resolver`` when one is given, otherwise they land in the
    inbox.  Sidecars that end up with no rows are removed.  Returns rows written per file.
    """
    grouped: dict[Path, list[dict]] = defaultdict(list)
    for row in rows:
        path = getattr(row, "path", None)
        if (path is None or path == PENDING) and resolver is not None:
            path = resolver.path_for(str(row.get("Form_ID", "")).strip()) or path
        grouped[path or PENDING].append(row)
    for stale in assignment_files():
        if stale not in grouped:
            stale.unlink()
    for path, group in grouped.items():
        write_file(path, group)
    return {path: len(group) for path, group in grouped.items()}


# --------------------------------------------------------------------------------- resolver


class SidecarResolver:
    """Map a child ID to the sidecar that owns it.

    Attested forms resolve through the identity registry (``Form_ID``, ``Legacy_ID``,
    ``Source_Key``, ``Source``, ``Original``, ``Gloss``); dictionary entries resolve by ID shape.
    Registry rows can be passed in (e.g. the registry ``assign_form_ids.py`` is about to write)
    or are loaded lazily from disk.

    Resolution order for an ``f_…`` form:

    1. legacy ``<file>-<row>`` id — the converter numbers CDIAL rows, rows without an etymon and
       rows under a numeric CDIAL etymon by input file, so a numeric (or appended-source stem)
       prefix names the file exactly;
    2. ``Source_Key`` — a ``legacy:<stem>:row:n`` key or a rich importer's immutable
       ``Entry_Key``, looked up in column 11 of the source CSVs;
    3. the citation key of ``Source`` — rows namespaced under a non-numeric etymon
       (``d1-2``, ``m1-1``, ``1907-2-1``) carry no file in their id, so find the source CSVs whose
       ``Source`` column cites that key, disambiguating several by the row's own form and gloss;
       a key cited by no survey CSV but by a dictionary input (``dedr``) resolves to that
       dictionary's sidecar.
    """

    _DICTIONARY_KEYS = {"dedr": DEDR_SIDECAR, "CDIAL": CDIAL_SIDECAR, "cdial": CDIAL_SIDECAR}

    def __init__(self, registry: Iterable[dict] | None = None, registry_path: Path = REGISTRY):
        self._registry_path = registry_path
        self._forms: dict[str, dict[str, str]] | None = None
        if registry is not None:
            self._index(registry)
        self._prefix_files: dict[str, str] | None = None
        self._csv_index: tuple[dict[str, str], dict[str, set[str]], set[tuple[str, str, str]]] | None = None

    _KEEP = ("Legacy_ID", "Source_Key", "Source", "Original", "Gloss")

    def _index(self, registry: Iterable[dict]) -> None:
        forms: dict[str, dict[str, str]] = {}
        for row in registry:
            form_id = row.get("Form_ID", "")
            if not form_id:
                continue
            # Prefer the active record; a retired tombstone still locates the source.
            if form_id not in forms or row.get("Status") == "active":
                forms[form_id] = {k: row.get(k, "") for k in self._KEEP}
        self._forms = forms

    @property
    def forms(self) -> dict[str, dict[str, str]]:
        if self._forms is None:
            with self._registry_path.open(encoding="utf-8", newline="") as handle:
                self._index(csv.DictReader(handle))
        assert self._forms is not None
        return self._forms

    @property
    def prefix_files(self) -> dict[str, str]:
        if self._prefix_files is None:
            self._prefix_files = legacy_prefix_files(ROOT)
        return self._prefix_files

    def _scan_sources(self) -> tuple[dict[str, str], dict[str, set[str]], set[tuple[str, str, str]]]:
        """One pass over the survey CSVs: Entry_Key → stem, citation key → stems, (stem, form, gloss)."""
        if self._csv_index is None:
            entry_keys: dict[str, str] = {}
            cited: dict[str, set[str]] = defaultdict(set)
            content: set[tuple[str, str, str]] = set()
            for path in sorted((ROOT / FORMS_DIR).glob("*.csv")):
                with path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.reader(handle):
                        if len(row) > 3:
                            content.add((path.stem, row[2].strip(), row[3].strip()))
                        if len(row) > 7:
                            for key in row[7].split(";"):
                                key = key.split("[", 1)[0].strip()
                                if key:
                                    cited[key].add(path.stem)
                        if len(row) > 10 and row[10].strip():
                            entry_keys[row[10].strip()] = path.stem
            self._csv_index = (entry_keys, dict(cited), content)
        return self._csv_index

    @property
    def entry_keys(self) -> dict[str, str]:
        return self._scan_sources()[0]

    @staticmethod
    def _dictionary_sidecar(source: str) -> Path | None:
        # data/<dictionary>/<file>.csv → data/<dictionary>/etymologies.csv
        parts = Path(source).parts
        if len(parts) == 3 and parts[0] == "data":
            return ROOT / parts[0] / parts[1] / "etymologies.csv"
        return None

    def _sidecar_for_input(self, source: str) -> Path | None:
        if not (ROOT / source).exists():
            return None
        if source.startswith(FORMS_DIR + "/"):
            return sidecar_for_source(source)
        return self._dictionary_sidecar(source)

    def path_for(self, child_id: str) -> Path | None:
        """Sidecar for ``child_id``; ``None`` when the form cannot be attributed to a source."""
        if not child_id.startswith("f_"):
            if re.match(r"\d", child_id):
                return CDIAL_SIDECAR
            if re.match(r"d\d", child_id):
                return DEDR_SIDECAR
            if re.match(r"m\d", child_id):
                return MUNDA_SIDECAR
            raise ValueError(f"cannot place etymology assignment for non-form child {child_id!r}")
        record = self.forms.get(child_id)
        if record is None:
            return None
        legacy_id, source_key = record["Legacy_ID"], record["Source_Key"]
        # 1. <file>-<row>
        if "-" in legacy_id:
            file = self.prefix_files.get(legacy_id.rsplit("-", 1)[0])
            if file:
                path = self._sidecar_for_input(file)
                if path:
                    return path
        # 2. immutable source record key
        if source_key.startswith("legacy:"):
            return sidecar_for_source(source_key.split(":")[1])
        if source_key:
            stem = self.entry_keys.get(source_key)
            if stem:
                return sidecar_for_source(stem)
        # 3. citation key of the form's Source, disambiguated by content
        keys = [k.split("[", 1)[0].strip() for k in record["Source"].split(";")]
        keys = [k for k in keys if k]
        _, cited, content = self._scan_sources()
        for key in keys:
            stems = cited.get(key, set())
            if len(stems) > 1:
                stems = {s for s in stems if (s, record["Original"], record["Gloss"]) in content}
            if len(stems) == 1:
                return sidecar_for_source(next(iter(stems)))
        for key in keys:
            if key in self._DICTIONARY_KEYS:
                return self._DICTIONARY_KEYS[key]
        return None


# ------------------------------------------------------------------------------------- CLI


def _misfiled(rows: list[AssignmentRow], resolver: SidecarResolver) -> list[tuple[AssignmentRow, Path | None]]:
    out = []
    for row in rows:
        target = resolver.path_for(row["Form_ID"].strip())
        if row.path == PENDING or (target is not None and target != row.path):
            out.append((row, target))
    return out


def cmd_check(_: argparse.Namespace) -> int:
    rows = read_assignments()
    resolver = SidecarResolver()
    misfiled = _misfiled(rows, resolver)
    per_file = Counter(str(r.path.relative_to(ROOT)) for r in rows)
    for path, count in sorted(per_file.items()):
        print(f"{count:7,d}  {path}")
    print(f"{len(rows):7,d}  total in {len(per_file)} sidecars")
    if misfiled:
        print(f"\n{len(misfiled)} row(s) are not in the sidecar of their source:")
        for row, target in misfiled[:25]:
            where = target.relative_to(ROOT) if target else "<unresolvable>"
            print(f"  {row['Form_ID']} -> {row['Etymon_ID']}: {row.path.relative_to(ROOT)} should be {where}")
        print("run `uv run python etymology_assignments.py sort` to file them")
        return 1
    return 0


def cmd_sort(_: argparse.Namespace) -> int:
    rows = read_assignments()
    resolver = SidecarResolver()
    moved = 0
    for row, target in _misfiled(rows, resolver):
        if target is not None:
            row.path = target
            moved += 1
    write_assignments(rows)
    unresolved = sum(1 for r in rows if r.path == PENDING)
    print(f"filed {moved} row(s); {unresolved} still pending")
    return 0


def cmd_locate(args: argparse.Namespace) -> int:
    resolver = SidecarResolver()
    for form_id in args.ids:
        path = resolver.path_for(form_id)
        print(f"{form_id}\t{path.relative_to(ROOT) if path else ''}")
    return 0


def cmd_import_legacy(args: argparse.Namespace) -> int:
    """One-time split of the former single overlay into per-source sidecars."""
    source = Path(args.file)
    rows = [AssignmentRow(r) for r in read_assignments([source])]
    resolver = SidecarResolver()
    unresolved = []
    for row in rows:
        row.path = resolver.path_for(row["Form_ID"].strip())
        if row.path is None:
            unresolved.append(row)
    if unresolved and not args.allow_pending:
        print(f"{len(unresolved)} row(s) cannot be attributed to a source; refusing to split:")
        for row in unresolved[:20]:
            print(f"  {row['Form_ID']} -> {row['Etymon_ID']}")
        print("re-run with --allow-pending to put them in the inbox")
        return 1
    existing = read_assignments()
    written = write_assignments(existing + rows)
    for path, count in sorted(written.items()):
        print(f"{count:7,d}  {path.relative_to(ROOT)}")
    print(f"{len(rows):7,d}  rows split into {len(written)} sidecars ({len(unresolved)} pending)")
    if args.remove:
        source.unlink()
        print(f"removed {source}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify every row sits in its source's sidecar").set_defaults(func=cmd_check)
    sub.add_parser("sort", help="file pending/misfiled rows into the right sidecars").set_defaults(func=cmd_sort)
    locate = sub.add_parser("locate", help="print the sidecar for one or more child IDs")
    locate.add_argument("ids", nargs="+")
    locate.set_defaults(func=cmd_locate)
    imp = sub.add_parser("import-legacy", help="split a single-file overlay into sidecars")
    imp.add_argument("file", nargs="?", default=str(LEGACY_OVERLAY))
    imp.add_argument("--allow-pending", action="store_true")
    imp.add_argument("--remove", action="store_true", help="delete the single file afterwards")
    imp.set_defaults(func=cmd_import_legacy)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
