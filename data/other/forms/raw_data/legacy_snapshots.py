"""Reproduce early hand-curated imports from pinned canonical CSV snapshots.

The nine sources registered here predate their extraction scripts.  Their
checked-in snapshots are therefore the immutable raw representation: this
importer verifies the pinned bytes, expands the legacy eight-column records to
the rich schema, assigns stable source-local keys from the frozen snapshot row,
and emits a per-record audit.  It does not claim that row numbers are printed
source locators; the source-specific checklists retain that legacy limitation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SNAPSHOT_DIR = ROOT / "data/other/forms/raw_data/legacy_snapshots"
AUDIT_DIR = ROOT / "source_checklists/audits"
RICH_COLUMNS = 15


@dataclass(frozen=True)
class Spec:
    filename: str
    sha256: str

    @property
    def unit_id(self) -> str:
        return Path(self.filename).stem

    @property
    def snapshot(self) -> Path:
        return SNAPSHOT_DIR / self.filename

    @property
    def installed(self) -> Path:
        return ROOT / "data/other/forms" / self.filename

    @property
    def audit(self) -> Path:
        return AUDIT_DIR / f"{self.unit_id}-snapshot-audit.csv"


SPECS = {
    spec.unit_id: spec
    for spec in (
        Spec("20220913-dhivehi.csv", "afa8f8bb10e2a7fa5f953544ae43de906a6f2a964f384a10275dbbc4a69cbb84"),
        Spec("20220913-khetrani.csv", "fcda1e1677e3e05396ccc03912c572f322fdc3b26d663d6c3501a2e1fb9c828d"),
        Spec("20220913-kholosi.csv", "bd778d155d3e82f9fbcb7b08fd717da543f1640fa70375ba6217ea5ad99d93d7"),
        Spec("20220913-konkani.csv", "f626055ddf5ae89cc59a78775babeeb307636a9507a62fc87e4c18a9f9c16e86"),
        Spec("20220913-kundalshahi.csv", "814abd00e97e796b370ff5286b190f47f964bbb7dc47560ea37291e713cfd517"),
        Spec("20220913-kvari.csv", "9dd40f0e91d883899ef01ad8c7e5d0f734dccc731513aaaa927fea10ae0d5a1c"),
        Spec("20220913-patyal.csv", "e1bc78830b1cb7a542e7decbe18a201104ef3aa424680074d5e55bdfb90aa1e0"),
        Spec("20220913-zadjali.csv", "a9cec49bf4e114c6242360a5dca856380fb4de894a5f59cc8c8d26aa2148034b"),
        Spec("20230524-sindhic.csv", "15b3ed313317cab08e8b0b16383e90ea21270fbf8c371a20c36e15477a8613e5"),
    )
}


def snapshot_rows(spec: Spec) -> list[list[str]]:
    payload = spec.snapshot.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != spec.sha256:
        raise ValueError(
            f"snapshot drift for {spec.unit_id}: expected {spec.sha256}, found {digest}"
        )
    with io.StringIO(payload.decode("utf-8"), newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or any(len(row) != 8 for row in rows):
        raise ValueError(f"{spec.unit_id}: expected nonempty eight-column snapshot")
    if any(not row[2].strip() for row in rows):
        raise ValueError(f"{spec.unit_id}: blank form in canonical snapshot")
    return rows


def entry_key(spec: Spec, row_number: int) -> str:
    return f"legacy:{spec.unit_id}:row:{row_number}"


def rich_rows(spec: Spec) -> list[list[str]]:
    return [
        [*row, "", "", entry_key(spec, row_number), "", "", "", ""]
        for row_number, row in enumerate(snapshot_rows(spec), 1)
    ]


def csv_bytes(rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode("utf-8")


def audit_bytes(spec: Spec) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "Status", "Reason", "Snapshot_Row", "Entry_Key", "Language_ID",
            "Parameter_ID", "Raw_Form", "Gloss", "Native", "Phonemic", "Notes", "Source",
        ]
    )
    for row_number, row in enumerate(snapshot_rows(spec), 1):
        writer.writerow(["ingested", "", row_number, entry_key(spec, row_number), *row])
    return stream.getvalue().encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def process(spec: Spec, *, install: bool, check: bool) -> None:
    installed = csv_bytes(rich_rows(spec))
    audit = audit_bytes(spec)
    if install:
        atomic_write(spec.installed, installed)
        atomic_write(spec.audit, audit)
    elif check:
        if spec.installed.read_bytes() != installed:
            raise ValueError(f"stale installed file for {spec.unit_id}; rerun with --install")
        if not spec.audit.exists() or spec.audit.read_bytes() != audit:
            raise ValueError(f"stale audit for {spec.unit_id}; rerun with --install")
    print(f"{spec.unit_id}: {len(rich_rows(spec))} rows")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("units", nargs="*", choices=sorted(SPECS))
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.install and args.check:
        parser.error("--install and --check are mutually exclusive")
    units = args.units or sorted(SPECS)
    for unit in units:
        process(SPECS[unit], install=args.install, check=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
