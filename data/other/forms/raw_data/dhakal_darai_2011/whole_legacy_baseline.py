"""Immutable pre-expansion inputs; never read live installed output as baseline."""
import csv
import hashlib
import io
from pathlib import Path
P=Path(__file__).resolve().parent
HASHES={'legacy-before-whole-recovery.csv': '89a59e3b08583b16c7b115f5e22422b0f798e7e7577f3059ffcbbf12f9080aa0', 'legacy-before-whole-recovery-profile.txt': 'c9ac1963856810b3bef56db905b5daac8d34bf66929382e3e0b91b0ba009116e'}


def checked_bytes(name):
    content=(P/name).read_bytes()
    assert hashlib.sha256(content).hexdigest()==HASHES[name], name
    return content


def legacy_rows():
    rows=list(csv.reader(io.StringIO(checked_bytes('legacy-before-whole-recovery.csv').decode())))
    assert len(rows)==485 and len({row[10] for row in rows})==485
    return rows


def legacy_profile():
    return checked_bytes('legacy-before-whole-recovery-profile.txt').decode()
