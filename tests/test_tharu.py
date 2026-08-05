import csv
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
BRACKETED_NUMBER = re.compile(r"^\(\d+\)$")
THARU_LANGUAGE_IDS = {
    "Rana",
    "Dang",
    "Chitwan",
    "Morang",
    "Tharu-BNM",
    "Tharu-BNT",
    "Tharu-RNK",
    "Tharu-RNS",
    "Tharu-RkM",
    "Tharu-RKB",
    "Tharu-TkN",
    "Tharu-KkP",
    "Tharu-SkP",
    "Tharu-DKS",
    "Tharu-DDK",
    "Tharu-DGC",
    "Tharu-DkR",
}


def test_webster_tharu_excludes_bracketed_reference_numbers():
    source = ROOT / "data/other/forms/20230530-tharu2.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert rows
    assert not any(BRACKETED_NUMBER.fullmatch(row[2]) for row in rows)


def test_compiled_tharu_excludes_bracketed_reference_numbers():
    source = ROOT / "cldf/forms.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        rows = (
            row
            for row in csv.DictReader(stream)
            if row["Language_ID"] in THARU_LANGUAGE_IDS
        )
        assert not any(BRACKETED_NUMBER.fullmatch(row["Form"]) for row in rows)
