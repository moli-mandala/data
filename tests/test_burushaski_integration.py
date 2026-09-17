"""Guard source records and grammatical labels through the complete CLDF build."""
import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "20220930-berger.csv",
    "20260726-berger-auto.csv",
    "20260726-yoshioka-eastern-burushaski.csv",
)


@pytest.fixture(scope="module")
def compiled():
    raw = {}
    for filename in SOURCES:
        with (ROOT / "data/other/forms" / filename).open(newline="") as stream:
            raw.update((r[10], r) for r in csv.reader(stream))
    with (ROOT / "data/form-identities.csv").open(newline="") as stream:
        identities = {
            r["Source_Key"]: r["Form_ID"] for r in csv.DictReader(stream)
            if r["Status"] == "active" and r["Source_Key"] in raw
        }
    wanted = set(identities.values())
    with (ROOT / "cldf/forms.csv").open(newline="") as stream:
        forms = {r["ID"]: r for r in csv.DictReader(stream) if r["ID"] in wanted}
    return raw, identities, forms


def test_every_source_record_survives_the_complete_build(compiled):
    raw, identities, forms = compiled
    assert raw.keys() == identities.keys(), sorted(raw.keys() - identities.keys())
    assert len(set(identities.values())) == len(raw)
    assert forms.keys() == set(identities.values())


def test_compiled_source_text_and_all_grammatical_labels_are_preserved(compiled):
    raw, identities, forms = compiled
    for key, source in raw.items():
        form = forms[identities[key]]
        assert form["Original"] == source[2], key
        assert form["Gloss"] == source[3], key
        assert form["Language_ID"] == source[0], key
        assert set(source[14].split()) <= set(form["Tags"].split()), key
        assert form["Source"] == source[7], key
        # The CLDF writer inserts spaces after punctuation and escapes newlines.
        normalize = lambda s: "".join(s.replace("\\n", "\n").split())
        assert normalize(form["Description"]) == normalize(source[6]), key


def test_historical_berger_alternatives_and_yasin_attestation_stay_distinct(compiled):
    _raw, ids, forms = compiled
    assert ids["berger-entry-328"] != ids["berger-entry-329"]
    assert ids["berger-entry-334"] != ids["berger-entry-334-dialect-1"]
    yasin = forms[ids["berger-entry-334-dialect-1"]]
    assert {"Burushaski-class-Y", "dialect:Bur:Berger-YS:Yasin"} <= set(yasin["Tags"].split())
