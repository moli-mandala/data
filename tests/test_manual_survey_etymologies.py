import csv
from pathlib import Path

from etymology_assignments import read_assignments


ROOT = Path(__file__).resolve().parents[1]
FORMS_DIR = ROOT / "data/other/forms"

SURVEYS = {
    # name: (source file, source references, raw rows, raw Parameter_IDs,
    #        compiled source-bearing forms, compiled accepted links)
    "chhattisgarhi": (
        "20230517-chattisgarhi.csv", {"chattisgarhi"}, 2_733, 2_120, 2_692, 2_079,
    ),
    "kannauji": (
        "20230526-kannauji.csv", {"kannauji"}, 3_033, 0, 3_033, 1_991,
    ),
    "rajasthani": (
        "20230521-rajasthani.csv",
        {"bagri", "dhundari", "hadothi", "marwari", "mewari", "mewati"},
        # 15,876 survey attestations plus the reviewed Mewari regional family heads and an
        # extracted compound component that cite `mewari` as their source (see data/etymologies.csv).
        16_522, 11_749, 15_887, 11_245,
    ),
    "bundeli": (
        "20230522-bundeli.csv", {"bundeli"}, 5_759, 4_071, 5_562, 3_951,
    ),
    "boehm_tharu": (
        "20230524-tharu.csv", {"boehm"}, 1_921, 1_628, 1_874, 1_589,
    ),
    "webster_tharu": (
        "20230530-tharu2.csv", {"webster"}, 3_560, 0, 3_560, 44,
    ),
}


def citation_keys(value):
    return {
        part.strip().split("[", 1)[0]
        for part in (value or "").split(";")
        if part.strip()
    }


def test_manual_survey_parameter_counts_are_not_dropped():
    for name, (filename, _references, row_count, linked_count, *_compiled) in SURVEYS.items():
        with (FORMS_DIR / filename).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == row_count, name
        assert sum(bool(row[1]) for row in rows) == linked_count, name


def test_manual_survey_etymologies_reach_the_compiled_graph():
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        forms = list(csv.DictReader(stream))
    with (ROOT / "cldf/edges.csv").open(encoding="utf-8", newline="") as stream:
        edge_kinds = {}
        for row in csv.DictReader(stream):
            if row["Rank"] == "1":
                edge_kinds.setdefault(row["Child_ID"], set()).add(row["Kind"])
    # Etymological links a survey attestation can carry. Compounds analysed into ordered
    # `component` edges (Chhattisgarhi pā̃nc-korī 'hundred') and synchronic `derived` forms
    # are attested too; only a form with no rank-1 edge at all is `unlinked`.
    accepted = {
        child for child, kinds in edge_kinds.items()
        if kinds & {"reflex", "borrowed", "variant"}
    }
    attested = {
        child for child, kinds in edge_kinds.items()
        if kinds & {"reflex", "borrowed", "variant", "component", "derived"}
    }

    for name, (_filename, references, _rows, _raw_links, form_count, link_count) in SURVEYS.items():
        survey_forms = {
            row["ID"] for row in forms if citation_keys(row.get("Source", "")) & references
        }
        assert len(survey_forms) == form_count, name
        # Source-owned links are the floor: later etymology review may link more of the
        # survey's unetymologised forms through the overlay, never fewer.
        assert len(survey_forms & accepted) >= link_count, name
        # forms.csv Status must agree with edges.csv: `unlinked` iff no rank-1 edge. Reviewed
        # regional family heads cite the survey too but are etyma (`entry`), not attestations.
        assert {
            row["ID"]
            for row in forms
            if row["ID"] in survey_forms and row["Status"] not in ("unlinked", "entry")
        } == survey_forms & attested, name


def test_source_owned_links_do_not_depend_on_duplicate_overlay_rows():
    with (ROOT / "cldf/forms.csv").open(encoding="utf-8", newline="") as stream:
        forms = {row["ID"]: row for row in csv.DictReader(stream)}
    assignments = read_assignments()

    # These surveys carry their etymologies in the source CSV's Parameter_ID. The overlay may
    # link forms the source left unetymologised (the 2026-09 joint SIL review did), but it must
    # never restate or contradict a source-owned link: a row for a source-linked form is only
    # allowed when it refines the head to one of its own section forms (e.g. 9238 -> 9238-2).
    source_owned = {"chattisgarhi", "bagri", "dhundari", "hadothi", "marwari", "mewari", "mewati"}
    source_links = {}
    for filename in ("20230517-chattisgarhi.csv", "20230521-rajasthani.csv"):
        with (FORMS_DIR / filename).open(encoding="utf-8", newline="") as stream:
            for raw in csv.reader(stream):
                if len(raw) > 3 and raw[1].strip():
                    source_links.setdefault((raw[0], raw[2], raw[3]), set()).add(raw[1].lstrip(">~"))
    with (ROOT / "data/form-identities.csv").open(encoding="utf-8", newline="") as stream:
        identity = {
            row["Form_ID"]: (row["Language_ID"], row["Original"], row["Gloss"])
            for row in csv.DictReader(stream) if row["Status"] == "active"
        }
    conflicts = []
    for row in assignments:
        if not citation_keys(forms[row["Form_ID"]].get("Source", "")) & source_owned:
            continue
        owned = source_links.get(identity.get(row["Form_ID"]), set())
        if owned and row["Etymon_ID"] not in owned and not any(
            row["Etymon_ID"].startswith(head + "-") for head in owned
        ):
            conflicts.append((row["Form_ID"], sorted(owned), row["Etymon_ID"]))
    assert conflicts == []

    kannauji_ids = {
        form_id for form_id, row in forms.items()
        if "kannauji" in citation_keys(row.get("Source", ""))
    }
    # 1,991 links were restored from the legacy NeoJambu database; later review adds more.
    assert sum(row["Form_ID"] in kannauji_ids for row in assignments) >= 1_991


def test_representative_chhattisgarhi_and_rajasthani_source_links_remain():
    with (FORMS_DIR / "20230517-chattisgarhi.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        chhattisgarhi = list(csv.reader(stream))
    with (FORMS_DIR / "20230521-rajasthani.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rajasthani = list(csv.reader(stream))

    assert ["NDu", "6557", "dɛh", "body", "", "dɛh", "", "chattisgarhi"] in chhattisgarhi
    assert any(
        row[0] == "marwari_gomat" and row[1] == "992" and row[2] == "hũ"
        for row in rajasthani
    )
    assert any(
        row[0] == "hadothi_patera" and row[1] == "9691" and row[2] == "mu"
        for row in rajasthani
    )
