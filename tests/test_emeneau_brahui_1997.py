import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1]
RAW = ROOT / "data/other/forms/raw_data"
SOURCE = "emeneau1997brahui"


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def identities():
    return rows(ROOT / "data/form-identities.csv")


def source_id(source_key: str) -> str:
    matches = [row["Form_ID"] for row in identities() if row["Source_Key"] == source_key]
    assert len(matches) == 1, source_key
    return matches[0]


def legacy_id(legacy: str) -> str:
    matches = [row["Form_ID"] for row in identities() if row["Legacy_ID"] == legacy]
    assert len(matches) == 1, legacy
    return matches[0]


def test_every_printed_page_has_a_parseable_agent_result_and_full_audit():
    counts = {}
    unit_ids = []
    for printed_page in range(440, 448):
        page = json.loads((RAW / f"emeneau_brahui_1997_agent/p{printed_page}.json").read_text())
        assert page["printed_page"] == printed_page
        assert page["pdf_page"] == printed_page - 438
        counts[str(printed_page)] = len(page["records"])
        unit_ids.extend(record["unit_id"] for record in page["records"])
    assert counts == {
        "440": 0, "441": 16, "442": 11, "443": 17,
        "444": 16, "445": 9, "446": 7, "447": 0,
    }
    assert len(unit_ids) == len(set(unit_ids)) == 76

    audit = rows(RAW / "20260819-emeneau-brahui-1997-audit.csv")
    assert {row["Unit_ID"] for row in audit} == set(unit_ids)
    assert Counter(row["Final_Status"] for row in audit) == {
        "entry_text_only": 44, "installed_form": 29, "context_only": 3,
    }
    assert sum(bool(row["Agent_Correction"]) for row in audit) == 18
    assert all(row["Review"] == "source-image-verified by editorial reconciliation" for row in audit)
    assert all(row["Material_Error"] == "no" for row in audit)


def test_manifest_and_reconciliation_describe_the_page_agent_pilot():
    manifest = json.loads((RAW / "20260819-emeneau-brahui-1997-manifest.json").read_text())
    assert manifest["pdf_sha256"] == "e2aa0c7a0063b83509cf402cb880de97a902b1195c7d5f69bb75d8775fb30dde"
    assert manifest["pdf_pages"] == 9
    assert manifest["pdf_redistributed"] is False
    assert manifest["extraction"]["agent_model"] == "gpt-5.6-luna"
    assert manifest["extraction"]["record_total"] == 76
    assert manifest["outputs"]["form_count"] == 19
    assert manifest["outputs"]["entry_text_count"] == 35

    reconciliation = json.loads(
        (RAW / "20260819-emeneau-brahui-1997-reconciliation.json").read_text()
    )
    assert reconciliation["correction_count"] == 18
    decisions = {row["unit_id"]: row["decision"] for row in reconciliation["corrections"]}
    assert "bāšt" in decisions["p442:s4:u03"]
    assert "taṛifing" in decisions["p444:s11:u01"]
    assert "cīkap-" in decisions["p444:sfoot:u01"]


def test_compiled_forms_preserve_source_notation_and_homonyms():
    forms = {row["ID"]: row for row in rows(ROOT / "cldf/forms.csv")}
    source_forms = [row for row in forms.values() if SOURCE in row["Source"]]
    assert len(source_forms) == 19
    assert "�" not in "".join("|".join(row.values()) for row in source_forms)

    begh = forms[source_id(f"{SOURCE}:p441:s2.1:begh")]
    hogh = forms[source_id(f"{SOURCE}:p441:s2.2:hogh")]
    assert (begh["Form"], begh["Original"]) == ("bēɣ-", "bēg̲h̲-")
    assert (hogh["Form"], hogh["Original"]) == ("hōɣ-", "hōg̲h̲-")

    sour = forms[source_id(f"{SOURCE}:p444:s11:tarifing:sour")]
    slaughter = forms[source_id(f"{SOURCE}:p444:s11:tarifing:slaughter")]
    assert sour["Form"] == slaughter["Form"] == "taṛifing"
    assert sour["Gloss"] == "to turn sour (of milk)"
    assert slaughter["Gloss"] == "to be slaughtered"

    article_tongue = forms[source_id(f"{SOURCE}:p446:fn6:dui")]
    ali_control = forms[source_id("ali-kobayashi2024:p698:e57")]
    assert article_tongue["Form"] == ali_control["Form"] == "dūī"
    assert article_tongue["Gloss"] == "tongue"
    assert "control" in ali_control["Gloss"] and article_tongue["ID"] != ali_control["ID"]


def test_accepted_links_and_corrections_are_rank_one():
    edges = rows(ROOT / "cldf/edges.csv")
    by_child = {}
    for edge in edges:
        by_child.setdefault(edge["Child_ID"], []).append(edge)

    accepted = {
        f"{SOURCE}:p441:s2.1:begh": "d5078",
        f"{SOURCE}:p441:s2.1:bel": "d5503",
        f"{SOURCE}:p441:s2.2:hogh": "d996",
        f"{SOURCE}:p441:s2.2:mux": "d4986",
        f"{SOURCE}:p441:s2.2:taf": "d3133",
        f"{SOURCE}:p442:s4:basht": "d4841",
        f"{SOURCE}:p444:s8.2:shurufing": "d2712",
        f"{SOURCE}:p444:s10:kirrefing": "d1595",
        f"{SOURCE}:p444:s11:tarifing:slaughter": "d3029",
        f"{SOURCE}:p445:s12:taring": "d3195",
        f"{SOURCE}:p445:s13:allai": "d235",
    }
    for key, target in accepted.items():
        assert any(
            edge["Parent_ID"] == target and edge["Kind"] == "reflex" and edge["Rank"] == "1"
            for edge in by_child[source_id(key)]
        )

    ulli = legacy_id("d500-2")
    hulli = source_id("ali-kobayashi2024:p705:e13")
    du = source_id("ali-kobayashi2024:p698:e53")
    assert [(e["Parent_ID"], e["Kind"], e["Rank"]) for e in by_child[ulli]] == [("d701", "reflex", "1")]
    assert [(e["Parent_ID"], e["Kind"], e["Rank"]) for e in by_child[hulli]] == [("d701", "reflex", "1")]
    assert [(e["Parent_ID"], e["Kind"], e["Rank"]) for e in by_child[du]] == [("6586", "borrowed", "1")]

    gadaba = source_id(f"{SOURCE}:p444:fn3:cikap")
    telugu = legacy_id("d2621-8")
    assert [(e["Parent_ID"], e["Kind"], e["Rank"]) for e in by_child[gadaba]] == [(telugu, "borrowed", "1")]

    duwi = source_id(f"{SOURCE}:p446:fn6:duwi")
    dui = source_id(f"{SOURCE}:p446:fn6:dui")
    assert [(e["Parent_ID"], e["Kind"], e["Rank"]) for e in by_child[duwi]] == [(dui, "variant", "1")]


def test_tentative_comparisons_are_ranked_and_stay_unlinked():
    forms = {row["ID"]: row for row in rows(ROOT / "cldf/forms.csv")}
    edges = rows(ROOT / "cldf/edges.csv")
    by_child = {}
    for edge in edges:
        by_child.setdefault(edge["Child_ID"], []).append(edge)

    expected = {
        f"{SOURCE}:p443:s6:puzza": [("d4477", "reflex", "2")],
        f"{SOURCE}:p443:s6:kuzing": [("d2687", "reflex", "2"), ("d1876", "reflex", "3")],
        f"{SOURCE}:p443:s7:pisfing": [("d4135", "reflex", "2"), ("d4183", "reflex", "3")],
        f"{SOURCE}:p444:s8.1:shupping": [("d2621", "reflex", "2")],
        f"{SOURCE}:p446:fn6:dui": [("5228", "borrowed", "2")],
    }
    for key, hypotheses in expected.items():
        form_id = source_id(key)
        assert forms[form_id]["Status"] == "unlinked"
        assert [(e["Parent_ID"], e["Kind"], e["Rank"]) for e in by_child[form_id]] == hypotheses


def test_entry_text_and_reference_outputs_are_resolvable():
    forms = {row["ID"] for row in rows(ROOT / "cldf/forms.csv")}
    blocks = [row for row in rows(ROOT / "cldf/entry-texts.csv") if row["Source"].startswith(SOURCE)]
    assert len(blocks) == 35
    assert all(row["Form_ID"] in forms for row in blocks)
    assert {"d500", "d701", "d5078", "6586", "14024", "5228"} <= {
        row["Form_ID"] for row in blocks
    }

    references = {row["ID"]: row for row in rows(ROOT / "cldf/references.csv")}
    assert "10\\.1017/S0041977X00032481" in references[SOURCE]["Source"]
    assert references[SOURCE]["Progress"].startswith("All lexical, etymological")
