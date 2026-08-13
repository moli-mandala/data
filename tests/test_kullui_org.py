import csv
import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "data/other/forms/raw_data/kullui_org.py"
SPEC = importlib.util.spec_from_file_location("kullui_org_importer", SCRIPT)
kullui = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = kullui
SPEC.loader.exec_module(kullui)


def article(source="Old Indo-Aryan", protoform="*bōll"):
    return {
        "id": 1119,
        "lexeme": "bolɳa",
        "orthography": "बोलणा",
        "origin": {"english": "Inherited from", "russian": None},
        "source": {"english": source, "russian": None},
        "protoform": protoform,
        "proto_meaning": {"english": "'speak'", "russian": None},
        "etymology": None,
        "ethnocultural": None,
        "translations": [{
            "grammar_info": "vtstndErgative verb:???",
            "translation_text": {"english": "say", "russian": "говорить"},
            "examples": [],
        }],
    }


def test_extracts_direct_and_embedded_oia_forms():
    assert kullui.extract_oia_etyma(article()) == ["bōll"]
    mandeali = article("Mandeali", "aḍḍi 'heel' < OIA*aḍḍi ʻ heel ʼ")
    assert kullui.extract_oia_etyma(mandeali) == ["aḍḍi"]
    assert kullui.extract_oia_etyma(article("Arabic", "'ādat")) == []


def test_cdial_matching_is_accent_insensitive_but_unique():
    key = kullui.normalize_oia("bhā́ga")
    index = {key: [("9430", "bhāga", ("portion",))]}
    assert kullui.match_cdial(["bhāga"], index) == ("9430", ["bhāga"], "matched")
    index[key].append(("9999", "bhā́ga", ("fortune lot",)))
    assert kullui.match_cdial(["bhāga"], index)[2] == "ambiguous"
    assert kullui.match_cdial(["bhāga"], index, "'fortune, lot'")[0] == "9999"


def test_rich_row_preserves_native_form_etymology_and_source_key():
    row = kullui.source_row(article(), "9321")
    assert len(row) == 15
    assert row[:6] == ["kul", "9321", "bolɳa", "say", "बोलणा", "bolɳa"]
    assert row[7] == "kullui-org[article 1119]"
    assert "Protoform: *bōll" in row[9]
    assert row[10] == "kullui:1119"
    assert {"verb", "tr", "inherited"} <= set(row[14].split())

    borrowed = article("Sanskrit", "bhāga")
    borrowed["origin"]["english"] = "Loanword from"
    assert kullui.source_row(borrowed, "9430")[1] == ">9430"


def test_installed_snapshot_has_stable_unique_article_keys():
    source = Path(__file__).parents[1] / "data/other/forms/20260813-kullui-org.csv"
    if not source.exists():
        return
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 2003
    assert {len(row) for row in rows} == {15}
    assert {row[0] for row in rows} == {"kul"}
    assert all(row[2] and row[5] == row[2] for row in rows)
    keys = [row[10] for row in rows]
    assert len(keys) == len(set(keys))
    assert all(key.startswith("kullui:") for key in keys)
    assert sum(bool(row[1]) for row in rows) == 1110
    assert sum(row[1].startswith(">") for row in rows) == 236
