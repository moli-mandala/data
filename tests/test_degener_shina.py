import csv
import importlib.util
import json
import unicodedata
from collections import Counter
from pathlib import Path

from segments import Tokenizer


ROOT = Path(__file__).parents[1]


def load_source():
    path = ROOT / "data/other/forms/raw_data/degener_shina_2008.py"
    spec = importlib.util.spec_from_file_location("degener_shina_2008", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


degener = load_source()
ENTRIES, PARSED, FORMS, AUDIT, STATUS = degener.build()
BY_KEY = {row["Entry_Key"]: row for row in FORMS}


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_transcription_covers_the_complete_glossary():
    assert len(ENTRIES) == 1561
    assert {e["page"] for e in ENTRIES} == set(range(243, 316))
    counts = Counter(e["page"] for e in ENTRIES)
    assert counts[243] == 23
    assert counts[246] == 14
    assert counts[295] == 25
    assert counts[315] == 8


def test_row_statuses_and_stable_unique_keys():
    assert len(FORMS) == 1577
    assert STATUS == {
        "installed_form": 1521,
        "installed_cross_reference_variant": 32,
        "audit_only_cross_reference": 8,
    }
    keys = [row["Entry_Key"] for row in FORMS]
    assert len(keys) == len(set(keys))
    assert all(key.startswith("degener-shina2008:p") for key in keys)


def test_language_dialect_and_locators():
    assert {row["Language_ID"] for row in FORMS} == {"Sh"}
    assert all("dialect:Sh:gil:Gilgit" in row["Tags"] for row in FORMS)
    assert all(row["Source"].startswith("degener-shina2008[p. ") for row in FORMS)


def test_representative_rows():
    baal = BY_KEY["degener-shina2008:p248:e02"]
    assert (baal["Form"], baal["Parameter_ID"], baal["Gloss"]) == (
        "baál", "9216", "child, boy")
    assert baal["Notes"] == "Kind, Junge"
    # light-verb construction is one headword, not an alternate
    assert BY_KEY["degener-shina2008:p247:e24"]["Form"] == "bal b-"
    assert "degener-shina2008:p247:e24:v2" not in BY_KEY
    # headline alternates become variant rows
    alt = BY_KEY["degener-shina2008:p284:e01:v2"]
    assert alt["Form"] == "maféer"
    assert alt["Variant_Of_Key"] == "degener-shina2008:p284:e01"
    assert "alternate" in alt["Tags"]
    # printed cross-reference resolves to a variant of its target
    kheer = BY_KEY["degener-shina2008:p280:e10"]
    assert kheer["Form"] == "kheer-"
    assert kheer["Variant_Of_Key"] == "degener-shina2008:p280:e04"
    # loanwords keep the arrow etymology and the loanword tag
    kheer_ge = BY_KEY["degener-shina2008:p280:e09"]
    assert kheer_ge["Form"] == "khéer ge"
    assert "loanword" in kheer_ge["Tags"]
    assert "xair" in kheer_ge["Etymology"]


def test_turner_link_policy():
    # a claim after a Burushaski comparandum still links
    ook = BY_KEY["degener-shina2008:p289:e19"]
    assert ook["Parameter_ID"] == "2538"
    # a work-plus-page citation after the claim does not block it
    yach = BY_KEY["degener-shina2008:p313:e14"]
    assert yach["Parameter_ID"] == "10395"
    # printed decimal sub-numbers link to the integer CDIAL parent
    balugun = BY_KEY["degener-shina2008:p248:e11"]
    assert balugun["Parameter_ID"] == "11503"
    assert "11503.3" in balugun["Etymology"]
    # hedged and multi-number claims stay prose
    assert BY_KEY["degener-shina2008:p243:e20"]["Parameter_ID"] == ""  # T. 145, 887
    assert BY_KEY["degener-shina2008:p270:e01"]["Parameter_ID"] == ""  # T. 14154?
    assert BY_KEY["degener-shina2008:p259:e18"]["Parameter_ID"] == ""  # zu T. 6298
    linked = sum(1 for row in FORMS if row["Parameter_ID"])
    assert linked == 529


def test_sound_profile_covers_every_installed_form():
    tokenizer = Tokenizer(str(ROOT / "conversion/degener-shina.txt"))
    for row in FORMS:
        out = tokenizer(unicodedata.normalize("NFC", row["Form"]),
                        column="IPA", segment_separator="", separator=" ")
        assert "�" not in out and "?" not in out, (row["Form"], out)
    assert tokenizer("ẓan th-", column="IPA", segment_separator="",
                     separator=" ") == "ẓan tʰ-"
    assert tokenizer("ac̣híi", column="IPA", segment_separator="",
                     separator=" ") == "aʦ̣ʰī̀"
    assert tokenizer("kaá~kaċ", column="IPA", segment_separator="",
                     separator=" ") == "kā́̃kaʦ"


def test_audit_is_complete_and_material_error_free():
    audit = rows(ROOT / "data/other/forms/raw_data/20260827-degener-shina-audit.csv")
    assert len(audit) == 1561
    assert all(row["Material_Error"] == "no" for row in audit)
    assert {row["Final_Status"] for row in audit} == {
        "installed_form", "installed_cross_reference_variant",
        "audit_only_cross_reference"}
    unresolved = [row["Unit_ID"] for row in audit
                  if row["Final_Status"] == "audit_only_cross_reference"]
    assert unresolved == ["p245:e09", "p258:e13", "p264:e06", "p274:e01",
                          "p278:e07", "p294:e21", "p299:e11", "p305:e10"]
    # every audit row keeps the raw headline so decisions are reconstructible
    assert all(row["Raw_Headline"] for row in audit)


def test_manifest_records_rights_scope_and_scan_identity():
    manifest = json.loads((ROOT / "data/other/forms/raw_data/"
                           "20260827-degener-shina-manifest.json").read_text())
    assert manifest["pdf_sha256"] == {
        "TN446831": degener.PDF1_SHA256, "TN447377": degener.PDF2_SHA256}
    assert manifest["pdf_redistributed"] is False
    assert manifest["outputs"]["form_count"] == 1577
    assert manifest["outputs"]["sample_count"] == 25
    assert manifest["extraction"]["transcription_uncertain_headword_records"] == 15
    assert manifest["extraction"]["transcription_uncertain_readings"] == 16
    assert manifest["glossary_printed_pages"] == [243, 315]
    assert "Gilgit" in manifest["scope"]["language_model"]


def test_installed_csv_matches_build():
    installed = list(csv.reader((ROOT / "data/other/forms/"
                                 "20260827-degener-shina.csv").open(encoding="utf-8")))
    assert len(installed) == len(FORMS)
    assert installed[0][2] == FORMS[0]["Form"]
    joined = "\n".join(",".join(row) for row in installed)
    assert "�" not in joined
    assert unicodedata.is_normalized("NFC", joined)


def test_uncertain_readings_are_typed_and_auditable():
    uncertain_rows = [row for row in FORMS if "uncertain" in row["Tags"].split()]
    assert len(uncertain_rows) == 15
    transcription = (ROOT / "data/other/forms/raw_data/"
                       "20260827-degener-shina-transcription.txt").read_text()
    source_lines = [line for line in transcription.splitlines()
                    if line.startswith(("H|", "C|", "X|", "XC|"))]
    assert sum(line.count("⟦") for line in source_lines) == 16


def test_compiled_rows_exist_when_cldf_has_been_built():
    forms_path = ROOT / "cldf/forms.csv"
    if not forms_path.exists():
        return
    compiled = rows(forms_path)
    installed = [row for row in compiled if "degener-shina2008" in row["Source"]]
    if not installed or forms_path.stat().st_mtime < (
            ROOT / "data/other/forms/20260827-degener-shina.csv").stat().st_mtime:
        return
    assert len(installed) == 1577
    assert "�" not in "".join("|".join(row.values()) for row in installed)

    source_keys = [
        row for row in rows(ROOT / "cldf/form-source-keys.csv")
        if row["Source_Key"].startswith("degener-shina2008:")
    ]
    assert len(source_keys) == 1577
    assert len({row["Source_Key"] for row in source_keys}) == 1577
    aliases = {
        row["Legacy_ID"]: row["Form_ID"]
        for row in rows(ROOT / "cldf/form-id-aliases.csv")
    }
    resolved_ids = {
        aliases.get(row["Legacy_ID"], row["Legacy_ID"])
        for row in source_keys
    }
    assert len(resolved_ids) == 1577
    assert resolved_ids <= {row["ID"] for row in compiled}
