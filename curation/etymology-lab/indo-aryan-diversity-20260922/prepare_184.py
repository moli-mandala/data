"""Prepare source-segmented Zargari four-fifths comparison."""
import csv
import json
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from etymology_assignments import read_assignments

D = Path("curation/etymology-lab/indo-aryan-diversity-20260922")
SOURCE = "rezaibaghbidi2003zargari"
FID = "f_re6aoervvpxke"
FIVE = "f_ahg7g4o5kv5ay"
FOUR = "f_7vled5oasq42i"
forms = {r["ID"]: r for r in csv.DictReader(open("cldf/forms.csv")) if SOURCE in r["Source"]}
linked = {r["Form_ID"]: r for r in read_assignments() if r["Status"] == "accepted"}
edges = list(csv.DictReader(open("cldf/edges.csv")))
already_linked = set(linked) | {r["Child_ID"] for r in edges if r["Rank"] == "1"}
form = forms[FID]
assert FID not in already_linked
assert (form["Form"], form["Original"], form["Gloss"]) == ("pānjundār-iśtār", "pāndžundār-ištār", "four fifths")
assert linked[FIVE]["Etymon_ID"] == "7655" and linked[FOUR]["Etymon_ID"] == "4655"
assert [(r["Child_ID"], r["Kind"]) for r in edges if r["Parent_ID"] == FID and r["Rank"] == "1"] == [("f_fcnyejf6wh7ck", "variant")]

key = [r["Source_Key"] for r in csv.DictReader(open("data/form-identities.csv")) if r["Form_ID"] == FID]
assert key == ["rezaibaghbidi2003zargari:p138:s3.5.3.1:i2"]
key = key[0]
source_rows = list(csv.reader(open("data/other/forms/20260825-rezai-baghbidi-zargari.csv")))
assert len([r for r in source_rows if r[0] == "Zarg" and r[3] == form["Gloss"] and r[7] == form["Source"] and r[10] == key]) == 1
raw_rows = list(csv.DictReader(open("data/other/forms/raw_data/20260825-rezai-baghbidi-zargari-audit.csv")))
raw = [r for r in raw_rows if r["Status"] == "ingest" and key in r["Emitted_Keys"].split() and r["Printed_Page"] == "138"]
assert len(raw) == 1 and raw[0]["Raw_Form"] == "pāndžundārištār/pāndž-ištār"

primary = {}
for page, html in enumerate(pickle.load(open("data/cdial/cdial.pickle", "rb")), 1):
    if not any(f"<number>{n}</number>" in html for n in {"7655", "4655"} - primary.keys()):
        continue
    for paragraph in BeautifulSoup(html, "html.parser").select("p"):
        number = paragraph.find("number")
        if number and number.get_text(strip=True) in {"7655", "4655"}:
            n = number.get_text(strip=True)
            primary[n] = [{"page": page, "url": f"https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page}", "html": str(paragraph), "text": paragraph.get_text(" ", strip=True)}]
    if len(primary) == 2:
        break
assert set(primary) == {"7655", "4655"}
assert "eur. panǰ" in primary["7655"][0]["text"] and "eur. pal. štar" in primary["4655"][0]["text"]
(D / "zargari-primary038.json").write_text(json.dumps(primary, ensure_ascii=False, indent=2) + "\n")

evidence = ("The survey explicitly defines its fraction compounds as the larger number in ablative/instrumental plural plus the smaller number, and prints pāndžundār-ištār/pāndž-ištār 'four fifth' (p. 138 §3.5.3.1). Its five-member is the separately installed Zargari pānj 'five' (already CDIAL 7655; Turner European Romani panǰ), and the four-member is iśtār 'four' (already CDIAL 4655 §1; Turner European Romani štar). The -undār ending is source case morphology, not an independent etymon. This is a synchronic two-component fraction, not a proposed reflex of a fused Sanskrit fraction. Original PDF, exact source audit and imported key checked; no transcription or identity change.")
accepted = [{"record": {k: form[k] for k in ("ID", "Language_ID", "Form", "Gloss", "Source", "Tags")}, "parent": FIVE, "kind": "component", "citation": f"CDIAL[7655];CDIAL[4655];{form['Source']}", "evidence": evidence, "components": [FIVE, FOUR]}]
(D / "pass184-decisions.json").write_text(json.dumps({"accepted": accepted, "held": []}, ensure_ascii=False, indent=2) + "\n")
(D / "pass184-source-audit.json").write_text(json.dumps([{"id": FID, "source_key": key, "raw_unit": raw[0]["Unit_ID"], "raw_form": raw[0]["Raw_Form"], "original": form["Original"], "compiled_form": form["Form"], "printed_page": raw[0]["Printed_Page"], "record_hash": raw[0]["Record_SHA256"], "components": [FIVE, FOUR], "child_variant": "f_fcnyejf6wh7ck"}], ensure_ascii=False, indent=2) + "\n")
(D / "pass184-review.md").write_text("\n".join([
    "# Zargari: four fifths",
    "",
    "One source-segmented fraction, two ordered component rows, under standing autonomous authorization. Printed p. 138 visually checked; raw/installed identity and both full Turner articles audited. No lexical import or rebuild.",
    "",
    "| Zargari form | Components | Evidence |",
    "|---|---|---|",
    f"| **{form['Form']}** ‘{form['Gloss']}’ | **pānj** ‘five’ (CDIAL **7655**) + **iśtār** ‘four’ (CDIAL **4655 §1**) | {evidence} |",
    "",
    "**Hold:** The neighboring dujendār-jek/duj-jek ‘half’ is genuinely compositional in the source, but its compiled rank-1 variant child `f_tothksktrrax4` is paš ‘half’. The PDF marks paš as an equivalent (`=paš`), not an alternate spelling of the two-numeral compound. Linking that parent to source numeral components would incorrectly propagate a two-numeral ancestry to independent paš. No compiled-edge edits under the no-rebuild instruction. The four-fifths parent has only the genuinely reduced compound pānj-iśtār as its rank-1 variant, so it is safe.",
    "",
    "Discovery: `zargari-primary038.json`, printed PDF image and `pass184-source-audit.json`. Save and validation pending.",
]) + "\n")
print("accepted", len(accepted), "components", len(accepted[0]["components"]))
