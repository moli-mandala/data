"""Preliminary isolated configuration and literal-profile fixture; never installs."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import unicodedata as ud

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "data/source_meta.py").exists())
STEM = "20260927-grierson-ahirwati"
PROFILE = "grierson-ahirwati-1908-literal-preliminary"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def clusters(value):
    result = []
    for char in value:
        if ud.combining(char) and result:
            result[-1] += char
        else:
            result.append(char)
    return result


def display(value):
    return ud.normalize("NFC", value.lower()).replace("ṅ", "ŋ")


def prepare():
    corpus, pins, counts = [], {}, collections.Counter()

    def add(path, category, keyfield, formfield, tagsfield=None):
        pins[str(path.relative_to(ROOT))] = sha(path)
        for row in records(path):
            corpus.append(dict(key=f'{category}:{row[keyfield]}', physical_key=row[keyfield],
                               original=row[formfield], category=category,
                               tags=row.get(tagsfield, []) if tagsfield else []))
            counts[category] += 1

    add(PACKAGE / "grammar/child-witnesses-reviewed.jsonl", "grammar", "witness_key", "source_form")
    path = PACKAGE / "specimens/specimen37/independent-review/metadata-proposal-revision2/candidate.csv"
    pins[str(path.relative_to(ROOT))] = sha(path)
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            assert len(row) == 15
            corpus.append(dict(key=f"no37:{row[10]}", physical_key=row[10], original=row[2],
                               category="no37", tags=row[14].split(), native=row[1]))
            counts["no37"] += 1
    add(PACKAGE / "specimen38/root-reviewed-revision/p238-atomic-reviewed.jsonl", "no38", "source_unit_key", "raw_form")
    add(PACKAGE / "specimen38/native-review/whole-utterances-proposal.jsonl", "no38_whole", "key", "emission_form", "tags")
    add(PACKAGE / "specimen38/native-review/quoted-emissions-reviewed-derivative.jsonl", "no38_quotes", "source_unit_key", "raw_form", "proposed_tags")
    add(PACKAGE / "outside-review/child-witnesses-reviewed.jsonl", "outside", "key", "source_form", "proposed_tags")
    add(PACKAGE / "outside-review/Hissar-whole-physical-atoms.jsonl", "outside_physical", "key", "raw_form", "proposed_tags")
    expected = dict(grammar=98, no37=390, no38=133, no38_whole=23, no38_quotes=18, outside=62, outside_physical=2)
    assert dict(counts) == expected, counts
    assert len({r["key"] for r in corpus}) == len(corpus)
    controls = ["Ṅ Ṅ ṅ ṅ", "Ñ ñ", "Ā̃ Ī̃ ā̃ ī̃", "karᵃṇō apᵃṇū aĩh", "-ā ā- –ā", "Jai-kaḷi- kai", "kai-ṭaraḥ", "‘Bōlō,’ (ē) ?", "पीलने बाँट-दियो", "ایک بیرےنے"]
    for n, form in enumerate(controls):
        corpus.append(dict(key=f"control:{n}", original=form, category="synthetic_control", tags=[]))
    rules, cluster_counts, symbols, native_symbols = {" ": "#"}, collections.Counter(), collections.Counter(), collections.Counter()
    for row in corpus:
        row["expected_display"] = display(row["original"])
        symbols.update(row["original"])
        native_symbols.update(row.get("native", ""))
        for norm in ("NFC", "NFD"):
            for cluster in clusters(ud.normalize(norm, row["original"])):
                rules[cluster] = "#" if cluster == " " else display(cluster)
                cluster_counts[cluster] += 1
    with (HERE / f"{PROFILE}.txt").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["Grapheme", "IPA"])
        writer.writerows(sorted(rules.items()))
    with (HERE / "coverage-corpus.jsonl").open("w") as stream:
        for row in corpus:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    inventory = dict(clusters=[dict(grapheme=c, count=n, codepoints=[f"U+{ord(x):04X}" for x in c]) for c, n in sorted(cluster_counts.items())],
                     Roman_and_control_codepoints=[dict(character=c, count=n, codepoint=f"U+{ord(c):04X}", name=ud.name(c, "UNNAMED")) for c, n in sorted(symbols.items())],
                     No37_native_codepoints=[dict(character=c, count=n, codepoint=f"U+{ord(c):04X}", name=ud.name(c, "UNNAMED")) for c, n in sorted(native_symbols.items())])
    (HERE / "unicode-inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    settings = f'''file: {STEM}.csv
defaults:
  transcription:
    profile: {PROFILE}
    input: form
    preserve_hyphens: true
  identity:
    dedupe_by_entry_key: true
  forms:
    split_alternates: false
  gloss:
    grammar_tags: false
sources:
  grierson1908ahirwati:
    reference:
      editor: OpenAI Codex
      ocr: false
      etymology_provenance: source
'''
    (HERE / f"{STEM}.yaml").write_text(settings)
    identities = json.loads((PACKAGE / "scope/historical-identity-proposal.json").read_text())
    outside = json.loads((PACKAGE / "outside-review/reviewed-identity-and-provenance-proposal.json").read_text())
    lects = [dict(ID=r["ID"], Language_ID=r["Language_ID"], Name=r["Name"], Location=r["Location"]) for r in identities["new_lects"]]
    lects += [dict(ID=r["id"], Language_ID=r["language"], Name=r["name"], Location=f"Historical {r['name'].split(' (')[0]}, LSI 1908 survey") for r in outside["proposed_historical_lects"]]
    import sys
    sys.path.insert(0, str(ROOT / "data"))
    from dialects import dialect_tag
    languages = {r["ID"]:r for r in csv.DictReader((ROOT / "data/cldf/languages.csv").open())}
    fields = next(csv.reader((ROOT / "data/cldf/dialects.csv").open()))
    with (HERE / "dialects-proposed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in lects:
            row.update(Source_Language_ID=row["ID"], Tag=dialect_tag(row["Language_ID"], row["ID"], row["Name"]), Glottocode="", Latitude="", Longitude="", Clade=languages[row["Language_ID"]]["Clade"], Quality="")
            writer.writerow(row)
    bibliography = '''@book{grierson1908ahirwati,
  author = {Grierson, George Abraham},
  title = {{Linguistic Survey of India. Volume IX, Part II. Indo-Aryan Family, Central Group. Specimens of the Rajasthani and Gujarati}},
  year = {1908},
  address = {Calcutta},
  publisher = {Superintendent Government Printing, India},
  url = {https://commons.wikimedia.org/wiki/File:Linguistic_Survey_of_India_Vol_9_Part_2.djvu},
  license = {Public-domain original 1908 edition; Commons source record declares Public Domain Mark 1.0. Internet Archive metadata separately says Copyright Permitted.},
  included = {Selected complete Ahirwati grammatical description pp. 49--51, Gurgaon specimen 37 pp. 233--236 and mixed Jhajjar/Rohtak specimen 38 pp. 237--239, with the explicit Bagri comparison spans on pp. 149 and 151. No Ahirwati column is printed in the volume's standard comparative tables.},
  provenance = {data/other/forms/raw_data/grierson_ahirwati_1908/scope/verified-scope-manifest.json; data/other/forms/raw_data/grierson_ahirwati_1908/scope/source-and-errata-review.json; data/other/forms/raw_data/grierson_ahirwati_1908/scope/major-addenda-applicability-review-20260927.json},
  note = {The original title retains THE before Rajasthani and Gujarati; the 1927 minor addenda p. 41 directs its omission. The source calls Ahirwati a Mewati variety and explicitly qualifies specimen 38 as mixed with Bangaru. The quoted Bagri comparisons distinguish Sirsa, Hissar and Bikaner. The author attributes the second-person plural present sai with nasalization to borrowing from Bagri; that attributed claim does not identify a specific printed donor occurrence.},
  jambu_editor = {OpenAI Codex},
  ocr = {No},
  etymology_provenance = {source}
}
'''
    (HERE / "references-proposed.bib").write_text(bibliography)
    status = dict(status="preliminary source-preserving house display configuration and coverage fixture; no installation approval", input_hashes=pins, counts=dict(counts), synthetic_controls=len(controls), fixture_cases=len(corpus), profile_rules=len(rules), policy="Root-reviewed lowercase and ṅ/Ṅ→ŋ house display only; all other marks/quantities/raised letters retained. Original strings retained exactly in corpus; blank Phonemic.",
                  pending=["No37 revision2 metadata/final-output acceptance; No38 native acceptance and metadata/boundary review; final combined candidate and refreshed symbols/pins; actual guarded importer command once implemented; canonical CSV/audit provenance after approved installation"],
                  source_roles=dict(primary="Grierson 1908 is the lexical/analysis citation", major_minor_addenda="Documentary applicability and title correction; no imported lexical correction in selected spans", OCR="Locator search only, not installed transcription input", authorities="No separate authority attestation emitted; contextual works remain documentary"),
                  canonical_changes=False, output_hashes={name:sha(HERE / name) for name in [f"{PROFILE}.txt", "coverage-corpus.jsonl", "unicode-inventory.json", f"{STEM}.yaml", "dialects-proposed.csv", "references-proposed.bib"]})
    (HERE / "proposal-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    prepare()
