"""Preliminary source-coverage fixture, not an accepted transcription profile."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import unicodedata as ud

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent
ROOT = HERE.parents[7]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(value):
    return ud.normalize("NFC", value.lower()).replace("ṅ", "ŋ")


def clusters(value):
    """Base plus following combining marks; all inventoried input is alphabetic."""
    result = []
    for character in value:
        if ud.combining(character) and result:
            result[-1] += character
        else:
            result.append(character)
    return result


def prepare():
    pins, corpus = {}, []
    counts = collections.Counter()
    for page in (564, 565, 568, 569):
        path = PACKAGE / "specimens" / f"p{page}-first-reading.jsonl"
        pins[str(path.relative_to(ROOT))] = sha(path)
        with path.open() as stream:
            for line in stream:
                for atom in json.loads(line)["target_atoms"]:
                    corpus.append(dict(key=atom["source_unit_key"], original=atom["raw_source_form"], category="specimen_physical_atom"))
                    counts["specimen_physical_atom"] += 1
    path = PACKAGE / "independent-grammar-review/independent-child-witnesses.jsonl"
    pins[str(path.relative_to(ROOT))] = sha(path)
    with path.open() as stream:
        for line in stream:
            witness = json.loads(line)
            corpus.append(dict(key=witness["witness_key"], original=witness["form"], category="grammar_child"))
            counts["grammar_child"] += 1
    path = PACKAGE / "outside-review/scope-decisions.json"
    pins[str(path.relative_to(ROOT))] = sha(path)
    for witness in json.loads(path.read_text())["physical_lexical_witnesses_outside_chapter"]:
        corpus.append(dict(key=witness["key"], original=witness["form_candidate"], category="outside_candidate", disposition=witness["disposition"]))
        counts["outside_candidate"] += 1
        for number, alternative in enumerate(witness.get("alternatives", []), 1):
            if alternative != witness["form_candidate"]:
                corpus.append(dict(key=f'{witness["key"]}:unaccepted-alternative:{number}', original=alternative, category="outside_alternative", disposition="unaccepted glyph alternative; not an emitted attestation"))
                counts["outside_alternative"] += 1
    assert counts["specimen_physical_atom"] == 561
    assert counts["grammar_child"] == 224
    assert counts["outside_candidate"] == 5
    assert len({row["key"] for row in corpus}) == len(corpus)
    controls = ["Ṅ Ṅ ṅ ṅ", "Ñ ñ", "Ā̃ Ī̃ ā̃ ī̃", "A ū", "annu(ṅg)", "āṇḍāt(ūn)", "āṇḍā(t)", "avan(d)", "avandu(n)", "Bāp- phōkān", "CH W ch w", "ṭ ṇ ḍ ḷ ṛ ś ṣ", "A,B;C.D? ‘E’ (F)!", "-a ā-"]
    for number, value in enumerate(controls):
        corpus.append(dict(key=f"synthetic-policy-control:{number}", original=value, category="synthetic_control"))
    inventory = collections.Counter()
    symbols = collections.Counter()
    rules = {" ": "#"}
    for row in corpus:
        row["expected_display"] = display(row["original"])
        for character in row["original"]:
            symbols[character] += 1
        for form in (ud.normalize("NFC", row["original"]), ud.normalize("NFD", row["original"])):
            for cluster in clusters(form):
                inventory[cluster] += 1
                rules[cluster] = "#" if cluster == " " else display(cluster)
    with (HERE / "grierson-kolami-basim-1906-preliminary.txt").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["Grapheme", "IPA"])
        writer.writerows(sorted(rules.items()))
    with (HERE / "coverage-corpus.jsonl").open("w") as stream:
        for row in corpus:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    inventory_report = dict(cluster_definition="Unicode base with following combining marks; both NFC and NFD inventoried", clusters=[dict(grapheme=c, count=n, codepoints=[f"U+{ord(x):04X}" for x in c]) for c, n in sorted(inventory.items())], codepoints=[dict(character=c, count=n, codepoint=f"U+{ord(c):04X}", name=ud.name(c, "UNNAMED")) for c, n in sorted(symbols.items())])
    (HERE / "unicode-inventory.json").write_text(json.dumps(inventory_report, ensure_ascii=False, indent=2) + "\n")
    status = dict(status="preliminary coverage fixture only; inputs unaccepted and final corrected candidate must refresh this fixture", input_hashes=pins, counts=dict(counts), synthetic_controls=len(controls), total_fixture_cases=len(corpus), profile_rules=len(rules), inventoried_clusters=len(inventory), inventoried_codepoints=len(symbols), policy="NFC lowercase and ṅ/Ṅ to ŋ only; all other quantities, optional notation, punctuation, ñ, ch, w, spaces and hyphens retained", boundary_policy="Direct production segments.Tokenizer API tested. Generic parser independently strips normal boundary punctuation; this fixture makes no parser-boundary approval claim.", original_policy="coverage-corpus original field copied exactly; display is separate, no input mutation", canonical_changes=False, output_hashes={name:sha(HERE / name) for name in ("coverage-corpus.jsonl", "unicode-inventory.json", "grierson-kolami-basim-1906-preliminary.txt")})
    (HERE / "fixture-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    return status


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False))
