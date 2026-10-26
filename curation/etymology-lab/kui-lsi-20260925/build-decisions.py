"""Prepare a bounded first Kui LSI research checkpoint without saving links."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "inventory.json").read_text())
assert len(records) == 151

proposals = {
    18: ("d4756", "The full DEDR article explicitly gives Kui mṛehenju ‘man, boy, husband’ and Kuwi mneha ‘man’. LSI mnēhēñju matches that man-family form in its survey transcription."),
    19: ("d400", "The full DEDR article explicitly gives Kui āsa ‘female, woman’ with plural -ska/-saka. LSI āsā matches the noun with survey vowel length."),
    47: ("d4841", "The full DEDR article explicitly gives Kui mio ‘above, aloft, over’. LSI miõ ‘up’ is its nasalized survey spelling."),
}

holds = {
    1: (["d784"], "The eight family in DEDR 784 has no printed Kui reflex; LSI āṭu needs a demonstrated correspondence for initial ā and the single stop."),
    2: ([], "LSI na ‘nine’ lacks a verified full DEDR reflex or immediate donor."),
    3: ([], "LSI daśu ‘ten’ resembles Indo-Aryan daśa, but the immediate donor into Kui was not verified."),
    4: ([], "LSI ābā ‘father’ does not match the printed Kui tanji in DEDR 3067; its own family or donor remains unresolved."),
    5: ([], "ābā ni is a genitive phrase based on an as-yet unsourced father stem; its grammatical suffix is not an independent etymon."),
    6: ([], "ābā ki is a dative phrase based on an as-yet unsourced father stem."),
    7: ([], "ābā bāhā ṭēkā is a multiword ablative response; the father stem and directional elements need separate analysis."),
    8: ([], "abbar ‘two fathers’ needs analysis of its numeral and kinship morphology before assigning a family."),
    9: ([], "ābāru is a plural of the unsourced father stem ābā."),
    10: ([], "ābār ni is an inflected plural father phrase; its base ābā is not sourced."),
    11: ([], "ābār ki is an inflected plural father phrase; its base ābā is not sourced."),
    12: ([], "ābār bāhā ṭēkā is a multiword plural ablative response, not a single kinship reflex."),
    13: ([], "LSI iyā ‘mother’ lacks a direct full DEDR Kui reflex or verified immediate donor."),
    14: ([], "LSI dādā ‘brother’ lacks a direct full DEDR Kui reflex; superficially similar DEDR 3020 has another meaning."),
    15: ([], "LSI tāmēsā ‘brother’ lacks a direct full DEDR Kui reflex or verified immediate donor."),
    16: (["d5151"], "DEDR 5151 has Kui bāi ‘who’, not ‘sister’; the LSI sister term requires another source or sense analysis."),
    17: ([], "LSI āŋgi ‘sister’ lacks a direct full DEDR Kui reflex or verified immediate donor."),
    20: (["d4756"], "LSI mriē ñju ‘son’ looks related to the male/man family DEDR 4756 but is a two-part response and needs analysis of its second element."),
    21: (["d4764"], "LSI mrāuskā ‘daughters’ may be plural or compound kinship morphology, but the printed family has not been reconciled to the whole form."),
    46: ([], "LSI ḍēŋgā ‘high’ lacks a direct full DEDR Kui reflex or verified donor."),
    48: ([], "LSI sāiṭi ‘down’ lacks a direct full DEDR Kui reflex or securely segmented base."),
    49: ([], "LSI soḍi ‘near’ lacks a direct full DEDR Kui reflex; similarly spelled DEDR items mean ‘run’ or ‘fireplace’."),
    50: ([], "LSI durā ‘far’ may be an Indo-Aryan loan, but its immediate donor was not established."),
}

animal = {
    22: "male horse", 23: "mare", 24: "horses", 25: "mares", 26: "bull", 27: "cow", 28: "bulls", 29: "cows",
    30: "male dog", 31: "bitch", 32: "dogs", 33: "bitches", 34: "he goat", 35: "female goat", 36: "goats",
    37: "male deer", 38: "female deer", 39: "deer",
}
for number, meaning in animal.items():
    candidates = ["d2199"] if number in range(26, 30) else []
    if number == 39:
        reason = "LSI sitāli ‘deer’ lacks a verified full DEDR Kui family or immediate donor."
    else:
        reason = f"The LSI {meaning} response is a multiword or inflected animal expression. Its full noun/modifier or numeral morphology is not yet sourced; no single-parent link represents the entire response."
    if number in range(26, 30):
        reason += " DEDR 2199 supplies Kui kōḍi ‘cow, ox’ but not the additional modifiers."
    holds[number] = (candidates, reason)

for number in range(40, 46):
    holds[number] = (["d3735"], "This good/bad person response is a multiword expression. DEDR 3735 gives Kui negi ‘good’, but the full noun, plural, or negative adjective components are not all verified.")

assert set(proposals).isdisjoint(holds)
assert set(proposals) | set(holds) == set(range(1, 51))
accepted, held = [], []
for number, record in enumerate(records[:50], 1):
    assert record["Source"].startswith("grierson-lsi1928")
    if number in proposals:
        parent, evidence = proposals[number]
        accepted.append(dict(record=record, parent=parent, kind="reflex", citation=record["Source"] + ";dedr[" + parent[1:] + "]", evidence=evidence, status="pending-review"))
    else:
        candidates, reason = holds[number]
        held.append(dict(record=record, candidates=candidates, reason=reason))
(P / "proposed-decisions.json").write_text(json.dumps(dict(status="pending-review", accepted=accepted, held=held), ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")

evidence_ids = {v[0] for v in proposals.values()} | {candidate for candidates, _ in holds.values() for candidate in candidates}
evidence = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt", encoding="utf-8", newline="") as handle:
    for article in csv.DictReader(handle):
        if article["Entry_ID"] in evidence_ids:
            evidence[article["Entry_ID"]] = article
assert set(evidence) == evidence_ids, sorted(evidence_ids - set(evidence))
(P / "primary-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(dict(proposed=len(accepted), held=len(held), unexamined=len(records)-50, evidenceArticles=len(evidence))))
