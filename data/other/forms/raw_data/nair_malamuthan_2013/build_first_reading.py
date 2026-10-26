"""Source-faithful first reading of the complete Malamuthan chapter (pp. 37–39).

This is a physical-source inventory, not an installation or a phonemic analysis.
Every printed comparison stays beside its target witness. Documentary claims
without an attributed Malamuthan form remain visible in the audit.
"""
import hashlib
import json
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[5] / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
EXPECTED_PDF = "4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af"
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == EXPECTED_PDF

units = []


def add(page, section, target="", gloss="", malayalam="", claim="", role="target", note=""):
    n = 1 + sum(u["printed_page"] == page and u["section"] == section for u in units)
    key = f"nair2013malamuthan:p{page}:{section}:{n:02d}"
    units.append({"unit_key": key, "printed_page": page, "pdf_page": page,
                  "section": section, "source_order": len(units) + 1,
                  "target_form": target, "source_gloss": gloss,
                  "malayalam_control": malayalam, "source_claim": claim,
                  "scope": role, "note": note,
                  "source_pdf_sha256": EXPECTED_PDF})


# The community labels occur in ethnographic English prose and are not called
# Malamuthan-language endonyms. The house term has similarly uncertain lect scope.
add(37, "ethnographic-context", "malamuttaan", "Malamuthans", role="documentary",
    claim="English ethnonym in the author's community description; no endonym claim.")
add(37, "ethnographic-context", "malakaaṛ", "Malakkar", role="documentary",
    claim="Alternative community name in English prose; target-language attribution not stated.",
    note="The visible final r has a dot below; source spells the long vowel with aa, not a macron.")
add(37, "ethnographic-context", "ellam", "house", role="scope-unresolved",
    claim="Author calls houses ellam in an ethnographic passage; the linguistic attribution is not explicit.")
add(37, "ethnographic-context", claim="Nilambur forest, Malappuram and Mukkam, Kozhikode are community locations, not exact elicitation sites.", role="documentary")
add(37, "ethnographic-context", claim="Social-distance and house-burning narrative is documentary context, not a lexical paradigm.", role="documentary")

for mal, tar, gloss in [
    ("varaṭṭe", "maraṭṭe", "let come"), ("viḷikkum", "miḷikkum", "will call"),
    ("vaḷa", "maḷa", "bangle"), ("viiṭŭ", "miitŭ", "house")]:
    add(37, "initial-v-to-m", tar, gloss, mal,
        "Source attributes initial Malayalam v > Malamuthan m; Tachanatu Muppan speech is compared without printed examples.")
add(37, "initial-v-to-m", claim="Tachanatu Muppans are mentioned for relative prevalence only; no separately printed witness.", role="comparison-context")
for tar, gloss in [("maavoo", "come"), ("ibalŭ", "these people"), ("ibaḷu", "this-she")]:
    add(37, "medial-v-b", tar, gloss,
        claim="Source says medial v is preserved or voiced b; it does not assign one rule to each individual form.")
for mal, usual, formal, gloss in [
    ("cekkan", "cekkaŋ", "cekkam", "boy"), ("kaḷḷan", "kallaŋ", "kallam", "thief")]:
    add(37, "final-n", usual, gloss, mal,
        "Source says Malayalam final alveolar n becomes m or ŋ and ŋ is usual.",
        note="Source prose says formal n replaces ŋ, but its printed formal examples end in m.")
    add(37, "final-n-formal", formal, gloss, mal,
        "Source labels this spelling formal speech, while describing an n replacement in prose.",
        note=f"Same printed comparison base as {usual}; formal n/m conflict retained.")
for mal, tar, gloss in [
    ("nakham", "nakan", "nail"), ("roomam", "rooman", "hair"),
    ("palam", "payan", "banana"), ("neeram", "neeran", "time")]:
    add(37, "final-m-to-n", tar, gloss, mal,
        "Source presents this pair under Malayalam final m > Malamuthan n.",
        note="The palam/payan pair has more than the described final consonant difference." if mal == "palam" else "")
add(37, "nr-cluster", "anṛoo", "", claim="Source gives this as a Malamuthan nṛ-cluster example; no gloss is supplied.")
add(37, "nr-cluster", claim="Source says many Malayalam nṛ clusters are pronounced ñj in Malamuthan.", role="comparison-context")
for mal, tar, gloss in [("enṛe", "eñja", "my"), ("ninṛe", "niñja", "your")]:
    add(37, "nr-to-nj", tar, gloss, mal, "Author's paired nṛ-to-ñj comparison.")
add(37, "alveolar-to-palatal", claim="The doubled alveolar comparison is introduced here and exemplified at the top of p. 38.", role="cross-page-context")

for mal, tar, gloss in [("neṛṛi", "necci", "forehead"), ("paaṛṛa", "paaca", "cockroach"), ("kuṛṛi", "kucci", "peg")]:
    add(38, "alveolar-to-palatal", tar, gloss, mal, "Source says doubled Malayalam alveolars become palatal in Malamuthan.")
for mal, tar, gloss in [("ari", "ayri", "rice"), ("nari", "nayri", "fox"),
                        ("vari", "mayri", "row"), ("kuru", "kuyri", "seed")]:
    add(38, "y-insertion", tar, gloss, mal,
        "Source says y is inserted and fully pronounced; its description 'after word initial vowels' does not fit every printed pair.")
for mal, tar, gloss in [("mala", "maya", "rain"), ("pula", "puya", "stream")]:
    add(38, "l-to-y", tar, gloss, mal, "Author's Malayalam l > Malamuthan y comparison.")
for mal, tar, gloss in [("kuli", "kuyj", "hole"), ("pulu", "puyju", "worm")]:
    add(38, "l-to-yj", tar, gloss, mal, "Author's Malayalam l > Malamuthan yj comparison.")
add(38, "l-to-v", "kavuttu", "neck", "kaluttu", "Author calls Malayalam l > Malamuthan v rare.")
for mal, tar, gloss in [("kayyu", "kayju", "hand"), ("miiśa", "miyja", "moustache")]:
    add(38, "yy-s-to-yj", tar, gloss, mal, "Source says Malayalam yy or s sometimes becomes Malamuthan yj.")

for tar, mal, gloss, feature in [
    ("oon", "avan", "he", "3sg m"), ("ooḷ", "avaḷ", "she", "3sg f"),
    ("ooḷu", "avaṛ", "they", "3pl"), ("iban", "ivan", "he", "3sg m"),
    ("ibaḷu", "ivaḷ", "she", "3sg f"), ("ibaḷu", "ivar", "they", "3pl")]:
    add(38, "pronouns", tar, gloss, mal, "Source pronoun table; feature follows the explicit English gloss.", note=feature)
for tar, mal, gloss, feature in [
    ("oonja", "avanṛe", "he", "gen"), ("ibañja", "ivanṛe", "he", "gen"),
    ("eñja", "enṛe", "my", "gen poss"), ("eñña", "enne", "I", "acc"),
    ("eññooṭu", "ennooṭu", "to me", "soc")]:
    add(38, "pronoun-cases", tar, gloss, mal,
        "Source explicitly labels the case in parentheses; he (Gen.) is retained verbatim.", note=feature)
for mal, tar in [("aa", "ayya"), ("ii", "iyya")]:
    add(38, "demonstratives", tar, "", mal, "The source says these Malayalam demonstratives become Malamuthan forms.")
add(38, "demonstratives", "-nee", "far remote", claim="Source explicitly calls this a far-remote demonstrative marker.")
for tar in ["itu", "atu", "neetu", "ibaḷu", "ooḷu", "neebaḷu"]:
    add(38, "demonstrative-series", tar, "", claim="Source lists three categories of third-person demonstratives, in two semicolon-separated series; no one-to-one glosses.")
add(38, "case-link", "-in", "", claim="Source names this Malamuthan link morph before accusative -e on the interrogative base.")
add(38, "case-link", "aarine", "who (Acc.)", "aare",
    "Malayalam aare is compared with Malamuthan aar-in-e; interrogative base -aarŭ and accusative -e are separately mentioned.")
add(38, "case-link", "-aarŭ", "", role="malayalam-control",
    claim="The printed interrogative base is explicitly Malayalam, not a separate Malamuthan lemma.")
add(38, "case-link", "-e", "accusative", role="malayalam-control",
    claim="The printed directly attaching suffix belongs to the Malayalam construction; source does not separately attest a Malamuthan -e row.")
for mal, tar, gloss in [("peṇṇinṛe", "peṇṇuṭa", "of the girl"), ("muukkinṛe", "muukkuṭa", "of the nose")]:
    add(38, "genitive-no-link", tar, gloss, mal, "Source says Malamuthan lacks a link morph present in the compared Malayalam genitive forms.")
add(38, "genitive-ca", "-ca", "genitive", claim="Source explicitly identifies -ca as a Malamuthan genitive suffix.")
for tar, mal, gloss in [("iñca", "enṛe", "my"), ("makaca", "makanṛe", "son’s"), ("kaṇṇica", "kanninṛe", "eye’s")]:
    add(38, "genitive-ca", tar, gloss, mal, "Source gives this as a Malamuthan genitive form with Malayalam comparison.")
add(38, "past-uttu", "-uttu", "past tense", claim="Source calls this a Malamuthan past-tense suffix and reports it also in Paniya without a Paniya form.")
for tar, mal, gloss in [("uututtoo", "uuti", "blew"), ("keṭṭuttoo", "keṭṭi", "tied"), ("keeṛuttoo", "keeṛi", "climbed")]:
    add(38, "past-uttu", tar, gloss, mal, "Source illustrates Malamuthan past -uttu with a Malayalam form.")
add(38, "past-uttu", claim="Paniya is mentioned as also having -uttu, but no Paniya witness is printed.", role="comparison-context")

add(39, "negative", "illa", "negative", "alla; illa",
    "Source says Malayalam alla and illa both correspond to Malamuthan illa; no separate equivalent for alla.")
for tar, mal, gloss in [("oonillaanṛoo", "avanalla", "not he"), ("ooḷillaanṛoo", "avaḷalla", "not she")]:
    add(39, "negative", tar, gloss, mal, "Author's paired negative-construction example.")
add(39, "bibliography", claim="Reports P. Somasekharan Nair, Malamuthante Malayalam, Samskarakeralam, January–March 1981, Trivandrum; reported authority only.", role="bibliographic-context")

keys = [u["unit_key"] for u in units]
assert len(keys) == len(set(keys))
assert all(unicodedata.is_normalized("NFC", u["target_form"]) for u in units)
assert {u["printed_page"] for u in units} == {37, 38, 39}
(HERE / "physical-units-first-reading.jsonl").write_text(
    "".join(json.dumps(u, ensure_ascii=False) + "\n" for u in units))
status = {"status": "complete-physical-first-reading-pending-independent-review",
          "source_pdf_sha256": EXPECTED_PDF, "printed_pages": [37, 38, 39],
          "physical_units": len(units),
          "scope_counts": {s: sum(u["scope"] == s for u in units) for s in sorted({u["scope"] for u in units})},
          "page_counts": {str(p): sum(u["printed_page"] == p for u in units) for p in (37, 38, 39)},
          "source_stage_installation": "pending"}
(HERE / "first-reading-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(status, ensure_ascii=False))
