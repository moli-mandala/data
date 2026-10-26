"""Source-local physical-unit inventory for Nair's Kani chapter, printed 32–33.

This does not install forms.  In particular, Malayalam and Tamil Nadu Kani
comparisons remain separate witnesses rather than inferred Kani etyma.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
PDF = Path(__file__).resolve().parents[7] / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
OUT = HERE / "physical-units-first-reading.jsonl"
STATUS = HERE / "scope-status.json"
units: list[dict] = []


def witness(form: str, gloss: str, scope: str, *, note: str = "") -> dict:
    return {"form": form, "gloss": gloss, "scope": scope, "note": note}


def add(page: int, section: str, number: int, kind: str, entries: list[dict],
        *, claim: str = "", disposition: str = "review_candidate", note: str = "") -> None:
    units.append({
        "source_unit_key": f"nair2013kani:p{page}:{section}:{number}",
        "printed_page": page,
        "pdf_page": page,
        "kind": kind,
        "source_witnesses": entries,
        "source_claim": claim,
        "disposition": disposition,
        "note": note,
        "review": "owner first reading; independent original and final-output audit pending",
    })


add(32, "identity", 1, "documentary", [
    witness("kaaṇi", "", "community name", note="Parenthetical source form with written aa and n underdot; language use not established"),
    witness("Naattukaanis", "", "plains community label", note="Literal printed community plural; no singular form inferred"),
    witness("Malankaanis", "", "forest community label", note="Literal printed community plural; no singular form inferred"),
    witness("malampaala", "", "Kerala Kani autonym", note="Source explicitly says Kanis of Kerala call their speech malampaala"),
    witness("malampaas´ai", "", "Tamil Nadu Kani autonym control", note="Preserves printed acute-like detached mark; its attachment and sibilant value are unresolved"),
], claim="Kerala Kani is said to differ from nearby Tamil Nadu Kani; plains groups largely switched to Malayalam, older forest speakers preserve original speech. The source lists Nedumangadu, Neyyattinkara, Kollam, Kulathuppuzha, Pattomavu, Kallar, Anappara, Mulayara, Kottur and Chonambara; no exact elicitation site.", disposition="documentary", note="Do not install community labels as lexical utterances merely from this paragraph. The self-name is explicit but its lexical gloss is not supplied.")

add(32, "vowel-shift", 0, "phonology context", [], claim="Source says Malayalam a corresponds to Kerala Kani e; Tamil Nadu Kani does not show this feature, while Mullukurumba of Wayanad reportedly does.", disposition="context")
for i, (ml, kn, gloss) in enumerate([
    ("añcu", "eñcu", "five"), ("aṭuppŭ", "eṭuppŭ", "stove"),
    ("aniyan", "eniyan", "younger brother"), ("kaḷḷan", "keḷḷan", "thief"),
    ("maavu", "meevu", "mango tree"), ("maaṭapraavŭ", "meeṭapraavŭ", "dove"),
], 1):
    add(32, "vowel-shift", i, "paired table", [witness(ml, gloss, "Malayalam control"), witness(kn, gloss, "Kerala Kani target")])

add(32, "v-consonant", 0, "phonology context", [], claim="Source says Malayalam/Tamil Nadu v becomes mainly m, rarely b in Kerala Kani, inconsistently among younger speakers.", disposition="context")
for i, (ml, kn, gloss) in enumerate([
    ("vanam", "manam", "forest"), ("veḷutta", "meḷutta", "white"),
    ("viṛṛu", "miṭṭu", "sold"), ("vaalu", "baalu", "tail"),
    ("veḷḷi", "beḷḷi", "silver"),
], 1):
    add(32, "v-consonant", i, "paired table", [witness(ml, gloss, "Malayalam control"), witness(kn, gloss, "Kerala Kani target")])

add(32, "retroflex-to-cc", 0, "phonology context", [], claim="Source contrasts Tamil Nadu Kani netti/kutti with Kerala Kani necci/kucci, describing replacement of ṛṛ with cc.", disposition="context")
for i, (literary, tamil, kerala, gloss) in enumerate([
    ("neṛṛi", "netti", "necci", "forehead"),
    ("kuṛṛi", "kutti", "kucci", "wooden peg"),
], 1):
    add(32, "retroflex-to-cc", i, "prose comparison", [
        witness(literary, gloss, "source comparison word"),
        witness(tamil, gloss, "Tamil Nadu Kani control"),
        witness(kerala, gloss, "Kerala Kani target"),
    ], note="The first source word's exact language is not stated independently; do not force it to Malayalam or Tamil Nadu Kani.")

add(32, "l-to-y", 0, "phonology context", [], claim="Source says underlined l becomes y as in Tamil Nadu Kani.", disposition="context")
for i, (ml, kn, gloss) in enumerate([
    ("kul̲i", "kuyi", "pit"), ("kool̲i", "kooyi", "fowl"),
    ("kal̲uta", "kayuta", "ass"),
], 1):
    add(32, "l-to-y", i, "paired table", [witness(ml, gloss, "Malayalam control", note="Original underlines l; PDF text layer omits underline"), witness(kn, gloss, "Kerala Kani target")])

add(32, "noun-increment", 0, "morphology", [witness("-in", "", "Kerala Kani target"), witness("-n", "", "Kerala Kani target")], claim="-in or -n added after noun stems except stems already ending in -n; source presents this before accusative table.")
for i, (ml, kn, gloss) in enumerate([
    ("aare", "aarine", "who"), ("kool̲iye", "kooyiine", "fowl"),
    ("makaḷe", "makaḷine", "daughter"),
], 1):
    add(33, "noun-increment", i, "paired accusative table", [witness(ml, gloss, "Malayalam control"), witness(kn, gloss, "Kerala Kani target")], claim="Both source glosses explicitly mark accusative.")

add(33, "present", 0, "morphology", [witness("-un̲n̲u", "", "Malayalam control", note="Original underlines both n glyphs"), witness("-ine", "", "Kerala Kani target")], claim="Malayalam present -unnu corresponds to Kani -ine; no general person endings asserted.")
for i, (kn, ml, gloss) in enumerate([
    ("meya peyyine", "mal̲a peyyun̲n̲u", "it is raining"),
    ("neearm utikkine", "suuryan udikkun̲n̲u", "the sun is rising"),
    ("paayine", "paayunnu", "speeds away"),
], 1):
    add(33, "present", i, "whole expression with parenthetic comparison", [witness(kn, gloss, "Kerala Kani target"), witness(ml, gloss, "Malayalam control")], note="Keep source full target expression; Malayalam parenthesis is comparison, not an extra Kani lexical analysis.")

add(33, "personal-endings", 1, "grammatical context", [], claim="Personal terminations are said generally absent in Kani, though a few forms preserve old remnants; no example forms in this paragraph.", disposition="context")
add(33, "negative", 0, "morphology", [witness("-aa", "", "Kerala Kani target", note="Source prints aa; do not split into stem" )], claim="Negative -aa shared with Old Malayalam, Paniya and Adiya; only Kani examples below are target attestations.")
for i, (kn, gloss) in enumerate([
    ("naṭakkaan parraa", "cannot walk, is not able to walk"),
    ("viḷiccaa", "will not call"),
    ("ekkaṛiyaa", "I do not know"),
], 1):
    add(33, "negative", i, "whole expression", [witness(kn, gloss, "Kerala Kani target")], note="Preserve whole attested expression; no inferred infinitive/stem or tense from English beyond source statement.")

for i, (kn, gloss) in enumerate([
    ("eccan", "elder brother"), ("ecci", "elder sister"),
    ("ittaakki", "if not"), ("uumakkiyaŋŋu", "tender tuber"),
    ("uuruvaalu", "fenugreek"), ("eyikuka", "be destroyed"),
    ("uriñña", "taste"),
], 1):
    add(33, "native-words", i, "lexical prose list", [witness(kn, gloss, "Kerala Kani target")], claim="Source calls these native words while noting most Kani vocabulary corresponds to Malayalam.")

add(33, "bibliography", 1, "documentary", [], claim="Source cites P. Somasekharan Nair, Kanikkarude Samsarabhasha (The speech of kanikkar), Samsakara Keralam; no year, volume, or pages printed.", disposition="documentary", note="Auxiliary authority is not independently collated; bibliographic spelling and missing details must remain as printed.")


def main() -> None:
    keys = [u["source_unit_key"] for u in units]
    assert len(keys) == len(set(keys))
    assert {u["printed_page"] for u in units} == {32, 33}
    with OUT.open("w", encoding="utf-8") as out:
        for unit in units:
            out.write(json.dumps(unit, ensure_ascii=False, sort_keys=True) + "\n")
    counts = Counter()
    for unit in units:
        counts[unit["disposition"]] += 1
        for entry in unit["source_witnesses"]:
            counts["witnesses"] += 1
            counts[entry["scope"]] += 1
    STATUS.write_text(json.dumps({
        "status": "complete owner first-reading and physical census; independent source/output audit pending; no canonical installation",
        "source_pdf": str(PDF),
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "printed_pages": [32, 33],
        "physical_units": len(units),
        "counts": dict(sorted(counts.items())),
        "first_reading_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "source_boundary": "Kani chapter starts printed/PDF32 and ends printed/PDF33 before Kurichiya on34; preface/contents are documentary context.",
        "identity_gate": "No existing canonical Kani/Kanikkaran base language or source-local dialect found in current Jambu registry; Kerala Kani and Tamil Nadu Kani must not be silently merged, and source location is broad rather than an exact elicitation site.",
        "installation_deferred": ["independent literal review", "language/variety identity", "bibliography and sound profile", "actual parser", "fresh output sample", "full CLDF build and full tests", "browser refresh only if user requests"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
