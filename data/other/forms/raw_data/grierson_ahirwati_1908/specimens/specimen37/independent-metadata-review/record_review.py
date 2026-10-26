"""Complete independent metadata review; guarded proposals only, no installation."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SPECIMEN = HERE.parent
PACKAGE = SPECIMEN.parents[1]
ROOT = next(p for p in HERE.parents if (p / "data/make_cldf.py").exists())
OWNER = SPECIMEN / "independent-review/metadata-proposal-revision2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def clean(value):
    return value.strip(' .;,?!‘’“”"').lower()


def write(name, rows):
    (HERE / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def review():
    path = OWNER / "candidate.csv"
    assert sha(path) == "fcd708e8edb9e164f28e8813bf69026bda262fa56b739fdde36814cd09862075"
    auditpath = OWNER / "candidate-audit.jsonl"
    assert sha(auditpath) == "df30840857b1e5899af75ecc436d91a29ed99904ace2624d768abbbef000af3f"
    source = list(csv.reader(path.open(newline="")))
    audit = records(auditpath)
    assert len(source) == len(audit) == 390
    assert [r[10] for r in source] == [r["source_unit_key"] for r in audit]
    assert all(r == a["row"] for r, a in zip(source, audit))
    grammarpath = PACKAGE / "grammar/child-witnesses-reviewed.jsonl"
    assert sha(grammarpath) == "5ac645b13ae1a3132ac094431fbce9376037705f93c0c59d9861e6e7ef999ee7"
    grammar = records(grammarpath)
    assert len(grammar) == 98
    roman = {r["key"]: r for r in records(SPECIMEN / "roman-atoms-first-reading.jsonl")}
    native = {r["key"]: r for r in records(SPECIMEN / "native-atoms-first-reading.jsonl")}
    assert len(roman) == 367 and len(native) == 365
    aliases = records(OWNER / "physical-key-aliases.jsonl")
    assert collections.Counter(r["physical_key"] for r in aliases) == collections.Counter([*roman, *native])
    assert len(records(OWNER / "reviewed-printer-continuations.jsonl")) == 4
    shared = records(OWNER / "shared-native-groups.jsonl")
    assert len(shared) == 1 and shared[0]["full_native_form"] == "बाँट-दियो।"
    wholes = {r["first_word_key"] + ":expression": r for r in records(SPECIMEN / "whole-utterances-first-reading.jsonl")}
    assert len(wholes) == 27
    decisions, corrections, precision_holds, projected = [], [], [], []
    protected = [0, 2, 3, 5, 8, 9, 10, 11, 12, 13]
    for ordinal, (old, a) in enumerate(zip(source, audit), 1):
        row = list(old)
        key, form, gloss = old[10], clean(old[2]), clean(old[3])
        tags = old[14].split()
        reasons, evidence = [], []
        # This is the actual production Row schema, not the author's displaced layout.
        assert old[4] == old[2]
        row[1], row[4] = "", old[1]
        reasons.append("Production schema repair: Parameter_ID is blank; exact intended source Native moves from index1 to index4. Roman Form/Original is unchanged.")
        kind = "whole" if key.endswith(":expression") else "lexical"
        if kind == "whole":
            assert old[3] == a["printed_interlinear_analysis"]
            assert old[2] == wholes[key]["proposed_whole_form"]
            row[6] = "Printed interlinear English analysis."
            if "?" in old[2] and "interr" not in tags:
                tags.append("interr")
                reasons.append("Original quotation has an explicit question mark; interr supplements sentential.")
            evidence.append("Original pp234–236 source punctuation and all27 accepted anchored sentence/speech scopes; nested quotations are deliberately overlapping.")
        else:
            if gloss == "that" and form in {"ak", "ki", "kē"}:
                tags = [t for t in tags if t not in {"pron", "demonstrative"}]
                reasons.append("Reporting/complement-clause connector in the exact source context; English that does not license demonstrative/pronoun tags for ak/ki/kē.")
            elif gloss == "that" and form in {"sō", "wō", "wāh"}:
                tags = [t for t in tags if t != "conj"]
                reasons.append("Exact source occurrence is a demonstrative in the quoted noun/referential phrase, not the reporting connector; remove conjunctive tag caused by shared English that.")
            if form == "wāhĩ" and gloss == "that-very":
                tags = [t for t in tags if t != "adv"] + ["pron", "demonstrative", "emph"]
                reasons.append("That-very modifies/references the same country in the source clause; demonstrative with explicit emphatic meaning, not temporal/manner adverb.")
            if gloss in {"in-opinion", "so-many"}:
                tags = [t for t in tags if t != "adv"]
                reasons.append("Keep nominal opinion or quantifier so-many analysis; no separate adverb POS follows from English phrase syntax.")
            if form == "chāw-chūchᵃḷā" and gloss == "rejoicing":
                tags = [t for t in tags if t != "adv"] + ["uncertain"]
                reasons.append("Rejoicing is a complement of to-make in the printed clause. English -ing alone cannot establish an adverb or participial subtype; lexical function remains qualified.")
            if gloss == "might-have-done":
                tags = [t for t in tags if t != "pret"] + ["modal", "uncertain"]
                reasons.append("Printed might-have-done is qualified irrealis/modal analysis, not an explicit target preterite classification; remove unsupported pret while retaining printed meaning.")
                row[6] = "The interlinear analysis reads ‘might-have-done’; the grammatical subtype of this source occurrence is not specified."
            if gloss in {"foreign-countries-to", "evil-behaviour-in", "labourer-to"}:
                tags.append("noun")
                if gloss == "foreign-countries-to": tags.append("pl")
                reasons.append("Printed lexical head is an explicit noun, and foreign-countries explicitly plural; retain existing relational tags without manufacturing a morpheme split.")
            if gloss in {"this-one-to", "this-one-of"}:
                tags += ["pron", "demonstrative"]
                reasons.append("Explicit interlinear this-one supplies a demonstrative-pronominal meaning; existing relational tag remains source-scoped.")
            if form == "apᵃṇā":
                tags += ["m", "obl"]
                reasons.append("p51 explicitly identifies exact apᵃṇā as masculine oblique; not inferred from the English possessor or natural gender.")
                evidence.append("lsi9.2:p51:own-pronoun-alternatives:w3")
            if form == "ghari":
                tags.append("m")
                reasons.append("p51 explicitly places exact ghari in the masculine consonant-final locative class; source grammatical gender, not English house gender.")
                evidence.append("lsi9.2:p51:consonant-noun-locative:w2")
            if form == "bōlyō":
                tags.append("participle")
                reasons.append("p237 explicitly calls bōlyō a past participle; same literal source form, with lexical occurrence citation retained separately.")
                evidence.append("lsi9.2:p237 mixed-lect linguistic commentary: bōlō/bōlyō are the explicitly named past participles, unlike Bangaru bōlā.")
            if row[6].startswith("Grierson gives apᵃṇū"):
                row[6] = "Grierson gives apᵃṇū/apᵃṇō as ‘own’ and apᵃṇā as the masculine oblique."
            if "agent/by" in row[6] or "agent case" in row[6]:
                row[6] = "The printed interlinear analysis reads ‘" + old[3] + "’."
                reasons.append("Preserve exact printed by analysis without falsely transferring p51 nē̃ morphology to taĩ or equating every by phrase with an agent-case label.")
            if "pret" in tags and form not in {"bōlyō", "thō", "thā"} and "thō" not in form:
                precision_holds.append(dict(key=key, exact_form=old[2], exact_gloss=old[3], present_tags=old[14], concern="Past English interlinear reading supports a past meaning, but the source gives no occurrence-specific preterite morphology label. Root should adjudicate whether broad pret discovery tags are intended; no suffix/root/tense derivation proposed."))
            if "pass" in tags and form != "karᵃṇō":
                precision_holds.append(dict(key=key, exact_form=old[2], exact_gloss=old[3], present_tags=old[14], concern="Passive English wording alone does not establish target passive morphology. Preserve source English interpretation; intrinsic voice classification needs source-scoped adjudication. dīṇū̃ must not be harmonized with p51 diṇū̃."))
        tags = list(dict.fromkeys(tags))
        row[14] = " ".join(tags)
        # Explicit auxiliary analysis citations, not additional invented attestations.
        pages = sorted({r["printed_page"] for r in a["source_record"]["Roman"]})
        if any("p51" in reason or "p51" in note for reason in reasons for note in [row[6]]) or evidence and any(":p51:" in str(e) for e in evidence):
            pages = sorted(set(pages) | {51})
        if form == "bōlyō" and kind == "lexical": pages = sorted(set(pages) | {237})
        if len(pages) > len({r["printed_page"] for r in a["source_record"]["Roman"]}):
            row[7] = "grierson1908ahirwati[pp. " + " and ".join(str(p) for p in pages) + ", specimen 37 and grammatical analysis]"
        for i in protected: assert row[i] == old[i], (key, i)
        # Raw source Native retained exactly, including the deliberate two blank bā̃ṭ/diyō rows.
        assert row[4] == old[1] and row[1] == ""
        changed = [i for i in range(15) if row[i] != old[i]]
        decision = dict(ordinal=ordinal, source_unit_key=key, kind=kind, exact_old_row=old, proposed_row=row,
                        changed_columns=changed, reasons=reasons, source_evidence=evidence,
                        grammar_scope="Interpretation of exact printed aligned English and explicit source grammar; no inferred roots, object case from me/him, participial subtype from generic -ing, or natural gender.",
                        preserved_typed_uncertainties=a["typed_uncertainties"], status="independent keyed proposal; root adjudication and owner regeneration pending")
        decisions.append(decision)
        corrections.append(dict(source_unit_key=key, old_row=old, new_row=row, changed_columns=changed, reasons=reasons))
        projected.append(row)
    assert len(projected) == 390 and [r[10] for r in projected] == [r[10] for r in source]
    write("all390-keyed-decisions.jsonl", decisions)
    write("protected-field-metadata-proposals.jsonl", corrections)
    write("tense-voice-precision-review-holds.jsonl", precision_holds)
    with (HERE / "protected-field-proposed.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows(projected)
    r3path = SPECIMEN / "independent-review/projection-revision3/candidate.csv"
    assert sha(r3path) == "a8141e9f2d2291d7f50929b3cb9fc3574cbf77669a2a19a92f4b4ee0d670fd99"
    r3rows = list(csv.reader(r3path.open(newline="")))
    assert len(r3rows) == 390
    r3proposals = []
    for old, r3, decision in zip(source, r3rows, decisions):
        assert r3[1] == "" and r3[4] == old[1]
        assert all(r3[i] == old[i] for i in range(15) if i not in {1, 4})
        new = decision["proposed_row"]
        changed = [i for i in range(15) if new[i] != r3[i]]
        assert set(changed) <= {6, 7, 14}
        if changed:
            r3proposals.append(dict(source_unit_key=r3[10], old_row=r3, new_row=new, changed_columns=changed,
                                   reasons=decision["reasons"][1:], source_evidence=decision["source_evidence"],
                                   preserved_typed_uncertainties=decision["preserved_typed_uncertainties"]))
    write("revision3-guarded-metadata-proposals.jsonl", r3proposals)
    projection = dict(status="PASS independent exact projection-only continuity check; metadata proposals remain unapproved", rows=390,
                      revision2_sha256=sha(path), revision3_sha256=sha(r3path),
                      protected_indices=[i for i in range(15) if i not in {1,4}],
                      metadata_proposal_rows=len(r3proposals), metadata_only_allowed_indices=[6,7,14],
                      proposal_sha256=sha(HERE/"revision3-guarded-metadata-proposals.jsonl"))
    (HERE/"revision3-projection-continuity-review.json").write_text(json.dumps(projection, ensure_ascii=False, indent=2)+"\n")
    report = dict(status="FAIL current candidate projection and metadata classes; exact guarded proposals prepared, not accepted installation output", counts=dict(rows=390, lexical=363, whole=27, roman_physical=367, native_physical=365, aliases=732, printer_joins=4, grammar_reviewed=98, native_shared_groups=1, projection_repairs=390,
                         additional_metadata_rows=sum(any(i not in {1,4} for i in d["changed_columns"]) for d in decisions), tense_voice_precision_holds=len(precision_holds)),
                  input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [path, auditpath, grammarpath, OWNER/"status.json", OWNER/"physical-key-aliases.jsonl", OWNER/"reviewed-printer-continuations.jsonl", OWNER/"shared-native-groups.jsonl", SPECIMEN/"first-reading-freeze.json", ROOT/"data/make_cldf.py"]},
                  original_evidence=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in sorted((ROOT/"tmp/jambu-next-source/ahirwati-native-specimens-20260927").glob("specimen37-p23[456]-owner-*.png"))],
                  protected_columns=protected, allowed_proposed_columns=[1,4,6,7,14], source_native_exact_relocated=True, literal_form_gloss_order_keys_unchanged=True,
                  findings=["Actual production schema index1 Parameter_ID/index4 Native is violated by all390 frozen rows; exact source-native relocation repairs it without adding etymons.", "English that cross-classifies reporting connectors and demonstratives; keyed context repairs supplied.", "Nominal/quantified readings were classified as adverbs from English phrase lists; specific corrections/hold supplied.", "might-have-done does not license an unqualified preterite subtype; broader English-past/passive precision questions remain explicit for root.", "Explicit source m/obl and past-participle descriptions, missing nominal/demonstrative functions, question scope and source grammar citation coverage reviewed.", "All27 whole source boundaries and exact printed interlinear meaning are retained; overlapping speech and native differences are not silently merged."],
                  review_authorship="Independent metadata/actual-CSV review of Asur-owned390-row proposal. Roman first transcription was authored by this reviewer and previously fully independently verified by Asur; this is not a fresh independent literal-transcription audit.",
                  projection_revision3=projection,
                  output_hashes={name:sha(HERE/name) for name in ["all390-keyed-decisions.jsonl", "protected-field-metadata-proposals.jsonl", "tense-voice-precision-review-holds.jsonl", "protected-field-proposed.csv", "revision3-guarded-metadata-proposals.jsonl", "revision3-projection-continuity-review.json"]}, canonical_changes=False)
    (HERE/"independent-full-metadata-review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    review()
