#!/usr/bin/env python3
"""Audit DDSA's Sheth transcription before installation.

Run with the cache acquired by sheth.py. This intentionally has no --install
switch until reference, morphology and transcription review gates are complete.
It emits proposed rich rows and a lossless article audit into a scratch directory.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[4]
SOURCE = "sheth1923"
DATE = "20260911"

# Verified against the printed sanketa-suci, frontmatter PDF page 2.
GRAMMAR = {
    "अ": "indecl", "अक": "verb intr", "उभ": "verb tr intr",
    "कर्म": "verb pass", "कवकृ": "participle pres pass", "कृ": "participle",
    "क्रि": "verb", "क्रिवि": "adv", "त्रि": "adj", "न": "noun n",
    "पुं": "noun m", "पुंन": "noun mn", "पुंस्त्री": "noun mf",
    "प्रयो": "verb caus", "ब": "pl", "भकृ": "participle fut",
    "भवि": "verb fut", "भूका": "verb pret", "भूकृ": "participle pp",
    "वकृ": "participle pres", "वि": "adj", "स": "pron",
    "संकृ": "conjunctive-participle", "सक": "verb tr", "स्त्री": "noun f",
    "स्त्रीन": "noun fn", "हेकृ": "participle",
}
LECTS = {"अप": "Ap", "अशो": "As", "शौ": "Pk", "मा": "Pk", "पै": "Pk", "चूपै": "Pk"}
MORPHOLOGY = re.compile(r"(?<!\w)(?:कवकृ|संकृ|भूकृ|वकृ|भकृ|हेकृ|प्रयो|भवि|भूका|कर्म)\s*\.")
DEV = re.compile(r"[\u0900-\u097f]")
QUOTATION = re.compile(r"['‘][^'’]*['’]")


def split_examples(text, crossreference_only=False):
    """A quoted gloss is not an example: e.g. 'देखो' अर्थ ... means 'look'."""
    examples = []
    def replace(match):
        before = text[:match.start()].rstrip()
        if before.endswith(";") or (crossreference_only and not before):
            examples.append(match.group())
            return ""
        return match.group()
    return QUOTATION.sub(replace, text).strip("।.;, "), examples


def plain(value):
    if hasattr(value, "get_text"):
        value = value.get_text(" ", strip=True)
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", str(value))).strip()


def grammar(text):
    """Consume only exact printed labels; leave mis-tagged definitions visible."""
    text = plain(text)
    labels = re.findall(r"\((अप|अशो|शौ|मा|पै|चूपै)\)", text)
    rest = re.sub(r"\((?:अप|अशो|शौ|मा|पै|चूपै)\)", "", text).strip()
    parts = [p for p in re.split(r"[.\s]+", rest) if p]
    if all(p in GRAMMAR for p in parts):
        return list(dict.fromkeys(t for p in parts for t in GRAMMAR[p].split())), labels, ""
    return [], labels, rest


def parse_article(markup, page, ordinal):
    key = f"{SOURCE}:p{page}:e{ordinal}"
    node = BeautifulSoup(markup, "html.parser")
    heads = node.find_all("hw")
    audit = {"entry_key": key, "page": page, "ordinal": ordinal, "raw_markup": markup,
             "status": "proposed", "review": [], "rows": [], "crossreferences": [],
             "printed_references": [], "grammar_labels": [], "lect_labels": [],
             "etymological_match": "not_attempted", "etymology_segments": [],
             "unresolved_subentry_regions": []}
    if len(heads) != 1:
        audit.update(status="excluded", review=["structure:headword-count"])
        return audit
    pairs = [plain(b) for b in heads[0].find_all("b")]
    if len(pairs) % 2 or not pairs:
        audit.update(status="excluded", review=["structure:unpaired-headwords"])
        return audit
    forms = list(zip(pairs[::2], pairs[1::2]))
    if any(not DEV.search(native) or DEV.search(roman) or not roman for native, roman in forms):
        audit.update(status="excluded", review=["transcription:invalid-headword-pair"])
        return audit
    heads[0].decompose()
    # DDSA encodes some language labels as references, including Apabhramsha.
    # Convert only a whole, exact language label; work citations such as
    # '(अप 12)' retain their source-reference meaning and their scope.
    for reference in node.find_all("reference"):
        if re.fullmatch(r"\((अप|अशो|शौ|मा|पै|चूपै)\)", plain(reference)):
            reference.name = "category"
    # The printed brace joins alternate heads; it is layout, not a definition.
    for text_node in list(node.find_all(string=True)):
        if plain(text_node) == "}":
            text_node.extract()
    etymologies = []
    for e in node.find_all("etymology"):
        text = plain(e)
        audit["etymology_segments"].append(text)
        # A compound's donor inside a numbered definition is not the donor of
        # the headword. Keep it in the lossless audit until the child is parsed.
        inside_sense = e.find_parent(["definition", "meaning"]) is not None
        if not inside_sense and (not text.startswith("°") or forms[0][0].startswith("°")):
            etymologies.append(text)
        else:
            audit["review"].append("etymology:subentry-scope-pending")
        e.decompose()
    tags, lects = [], []
    sense_grammar = {}
    for e in node.find_all("category"):
        text = plain(e)
        audit["grammar_labels"].append(text)
        parsed, labels, remainder = grammar(text)
        owner = e.find_parent(["definition", "meaning"])
        if owner is not None:
            while owner.find_parent(["definition", "meaning"]) is not None:
                owner = owner.find_parent(["definition", "meaning"])
            local = sense_grammar.setdefault(id(owner), {"tags": [], "lects": []})
            local["tags"].extend(parsed); local["lects"].extend(labels)
        else:
            tags.extend(parsed); lects.extend(labels)
        if remainder:
            audit["review"].append("grammar:unrecognized-or-mistagged:" + remainder)
            # Keep source text in place instead of silently discarding malformed tags.
            e.unwrap()
        else:
            e.decompose()
    head_references, sense_references = [], {}
    for e in node.find_all("reference"):
        text = plain(e)
        audit["printed_references"].append(text)
        owner = e.find_parent(["definition", "meaning"])
        if owner is not None:
            while owner.find_parent(["definition", "meaning"]) is not None:
                owner = owner.find_parent(["definition", "meaning"])
            sense_references.setdefault(id(owner), []).append(text)
        else:
            head_references.append(text)
        e.decompose()
    head_crossreferences, sense_crossreferences = [], {}
    for e in node.find_all("var"):
        text = plain(e)
        audit["crossreferences"].append(text)
        owner = e.find_parent(["definition", "meaning"])
        if owner is not None:
            while owner.find_parent(["definition", "meaning"]) is not None:
                owner = owner.find_parent(["definition", "meaning"])
            sense_crossreferences.setdefault(id(owner), []).append(text)
        else:
            head_crossreferences.append(text)
        e.decompose()
    audit["lect_labels"] = lects
    if any(label == "चूपै" for label in lects):
        audit["review"].append("dialect:registration-pending")
    languages = list(dict.fromkeys(LECTS[l] for l in lects))
    lang = languages[0] if len(languages) == 1 else "Pk"
    if len(languages) > 1:
        audit["review"].append("dialect:multiple-source-lects")
    definitions = [e for e in node.find_all(["definition", "meaning"])
                   if e.find_parent(["definition", "meaning"]) is None]
    if definitions and head_crossreferences:
        audit["review"].append("scope:crossreference-outside-numbered-senses")
    audit['unscoped_references'] = head_references if definitions else []
    # A numbered definition can contain form-specific POS or paradigms. Preserve
    # its sense boundary and flag unresolved structure, never spread that label
    # back onto unrelated senses.
    units = []
    for i, e in enumerate(definitions, 1):
        text = plain(e)
        number = re.match(r"^([०-९0-9]+(?:[-–][०-९0-9]+)?)\s*[.।]?\s*", text)
        local = dict(sense_grammar.get(id(e), {}))
        local['crossreferences'] = sense_crossreferences.get(id(e), [])
        local['references'] = sense_references.get(id(e), [])
        units.append((i, text[number.end():] if number else text,
                      number.group(1) if number else "", local))
        e.decompose()
    outside = plain(node)
    notes = []
    if units and outside.strip("।.;, "):
        notes.append(outside)
        audit["review"].append("scope:prose-outside-numbered-senses")
    if not units:
        units = [(1, outside, "", {'references': head_references})]
    for sense, definition, printed_sense, local_grammar in units:
        sense_review = []
        # A quotation following an otherwise definition-less See entry is a
        # usage example, not the meaning of that headword (e.g. pōma, p. 618).
        crossreferences = head_crossreferences + local_grammar.get('crossreferences', [])
        gloss, quotes = split_examples(definition, bool(crossreferences))
        if re.fullmatch(r"(?:(?:ऊपर|नीचे|आगे|पीछे)\s+)?देखो(?:\s+.*)?", gloss):
            crossreferences = crossreferences + [gloss]
            gloss = ""
        # DDSA leaves compounds inside <definition>. Do not put a compound's
        # definition on its parent. The child region stays lossless and flagged
        # for expansion; this proposal cannot be installed before that gate.
        compound = re.search(r'(?<!\S)°(?=[\u0900-\u097f])', gloss)
        if compound:
            audit['unresolved_subentry_regions'].append({
                'sense': sense, 'text': gloss[compound.start():]})
            gloss = gloss[:compound.start()].rstrip(' ।.;,')
            sense_review.append('structure:embedded-subentry-or-reference')
        if MORPHOLOGY.search(gloss):
            sense_review.append("morphology:embedded-forms")
        if re.search(r"[°=]|(?:^|[।;])\s*(?:देखो|देखिये)", gloss):
            sense_review.append("structure:embedded-subentry-or-reference")
        inline = re.match(r"^(पुंन|पुंस्त्री|स्त्रीन|स्त्री|पुं|न|वि|सक|अक|अ)\.\s*", gloss)
        sense_tags = local_grammar.get("tags") or tags[:]
        sense_lects = local_grammar.get("lects") or lects
        sense_languages = list(dict.fromkeys(LECTS[l] for l in sense_lects))
        sense_lang = sense_languages[0] if len(sense_languages) == 1 else lang
        if len(sense_languages) > 1 or "चूपै" in sense_lects:
            sense_review.append("dialect:sense-specific-label-review")
        if inline:
            # New POS is confined to this sense. It overrides the head's category.
            sense_tags = GRAMMAR[inline.group(1)].split()
            gloss = gloss[inline.end():]
        for variant, (native, roman) in enumerate(forms, 1):
            rowkey = f"{key}:s{sense}:v{variant}"
            review = list(dict.fromkeys(audit["review"] + sense_review))
            if "�" in roman:
                review.append("transcription:replacement-character")
            citation = f"{SOURCE}[DDSA web page {page}, article {ordinal}"
            if printed_sense:
                citation += f", sense {printed_sense}"
            citation += "]"
            rowtags = list(dict.fromkeys(sense_tags + (["uncertain"] if review else [])))
            # Source comparisons are prose until a unique, semantically compatible
            # target is reviewed. No cognacy is inferred from Sanskrit equivalents.
            etymology = "; ".join(dict.fromkeys(etymologies))
            row = [sense_lang, "", roman, gloss, native, "", " ".join(notes + quotes),
                   citation, "", etymology, rowkey,
                   f"{key}:s{sense}:v1" if variant > 1 else "", "", "", " ".join(rowtags)]
            audit["rows"].append({"row": row, "review": review,
                                  "printed_sense": printed_sense, "lect_labels": sense_lects,
                                  "references": local_grammar.get('references', []),
                                  "crossreferences": crossreferences})
    return audit


def propose(cache, output, allow_partial=False, seed=20260911):
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest["complete"] and not allow_partial:
        raise ValueError("Snapshot incomplete; use --allow-partial only for parser development")
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "audit.jsonl"
    counts = collections.Counter()
    reviews = collections.Counter()
    samples = []
    randomizer = random.Random(seed)
    with audit_path.open("w") as af, (output / "proposed.csv").open("w") as cf:
        writer = csv.writer(cf)
        for page in manifest["pages"]:
            raw = (cache / f"{page['page']:04d}.html").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == page["sha256"]
            soup = BeautifulSoup(raw.decode("utf-8"), "html.parser")
            heads = soup.find_all("hw")
            assert len(heads) == page["headwords"]
            for ordinal, head in enumerate(heads, 1):
                record = parse_article(str(head.parent), page["page"], ordinal)
                af.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["articles"] += 1
                counts[record["status"]] += 1
                for proposed in record["rows"]:
                    writer.writerow(proposed["row"])
                    counts["rows"] += 1
                    reviews.update(proposed["review"])
                if len(samples) < 20:
                    samples.append(record)
                else:
                    n = randomizer.randrange(counts["articles"])
                    if n < 20:
                        samples[n] = record
    report = {"source": SOURCE, "snapshot_complete": manifest["complete"],
              "counts": dict(counts), "review_classes": dict(reviews), "audit_seed": seed,
              "installation_ready": False,
              "deferred_gates": ["reference-resolution", "morphology", "dialect-registration",
                                  "sound-profile", "source-audit", "tests", "build", "browser-QA"]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "sample.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/sheth-ddsa-20260911")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/sheth-proposal-20260911")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    report = propose(args.cache, args.output, args.allow_partial, args.seed)
    print(json.dumps({k:v for k,v in report.items() if k != "review_classes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
