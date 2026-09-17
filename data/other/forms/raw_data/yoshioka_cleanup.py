#!/usr/bin/env python3
"""Rebuild Yoshioka from a pinned, font-decoded, positioned source snapshot.

The embedded Gentium cmap, not OCR, supplies missing PDF Unicode mappings.
Legacy keys are anchored to the original page/line; additions use physical keys.
Run --extract once with the source PDF, then run offline from the snapshot.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PACKAGE = HERE / "yoshioka_2026"
SNAPSHOT = PACKAGE / "source-lines.jsonl.gz"
PDF_SHA = "020d343e067d7e46e91eacbf9d30f79844625a7f25f3320772002dc5361e4574"
EASTERN_TAG = 'dialect:Bur:Yoshioka-EB:Eastern%20Burushaski'


def dialect_tag(code):
    return f'dialect:Bur:Yoshioka-{code}:{quote(legacy.DIALECTS[code], safe="")}'
# Source-image reviewed continuation lines that the old left-margin heuristic
# promoted to dictionary entries. Their text belongs to the preceding article.
CONTINUATIONS = {337, 469, 503, 522, 575, 803, 1022, 1101, 1546, 1672,
                 1722, 1732, 1929, 1954, 2061, 2364, 2446, 2506, 2593, 2655}
SPEC = importlib.util.spec_from_file_location("yoshioka_legacy", HERE / "yoshioka.py")
legacy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = legacy
SPEC.loader.exec_module(legacy)


def canonical(text):
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip())


def decoded_resource_manager():
    from pdfminer.pdfinterp import PDFResourceManager
    from pdfminer.pdffont import PDFCIDFont, TrueTypeFont
    from pdfminer.pdftypes import dict_value, stream_value
    from pdfminer.psparser import literal_name

    class Manager(PDFResourceManager):
        def get_font(self, objid, spec):
            font = super().get_font(objid, spec)
            if (isinstance(font, PDFCIDFont) and "Gentium" in str(font.fontname)
                    and not getattr(font, "yoshioka_decoded", False)):
                descriptor = dict_value(spec.get("FontDescriptor", {}))
                if "FontFile2" in descriptor:
                    if literal_name(spec.get("CIDToGIDMap")) != "Identity":
                        raise ValueError("Unexpected Gentium CID-to-glyph map")
                    data = stream_value(descriptor["FontFile2"]).get_data()
                    font.unicode_map = TrueTypeFont(str(font.fontname), io.BytesIO(data)).create_unicode_map()
                    font.yoshioka_decoded = True
            return font
    return Manager()


def positioned_lines(page, number):
    """Group baselines, attach raised numbers and combining marks before sorting."""
    chars = [dict(c) for c in page.chars if 75 < c["top"] < 752
             and c.get("size", 0) > 2 and "MSPGothic" not in c.get("fontname", "")]
    groups = []
    ordinary = [c for c in chars if c["size"] >= 9 and not unicodedata.combining(c["text"][0])]
    for c in sorted(ordinary, key=lambda c: (c["bottom"], c["x0"])):
        match = next((g for g in reversed(groups[-5:]) if abs(g[0]["bottom"] - c["bottom"]) < 4.5), None)
        if match is None:
            groups.append([c])
        else:
            match.append(c)
    ordinary_ids = {id(c) for c in ordinary}
    for c in chars:
        if id(c) in ordinary_ids:
            continue
        if unicodedata.combining(c["text"][0]):
            bases = [x for x in ordinary if x["text"].strip() and x["fontname"] == c["fontname"]
                     and abs(x["bottom"] - c["bottom"]) < 4.5 and x["x0"] < c["x0"]]
            if bases:
                base = min(bases, key=lambda x: abs(x["x1"] - c["x0"]))
                base["text"] += c["text"]
                continue
        if groups:
            nearest = min(groups, key=lambda g: abs(g[0]["bottom"] - c["bottom"]))
            if abs(nearest[0]["bottom"] - c["bottom"]) < 8:
                nearest.append(c)
    result = []
    for group in sorted(groups, key=lambda g: g[0]["bottom"]):
        group.sort(key=lambda c: c["x0"])
        tokens, current = [], []
        def flush():
            if not current:
                return
            text = canonical("".join(c["text"] for c in current))
            if text:
                form = any("Gentium" in c["fontname"] for c in current)
                tokens.append({"text": text, "form": form,
                    "italic": any("Italic" in c["fontname"] for c in current if "Gentium" in c["fontname"]),
                    "small": max(c["size"] for c in current) < 11,
                    "x": round(min(c["x0"] for c in current), 3)})
            current.clear()
        for c in group:
            if c["text"].isspace():
                # Word's positioned underdot uses an overlapping space as a
                # drawing carrier. It is not a boundary inside j̣a-, etc.
                overlaps = any(x['text'].strip() and x['x0'] < c['x1']-0.5
                               and x['x1'] > c['x0']+0.5 for x in group)
                if not overlaps:
                    flush()
            else:
                current.append(c)
        flush()
        if tokens:
            result.append({"pdf_page": number, "top": round(min(c["top"] for c in group), 3),
                           "tokens": tokens, "text": canonical(" ".join(t["text"] for t in tokens))})
    return result


def extract_snapshot(pdf, path=SNAPSHOT):
    import pdfplumber
    if hashlib.file_digest(pdf.open("rb"), "sha256").hexdigest() != PDF_SHA:
        raise ValueError("Yoshioka PDF differs from the pinned 626-page source")
    serial = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with pdfplumber.open(pdf) as old, pdfplumber.open(pdf) as new, temporary.open("wb") as raw:
        new.rsrcmgr = decoded_resource_manager()
        if len(new.pages) != 626:
            raise ValueError("Unexpected Yoshioka page count")
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            for number in legacy.VOCABULARY_PAGES:
                anchors = []
                for line in legacy.native_lines(old.pages[number-1], number):
                    if legacy.analyze_candidate(line):
                        serial += 1
                        anchors.append((line.top, f"yoshioka-entry-{serial}", line.text))
                lines = positioned_lines(new.pages[number-1], number)
                assigned = set()
                for line in lines:
                    matches = [(abs(top-line["top"]), key, text) for top, key, text in anchors if abs(top-line["top"]) < 3]
                    if matches:
                        _, key, _ = min(matches)
                        if key not in assigned:
                            line["legacy_key"] = key
                            line["merged_legacy"] = [{"key": k, "text": text} for _, k, text in sorted(matches) if k != key]
                            assigned.update(k for _, k, _ in matches)
                    zipped.write((json.dumps(line, ensure_ascii=False, sort_keys=True)+"\n").encode())
                if assigned != {key for _, key, _ in anchors}:
                    raise ValueError(f"Unmatched original entry anchors on PDF page {number}")
                old.pages[number-1].close()
                new.pages[number-1].close()
                if number % 10 == 0:
                    print(f"Snapshotted PDF page {number}", flush=True)
    if serial != 3212:
        raise ValueError(f"Legacy identity inventory changed: {serial}")
    temporary.replace(path)


def load_snapshot(path=SNAPSHOT):
    if not path.exists():
        raise FileNotFoundError(f"Missing source snapshot {path}; run --extract with the pinned PDF")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream]
    if {l['pdf_page'] for l in lines} != set(legacy.VOCABULARY_PAGES):
        raise ValueError("Source snapshot is incomplete")
    return lines


def roman(number):
    result = ""
    for value, symbol in [(100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
                          (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while number >= value:
            result += symbol
            number -= value
    return result


def token_codes(token):
    if token["form"]:
        return []
    value = token["text"].strip("(),;:")
    parts = value.split(".")
    known = legacy.GRAMMAR_CODES | {"INSTRUCTIVE", "ADJVLZ", "SG-only", "PL-only"}
    return parts if all(p in known for p in parts) else []


def meaning_index(tokens):
    for i, token in enumerate(tokens):
        text = token["text"]
        if text in {"||", "¶"}:
            return i
        if token["form"] or token_codes(token) or not re.search("[A-Za-z]", text):
            continue
        if text in {"and", "or"} and i+1 < len(tokens) and (tokens[i+1]["form"] or token_codes(tokens[i+1])):
            continue
        return i
    return len(tokens)


def entry_records(lines):
    records, current = [], None
    for line in lines:
        tokens = line["tokens"]
        if line.get('legacy_key') in {f'yoshioka-entry-{i}' for i in CONTINUATIONS}:
            if current is None:
                raise ValueError('Continuation lacks its preceding article')
            current['lines'].append(line)
            current['tokens'].extend(tokens)
            current.setdefault('continuations', []).append(line['legacy_key'])
            continue
        # Indented senses can omit their headword. A new class/POS label after
        # the previous article's reference/origin introduces a separate sense.
        previous_finished = current and meaning_index(current['tokens']) < len(current['tokens'])
        sense = (previous_finished and not line.get('legacy_key') and tokens[0]['x'] < 220
                 and bool(set(token_codes(tokens[0])) & (legacy.NOUN_CLASSES | set(legacy.POS_TAGS) | set(legacy.DIALECTS)))
                 and meaning_index(tokens) < len(tokens)
                 and not tokens[meaning_index(tokens)]['text'].startswith(('‘', '“', "'", '"')))
        meaning = meaning_index(tokens)
        new_head = (tokens[0]["form"] and tokens[0]["x"] < 170
                    and not tokens[0]["text"].startswith(("(", "-"))
                    and ((meaning < len(tokens)
                          and tokens[meaning]['text'] not in {"||", "¶"}
                          and not tokens[meaning]['text'].startswith(('‘', '“', "'", '"')))
                         or (meaning == len(tokens) and previous_finished
                             and any(set(token_codes(t)) & set(legacy.POS_TAGS) for t in tokens[1:]))))
        if line.get("legacy_key") or new_head or sense:
            inherited_head = next((t for t in current['tokens'] if t['form']), None) if sense else None
            sense_of = current['key'] if sense else ''
            key = line.get("legacy_key") or f"yoshioka:p{line['pdf_page']-326}:y{round(line['top']*1000)}"
            current = {"key": key, "pdf_page": line["pdf_page"], "top": line["top"],
                       "lines": [], "tokens": []}
            if sense:
                current['tokens'].append(dict(inherited_head))
                current['sense_of'] = sense_of
            records.append(current)
        if current is None:
            raise ValueError("Unassigned source text before first entry")
        current["lines"].append(line)
        current["tokens"].extend(tokens)
    return records


def split_forms(text):
    # A shared compound tail is distributed only within the printed slash list.
    # Commas delimit complete alternatives in the source.
    result = []
    for item in re.split(r"[,;]\s*", text):
        item = item.strip(" ,;")
        if not item:
            continue
        if "/" in item:
            words = item.split()
            choices = [w.split("/") for w in words]
            import itertools
            result.extend(" ".join(parts) for parts in itertools.product(*choices))
        else:
            result.append(item)
    return list(dict.fromkeys(canonical(x) for x in result))


def citations(note, page, form):
    refs = [f"yoshioka2012[p. {roman(page-326)}, s.v. {form}]"]
    for match in re.finditer(r"\bB\.(\d+(?:[-–]\d+)?(?:,\s*\d+)*)(?:\s*\(([^)]*)\))?", note):
        locator = f"p. {match[1]}"
        if match[2]:
            locator += ", " + match[2].replace(";", ",").replace("[", "(").replace("]", ")")
        refs.append(f"berger[{locator}, cited by Yoshioka]")
    for match in re.finditer(r"\bAA\.#(\d+(?:,\s*\d+)*)", note):
        refs.append(f"ilcaa1967[item {match[1]}, cited by Yoshioka]")
    return ";".join(dict.fromkeys(refs))


def scoped_tags(codes):
    tags = legacy.grammatical_tags(" ".join(codes))
    if any(code in legacy.POS_TAGS or code == "ONO" for code in codes):
        tags = [t for t in tags if t != "noun"]
    return tags


def parse_record(record):
    tokens = [dict(t) for t in record["tokens"]]
    # A handful of suffixes were typeset in Times instead of Gentium.
    for token in tokens:
        if not token['form'] and re.fullmatch(r'-[^\W\d_]+[,;]?', token['text']):
            token['form'] = True
    for previous, token in zip(tokens, tokens[1:]):
        if (set(token_codes(previous)) & set(legacy.DIALECTS) and not token['form']
                and re.search(r'[^\x00-\x7f]', token['text']) and token['text'].isalpha()):
            token['form'] = True  # chiír / gurpáltiŋ, exceptionally set in Times
    # Two roman roots use Times rather than Gentium; the following italic stem
    # and source-line anchor disambiguate them from English definitions.
    if (len(tokens) > 1 and not tokens[0]['form'] and tokens[1]['form']
            and not token_codes(tokens[0]) and re.fullmatch(r"[^\W\d_]+", tokens[0]['text'])):
        tokens[0]['form'] = True
    raw = canonical(" ".join(t["text"] for t in tokens))
    lexical = raw.split("||", 1)[0].split("¶", 1)[0].strip()
    reference = raw.split("||", 1)[1].split("¶", 1)[0].strip() if "||" in raw else ""
    etymology = raw.split("¶", 1)[1].strip() if "¶" in raw else ""
    stop = next((i for i, t in enumerate(tokens) if t["text"] in {"||", "¶"}), len(tokens))
    lexical_tokens = tokens[:stop]
    start = meaning_index(lexical_tokens)
    preamble, gloss_tokens = lexical_tokens[:start], lexical_tokens[start:]
    gloss = canonical(" ".join(t["text"] for t in gloss_tokens))
    if (preamble and set(token_codes(preamble[-1])) & legacy.NOUN_CLASSES
            and any(set(token_codes(t)) & legacy.NOUN_CLASSES for t in gloss_tokens)):
        # Inline class-specific definitions retain their first label too.
        gloss = preamble[-1]['text'] + ' ' + gloss
    prefix_labels = []
    offset = 0
    while offset < len(preamble) and token_codes(preamble[offset]):
        prefix_labels.extend(token_codes(preamble[offset]))
        offset += 1
    initial = []
    for token in preamble[offset:]:
        if not token["form"]:
            break
        if initial and initial[0]["italic"] != token["italic"]:
            break
        initial.append(token)
    heads = split_forms(" ".join(t["text"] for t in initial))
    # Source roots in roman type may precede a list of bold-italic stems.
    stem_start = offset + len(initial)
    stems = []
    while stem_start < len(preamble) and preamble[stem_start]["form"]:
        stems.append(preamble[stem_start]["text"])
        stem_start += 1
    alternate_stems = split_forms(" ".join(stems))
    codes = prefix_labels.copy()
    if not prefix_labels:
        for token in preamble[stem_start:]:
            if token["form"] or any(c in legacy.GRAMMATICAL_TAGS for c in token_codes(token)):
                break
            codes.extend(token_codes(token))
    # Explicit word class can follow an entire pronoun paradigm (alét).
    for token in preamble[stem_start:]:
        codes.extend(c for c in token_codes(token) if c in legacy.POS_TAGS or c == "ONO")
    all_codes = [c for token in preamble[stem_start:] for c in token_codes(token)]
    if not set(all_codes) & set(legacy.GRAMMATICAL_TAGS):
        # A common class can follow the complete list of dialect variants.
        codes.extend(c for c in all_codes if c in legacy.NOUN_CLASSES)
    tags = scoped_tags(list(dict.fromkeys(codes)))
    if any(re.fullmatch(r'\(SG\)[,;]?', token['text']) for token in preamble[stem_start:]):
        tags.append('sg')
    leading_codes = []
    for token in preamble[stem_start:]:
        if token['form']:
            break
        leading_codes.extend(token_codes(token))
    # SG PL preceding DOUBLE PL marks a number-invariant headword, followed
    # by an additional plural formation (arzóq, ḍaámal).
    if any(leading_codes[i:i+3] in (['SG', 'PL', 'DOUBLE'], ['SG', 'PL', 'PL'])
           for i in range(len(leading_codes)-2)):
        tags.extend(['sg', 'pl'])
    if any(leading_codes[i:i+3] == ['PL', 'DOUBLE', 'PL']
           for i in range(len(leading_codes)-2)):
        tags.append('pl')
    if not any(t['form'] for t in preamble[stem_start:]):
        # Unscoped labels describe the headword. Dotted argument specifications
        # (Y.PL.OBJ, etc.) remain in Source grammar, not headword number/class.
        head_codes = [c for t in preamble[stem_start:] if '.' not in t['text']
                      for c in token_codes(t) if c in legacy.GRAMMATICAL_TAGS]
        tags.extend(scoped_tags(head_codes))
    # Dialect labels after a paradigm marker describe its forms, not the head.
    dialects = [EASTERN_TAG]
    dialects.extend(dialect_tag(c) for c in codes if c in legacy.DIALECTS)
    tags.extend(dialects)
    tags.extend(legacy.loan_tags("¶ " + etymology) if etymology else [])
    morphology_tokens = preamble if prefix_labels else preamble[stem_start:]
    morphology = canonical(" ".join(t["text"] for t in morphology_tokens))
    review = []
    excluded = []
    if not heads or not any(c.isalpha() for c in heads[0]):
        excluded.append("missing-headword")
    if not gloss:
        excluded.append("missing-definition-or-continuation")
    if any("�" in x or re.search(r"\d", x) for x in heads):
        excluded.append("damaged-headword")
    if "�" in raw:
        review.append("unmapped-source-character")
    if '?' in etymology and any(t.startswith('loan:') for t in tags):
        review.append('source-uncertain-donor')
    audit = {"Entry_Key": record["key"], "PDF_Page": record["pdf_page"],
             "Printed_Page": roman(record["pdf_page"]-326), "Top": record["top"],
             "Raw_Text": raw, "Form": heads[0] if heads else "", "Gloss": gloss,
             "Morphology": morphology, "Reference_Note": reference, "Etymology": etymology,
             "Status": "excluded" if excluded else "installed", "Review": ";".join(excluded+review),
             "Emitted_Keys": "", "Crossreference": "", "Crossreference_Target": ""}
    rows = []
    if not excluded:
        # Retain suffix paradigms verbatim; full, explicitly labelled forms are
        # emitted below. No stem+suffix spelling is invented.
        notes = "Source grammar: " + morphology if morphology else ""
        base = ["Bur", "", heads[0], gloss, "", "", notes,
                citations(reference, record["pdf_page"], heads[0]), "", etymology,
                record["key"], "", "", "", " ".join(dict.fromkeys(tags + (["uncertain"] if review else [])))]
        rows.append(base)
        for i, form in enumerate(dict.fromkeys(heads[1:] + alternate_stems), 1):
            if form == base[2]:
                continue
            row = base.copy()
            row[2], row[10], row[11] = form, f"{base[10]}:variant:{i}", base[10]
            row[14] = " ".join(dict.fromkeys(tags + ["alternate"]))
            rows.append(row)
        # Track number/class/aspect scope between printed inflectional forms.
        active = []
        forms = []
        counter = 0
        def emit_paradigm():
            nonlocal counter
            if not forms:
                return
            for form in split_forms(" ".join(forms)):
                if not active or form.startswith(("-", "<")) or not any(c.isalpha() for c in form):
                    continue
                if form in {r[2] for r in rows} and not active:
                    continue
                counter += 1
                row = base.copy()
                row[2], row[10], row[11] = form, f"{base[10]}:inflection:{counter}", base[10]
                replaces_class = any(c in legacy.NOUN_CLASSES for c in active)
                replaces_dialect = any(c in legacy.DIALECTS for c in active)
                replaces_number = any(c in {'SG', 'PL'} for c in active)
                inflection_tags = [t for t in tags if not (replaces_class and (t.startswith("Burushaski-class-") or t in {"m", "f"}))
                                   and not (replaces_number and t in {'sg', 'pl', 'double-plural'})
                                   and not (replaces_dialect and t.startswith('dialect:') and t != EASTERN_TAG)]
                inflection_tags += scoped_tags(active)
                if "noun" not in tags:
                    inflection_tags = [t for t in inflection_tags if t != "noun"]
                inflection_tags += [dialect_tag(c) for c in active if c in legacy.DIALECTS]
                row[14] = " ".join(dict.fromkeys(inflection_tags + ["alternate"]))
                rows.append(row)
            forms.clear()
        for token in morphology_tokens:
            cs = token_codes(token)
            if cs:
                after_forms = bool(forms)
                after_suffix = bool(forms) and all(f.strip(',').startswith('-') for f in forms)
                emit_paradigm()
                for c in cs:
                    if c in {"SG", "PL", "IPFV", "PFV", "CP", "IMP", "NEG"}:
                        active = [x for x in active if x not in legacy.GRAMMATICAL_TAGS
                                  or (c == 'PL' and x == 'DOUBLE' and not after_forms)]
                    if c in legacy.NOUN_CLASSES:
                        active = [x for x in active if x not in legacy.NOUN_CLASSES]
                    if c in legacy.DIALECTS:
                        active = [x for x in active if x not in legacy.DIALECTS]
                        if after_forms:
                            # A new dialect's stem follows a finished verbal
                            # paradigm. Do not extend its aspect/mood marker
                            # onto that dialectal stem without a repeated label.
                            active = [x for x in active if x not in {'IPFV', 'PFV', 'CP', 'P', 'PP', 'IMP', 'NEG'}]
                        if after_suffix:
                            # A dialectal headword follows the finished suffix
                            # paradigm: siṇc X PL -kó GA ṣiṇc is not plural ṣiṇc.
                            active = [x for x in active if x not in legacy.GRAMMATICAL_TAGS and x not in legacy.NOUN_CLASSES]
                    active.append(c)
            elif token["form"]:
                forms.append(token["text"])
        emit_paradigm()
        audit["Emitted_Keys"] = "|".join(r[10] for r in rows)
    return rows, audit


def row_sha256(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def crossreference_decisions():
    review = json.loads((PACKAGE / 'crossreference-decisions.json').read_text())
    if review['pdf_sha256'] != PDF_SHA:
        raise ValueError('Cross-reference review uses a different PDF')
    return review['decisions']


def apply_reviewed_crossreferences(rows, audits, decisions=None):
    """Apply only image-reviewed decisions, rejecting changed evidence first."""
    decisions = crossreference_decisions() if decisions is None else decisions
    by_key = {r[10]: r for r in rows}
    by_audit = {a['Entry_Key']: a for a in audits}
    expected = {key for a in audits if 'unresolved-crossreference' in a['Review'].split(';')
                for key in a['Emitted_Keys'].split('|')}
    if {d['entry_key'] for d in decisions} != expected:
        raise ValueError('Changed cross-reference review inventory')
    seen = set()
    # Validate the whole batch before editing any rows or copying target data.
    for decision in decisions:
        key = decision['entry_key']
        if key in seen:
            raise ValueError(f'Duplicate cross-reference decision: {key}')
        seen.add(key)
        if key not in by_key or row_sha256(by_key[key]) != decision['expected_source_row_sha256']:
            raise ValueError(f'Changed cross-reference source row: {key}')
        audit = by_audit[decision['source_entry_key']]
        if (audit['Raw_Text'] != decision['source_raw_text']
                or int(audit['PDF_Page']) != decision['source_pdf_page']
                or audit['Crossreference'] != decision['printed_target']
                or by_key[key][2] != decision['source_form']):
            raise ValueError(f'Changed cross-reference source evidence: {key}')
        for evidence in decision['target_evidence'] + decision['root_evidence']:
            target_audit = by_audit[evidence['source_key']]
            if (target_audit['Raw_Text'] != evidence['raw_text']
                    or int(target_audit['PDF_Page']) != evidence['pdf_page']):
                raise ValueError(f'Changed cross-reference target evidence: {key}')
            if 'row_sha256' in evidence:
                target = by_key.get(evidence['entry_key'])
                if target is None or row_sha256(target) != evidence['row_sha256']:
                    raise ValueError(f'Changed cross-reference target row: {key}')
        evidence_keys = {e['entry_key'] for e in decision['target_evidence']}
        if (set(decision['candidate_keys']) != evidence_keys
                or set(decision['root_keys']) != {e['source_key'] for e in decision['root_evidence']}):
            raise ValueError(f'Unreviewed cross-reference candidates: {key}')
        if decision['decision'] == 'resolved':
            target_key = decision['target_key']
            if target_key not in evidence_keys or target_key not in decision['candidate_keys']:
                raise ValueError(f'Unreviewed cross-reference target: {key}')
            target = by_key[target_key]
            permitted_form = by_key[key][2] + ('-' if decision['allow_missing_terminal_hyphen'] else '')
            if target[2] != permitted_form or not target[3] or by_audit[next(
                    e['source_key'] for e in decision['target_evidence'] if e['entry_key'] == target_key
                    )]['Crossreference']:
                raise ValueError(f'Cross-reference target is not the reviewed lexical form: {key}')
        elif (decision['decision'] != 'ambiguous' or decision['target_key']
                or len(decision['candidate_keys']) < 2):
            raise ValueError(f'Invalid cross-reference decision: {key}')
    grammar_tags = {'noun'} | {
        tag for mapping in (legacy.POS_TAGS, legacy.GRAMMATICAL_TAGS)
        for values in mapping.values() for tag in values
    }
    for decision in decisions:
        row = by_key[decision['entry_key']]
        tags = row[14].split()
        if decision['decision'] == 'resolved':
            target = by_key[decision['target_key']]
            row[11], row[3] = target[10], target[3]
            # The identical printed form has the target's grammatical analysis.
            # Keep both citations and mark where the additional grammar came from.
            row[7] = ';'.join(dict.fromkeys(row[7].split(';') + target[7].split(';')))
            evidence = next(e for e in decision['target_evidence'] if e['entry_key'] == target[10])
            morphology = by_audit[evidence['source_key']]['Morphology']
            if morphology:
                row[6] += '; Referenced entry grammar: ' + morphology
            tags = [t for t in tags if t != 'uncertain'] + ['alternate']
            tags.extend(t for t in target[14].split() if t in grammar_tags
                        or t.startswith(('Burushaski-class-', 'dialect:')))
        else:
            # An index list can mix senses. Its second form must not inherit
            # the first form's newly resolved sense through the old list edge.
            row[11], row[3] = '', ''
            tags = [t for t in tags if t != 'alternate'] + ['uncertain']
            meanings = [by_key[k][3] for k in decision['candidate_keys']]
            row[6] += '; Reference is ambiguous between ' + ' / '.join(meanings) + '.'
        row[14] = ' '.join(dict.fromkeys(tags))
    # Primary-entry accounting must include every form in a multi-form index.
    for audit in audits:
        audit['Crossreference_Resolution'] = ''
        audit['Crossreference_Form_Targets'] = ''
        if not audit['Crossreference']:
            continue
        emitted = [by_key[k] for k in audit['Emitted_Keys'].split('|')]
        resolved = sum(bool(r[11] and r[3]) for r in emitted)
        audit['Crossreference_Resolution'] = (
            'resolved' if resolved == len(emitted) else 'partial' if resolved else 'unresolved')
        audit['Crossreference_Form_Targets'] = '|'.join(f'{r[10]}={r[11]}' for r in emitted if r[11] and r[3])
        primary = by_key[audit['Entry_Key']]
        audit['Crossreference_Target'], audit['Gloss'] = primary[11], primary[3]
        reasons = [r for r in audit['Review'].split(';') if r and r != 'unresolved-crossreference']
        if resolved != len(emitted):
            reasons.append('ambiguous-crossreference')
        audit['Review'] = ';'.join(reasons)


def compile_records(records):
    rows, audits = [], []
    for record in records:
        emitted, audit = parse_record(record)
        rows.extend(emitted)
        audits.append(audit)
        for continuation in record.get('continuations', []):
            line = next(l for l in record['lines'] if l.get('legacy_key') == continuation)
            continuation_audit = dict.fromkeys(audit, '')
            continuation_audit.update(Entry_Key=continuation, PDF_Page=line['pdf_page'],
                Printed_Page=roman(line['pdf_page']-326), Top=line['top'], Raw_Text=line['text'],
                Status='merged-continuation', Review='source-reference-or-comparison-continuation',
                Emitted_Keys=audit['Emitted_Keys'])
            audits.append(continuation_audit)
        for line in record['lines']:
            for merged in line.get('merged_legacy', []):
                merged_audit = dict.fromkeys(audit, '')
                merged_audit.update(Entry_Key=merged['key'], PDF_Page=line['pdf_page'],
                    Printed_Page=roman(line['pdf_page']-326), Top=line['top'], Raw_Text=merged['text'],
                    Status='merged-fragment', Review='same-printed-entry',
                    Emitted_Keys=audit['Emitted_Keys'])
                audits.append(merged_audit)
    # Several class-separated senses share one citation printed at the end.
    # Preserve that source reference on earlier senses lacking their own.
    by_audit_key = {audit['Entry_Key']: audit for audit in audits}
    for record in reversed(records):
        if not record.get('sense_of'):
            continue
        audit, parent = by_audit_key[record['key']], by_audit_key[record['sense_of']]
        if audit['Reference_Note'] and not parent['Reference_Note']:
            parent['Reference_Note'] = audit['Reference_Note']
            parent['Review'] = ';'.join(filter(None, [parent['Review'], 'shared-sense-reference']))
            for row in rows:
                if row[10] == parent['Entry_Key'] or row[10].startswith(parent['Entry_Key']+':'):
                    row[7] = citations(parent['Reference_Note'], int(parent['PDF_Page']), parent['Form'])
    keys = {r[10] for r in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate emitted Yoshioka key")
    by_form = defaultdict(list)
    for row in rows:
        if ":variant:" not in row[10] and ":inflection:" not in row[10]:
            by_form[row[2]].append(row)
    audits_by_key = {a["Entry_Key"]: a for a in audits}
    for row in rows:
        match = re.fullmatch(r"see\s+(.+?)[.]?", row[3])
        if not match:
            continue
        target_form = canonical(match[1])
        candidates = [r for r in by_form.get(target_form, []) if r[10] != row[10]]
        audit = audits_by_key.get(row[10])
        row[6] = "; ".join(x for x in [row[6], "See " + target_form + "."] if x)
        row[3] = ""
        if len(candidates) == 1 and not candidates[0][3].startswith("see "):
            target = candidates[0]
            row[11], row[3] = target[10], target[3]
            row[14] = " ".join(dict.fromkeys(row[14].split() + ["alternate"]))
            if audit:
                audit["Crossreference_Target"] = target[10]
                audit["Gloss"] = row[3]
        else:
            row[14] = " ".join(dict.fromkeys(row[14].split() + ["uncertain"]))
            if audit:
                audit["Review"] = ";".join(x for x in [audit["Review"], "unresolved-crossreference"] if x)
        if audit:
            audit["Crossreference"] = target_form
    apply_reviewed_crossreferences(rows, audits)
    parents = {r[10]: r[11] for r in rows if r[11]}
    for key in parents:
        seen = set()
        node = key
        while node in parents:
            if node in seen:
                raise ValueError(f"Variant cycle at {key}")
            seen.add(node)
            node = parents[node]
        if node not in keys:
            raise ValueError(f"Missing variant parent {node}")
    by_key = {row[10]: row for row in rows}
    for audit in audits:
        audit['Raw_Text_SHA256'] = hashlib.sha256(audit['Raw_Text'].encode()).hexdigest()
        audit['Emitted_Row_SHA256'] = '|'.join(
            row_sha256(by_key[key])
            for key in audit['Emitted_Keys'].split('|') if key)
    return rows, audits


def write_outputs(output, rows, audits):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "forms.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    with (output / "audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audits[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audits)
    with (output / 'form-aliases.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Retired_Source_Key', 'Target_Source_Key', 'Reason'], lineterminator='\n')
        writer.writeheader()
        for audit in audits:
            if audit['Status'].startswith('merged-'):
                writer.writerow({'Retired_Source_Key': audit['Entry_Key'],
                    'Target_Source_Key': audit['Emitted_Keys'].split('|')[0],
                    'Reason': audit['Review']})
    by_key = {r[10]: r for r in rows}
    reviewed = crossreference_decisions()
    with (output / 'crossreference-audit.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Entry_Key', 'Source_Entry_Key', 'Form',
            'Printed_Target', 'Decision', 'Target_Key', 'Gloss', 'Candidate_Keys', 'Reason',
            'Source_PDF_Page', 'Target_PDF_Pages', 'Emitted_Row_SHA256'], lineterminator='\n')
        writer.writeheader()
        for decision in reviewed:
            row = by_key[decision['entry_key']]
            writer.writerow(dict(Entry_Key=row[10], Source_Entry_Key=decision['source_entry_key'],
                Form=row[2], Printed_Target=decision['printed_target'], Decision=decision['decision'],
                Target_Key=row[11], Gloss=row[3], Candidate_Keys='|'.join(decision['candidate_keys']),
                Reason=decision['reason'], Source_PDF_Page=decision['source_pdf_page'],
                Target_PDF_Pages='|'.join(str(p) for p in sorted({e['pdf_page'] for e in decision['target_evidence']})),
                Emitted_Row_SHA256=row_sha256(row)))
    counts = {"source_records": len(audits), "installed_rows": len(rows),
              "statuses": dict(Counter(a["Status"] for a in audits)),
              "review": dict(Counter(a["Review"] for a in audits)),
              "linked_variants": sum(bool(r[11]) for r in rows),
              "resolved_crossreferences": sum(a['Crossreference_Resolution'] == 'resolved' for a in audits),
              "crossreference_entries": dict(Counter(a['Crossreference_Resolution'] for a in audits if a['Crossreference'])),
              "reviewed_crossreference_forms": dict(Counter(d['decision'] for d in reviewed))}
    (output / "counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--pdf", type=Path, default=legacy.DEFAULT_PDF)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "tmp/yoshioka-20260914/proposal")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    if args.extract:
        extract_snapshot(args.pdf, args.snapshot)
    rows, audits = compile_records(entry_records(load_snapshot(args.snapshot)))
    write_outputs(args.output_dir, rows, audits)
    if args.install:
        import shutil
        shutil.copyfile(args.output_dir / "forms.csv", ROOT / "data/other/forms/20260726-yoshioka-eastern-burushaski.csv")
        shutil.copyfile(args.output_dir / "audit.csv", PACKAGE / "audit.csv")
        shutil.copyfile(args.output_dir / "counts.json", PACKAGE / "counts.json")
        shutil.copyfile(args.output_dir / 'crossreference-audit.csv', PACKAGE / 'crossreference-audit.csv')
        alias_dir = ROOT / 'data/other/form_aliases'
        alias_dir.mkdir(exist_ok=True)
        shutil.copyfile(args.output_dir / 'form-aliases.csv', alias_dir / '20260914-yoshioka.csv')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
