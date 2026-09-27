"""Prepare source-only Kharia rows from the Munda Lexical Archive snapshot.

This command never builds Jambu's database.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source-wayback.txt"
CSV = HERE.parents[1] / "20260925-donegan-stampe-kharia.csv"
AUDIT = HERE / "audit.jsonl"
SOURCE_SHA256 = "4e827a17e08103d4ddf302934af868cdc5275896696c261e3524981af9f79a4c"
SOURCE_KEY = "DSKH"
POS = {
    "N": "noun", "NI": "noun", "NA": "noun", "NK": "noun",
    "V": "verb", "VI": "verb", "VT": "verb",
    "ADJ": "adj", "ADV": "adv", "NUM": "num", "PP": "postp",
    "PRON": "pron", "CONJ": "conj", "INTERJ": "interj",
}
WITNESSES = {"A", "B", "D", "P", "AB", "AD", "BD", "ABD", "DP"}
EXISTING_DSKH_IDS = {"12601", "12711", "32441", "3721", "10041"}


def raw_records(data: bytes) -> list[dict]:
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("Kharia archive snapshot SHA-256 changed")
    parts = data.decode("ascii").split("\f")
    if len(parts) != 2 or "Munda Lexical Archive" not in parts[0]:
        raise ValueError("Kharia archive boundaries changed")
    body = parts[1]
    matches = list(re.finditer(r"#(\d+)\.", body))
    if len(matches) != 3620 or len({m.group(1) for m in matches}) != 3620:
        raise ValueError("Expected 3620 uniquely numbered Kharia records")
    records = []
    previous = 0
    for ordinal, match in enumerate(matches, 1):
        records.append({"ordinal": ordinal, "source_id": match.group(1),
                        "raw": body[previous:match.end()].strip()})
        previous = match.end()
    if body[previous:].strip():
        raise ValueError("Unexpected unnumbered Kharia source text")
    return records


def legacy_parse(record: dict) -> tuple[list[str] | None, dict]:
    raw = record["raw"]
    source_id = record["source_id"]
    key = f"kharia-mla2004:{source_id}"
    audit = {"entry_key": key, "ordinal": record["ordinal"],
             "source_id": source_id, "raw": raw}
    header = raw.split("{", 1)[0]
    heads = re.findall(r"<([^<>]+)>", header)
    gloss_match = re.search(r"``(.*?)''", raw, re.S)
    labels = re.findall(r"\{([^}]+)\}", raw[:gloss_match.start()] if gloss_match else header)
    witness_match = re.fullmatch(r"\s*<[^<>]+>\(([A-Z]+)\)\s*", header)
    witness = witness_match.group(1) if witness_match else None
    audit.update({"headwords": heads, "source_labels": labels, "source_witness": witness})

    def skip(reason: str):
        audit.update({"status": "excluded", "reason": reason})
        return None, audit

    if source_id in EXISTING_DSKH_IDS:
        return skip("archive ID already cited through Rau 2019 Munda rows")
    if len(heads) != 1 or witness not in WITNESSES:
        return skip("not one unqualified archive-witness headword")
    form = heads[0]
    if not re.fullmatch(r"[a-z-]+", form):
        return skip("source transcription or segmentation needs review")
    if not gloss_match:
        return skip("no bounded lexical gloss")
    source_gloss = gloss_match.group(1)
    if not source_gloss.strip("? ") or any(s in source_gloss for s in ("?", "<", ">", "^G")):
        return skip("uncertain, placeholder, or embedded-form gloss")
    if any(s in source_gloss for s in ("[", "]", ";", "%", "`")):
        return skip("multi-sense, witness-qualified, or marked-up gloss")
    if len(labels) != 1 or labels[0] not in POS:
        return skip("unmapped or mixed grammatical label")
    gloss = re.sub(r"\s+", " ", source_gloss.replace("^", "").replace("_", " ")).strip()
    if not gloss or "\ufffd" in gloss or len(gloss) > 100:
        return skip("empty, corrupt, or extended gloss")
    tags = [POS[labels[0]]]
    if "??" in raw:
        tags.append("uncertain")
        audit["review_reason"] = "source editorial uncertainty; scope may concern analysis, not form"
    citation = f"{SOURCE_KEY}[entry {source_id}]"
    row = ["kh", "", form, gloss, "", "", "", citation, "", "", key,
           "", "", "", " ".join(tags)]
    audit.update({"status": "ingested", "form": form, "gloss": gloss,
                  "tags": tags, "citation": citation})
    return row, audit



# Full snapshot parser. Legacy selection above only pins previously installed keys.
LEGACY = {}


def physical_records(data):
    raw_records(data)
    lines = data.decode('ascii').split('\f')[1].splitlines()
    records = []
    for line_number, line in enumerate(lines, 1):
        raw = line.strip()
        if not raw: continue
        if raw == '21.' and records[-1]['raw'].endswith('#93'):
            records[-1]['raw'] += '\n' + raw
            records[-1]['line_end'] = line_number
            records[-1]['source_id'] = '9321'
            records[-1]['source_ids'] = ['9321']
            records[-1]['line_repair'] = 'Numeric ID9321 split across two physical lines'
            continue
        ids = re.findall(r'#(\d+)\.', raw)
        records.append({'line': line_number, 'line_end': line_number, 'source_id': ids[0] if ids else None,
                        'source_ids': ids, 'raw': raw})
    assert len(records) == 3631
    assert sum(bool(x['source_id']) for x in records) == 3620
    assert len({sid for x in records for sid in x['source_ids']}) == 3621
    return records


def clean(text):
    return re.sub(r'\s+', ' ', text.replace('^', '').replace('_', ' ').replace('%', '').replace('<', '').replace('>', '')).strip()


def repair_markup(raw, source_id):
    text, repairs = raw, []
    replacements = {
        '101': ('{ADV ``', '{ADV} ``'),
        '211': ('reflexive [B].', "reflexive [B]''."),
        '5013': ('{VT} to ^remove', '{VT} ``to ^remove'),
        '19852': ("{N.  ^sun-^moon, the supreme ^spirit'", "{N} ``^sun-^moon, the supreme ^spirit''"),
        '26531': ("(%Phaseolus Roxburghii).  *@'", "(%Phaseolus Roxburghii)''.  *@"),
        '31571': ("{VI.  ~to ^tremble, to ^shiver'", "{VI} ``to ^tremble, to ^shiver''"),
        '29651': (',,su?trom>', ',,<su?trom>'),
        '15761': ("to ^chip'' [D]", "to ^chip [D]''"),
        '12652': ("^yesterday's'--|", "^yesterday's''.--|"),
        '8042': ("^rest`.", "^rest''."),
        '28271': ("^straighten.  |", "^straighten''.  |"),
        '28310': ("^straighten.  |", "^straighten''.  |"),
    }
    if source_id in replacements:
        before, after = replacements[source_id]
        if before not in text: raise ValueError(('repair mismatch', source_id, before))
        text = text.replace(before, after, 1)
        repairs.append({'before': before, 'after': after, 'reason': 'Recoverable source markup delimiter; lexical letters retained'})
    return text, repairs


def grammar(label):
    core = re.sub(r'\([^)]*\)', '', label).strip(' `?')
    tags = []
    mapping = {'NI': ['noun', 'inanimate'], 'NA': ['noun', 'animate'], 'NK': ['noun', 'kinship'],
       'N': ['noun'], 'V': ['verb'], 'VI': ['verb', 'intr'], 'VT': ['verb', 'tr'],
       'VIRR': ['verb'], 'ADJ': ['adj'], 'ADV': ['adv'], 'NUM': ['num'], 'PP': ['postp'],
       'PRON': ['pron'], 'CONJ': ['conj'], 'INTERJ': ['interj'], 'INTERR': ['interr'],
       'INTER': ['interj'], 'DEM': ['demonstrative'], 'NEG': ['neg'], 'PX': ['prefix'],
       'SX': ['suffix'], 'PART': ['part'], 'PARTICLE': ['part'], 'PN': ['proper-noun'],
       'ONOM': ['onomatopoeia'], 'STAT': ['adj', 'verb'],
       'SUB': ['conj'], 'IV': ['verb', 'intr'], 'NCF': ['noun']}
    for token in re.findall(r'[A-Z]+', core): tags.extend(mapping.get(token, []))
    if core in ('INTER ADV', 'INTER ADJ'):
        tags = [t for t in tags if t != 'interj'] + ['interr']
    if label.startswith('VT') and 'caus.' in label: tags.append('caus')
    if label.startswith('VIRR') and '3 sg. pres.' in label: tags += ['3sg', 'pres']
    return list(dict.fromkeys(tags))


def scope_gloss(text, witness):
    # A witness qualifier within a parenthesis narrows only that explanation.
    def parenthetical(match):
        content = match.group(1)
        qualifiers = re.findall(r'\[([ABDP]+)\]', content)
        if qualifiers and not any(set(witness).intersection(q) for q in qualifiers): return ''
        return '(' + re.sub(r'\s*\[[ABDP]+\]', '', content) + ')'
    text = re.sub(r'\(([^()]*)\)', parenthetical, text)
    # Two cool/lukewarm entries omit the separator after the D qualification.
    text = re.sub(r'(\[[ABDP]+\])\s+(?=[^,;\s])', r'\1, ', text)
    pieces, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == '(': depth += 1
        elif ch == ')': depth = max(0, depth - 1)
        elif ch in ',;' and depth == 0:
            pieces.append((text[start:i], ch)); start = i + 1
    pieces.append((text[start:], ''))
    kept = []
    for piece, separator in pieces:
        qualifier = re.search(r'\[([ABDP]+)\]\s*$', piece)
        if qualifier:
            if not set(witness).intersection(qualifier.group(1)): continue
            piece = piece[:qualifier.start()]
        if clean(piece): kept.append((clean(piece), separator))
    return ''.join(p + (sep + ' ' if i < len(kept)-1 else '') for i, (p, sep) in enumerate(kept)).strip()


def gloss_grammar(gloss, tags):
    flags = []
    for word, tag in [('emphatic', 'emph'), ('inclusive', 'inclusive'), ('exclusive', 'exclusive')]:
        if re.search(r'\b'+word+r'\b', gloss):
            tags.append(tag)
            gloss = gloss.replace('('+word+')', '').strip()
            if gloss.endswith(', '+word): gloss = gloss[:-len(', '+word)]
    for word, tag in [('plural', 'pl'), ('dual', 'du'), ('singular', 'sg'), ('reflexive', 'refl'), ('conditional', 'conditional')]:
        if '('+word+')' in gloss:
            tags.append(tag); gloss = gloss.replace('('+word+')', '').strip()
    if gloss == 'causative': tags.append('caus')
    for marker, tag in [('tr.', 'tr'), ('intr.', 'intr'), ('caus.', 'caus')]:
        if '(' + marker + ')' in gloss:
            tags.append(tag); gloss = clean(gloss.replace('(' + marker + ')', ''))
    grammatical = {'possessive': 'poss', 'genitive': 'gen', 'accusative': 'acc', 'reflexive': 'refl',
                   'imperative': 'impv', 'prohibitive': 'prohibitive'}
    if any(t in tags for t in ('pron', 'prefix', 'suffix', 'interr')):
        for word, tag in grammatical.items():
            if re.search(r'\b' + word + r'\b', gloss.split('of verbs taking')[0]): tags.append(tag)
        person = {'1st': '1', 'first': '1', '1': '1', '2nd': '2', 'second': '2', '2': '2', '3rd': '3', 'third': '3', '3': '3'}
        number = {'singular': 'sg', 'sg.': 'sg', 'plural': 'pl', 'pl.': 'pl', 'dual': 'du', 'du.': 'du'}
        for match in re.finditer(r'\b(1st|first|1|2nd|second|2|3rd|third|3)(?: person)? (singular|plural|dual|sg\.|pl\.|du\.)', gloss):
            n = number[match[2]]
            tags.append(person[match[1]] + n if n != 'du' else 'du')
            if n == 'du': flags.append('person_in_dual_preserved_in_source_gloss')
        # Only remove purely grammatical parenthetical labels; functional glosses remain.
        gloss = re.sub(r'\((?:1st|2nd|3rd) person (?:singular|plural)\)', '', gloss)
    return clean(gloss), tags, flags




def gloss_usage(gloss, tags):
    notes = []
    def constraint(match):
        text = match[1]
        if re.search(r'contemptuous|addressed to|^(?:with|used with|added to) ', text):
            if 'contemptuous' in text: tags.append('pejorative')
            notes.append('Source usage constraint: ' + text)
            return ''
        return match[0]
    gloss = re.sub(r'\(([^()]*)\)', constraint, gloss)
    # These are definitions of grammatical morphemes, not tense words in ordinary meanings.
    if any(t in tags for t in ('suffix', 'prefix')):
        main = re.split(r'\b(?:used with|with (?:neutral|active)|of verbs taking|added to)\b', gloss, 1)[0]
        terms = [('future tense','fut'), ('present tense','pres'), ('past tense','pret'),
                 ('present immediate','pres'), ('present perfect','perfect'),
                 ('imperative','impv'), ('infinitive','inf'), ('gerundival','ger'),
                 ('reciprocal','reciprocal'), ('reflexive','refl'), ('intensive','intensive'),
                 ('dual','du'), ('plural','pl'), ('classifier','classifier')]
        for word, tag in terms:
            if re.search(r'\b'+word+r'\b', main): tags.append(tag)
        usage = re.search(r'\b((?:used with|with (?:neutral|active)) .*)$', gloss)
        if usage:
            notes.append('Source usage constraint: ' + usage[1])
            gloss = gloss[:usage.start()].strip()
    if '(interrogative particle)' in gloss:
        tags += ['interr','part']; gloss = gloss.replace('(interrogative particle)', '')
    return clean(gloss), tags, notes


def commentary_grammar(commentary, witness):
    """Parse explicit head-level editorial labels, without tagging quoted comparisons."""
    tags, notes, evidence = [], [], []
    for match in re.finditer(r'(?:^|\s)!([^.!#]+)', commentary):
        claim = match[1].strip()
        lower = claim.replace('^', '').lower()
        qualifier = re.search(r'\[([ABDP]+)\]', claim)
        if qualifier and not set(witness).intersection(qualifier[1]): continue
        assigned = []
        if lower.startswith('auxiliary'): assigned.append('auxiliary')
        elif lower.startswith('postposition'): assigned.append('postp')
        elif lower == 'nk': assigned += ['noun', 'kinship']
        elif lower.startswith('neutral verb'): assigned.append('verb')
        elif lower.startswith('causal of'): assigned.append('caus')
        elif lower.startswith('prefixed to'): assigned.append('prefix')
        elif lower.startswith('term of contempt'): assigned.append('pejorative')
        elif lower.startswith(('emphatic', 'nominative and accusative')):
            for term, tag in [('emphatic','emph'), ('nominative','nom'), ('accusative','acc'), ('interrogative','interr')]:
                if term in lower: assigned.append(tag)
        usage = lower.startswith(('bound form', 'verbal affix', 'used only', 'used with', 'used alone', 'with verbs', 'probably used'))
        if assigned or usage:
            evidence.append({'source_claim':claim, 'witness':witness, 'tags':assigned,
                             'status':'structured_grammar' if assigned else 'source_usage_constraint'})
            if usage: notes.append('Source usage constraint: ' + claim)
            tags.extend(assigned)
    return tags, notes, evidence


def parse_unit(record):
    sid = record['source_id']
    key = f'kharia-mla2004:{sid}' if sid else f'kharia-mla2004:unnumbered:{record["line"]}'
    raw = record['raw']
    text, repairs = repair_markup(raw, sid)
    audit = dict(record, entry_key=key, repairs=repairs, rows=[], sense_holds=[])
    if re.fullmatch(r'\*+|- <[^>]+> -', text):
        audit.update(status='excluded_context', reason='Nonlexical alphabetic heading or section divider')
        return [], audit
    start = text.find('{')
    header = text[:start] if start >= 0 else re.split(r'\s+(?:\?\?|#|\*@)', text, 1)[0]
    grouping = re.match(r'^(?:\^T2)?\s*(<[^>]+>)::(?=<)', header)
    audit['grouping_context'] = grouping[1] if grouping else ''
    if grouping: header = header[grouping.end():]
    heads = []
    for match in re.finditer(r'(?<!<)<([^<>]+)>(?!>)(?:\(([^)]*)\))?', header):
        heads.append({'form': match[1], 'witness': match[2] or '', 'start': match.start(), 'end': match.end()})
    if not heads: raise ValueError(('unparsed head', sid, text))
    audit['headwords'] = heads
    audit['header'] = header
    senses = []
    pos = start
    while pos >= 0:
        match = re.match(r'((?:\{[^}]*\}\s*)+)``(.*?)\'\'', text[pos:])
        if not match:
            if sid == '6651': break  # head survives, source supplies only a damaged empty label
            raise ValueError(('unparsed definition', sid, text[pos:]))
        senses.append({'number': len(senses)+1, 'label': '; '.join(re.findall(r'\{([^}]*)\}', match[1])), 'source_gloss': match[2]})
        pos += match.end()
        next_sense = re.match(r"['.]?\s*(?=\{)", text[pos:])
        if next_sense: pos += next_sense.end()
        else: break
    if not senses: senses = [{'number': 1, 'label': '', 'source_gloss': ''}]
    audit['senses'] = senses
    tail = text[pos:] if pos >= 0 else text[len(header):]
    audit['source_commentary'] = tail
    # Citation-only macro *@ is retained in raw audit, not presented as linguistic analysis.
    commentary = re.sub(r'#\s*\d+(?:\n\d+)?\.?|\*@\.?', '', tail).strip()
    commentary = re.sub(r"^[.'\s]+", '', commentary)
    rows = []
    for sense in senses:
        first_by_witness = {}
        for hi, head in enumerate(heads, 1):
            witness = head['witness']
            scoped = list(witness) if re.fullmatch(r'[ABDP]+', witness) and re.search(r'\[[ABDP]+\]', sense['source_gloss'] + (' [D]' if sid == '26781' else '')) else [witness]
            for wi, w in enumerate(scoped):
                gloss = scope_gloss(sense['source_gloss'], w)
                child = key if sense['number'] == 1 and hi == 1 else f'{key}:sense:{sense["number"]}:head:{hi}'
                if wi: child += ':witness:' + w
                flags = []
                if not gloss.strip(' ?.G'):
                    gloss = ''; flags.append('source_gloss_empty_or_witness_alignment_unresolved')
                tags = grammar(sense['label'])
                comment_tags, usage_notes, comment_evidence = commentary_grammar(commentary, w)
                tags.extend(comment_tags)
                if sense['label'] == 'V(VI [B])' and 'B' in w: tags.append('intr')
                if '(BND)' in sense['label']: usage_notes.append('Source usage constraint: bound form (BND)')
                if 'TAG' in sense['label'] and 'echoword' in gloss: tags.append('echo')
                gloss, tags, grammar_flags = gloss_grammar(gloss, tags)
                flags += grammar_flags
                gloss, tags, gloss_notes = gloss_usage(gloss, tags)
                usage_notes += gloss_notes
                if not re.fullmatch(r'[ABDP]+', w): flags.append('source_witness_unresolved')
                if re.search(r'[^a-z -]', head['form']): flags.append('source_transcription_preserved')
                if '?' in sense['source_gloss'] or '??' in text: flags.append('source_editorial_or_gloss_uncertainty')
                if not tags or '?' in sense['label']: flags.append('source_grammar_unresolved')
                if sid == '12820': flags.append('source_person_number_contradiction_we_vs_second_person')
                if repairs or sid == '6651': flags.append('source_markup_repaired_or_damaged')
                if re.search(r'\*(?:Hi\.|Loan\b)', commentary): tags.append('loanword')
                if flags: tags.append('uncertain')
                analysis = commentary
                if 'caus.' in sense['label'] or label_analysis(sense['label']):
                    analysis = 'Source grammatical analysis: ' + sense['label'] + ('. ' + commentary if commentary else '')
                citation = f'DSKH[entry {sid}]' if sid else f'DSKH[body line {record["line"]}]'
                if child not in LEGACY: citation = citation[:-1] + f', witness {w or "unspecified"}, sense {sense["number"]}]'
                variant = ''
                if w in first_by_witness and (',,' in header or ';;' in header) and '/' not in header and '\\' not in header and not re.search(r'\?\?.*phrase', text, re.I):
                    variant = first_by_witness[w]
                first_by_witness.setdefault(w, child)
                row = ['kh', '', head['form'], gloss, '', '', '', citation, '; '.join(usage_notes), analysis, child, variant, '', '', ' '.join(dict.fromkeys(tags))]
                rows.append(row)
                audit['rows'].append({'entry_key':child, 'form':head['form'], 'gloss':gloss, 'source_gloss':sense['source_gloss'], 'label':sense['label'], 'witness':w, 'tags':row[14], 'uncertainty_types':flags, 'commentary_grammar':comment_evidence})
    audit['status'] = 'ingested'
    return rows, audit


def label_analysis(label):
    return bool(re.search(r'[<(]|STAT|SUB|IV|IX|NCF|TAG|^NP$', label))


def commentary_causatives(rows, audit):
    """Account for every directly glossed Caus. mention without collapsing homonyms."""
    original_rows = list(rows)
    by_key = {r[10]: r for r in rows}
    for item in audit:
        if item['status'] != 'ingested': continue
        blocks = list(re.finditer(r'\bCaus\.\s*(.*?)(?=\s+(?:Cf\.|\||\*@|\?\?|#)|$)', item['source_commentary']))
        if not blocks: continue
        claims = []
        for block in blocks:
            matches = list(re.finditer(r"((?:<[^<>]+>(?:\([^)]*\))?(?:[, ]*)?)+)\s*`([^']*)'", block[1]))
            if not matches: raise ValueError(('unparsed causative mention',item['source_id'],block[1]))
            for mention in matches:
                header, source_gloss = mention.groups()
                for head in re.finditer(r'<([^<>]+)>(?:\(([^)]*)\))?', header):
                    form, witness = head[1], head[2] or ''
                    gloss, tags, flags = gloss_grammar(clean(source_gloss), ['verb','caus'])
                    candidates = [r for r in original_rows if r[2] == form and r[3].rstrip('.') == gloss.rstrip('.')]
                    claim = {'source_form':form,'source_witness':witness,'source_gloss':source_gloss,
                             'candidate_keys':[r[10] for r in candidates],'source_span':mention[0]}
                    if candidates:
                        claim.update(status='represented_by_existing_lexical_rows',reason='Exact literal head and complete cleaned gloss already extracted; numeric-record identities remain separate')
                    else:
                        child = item['entry_key'] + ':causative:' + str(len(claims)+1)
                        if re.search(r'[^a-z -]', form): flags.append('source_transcription_preserved')
                        if not re.fullmatch('[ABDP]+',witness): flags.append('source_witness_unresolved')
                        if '??' in item['raw']: flags.append('source_editorial_uncertainty')
                        if flags: tags.append('uncertain')
                        cite = f'DSKH[entry {item["source_id"]}, causative mention {len(claims)+1}, witness {witness or "unspecified"}]'
                        own = [by_key[o['entry_key']] for o in item['rows']]
                        verbal = [r for r in own if 'verb' in r[14].split()]
                        parents = verbal or own
                        parent = parents[0][10] if len(parents) == 1 else ''
                        row = ['kh','',form,gloss,'','','',cite,'','Source explicitly labels this form Caus. under '+', '.join(h['form'] for h in item['headwords'])+'.',child,'','',parent,' '.join(dict.fromkeys(tags))]
                        rows.append(row)
                        claim.update(status='new_lexical_child',entry_key=child,parent_candidates=[r[10] for r in parents],accepted_parent=parent)
                        item.setdefault('supplementary_rows',[]).append({'entry_key':child,'form':form,'gloss':gloss,'label':'source Caus.','witness':witness,'tags':row[14],'uncertainty_types':flags})
                    claims.append(claim)
        item['causative_mentions'] = claims
    return rows


# Independently glossed forms in explicit Syn./Cf./E.g./Also contexts, reviewed
# against the complete source. Components and hypothetical analyses are separate.
LEXICAL_COMMENTARY = {
 '2181': {'bote'}, '13731': {'jhalob'}, '14221': {'jorme-'}, '14361': {"te'D-"},
 '17991': {'koTa tuyu'}, '20301': {'ku~Rab-te'}, '22411': {'mugsiG'},
 '22501': {'r[an]ab'}, '28271': {'seGbor'}, '28310': {'seGbor'},
 '30641': {'tetrna'}, '30651': {'teja'}, '29161': {'pag-sor'},
 '9301': {'gam Oy~EG','O~e',"Oy~EGgO'D"},
 '9321': {'gam Oy~EG','O~e',"Oy~EGgO'D"},
}


def commentary_mentions(rows, audit):
    forms = {r[2] for r in rows}
    for item in audit:
        if item['status'] != 'ingested': continue
        sid = item['source_id']; tail = item['source_commentary']; decisions = []
        for match in re.finditer(r"(?<!<)<([^<>]+)>(?!>)(?:\(([^)]*)\))?\s*`([^']*)'", tail):
            form, witness, gloss = match[1], match[2] or '', clean(match[3])
            before = tail[:match.start()]
            decision = {'form':form,'source_witness':witness,'source_gloss':match[3],'source_span':match[0]}
            if form in LEXICAL_COMMENTARY.get(sid,set()):
                if gloss == 'id.':
                    meanings = list(dict.fromkeys(r['gloss'] for r in item['rows'] if r['gloss']))
                    if len(meanings) != 1: raise ValueError(('ambiguous id gloss',sid,meanings))
                    gloss = meanings[0]
                    decision['gloss_resolution'] = 'Explicit id. refers to the sole preceding lexical sense'
                key = item['entry_key'] + ':commentary:' + str(len(decisions)+1)
                tags = ['verb'] if gloss.startswith('to ') else []
                flags = ['source_commentary_attestation']
                if re.search(r'[^a-z -]',form): flags.append('source_transcription_preserved')
                if sid in ('9301','9321'):
                    witness = 'PVersuch'; flags.append('source_qualified_witness_transcription')
                if not tags: flags.append('source_grammar_unspecified')
                tags.append('uncertain')
                cite = f'DSKH[entry {sid}, commentary mention {len(decisions)+1}]'
                row = ['kh','',form,gloss,'','','',cite,'','Explicitly glossed source comparison/example; source notation and uncertainty retained.',key,'','','',' '.join(tags)]
                rows.append(row)
                decision.update(status='new_lexical_child',entry_key=key)
                item.setdefault('supplementary_rows',[]).append({'entry_key':key,'form':form,'gloss':gloss,'label':'source comparison/example','witness':witness,'tags':row[14],'uncertainty_types':flags})
            elif form in forms:
                decision.update(status='represented_form_with_source_analysis',reason='Literal form already extracted; complete context-specific analysis remains in source commentary')
            elif sid in ('91','141','2241','2361','2381','4651','8441','18460','29391'):
                decision.update(status='excluded_other_language',reason='Explicit Hindi/Juang/Sora comparison, not a Kharia headword')
            elif sid == '32181':
                decision.update(status='unresolved_comparison_language',reason='Tilde-marked comparative synonym list uses K/M/P witness system also found in Juang comparisons; no unsupported Kharia language assignment')
            elif not gloss.strip(' ?.'):
                decision.update(status='analysis_placeholder',reason='Source component is unglossed or only question-mark glossed; retain raw analysis without invented meaning')
            elif sid == '23751':
                decision.update(status='hypothetical_analysis',reason='Source explicitly proposes a questioned causative derivation; not asserted as an independently attested lexical form')
            elif sid == '32461':
                decision.update(status='connected_usage_example',reason='Complete negative finite sentence illustrating the prefix, retained as source usage rather than a standalone lemma')
            elif '|' in before or re.search(r'\bfrom\s*$',before) or before.lstrip(' .').startswith('(') or '<-' in match[0] or form in ('a-','-iJ'):
                decision.update(status='source_morphological_analysis',reason='Explicit component/affix analysis retained in Etymology and raw audit; not promoted into an independently attested word')
            else:
                raise ValueError(('unreviewed commentary form',sid,match[0],before[-70:]))
            decisions.append(decision)
        if decisions: item['commentary_mentions'] = decisions
    return rows


def prepare():
    LEGACY.clear()
    for record in raw_records(SOURCE.read_bytes()):
        row, _ = legacy_parse(record)
        if row: LEGACY[row[10]] = row
    assert len(LEGACY) == 829
    rows, audit = [], []
    for record in physical_records(SOURCE.read_bytes()):
        emitted, item = parse_unit(record)
        rows.extend(emitted); audit.append(item)
    keys = {r[10] for r in rows}
    assert len(keys) == len(rows) and set(LEGACY) <= keys
    by_form = collections.defaultdict(list)
    by_key = {r[10]:r for r in rows}
    for row in rows: by_form[row[2]].append(row)
    for item in audit:
        for out in item['rows']:
            if 'caus. of' not in out['label']: continue
            parents = re.findall(r'<([^<>]+)>', out['label'])
            candidates = [r for p in parents for r in by_form[p] if r[10] != out['entry_key']]
            evidence = {'source_label':out['label'], 'parent_forms':parents, 'candidates':[r[10] for r in candidates],
                        'status':'unique' if len(candidates)==1 else 'ambiguous' if candidates else 'unmatched'}
            uncertain_claim = '?' in re.sub(r'<[^>]*>', '', out['label'])
            if uncertain_claim: evidence['status'] = 'source_uncertain'
            if len(candidates)==1 and not uncertain_claim:
                quoted = re.findall(r"`([^']*)'", out['label'])
                claimed_gloss = quoted[-1] if quoted and quoted[-1].strip() else out['gloss']
                stop = {'to', 'be', 'the', 'a', 'an', 'of', 'in', 'on', 'one', 's', 'have', 'cause', 'make', 'id'}
                tokens = lambda text: set(re.findall(r'[a-z]+', text.lower())) - stop
                shared = tokens(claimed_gloss) & tokens(candidates[0][3])
                evidence['claimed_parent_gloss'] = claimed_gloss
                evidence['candidate_gloss'] = candidates[0][3]
                evidence['shared_semantic_tokens'] = sorted(shared)
                if shared:
                    by_key[out['entry_key']][13]=candidates[0][10]
                    evidence['accepted_parent']=candidates[0][10]
                else:
                    evidence['status']='semantic_conflict_or_unverified'
                    out['uncertainty_types'].append('derivation_semantics_unverified')
                    if 'uncertain' not in by_key[out['entry_key']][14].split(): by_key[out['entry_key']][14] += ' uncertain'
                    out['tags']=by_key[out['entry_key']][14]
            out['derivation']=evidence
    rows = commentary_causatives(rows, audit)
    rows = commentary_mentions(rows, audit)
    assert len({r[10] for r in rows}) == len(rows)
    alignment = json.loads((HERE/'same-source-reuse-alignment.json').read_text())
    for item in audit:
        related = [a for a in alignment['alignments'] if a['source_id'] == item['source_id']]
        if related: item['prior_same_source_attestations'] = related
    return rows, audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args=parser.parse_args(); rows,audit=prepare()
    target=CSV if args.install else HERE/'staged.csv'
    with target.open('w',newline='',encoding='utf-8') as stream: csv.writer(stream).writerows(rows)
    (AUDIT if args.install else HERE/'staged-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
    print(json.dumps({'physical_units':len(audit),'rows':len(rows),'statuses':collections.Counter(a['status'] for a in audit),'blank_gloss_rows':sum(not r[3] for r in rows),'derivation_edges':sum(bool(r[13]) for r in rows)}))


if __name__ == '__main__': main()
