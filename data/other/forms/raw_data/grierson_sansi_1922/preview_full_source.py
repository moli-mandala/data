"""Assemble the full reviewed Sansi source; proposal only, never canonical writes.

A source cell remains the identity boundary. Explicit alternatives expand through
source-specific rules; only identical specimen attestations within one lect and
register reuse an entry. Comparison controls and bound endings remain in audit.
"""
import csv
import importlib.util
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = 'grierson1922lsi11'
spec = importlib.util.spec_from_file_location('sansi_table_grammar', HERE / 'table_grammar.py')
grammar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grammar)
REGIONS = {
    'Northern Panjab': ('sansi_northern_punjab_lsi1922', 'Northern_Punjab'),
    'Panjab': ('sansi_panjab_lsi1922', 'Panjab'),
    'Saharanpur': ('sansi_saharanpur_lsi1922', 'Saharanpur'),
    'United Provinces': ('sansi_united_provinces_lsi1922', 'United_Provinces'),
    'Gujrat': ('sansi_gujrat_lsi1922', 'Gujrat'),
    'Gurdaspur': ('sansi_gurdaspur_lsi1922', 'Gurdaspur'),
    'Sialkot': ('sansi_sialkot_lsi1922', 'Sialkot'),
}


def read(name):
    with (HERE / name).open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert all(r['review_status'].startswith('second_reading') or 'uncertain' in r['review_status'] for r in rows)
    return rows


def nfc(s):
    return unicodedata.normalize('NFC', s.strip())


def regional_tags(attribution):
    label = attribution.replace(' district', '').replace(' argot', '')
    return [f'dialect:Sansi:{REGIONS[x][0]}:{REGIONS[x][1]}'
            for x in label.split(';') if x in REGIONS]


def source_tags(text):
    tokens = re.split(r'[;\s]+', text.strip())
    aliases = {'v': 'verb', 'past': 'pret', 'ptcp': 'participle',
               'cvb': 'conjunctive-participle', 'imp': 'impv',
               'subj': 'subjunctive', 'masc': 'm', 'fem': 'f'}
    result = []
    for t in tokens:
        if not t:
            continue
        if t in {'1', '2', '3'}:
            number = 'sg' if 'sg' in tokens else 'pl' if 'pl' in tokens else ''
            result.append(t + number if number else {'1':'first-person','2':'second-person','3':'third-person'}[t])
        else:
            result.append(aliases.get(t, t))
    return list(dict.fromkeys(result))


def row(key, form, gloss, citation, attribution='', tags=(), notes='', etymology=''):
    return ['Sansi', '', nfc(form), gloss, '', '', notes, citation, '', etymology,
            key, '', '', '', ' '.join(dict.fromkeys(regional_tags(attribution) + list(tags)))]


def alternatives(item, register, raw):
    """Only expand alternatives that the reviewed source actually prints."""
    ordinary = register == 'ordinary'
    if item in {54, 59, 168, 175, 217} or (ordinary and item in {76, 85, 97}) or (not ordinary and item in {65, 71, 219}):
        cleaned = re.sub(r'\s*\((?:sing\.|plur\.|Little)\)', '', raw)
        return [x.strip() for x in cleaned.split(',')]
    if item == 82 and not ordinary:
        return ['Khlōṇā', 'khaḷā hōpṇā', 'raḷā hōpṇā']
    if item == 102 and ordinary:
        return ['Bappā-gā', 'Bappā-gē', 'Bappā-gī', 'Bappā-gīā̃']
    if item == 119 and ordinary:
        return [a+' '+b for a in ['Chaṅgā','nēk'] for b in ['ādmī','banda']]
    if item in {120,124} and ordinary:
        return ['Chaṅgē '+x for x in (['ādmīā-gā','bandē-gā'] if item==120 else ['ādmī','bandē'])]
    if item in {129,131} and not ordinary:
        return [x+' '+('bōrā' if item==129 else 'bōrī') for x in (['Nhaiṛā','nharāb'] if item==129 else ['Nhaiṛī','nharāb'])]
    if item == 133:
        context, adjective = raw.split('] ')
        return ['['+x+'] '+adjective for x in context[1:].split(' or ')]
    if item == 162 or (ordinary and item in {192,193}) or item == 203:
        main = raw.split(' (')[0]
        return [main, main.rsplit(' ',1)[0]+' sīyyā']
    if item == 191:
        return [x.strip().removeprefix('or ') for x in raw.split(',')]
    if item == 211 and ordinary:
        return [raw.split(' (')[0]]
    if 211 <= item <= 216 and not ordinary:
        main, alternate = raw.split(' (')
        return [main, main.split(' ',1)[0]+' '+alternate.rstrip(')').removeprefix('or ')]
    # The alternatives below replace one explicitly bounded constituent,
    # preserving the rest of the printed sentence in each emitted answer.
    replacements = {
        (225,'argot'): ('buskiā (or khapṇiā)', ['buskiā','khapṇiā']),
        (231,'argot'): ('buskiā (or khapṇiā)', ['buskiā','khapṇiā']),
        (228,'ordinary'): ('kōṭlē (baint or sōṭē)', ['kōṭlē','baint','sōṭē']),
        (228,'argot'): ('nōṭlē (nhōṭē)', ['nōṭlē','nhōṭē']),
        (232,'argot'): ('balnē (or rukṇā or lābē)', ['balnē','rukṇā','lābē']),
        (234,'argot'): ('rukṇā (baluā, lābā)', ['rukṇā','baluā','lābā']),
        (241,'argot'): ('Ḍhāmē-(or nādā)-gē', ['Ḍhāmē-gē','nādā-gē']),
    }
    if (item,register) in replacements:
        old, new = replacements[item,register]
        assert old in raw, (item,register,raw)
        return [raw.replace(old,x) for x in new]
    return [raw]


def table_notes(item, register, c):
    notes = []
    if item in {133,136}:
        notes.append('The source prompt explicitly marks comparative degree; bracketed comparison context is retained.')
    if item in {134,137}:
        notes.append('The source prompt explicitly marks superlative degree; bracketed comparison context is retained.')
    if item == 76 and register == 'ordinary':
        notes.append('The printed qualifier “Little” applies to the first answer Chiṛiyā.')
    if item == 211 and register == 'ordinary':
        notes.append('The source gives gēā as the pronunciation of gayā.')
    if item == 162:
        notes.append('The source prints “etc.” after the two explicit alternatives; no additional form is inferred.')
    if item == 97 and register == 'argot':
        notes.append('The argot cell prints Jēkar jē without the comma present in the ordinary column; the literal expression is retained.')
    if item in {113,115,125} and (register == 'argot' or item == 113):
        notes.append(c['note'])
    return ' '.join(notes)


def generate():
    out, audit = [], []
    table = read('table-staged.tsv')
    assert [int(r['prompt']) for r in table] == list(range(1,242))
    for c in table:
        item = int(c['prompt'])
        for register in ['ordinary','argot']:
            raw = nfc(c[register])
            base = f'{SOURCE}:sasi_{register}:{item}'
            answers = alternatives(item,register,raw) if raw else []
            keys = []
            for j, form in enumerate(answers,1):
                # Legacy pilot multi-answer keys use :answer:1, not a bare first key.
                key = base if len(answers)==1 else f'{base}:answer:{j}'
                keys.append(key)
                tags = grammar.explicit_table_tags(item)
                if register == 'argot': tags.append('argot')
                if item in {168,175,217}: tags += ['impv', '2sg' if j==1 else '2pl', 'sg' if j==1 else 'pl']
                if item in {133,134,136,137}: tags.append('degree')
                if item in {113,115,125} and (register=='argot' or item==113): tags.append('uncertain')
                gloss = ('Little bird' if j==1 else 'Bird') if item==76 and register=='ordinary' else c['gloss']
                out.append(row(key,form,gloss,f'{SOURCE}[p. {c["page"]}, item {item}, {register}]',
                               'Northern Panjab' if register=='ordinary' else '', tags,table_notes(item,register,c)))
            audit.append(dict(c,section='table',source_cell_key=base,register=register,raw_cell=raw,
                              entry_keys=keys,reuse_entry_keys=[],status='ingested' if keys else 'source_blank',
                              english_printed_page=int(c['page'])))

    prose = read('prose-staged.tsv')
    by_unit = {(c['page'],c['unit']):c for c in prose}
    def prose_key(c):
        return f'{SOURCE}:sansi:prose:p{c["page"]}:{c["unit"]}'
    explicit_parents = {
        ('60','ten-prefix'): ('60','ten-base'), ('60','ten-reduced'): ('60','ten-base'),
        **{('60',f'disguise-hair-{i}'):('60','disguise-hair-0') for i in [1,2]},
        **{('60',f'disguise-foot-{i}'):('60','disguise-foot-0') for i in [1,2,3]},
        **{('63',f'my{i}'):('63','my-base') for i in [1,2,3]},
    }
    for c in prose:
        key = prose_key(c)
        emit = c['status'] in {'target','target_contextual_attribution'}
        tags = source_tags(c['tags'])
        if c['attribution']=='argot' or c['attribution'].endswith(' argot'): tags.append('argot')
        notes = c['source_claim']
        etymology = ''
        parent_key = ''
        relation = None
        paired = explicit_parents.get((c['page'],c['unit']), (c['page'],c['unit']+'-base'))
        base = by_unit.get(paired)
        if base and 'argot' in tags:
            # Explicit source transformations only, never inferred by form matching.
            # A source's tentative etymological comparison remains unlinked.
            tentative = any(x in c['source_claim'].lower() for x in ['perhaps','tentative','may be','apparently'])
            target_base = base['status'] in {'target','target_contextual_attribution'}
            compatible = c['gloss'] == base['gloss']
            relation = dict(type='source_argot_transformation', target_source_unit=prose_key(base),
                            target_form=base['form'], target_gloss=base['gloss'],
                            target_attribution=base['attribution'], tentative=tentative,
                            disposition='linked' if target_base and compatible and not tentative else
                            'tentative_source_claim' if tentative else 'comparison_control' if not target_base else 'different_sense')
            etymology = f"The source compares argot {c['form']} with {base['form']} ({base['gloss']})."
            if tentative: etymology += ' The source qualifies this explanation as tentative.'
            if base['status']=='target_contextual_attribution':
                etymology += ' The base is attributed to ordinary Sansi from the source context; this attribution is uncertain.'
                tags.append('uncertain')
            if relation['disposition']=='linked':
                parent_key=prose_key(base)
                tags.append('derived')
            notes = ''
        elif c['unit'].endswith('-base') and (c['page'],c['unit'][:-5]) in by_unit:
            partner=by_unit[c['page'],c['unit'][:-5]]
            notes=f"Printed comparison base for argot {partner['form']}."
            if c['status']=='target_contextual_attribution':
                notes += ' Attribution to ordinary Sansi follows the source’s general statement that the argot alters ordinary words; this particular base is not separately language-labelled.'
        if c['page']=='60' and c['unit'] in {'poison','advocate','lower-leg'}:
            tags.append('figurative')
        if c['unit']=='water-kolhati': notes='Explicit comparison with Sansi argot chaī̃.'
        if c['unit']=='argot-name': tags.append('proper-noun')
        if c['gloss']=='Kashmir': tags.append('proper-noun')
        if c['page']=='63' and c['unit'].startswith('copula'):
            tags.append('copula')
        notes=notes.replace(' Explicit repeated example in vowel-change discussion; reconcile exact form/gloss only with locators.', '').replace('Explicit repeated example in vowel-change discussion; reconcile exact form/gloss only with locators.', 'Example in the source’s vowel-change discussion.').replace('Explicit recap of source variants; retain repeated attestation for exact-form/gloss reconciliation.', 'The source explicitly recapitulates these variants.').replace('each explicit form retained.', 'explicit forms printed.').replace('retain each printed alternative.', 'explicit alternatives printed.')
        if emit:
            gloss='be' if c['page']=='63' and c['unit'].startswith('copula') else c['gloss']
            emitted=row(key,c['form'],gloss,f'{SOURCE}[p. {c["page"]}, prose {c["unit"]}]',c['attribution'],tags,notes,etymology)
            emitted[13]=parent_key
            out.append(emitted)
        reuse=[]
        if c['status']=='pronunciation_evidence':
            target_unit=c['unit'].replace('approximation','pronunciation')
            target=next(r for r in out if r[10]==prose_key(by_unit[c['page'],target_unit]))
            target[5]=c['form']
            target[6]+=' The Phonemic field retains the source’s approximate pronunciation notation literally; it is not an IPA conversion.'
            reuse=[target[10]]
        audit.append(dict(c,section='prose',source_cell_key=key,entry_keys=[key] if emit else [],reuse_entry_keys=reuse,
                          status='ingested' if emit else c['status'],source_status=c['status'],
                          structured_relation=relation,final_notes=notes,etymology=etymology))

    seen = {}
    for c in read('specimen-staged.tsv'):
        key = f'{SOURCE}:sansi:specimen:{c["unit"]}'
        form, gloss = nfc(c['form']), c['gloss']
        control = c['lect'].startswith('District Kheri')
        register = 'argot' if c['specimen'].startswith('argot') else 'ordinary'
        pair = (c['lect'],register,form,gloss)
        citation = f'{SOURCE}[p. {c["page"]}, specimen {c["specimen"]}, line {c["line"]}, aligned word {c["word"]}]'
        tags = ['argot'] if register=='argot' else []
        notes = c['note'].replace('Aligned Roman/English source token; no inferred lemma.', '').strip()
        if 'uncertain' in c['review_status']: tags.append('uncertain')
        keys,reuse=[],[]
        if not control:
            if pair in seen:
                old = out[seen[pair]]
                old[7] += ';'+citation
                old[14] = ' '.join(dict.fromkeys(old[14].split()+tags))
                if notes and notes not in old[6]: old[6] = ' '.join([old[6],notes]).strip()
                reuse=[old[10]]
            else:
                seen[pair]=len(out)
                keys=[key]
                out.append(row(key,form,gloss,citation,c['lect'],tags,notes))
        audit.append(dict(c,section='specimen',source_cell_key=key,register=register,
                          entry_keys=keys,reuse_entry_keys=reuse,
                          status='non_target_Hindostani_control' if control else 'reused_identical_lect_register_form_gloss' if reuse else 'ingested'))
    assert len(audit)==482+573+1507
    keys={r[10] for r in out}
    assert len(keys)==len(out)
    assert all(k in keys for a in audit for k in a['entry_keys']+a['reuse_entry_keys'])
    old = HERE.parents[1] / '20260925-grierson-sansi.csv'
    with old.open() as f: legacy={r[10] for r in csv.reader(f)}
    assert legacy <= keys, legacy-keys
    return out,audit

if __name__=='__main__':
    rows,audit=generate()
    with (HERE/'full-preview.csv').open('w',newline='') as f:
        csv.writer(f,lineterminator='\n').writerows(rows)
    with (HERE/'full-preview-audit.jsonl').open('w') as f:
        for a in audit: f.write(json.dumps(a,ensure_ascii=False)+'\n')
    print(f'{len(audit)} source units; {len(rows)} proposed forms; no canonical writes')
