"""Apply keyed author-CSV decisions without modifying the first readings."""
import copy
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
source = P / 'witness-expanded-proposed.jsonl'
rows = [json.loads(line) for line in source.open()]
decisions = json.loads((P / 'structured-metadata-decisions-20260927.json').read_text())
pos = {'noun':['noun'], 'verb':['verb'], 'adjective':['adj'], 'numeral':['num'],
       'pronoun':['pron'], 'adverb':['adv'], 'postposition':['postp'], 'adposition':[],
       'noun, adjective':['noun','adj'], 'conjunction':['conj'], 'determiner':['determiner'],
       'preposition':['prep'], 'particle':['part'], '?':['uncertain']}
changes = []

def note(c, text):
    c['notes'] = ' '.join(x for x in (c.get('notes','').strip(), text.strip()) if x)

def uncertain(c, reason):
    c['tags'].append('uncertain')
    c.setdefault('uncertainty_reasons', []).append(reason)
    note(c, reason)

for row in rows:
    raw = row['source_record']
    if row['representation'] != 'author repository CSV' or raw['status'] == 'blank':
        continue
    n = raw['physical_csv_row']
    lexical = raw['source_file'] == 'lexicon.csv'
    before = copy.deepcopy(row['candidates'])
    for c in row['candidates']:
        if lexical:
            fields = dict(zip(raw['raw_column_names'][:14], raw['raw_fields']))
            # Never reuse mutable POS lists from the first-reader: feminine is
            # licensed only by this record's own printed Form field.
            c['tags'] = list(pos[fields['Part of Speech']])
            if fields['Form'] == 'feminine':
                c['tags'].append('f')
            if ':paradigm:' in c['entry_key']:
                paradigm = next(p for p in raw['paradigm_candidates'] if p['source_unit_key'] == c['entry_key'])
                c['tags'] += {'f':['f'], 'pl':['pl'], 'pl.obl':['pl','obl'], 'comp':['degree']}[paradigm['source_grammar']]
            else:
                for group in ('lexicon_gloss_decisions','lexicon_function_decisions'):
                    d = decisions[group].get(str(n), {})
                    if 'gloss' in d:
                        c['gloss'] = d['gloss']
                    c['tags'] += d.get('add_tags', [])
                    if d.get('notes'):
                        note(c, d['notes'])
                    if d.get('uncertainty_type'):
                        uncertain(c, d['uncertainty_type'])
                if n in decisions['lexicon_head_uncertainty_rows']:
                    assert c['form'].endswith('?'), (n,c['form'])
                    c['form'] = c['form'][:-1].rstrip()
                    uncertain(c, 'The source questions this headword.')
                if n in decisions['lexicon_notes_to_etymology_rows']:
                    c['etymology'] = ' '.join(x for x in (c.get('etymology',''), fields['Notes']) if x)
                    c['notes'] = c.get('notes','').replace(fields['Notes'], '', 1).strip()
                if n in decisions['lexicon_notes_with_transcription_or_lexical_uncertainty_rows']:
                    uncertain(c, 'The source records a qualification or uncertainty: '+fields['Notes'])
        else:
            d = decisions['sentence_conflicts'].get(str(n))
            if d:
                uncertain(c, d['reason'])
            if n == 18:
                c['etymology'] = raw['notes']
                c['notes'] = ''
            if n == 77:
                uncertain(c, 'The source questions the garment translation robe.')
            if raw.get('interlinear_analysis'):
                c['source_interlinear_analysis'] = raw['interlinear_analysis']
        if ' ' in c['form']:
            c['tags'].append('multiword-expression')
        if c.get('variant_of_key'):
            c['tags'].append('alternate')
        c['tags'] = list(dict.fromkeys(c['tags']))
    # Tildes explicitly separate lexical alternatives. IPA2 retains its
    # speaker label; no undocumented equivalence of M and a dialect is made.
    if lexical and ' ~ ' in raw['form']:
        assert len(row['candidates']) == 1
        base = row['candidates'][0]
        forms = raw['form'].split(' ~ ')
        assert n in (44,46,47) and len(forms) == 2
        ipa = ['pu','pʰu'] if n == 46 else [raw['phonetic'], '']
        row['candidates'] = []
        for i, form in enumerate(forms):
            c = copy.deepcopy(base)
            c.update(entry_key=base['entry_key']+f':answer{i+1}', form=form, phonetic=ipa[i])
            c['tags'] = [t for t in c['tags'] if t != 'multiword-expression']
            if i:
                c['variant_of_key'] = row['candidates'][0]['entry_key']
                c['tags'].append('alternate')
            note(c, 'The source lists these headword alternatives: '+raw['form']+'.')
            if n != 46:
                note(c, 'The separately labelled IPA 2 pronunciation is retained in the source evidence; no speaker or dialect identity is inferred.')
            row['candidates'].append(c)
        row['alternate_decision'] = dict(raw_form=raw['form'], raw_phonetic=raw['phonetic'],
                                        labelled_pronunciation=raw['pronunciation_evidence'].get('IPA 2',''),
                                        rule='Explicit tilde list; primary IPA retained only on first printed alternative; optional p(ʰ)u expanded literally.')
    if before != row['candidates']:
        changes.append(dict(source_unit_key=row['source_unit_key'], before=before, after=row['candidates']))

out = P/'witness-metadata-reviewed-proposed.jsonl'
out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(P/'structured-metadata-application.json').write_text(json.dumps({
    'status':'proposal; independent output review pending',
    'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'decisions_sha256':hashlib.sha256((P/'structured-metadata-decisions-20260927.json').read_bytes()).hexdigest(),
    'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
    'changes':changes},ensure_ascii=False,indent=2)+'\n')
print('units',len(rows),'candidates',sum(len(r['candidates']) for r in rows),'changed_units',len(changes))
