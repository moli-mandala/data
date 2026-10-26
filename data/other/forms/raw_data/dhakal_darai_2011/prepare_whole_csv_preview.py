"""Whole-source review preview, with no canonical installation command.

Occurrence rows are deliberately retained pending citation-preserving reuse review.
Scholarly Notes are a review surface here, not yet an accepted editorial export.
"""
import csv
import collections
import hashlib
import json
import re
from pathlib import Path

from whole_legacy_baseline import legacy_rows, legacy_profile

P = Path(__file__).resolve().parent
LECTS = {name: f'dialect:Darai:darai_{name.lower()}:{name}'
         for name in ['Chitwan', 'Tanahun', 'Nawalparasi']}
NOTE_FIELDS = ['note', 'chapter4_editorial_context', 'source_interpretation_uncertainty',
               'source_analysis_uncertainty', 'source_semantic_discrepancy',
               'encoding_uncertainty', 'transcription_uncertainty']
CLAIM_LABELS = {'source_grammar':'grammatical label', 'source_use':'use',
    'source_function':'function', 'source_category':'category', 'source_role':'role',
    'source_verb_type':'verb class', 'source_mood':'mood', 'source_tense':'tense'}


def notes_for(row):
    claims = row['source_claims']
    notes = [claims[k] for k in NOTE_FIELDS if k!='note' and isinstance(claims.get(k), str) and claims[k]]
    if row.get('editorial_source_note'):notes.insert(0,row['editorial_source_note'])
    notes=[note.replace('Higher-resolution independent review pending.','').strip() for note in notes]
    notes.extend(f'Source {label}: {claims[k]}.' for k,label in CLAIM_LABELS.items() if claims.get(k))
    if row['proposed_grammar'].get('labels') and row['source_gloss'].strip():
        notes.append('Printed interlinear analysis: '+row['source_gloss']+'.')
    if row.get('interlinear_scope_decision', {}).get('note'):
        notes.append(row['interlinear_scope_decision']['note'])
    if row.get('source_heading_decision'):
        notes.append(row['source_heading_decision']['source_heading_note'])
    if claims.get('source_surface_form') and claims['source_surface_form'] != row['form']:
        notes.append('Corresponding printed surface form: '+claims['source_surface_form']+'.')
    notes.extend(row.get('source_role_notes', []))
    attribution=row['source_attribution']
    if attribution.get('witness'):
        notes.append('Source attribution: '+attribution['witness']+'.')
    context=attribution.get('corpus_context')
    if context:
        notes.append('Source narrative: '+context['title']+'; speaker: '+context['speaker']+'.')
        if context.get('recording_locality'):
            notes.append('Source recording locality: '+context['recording_locality']+'.')
    if claims.get('text_id'):
        notes.append('Source text identifier: '+claims['text_id']+'.')
    if claims.get('text_ids'):
        notes.append('Source corpus references: '+', '.join(claims['text_ids'])+'.')
    if claims.get('speaker_turn_label'):
        notes.append('Source speaker-turn label: '+claims['speaker_turn_label']+'.')
    if row.get('source_translation_type'):
        notes.append('Source '+row['source_translation_type']+' translation of the proverb.')
    if claims.get('source_solution'):
        notes.append('Source riddle solution: '+claims['source_solution']['raw'].strip('"'))
    if row.get('source_notation_decision') and row['source_notation_decision'].get('expanded_forms') is None:
        notes.append('The source prints a compressed group whose alternative scope remains uncertain: '+row['form']+'.')
    return notes


def citation_pages(row, source):
    """Use physical evidence pages, not all pages of a cited comparison/context."""
    if row.get('editorial_statement'):
        group=row['source_claims']['source_expression_groups'][row['editorial_statement']-1]
        pages={r['printed_page'] for r in group['source_surface_lines']}
        pages.add(group['source_translation_line']['printed_page'])
    else:
        pages=set(source.get('owner_review_pages', []))
        for location in source.get('source_locations', []):
            if isinstance(location,dict) and location.get('printed_page'):
                pages.add(location['printed_page'])
        if not pages:pages={row['printed_page']}
    return sorted(pages,key=lambda p:(not isinstance(p,int),p))


def build():
    old={r[10]:r for r in legacy_rows()}
    routes={r['source_unit_key']:r for r in map(json.loads,(P/'whole-transcription-proposal.jsonl').open())}
    evidence_cache={}
    metadata=[json.loads(line) for line in (P/'whole-expanded-metadata-review.jsonl').open()]
    graph_decisions=json.loads((P/'whole-source-graph-decisions.json').read_text())
    variant_decisions=graph_decisions['variant_decisions']
    variants={r['child_key']:r for r in variant_decisions}
    explicit_derivations={r['child_key']:r for r in graph_decisions.get('derivation_decisions',[])}
    source_forms={r['source_unit_key']:r['form'] for r in metadata}
    derivations=collections.defaultdict(dict)
    for row in metadata:
        group=row['source_claims'].get('derivation_group')
        if group:
            assert row['kind'] not in derivations[group]
            derivations[group][row['kind']]=row['source_unit_key']
    output={'author-transcription':[], 'literal-preservation':[]}
    for row in metadata:
        if row['disposition'].startswith('audit-only'):
            continue
        key=row['source_unit_key'];route=routes[key]
        name=row['evidence_file']
        if name not in evidence_cache:
            evidence_cache[name]=[json.loads(line) for line in (P/name).open()]
        source=evidence_cache[name][row['evidence_line']-1]
        pages=citation_pages(row,source)
        citation=';'.join(f'dhakal2011darai[p. {page}]' for page in pages)
        values=old[key][:] if key in old else ['Darai','',row['form'],row['proposed_grammar']['gloss'],'',route['proposed_phonemic'],'',citation,'','',key,'','','','']
        notes=([values[6]] if values[6] else [])+notes_for(row)
        if route['route']!='author-transcription':
            notes.extend(route['transcription_decisions'])
        else:
            notes.extend(n for n in route['transcription_decisions'] if n.startswith('Source '))
        values[6]=' '.join(dict.fromkeys(notes))
        tags=re.split(r'[;\s]+',values[14].strip()) if values[14].strip() else []
        lect=row['source_attribution'].get('lect')
        if lect in LECTS:tags.append(LECTS[lect])
        tags.extend(row['proposed_grammar']['tags'])
        if row['kind'] in {'whole-expression','expressions','diagram-expression'} and len(row['form'].split())>1:
            tags.append('multiword-expression')
        if row['open_review'] or route['route']=='literal-unresolved':tags.append('uncertain')
        values[14]=' '.join(dict.fromkeys(tags))
        if key in variants:
            decision=variants[key]
            assert row['form']==decision['child_form']
            assert source_forms[decision['parent_key']]==decision['parent_form']
            assert not values[11]
            values[11]=decision['parent_key']
        if row['kind']=='derivation-output':
            group=derivations[row['source_claims']['derivation_group']]
            assert set(group)=={'derivation-base','derivation-affix','derivation-output'}
            values[13]='|'.join(group[k] for k in ['derivation-base','derivation-affix'])
            values[14]+=' adv derived'
            values[6]+=' Source explicitly derives this adverb by suffixing se to the listed base.'
        if key in explicit_derivations:
            decision=explicit_derivations[key]
            assert row['form']==decision['child_form']
            assert source_forms[decision['parent_key']]==decision['parent_form']
            assert not values[13]
            values[13]=decision['parent_key']
            values[6]+=' '+decision['source_assertion']
        target='author-transcription' if route['route']=='author-transcription' else 'literal-preservation'
        output[target].append(values)
    assert len({r[10] for group in output.values() for r in group})==sum(map(len,output.values()))
    assert set(old)<={r[10] for group in output.values() for r in group}
    return output


if __name__=='__main__':
    groups=build();files={}
    for route,rows in groups.items():
        file=P/('whole-preview-'+route+'.csv')
        with file.open('w',newline='') as stream:csv.writer(stream).writerows(rows)
        files[file.name]={'rows':len(rows),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    report={'status':'Whole-source occurrence preview; not frozen or installed', 'files':files,
        'pending':['Editorial Notes cleanup and all claim retention checks','Citation-preserving reuse and all source-unit accounting','Metadata/profile/actual parser tests','Independent original-versus-output audit'],
        'canonical_changed':False}
    (P/'whole-csv-preview-status.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
