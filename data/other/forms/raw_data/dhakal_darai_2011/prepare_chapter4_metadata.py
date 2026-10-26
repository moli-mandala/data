"""Propose Chapter 4 source analysis; not an installer or independent approval."""
import json
from pathlib import Path
from interlinear_grammar import parse
from chapter4_editorial_notes import scholarly_notes

P = Path(__file__).resolve().parent
KEY = 'dhakal2011:'
NOTES = {
    'p105:ch4:surface:l73:word:l76:w1': 'The source translation identifies Dawa as a death ritual.',
    'p105:ch4:surface:l46:word:l52:w1': 'The author identifies this postposition as borrowed from Nepali.',
    'p106:ch4:surface:l55:word:l58:w1': 'The source translation identifies Saur as a kind of fish.',
    'p106:ch4:surface:l29:word:l32:w1': 'The source prints the English gloss kiing-COM.',
    'p110:ch4:surface:l46:word:l49:w1': 'In this example, the present is offered when somebody dies.',
}


def normalize_tags(tags):
    result = []
    person = next((t for t in tags if t in {'1','2','3'}), None)
    number = next((t for t in tags if t in {'sg','pl'}), None)
    for tag in tags:
        if tag=='non-honorific':
            continue  # Preserved as a source qualification below, not informal.
        if tag in {'1','2','3'}:
            result.append(tag + number if number else {'1':'first-person','2':'second-person','3':'third-person'}[tag])
        elif person and tag == number:
            continue
        else:
            result.append({'inst':'instr','dem':'demonstrative','com':'comitative'}.get(tag,tag))
    return list(dict.fromkeys(result))


def build():
    records = [json.loads(line) for line in (P/'chapter4-adjudicated-records.jsonl').read_text().splitlines()]
    interlinear = [json.loads(line) for line in (P/'chapter4-interlinear-reviewed.jsonl').read_text().splitlines()]
    result = []
    adjudicated = set(json.loads((P/'chapter4-adjudication-reconciliation-20260927.json').read_text())['corrections_applied'])
    for source in records:
        record = dict(source)
        excluded = source['status'] in {'source-blank','zero-morphology-evidence','structural-evidence','comparison-control','segment-evidence','source-unacceptable','root-evidence'}
        record['disposition'] = 'audit-only' if excluded else 'proposed-attestation'
        if source['status']=='root-evidence':
            record['disposition_reason']='Source explicitly attributes tə root discussion to Middle Indo-Aryan; not a securely attributed Darai morpheme.'
        analysis = parse(source.get('gloss','')) if source['record_type']!='expressions' else {'gloss':source['gloss'],'tags':[],'labels':[]}
        tags = normalize_tags(source.get('tags',[]) + analysis['tags'])
        if source['source_unit_key'] in adjudicated:
            tags = [tag for tag in tags if tag != 'uncertain']
        if source['form'].startswith('-') and not excluded:
            tags.append('suffix')
        if source['form'].endswith('-') and not excluded:
            tags.append('stem')
        if source['record_type']=='expressions':
            tags += ['sentential']
            if ' ' in source['form']:tags.append('multiword-expression')
        record['proposed_grammar'] = dict(analysis,tags=list(dict.fromkeys(tags)))
        record['proposed_notes'] = scholarly_notes(source)
        if not excluded:
            record['lect'] = source.get('lect') or 'Chitwan'
            record['lect_basis'] = 'Explicit source attribution where present; otherwise the author’s thesis default on page 8, recorded in whole-thesis-attribution-policy.json.'
        if 'non-honorific' in source.get('tags',[]):
            record['source_register_qualification']='The source classifies this form as non-honorific.'
        record['metadata_status'] = 'Proposed; source notes, register, morphology and all grammar claims require final reconciliation'
        result.append(record)
    for source in interlinear:
        record = dict(source)
        conflict = 'source_interpretation_uncertainty' in source
        ambiguous = source.get('needs_manual_alignment',False)
        analysis = parse(source['candidate_gloss'], ambiguous=ambiguous, conflicting=conflict)
        excluded = source.get('source_example_status')=='source-unacceptable'
        record['disposition']='audit-only' if excluded else 'proposed-attestation'
        record['proposed_grammar']=analysis
        record['proposed_phonemic']=source['form']
        notes=[]
        if analysis.get('labels'):
            notes.append('Source interlinear labels: '+', '.join(analysis['labels'])+'.')
        for field in ['uncertainty','source_semantic_discrepancy','source_interpretation_uncertainty']:
            if source.get(field):notes.append(source[field])
        extra=NOTES.get(source['source_unit_key'].removeprefix(KEY))
        if extra:notes.append(extra)
        if 'ø' in source['form']:
            assert source['form'].endswith('-ø')
            record['proposed_phonemic']=source['form'][:-2]
            notes.append('The source writes a null nominative marker as -ø in its segmented analysis; it is not a pronounced vowel.')
            record['null_marker_policy']='Retain the entire literal segmented form and raw evidence; omit the explicitly null marker from proposed phonemic input only.'
        if source['source_unit_key']==KEY+'p105:ch4:surface:l46:word:l52:w1':
            analysis['tags'].append('loanword')
        record['proposed_notes']=' '.join(notes)
        record['lect']='Chitwan'
        record['metadata_status']='Proposed source-label interpretation; independent review pending'
        result.append(record)
    assert len(result)==1328 and len({r['source_unit_key'] for r in result})==1328
    return result


if __name__=='__main__':
    rows=build()
    (P/'chapter4-metadata-proposal.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'review units;',sum(r['disposition']=='proposed-attestation' for r in rows),'proposed attestations; no canonical installation')
