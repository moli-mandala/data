"""Prepare review metadata for every indexed Darai unit; never install or export.

All source claims remain accessible through the pinned evidence pointer. This pass
classifies explicit controls and proposes exact source-label mappings. It does not
invent POS from English definitions or collapse repeated attestations.
"""
import collections
import hashlib
import json
from pathlib import Path
from interlinear_grammar import parse, parse_source_grammar
from chapter4_editorial_notes import scholarly_notes
P=Path(__file__).resolve().parent
CONTROL_STATUS={'name-evidence','structural-evidence','comparison-control','segment-evidence',
 'zero-morphology-evidence','source-blank','excluded','source-unacceptable','root-evidence'}
CONTROL_DISPOSITION={'source-absence-control','foreign-comparison-control','structural-control',
 'nonlexical-source-cell','non-Darai-context-control','non-Darai-English-gloss-control',
 'English-gloss-control','unacceptable-source-control','foreign-language-quotation-control',
 'zero-morphology-evidence','phonological-context','source-unacceptable'}
CLAIM_FIELDS=['note','uncertainty','source_interpretation_uncertainty','source_semantic_discrepancy',
 'encoding_uncertainty','transcription_uncertainty','source_analysis_uncertainty','source_function',
 'source_use','source_grammar','subject_label','object_label','source_subject','source_object',
 'source_receptor','source_tense','source_mood','source_agreement','source_agreement_type',
 'source_person_number','source_category','source_role','source_language','source_dialect',
 'attribution','source_reference','text_ids','text_id','source_surface_form',
 'source_analysis_extra_fragments','source_solution','source_senses','source_alternates',
 'source_expression_groups','source_translation_groups','source_table','table_context',
 'source_header','source_verb_type','root_source_unit_key','source_register_qualification',
 'lect','witness','phonemic_status','consultant_provenance','dialect_policy','derivation_group']

def build():
    index=[json.loads(l) for l in (P/'whole-evidence-index.jsonl').open()]
    grouped=collections.defaultdict(list)
    for row in index:grouped[row['evidence_file']].append(row)
    output=[]
    for name,items in grouped.items():
        sources=[json.loads(l) for l in (P/name).open()]
        for item in items:
            source=sources[item['evidence_line']-1]
            assert source.get('source_unit_key',source.get('key'))==item['source_unit_key']
            control=source.get('status') in CONTROL_STATUS or source.get('disposition') in CONTROL_DISPOSITION
            control=control or source.get('source_example_status')=='source-unacceptable'
            control=control or item['kind'] in {'source-context-control','paradigm-absence-control','source-absence-control'}
            aligned=item['kind']=='aligned-word' or ':word:' in item['source_unit_key']
            conflict=any(source.get(k) for k in ['source_interpretation_uncertainty','source_analysis_uncertainty'])
            if aligned:
                # Resolve contradictory labels individually after parsing. A
                # lexical gloss mismatch does not invalidate unrelated person.
                reviewed_alignment = [d for d in source.get('review_decisions', [])
                    if d.get('key') == item['source_unit_key']
                    and ('printed_gloss' in d or 'gloss' in d)
                    and d.get('printed_gloss', d.get('gloss')) == item['gloss']]
                ambiguous = source.get('needs_manual_alignment', False) and not reviewed_alignment
                grammar=parse(item['gloss'], ambiguous=ambiguous)
                if reviewed_alignment:
                    grammar['alignment_adjudication'] = reviewed_alignment
            elif source.get('source_grammar'):
                grammar=parse_source_grammar(source['source_grammar']);grammar['gloss']=item['gloss']
            else:
                grammar={'gloss':item['gloss'],'tags':source.get('tags',[]),'labels':[]}
            # Existing accepted row tags remain intact until explicit reconciled change.
            if source.get('output'):
                grammar['existing_tags']=source['output'][14].split()
            claims={k:source[k] for k in CLAIM_FIELDS if source.get(k) not in (None,'',[],{})}
            if name=='chapter4-adjudicated-records.jsonl':
                note=scholarly_notes(source)
                if note:claims['chapter4_editorial_context']=note
            issues=[]
            if source.get('unresolved_glyphs') or source.get('encoding_uncertainty') or source.get('transcription_uncertainty'):
                issues.append('transcription or encoding uncertainty: no inferred phonemic value')
            if conflict:issues.append('source analysis conflict requires role-scoped metadata')
            if source.get('source_alternates') and len(source['source_alternates'])>1:issues.append('explicit source alternatives require expansion or attestation reuse')
            if source.get('source_expression_groups'):issues.append('multiple translated statements require explicit output grouping')
            if not control and not item['form']:issues.append('blank target form requires disposition')
            output.append(dict(source_unit_key=item['source_unit_key'],evidence_file=name,evidence_line=item['evidence_line'],
                printed_page=item['printed_page'],pdf_page=item['pdf_page'],kind=item['kind'],
                form=item['form'],source_gloss=item['gloss'],proposed_grammar=grammar,
                source_claims=claims,source_status=source.get('status'),source_disposition=source.get('disposition'),
                disposition='audit-only explicit control' if control else 'candidate attestation; review pending',
                reuse_source_unit_key=item.get('source_reuse_key'),parent_expression_key=item.get('parent_expression_key'),
                open_review=issues,metadata_status='Not frozen; source claims, roles, variants and profiles require full reconciliation'))
    assert len(output)==len(index)
    return output

if __name__=='__main__':
    rows=build();f=P/'whole-metadata-review.jsonl'
    f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report=dict(status='First whole-source metadata proposal; not accepted lexical export',units=len(rows),
      dispositions=dict(collections.Counter(r['disposition'] for r in rows)),
      open_review=dict(collections.Counter(x for r in rows for x in r['open_review'])),
      metadata_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
      evidence_index_sha256=hashlib.sha256((P/'whole-evidence-index.jsonl').read_bytes()).hexdigest())
    (P/'whole-metadata-review-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
