"""Expand printed paradigm abbreviations into reviewable source-local candidates.

No canonical export: grammar/tag/gloss reconciliation and final audits are separate.
"""
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

P = Path(__file__).resolve().parent
NOUNS = {'horse', 'house', 'elephant', 'girl', 'sister'}


def main():
    manifest = json.loads((P/'reviewed-draft-manifest-20260927.json').read_text())
    for name, digest in manifest['input_sha256'].items():
        assert hashlib.sha256((P/name).read_bytes()).hexdigest() == digest, 'Regenerate reviewed drafts first: '+name
    for name, digest in manifest['output_sha256'].items():
        assert hashlib.sha256((P/name).read_bytes()).hexdigest() == digest, name
    units = [json.loads(s) for s in (P/'reviewed-draft-audit-20260927.jsonl').read_text().splitlines()]
    grammar = [u['reconciled'] for u in units if ':grammar:' in u['source_unit_key']]
    stems = {}
    for row in grammar:
        if row['group'] in NOUNS and row['item']=='1':
            assert '-' in row['singular_or_form']
            stems[row['group']] = row['singular_or_form'].split('-',1)[0]
    conditional = next(r for r in grammar if (r['group'],r['item'])==('fall','3'))
    stems['fall-conditional'] = conditional['singular_or_form'].split('-',1)[0]
    output = []
    for unit in units:
        row = unit['reconciled']
        result = dict(source_unit_key=unit['source_unit_key'], printed_page=row['page'], source_review=unit, candidates=[])
        if ':context:' in unit['source_unit_key']:
            result['disposition'] = 'attributed-context-only'
        elif ':sentences:' in unit['source_unit_key']:
            result['disposition'] = 'complete-sentence-candidate'
            result['candidates'] = [dict(candidate_key=unit['source_unit_key'], form=unicodedata.normalize('NFC',row['form']), literal_translation=row['literal_translation'], source_review_note=row['review_note'])]
        else:
            result['source_label'] = row['source_label']
            result['gloss_or_scope'] = row['gloss_or_scope']
            if row['source_label'].startswith('control;'):
                assert all(not row[f] or row[f]=='etc.' for f in ['singular_or_form','plural'])
                result['disposition'] = 'printed-continuation-control-no-invented-cells'
            else:
                result['disposition'] = 'grammar-candidates-interpretation-pending'
                for field in ['singular_or_form','plural']:
                    if not row[field]:
                        continue
                    pieces = row[field].split(';')
                    local_stem = stems.get(row['group'])
                    if row['group']=='fall' and row['item'] in {'3','4','5'}:
                        local_stem = stems['fall-conditional']
                    if row['group']=='intro-future' and row['item'] in {'1','2','3'}:
                        local_stem = row[field].split('-',1)[0]
                    for index, literal in enumerate(pieces,1):
                        literal=literal.strip()
                        form=literal
                        expansion='literal source form or suffix retained'
                        if local_stem is not None:
                            if literal.startswith('-'):
                                form=local_stem+literal[1:]
                                expansion='printed suffix attached to explicitly printed paradigm stem'
                            elif '-' in literal:
                                assert literal.split('-',1)[0]==local_stem, (row,literal)
                                form=literal.replace('-','',1)
                                expansion='printed stem/ending separator removed'
                        result['candidates'].append(dict(candidate_key=f"{unit['source_unit_key']}:{field}:{index}",form=unicodedata.normalize('NFC',form),literal_cell_piece=literal,source_field=field,printed_position=index,expansion=expansion,stem_evidence=local_stem))
        output.append(result)
    assert len(output)==309
    candidates=[c for u in output for c in u['candidates']]
    assert len({c['candidate_key'] for c in candidates})==len(candidates)
    assert all(c['form'] and c['form']!='etc.' and '\ufffd' not in c['form'] for c in candidates)
    (P/'whole-expansion-draft-20260927.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n'for u in output))
    summary=dict(status='source-local expansion draft only',groups=len(output),candidate_forms=len(candidates),noun_stems=stems,
                 control_groups=sum(not u['candidates'] for u in output),input_manifest_sha256=hashlib.sha256((P/'reviewed-draft-manifest-20260927.json').read_bytes()).hexdigest(),
                 remaining='Independent readings, semantic/grammar/number/gender interpretation, typed uncertainties, stable canonical-key choice, legacy288 integration, profile and final audit remain required before installation.')
    (P/'whole-expansion-summary-20260927.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
