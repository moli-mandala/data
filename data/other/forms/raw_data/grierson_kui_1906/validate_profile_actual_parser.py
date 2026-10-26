"""Exercise the unmodified generic parser in a small source-local scratch tree."""
import collections
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import unicodedata as ud
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[4]
SCRATCH=DATA.parent/'tmp/kui-specimens/parser-sandbox'
STEM='20260927-kui-isolated-proposal'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return [json.loads(x) for x in p.read_text().splitlines()]

def main():
    inp=HERE/'table-metadata-candidate.json'
    table=json.loads(inp.read_text())['readings']
    table_by_prompt={r['prompt_id']:r for r in table}
    assert len(table_by_prompt)==233
    assert 'sentential' not in table_by_prompt[241]['tags']
    assert all('sentential' in table_by_prompt[p]['tags'] for p in range(220,241))
    assert all('participle' not in table_by_prompt[p]['tags'] for p in (170,171,177,178,218,219))
    status=json.loads((HERE/'table-metadata-status.json').read_text())
    assert sha(inp)==status['output_sha256']
    prompts={r['prompt_id']:r for r in read(HERE/'table-prompt-inventory.jsonl')}
    atoms=read(HERE/'specimens/interlinear-reviewed.jsonl')
    expressions=read(HERE/'specimens/expression-spans-reviewed.jsonl')
    rows=[]
    for r in table:
        p=prompts[r['prompt_id']]
        for i,f in enumerate(r['forms'],1):
            tags=list(r['tags'])
            if ' ' not in f:
                tags=[t for t in tags if t!='multiword-expression']
            rows.append(['Kui','',f,p['english_prompt'],'','',
                ' '.join(r.get('source_notes_additions',[])),
                f'grierson1906kui[p. {p["printed_page"]}, table prompt {r["prompt_id"]}, answer {i}]',
                '','',p['key']+f':answer{i}','','','',' '.join(tags)])
    for r in atoms:
        rows.append(['Kui','',r['raw_source_form'],r['raw_gloss'],'','',r['source_note'],
            f'grierson1906kui[p. {r["printed_page"]}, specimen {r["specimen"]}, line {r["physical_line"]}, word {r["word"]}]',
            '','',r['source_unit_key'],'','','','uncertain' if r['typography_uncertainty'] else ''])
    for r in expressions:
        rows.append(['Kui','',r['raw_source_form'],r['raw_interlinear_gloss'],'','','',
            f'grierson1906kui[pp. {",".join(map(str,r["printed_pages"]))}, {r["source_unit_key"]}]',
            '','',r['source_unit_key'],'','','','multiword-expression' if ' ' in r['raw_source_form'] else ''])
    assert len(rows)==len({r[10] for r in rows})==1225
    target=SCRATCH/'data/other/forms'
    target.mkdir(parents=True,exist_ok=True)
    (SCRATCH/'conversion').mkdir(exist_ok=True)
    codepins={}
    for name in ['make_cldf.py','source_meta.py']:
        shutil.copyfile(DATA/name,SCRATCH/name)
        codepins[name]=sha(DATA/name)
    profile=HERE/'metadata-profile-proposal/literal-preservation-proposed.txt'
    shutil.copyfile(profile,SCRATCH/'conversion/kui-literal-proposal.txt')
    settings=dict(file=STEM+'.csv', defaults=dict(
        transcription=dict(profile='kui-literal-proposal',input='form',preserve_hyphens=True),
        identity=dict(dedupe_by_entry_key=True),forms=dict(split_alternates=False),gloss=dict(grammar_tags=False)),
        sources=dict(grierson1906kui=dict(forms=dict(split_alternates=False),gloss=dict(grammar_tags=False),identity=dict(dedupe_by_entry_key=True))))
    (target/(STEM+'.yaml')).write_text(yaml.safe_dump(settings,sort_keys=False))
    os.chdir(SCRATCH)
    sys.path[:0]=[str(SCRATCH),str(DATA)]
    import make_cldf
    checks={}
    display_deltas=[]
    for normalization in ('NFC','NFD'):
        inputs=[[ud.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
        csvpath=target/(STEM+'.csv')
        with csvpath.open('w',newline='') as f:
            csv.writer(f).writerows(inputs)
        errors=io.StringIO()
        parsed,stats=make_cldf.parse_file(str(csvpath),errors,name=STEM)
        assert not errors.getvalue(),errors.getvalue()
        assert len(parsed)==stats['converted']==len(rows)
        assert [r.entry_key for r in parsed]==[r[10] for r in rows]
        for a,r in zip(parsed,inputs):
            assert (a.old_form,a.gloss,a.native,a.ipa,a.notes,a.source,a.tags)==(r[2],r[3],r[4],r[5],r[6],r[7],r[14]),a.entry_key
            expected=ud.normalize('NFC',r[2].strip('1234⁴5⁵67⁷,;.'))
            assert ud.normalize('NFC',a.form)==expected,(a.entry_key,a.form,expected)
            if normalization=='NFC' and a.form!=r[2]:
                display_deltas.append(dict(key=a.entry_key,original=r[2],display=a.form,reason='Generic parser strips boundary punctuation/digits before profile; Original remains exact.'))
        checks[normalization]=dict(rows=len(parsed),converted=stats['converted'],errors=0,ordered_keys=True,original_and_all_supplied_fields_preserved=True)
    covered=[k for e in expressions for k in e['atom_keys']]
    non_titles=[r['source_unit_key'] for r in atoms if r['unit_type']!='title_atom']
    assert collections.Counter(covered)==collections.Counter(non_titles)
    continuations=json.loads((HERE/'specimens/explicit-continuations-first-reading.json').read_text())
    affected=[]
    for c in continuations:
        matches=[e for e in expressions if all(k in e['atom_keys'] for k in c['atom_keys'])]
        assert len(matches)==1
        affected.append(dict(expression_key=matches[0]['source_unit_key'],physical_keys=c['atom_keys'],
            issue='Current span joins every physical atom with a space, yielding a space inside this printed continued word.',
            required_emission=c['combined_form'],disposition='Keep physical atoms in audit; exporter must use explicit joined lexical form and both locators.'))
    report=dict(status='actual parser fixture passed; integration and canonical registration remain pending',
        inputs={str(p.relative_to(HERE)):sha(p) for p in [inp,profile,HERE/'specimens/interlinear-reviewed.jsonl',HERE/'specimens/expression-spans-reviewed.jsonl']},
        parser_code_sha256=codepins,checks=checks,display_changes=len(display_deltas),display_delta_records=display_deltas,
        expression_accounting=dict(expressions=77,non_title_atoms=884,exact_partition=True,continuation_issues=affected),
        metadata_checks=dict(table_written_cells=233,table_rows=258,nonfinite_subtypes_not_inferred=True,fragment241_not_sentential=True),
        pending=['Apply two explicit continued-word joins in lexical/expression emission; retain all physical keys and locators.',
            'Type all77 expressions from source syntax/translation: punctuation segmentation alone does not prove sentential status.',
            'Decide how standalone interlinear atoms with source punctuation enter final display, retaining exact Original; tested generic route strips only its existing boundary set.',
            'Resolve final auxiliary bibliography, source claims, approved dialect registration and field-level uncertainty propagation.',
            'Integrate latest grammar/outside-comparison revisions, then refresh inventory/profile and run full integrated actual-parser/fresh audit.',
            'Canonical install/fullbuild/database/browser are not performed by this fixture.'],
        fixture_scope='Only table and four specimens,1225 rows. Grammar/comparands retain tokenizer coverage only; no invented control-language mappings.',
        canonical_changes=False)
    (HERE/'metadata-profile-proposal/actual-parser-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(checks=checks,display_changes=len(display_deltas),continuation_issues=len(affected)),indent=2))

if __name__=='__main__':
    main()
