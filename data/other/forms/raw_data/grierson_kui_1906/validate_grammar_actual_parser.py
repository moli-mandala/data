"""Isolated actual-parser check of all resolved, nonblank grammar candidates."""
import ast
import collections
import csv
import io
import json
import os
import shutil
import sys
import unicodedata as ud
import yaml
from validate_profile_actual_parser import HERE, DATA, sha, read

def main():
    source=HERE/'grammar-metadata-candidate.jsonl'
    assert sha(source)=='2806561d2a09c15db6a98ff921c799c932301f4f0427a0a18ec334f779a5050f'
    records=read(source)
    assert len(records)==432
    evidence=HERE/'metadata-profile-proposal'
    prior=evidence/'grammar-parser-accounting-before-conjunctive-tag.jsonl'
    if not prior.exists():
        old_report=json.loads((evidence/'grammar-actual-parser-review.json').read_text())
        assert old_report['input_sha256']=='c9e7ec01d6a2279def3b31d3d9cd53d6a53e11166c554bc639cdb4b93c9d6567'
        shutil.copyfile(evidence/'grammar-parser-accounting.jsonl',prior)
        shutil.copyfile(evidence/'grammar-actual-parser-review.json',evidence/'grammar-actual-parser-before-conjunctive-tag.json')
    old={r['source_unit_key']:r['source_record'] for r in read(prior)}
    delta=[]
    for r in records:
        for field in set(old[r['source_unit_key']])|set(r):
            if old[r['source_unit_key']].get(field)!=r.get(field):
                delta.append(dict(key=r['source_unit_key'],field=field,old=old[r['source_unit_key']].get(field),new=r.get(field)))
    assert len(delta)==1 and delta[0]['key']=='grierson1906kui:p469:chinna-negative-participle:1'
    assert delta[0]['field']=='tags' and delta[0]['new']==delta[0]['old']+['conjunctive-participle']
    mapping_path=HERE/'source-lect-mapping-approved-20260927.json'
    routes={r['key']:r for r in json.loads(mapping_path.read_text())['mappings']}
    external={'external-Gondi':'Gondi','external-Kanarese':'Kannada','external-Tamil':'Tamil','external-Telugu':'Telugu'}
    languages={r['ID'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
    tree=ast.parse((DATA/'tags.py').read_text())
    allowed=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='GRAMMATICAL_TAGS' for t in n.targets))
    allowed |= next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='GENDER_TAGS' for t in n.targets))
    rows=[]
    audit=[]
    for r in records:
        assert not set(r['tags'])-allowed,(r['source_unit_key'],set(r['tags'])-allowed)
        item=dict(source_unit_key=r['source_unit_key'],source_record=r,route_key=r['route_key'])
        if not r['form']:
            item.update(disposition='documentary_blank',reason='Printed empty paradigm cell; no invented form.')
            audit.append(item)
            continue
        route=routes.get(r['route_key'],{})
        base=route.get('proposed_base') or external.get(r['route_key'])
        if not base:
            item.update(disposition='held_unresolved_route',reason=route.get('disposition','Joint Kanarese/Tamil attribution is not a unique language mapping.'))
            audit.append(item)
            continue
        assert base in languages,(r['source_unit_key'],base)
        if route:
            assert route['accepted'] is True
        tags=list(r['tags'])
        uncertainty=[]
        if r.get('uncertainty'):
            uncertainty.append(dict(type='source-typography-or-scope',reason=r['uncertainty']))
        if route.get('uncertainty_type'):
            uncertainty.append(dict(type=route['uncertainty_type'],reason=route['provenance_rule']))
        if uncertainty and 'uncertain' not in tags:
            tags.append('uncertain')
        note=r['source_note']
        if r.get('witness'):
            note+=' Source witness: '+r['witness']+'.'
        if route.get('uncertainty_type'):
            note+=' '+route['provenance_rule']
        row=[base,'',r['form'],r['gloss'],'','',note,
             f'grierson1906kui[p. {r["printed_page"]}, {r["block"]}, {r["source_unit_key"].rsplit(":",1)[1]}]',
             '','',r['source_unit_key'],'','','',' '.join(tags)]
        rows.append(row)
        item.update(disposition='parser_fixture_emission',language_id=base,
                    proposed_dialect_id=route.get('proposed_dialect_id'),
                    typed_uncertainty=uncertainty,emitted_tags=tags,
                    route_evidence='Approved source lect mapping' if route else 'Explicit single-language source label and existing canonical base registry',
                    dialect_registration_pending=bool(route.get('proposed_dialect_id')))
        audit.append(item)
    counts=collections.Counter(r['disposition'] for r in audit)
    assert counts==dict(parser_fixture_emission=413,documentary_blank=16,held_unresolved_route=3),counts
    assert len({r[10] for r in rows})==413
    scratch=DATA.parent/'tmp/kui-specimens/grammar-parser-sandbox'
    target=scratch/'data/other/forms'
    target.mkdir(parents=True,exist_ok=True)
    (scratch/'conversion').mkdir(exist_ok=True)
    for name in ('make_cldf.py','source_meta.py'):
        shutil.copyfile(DATA/name,scratch/name)
    profile=HERE/'metadata-profile-proposal/literal-preservation-proposed.txt'
    shutil.copyfile(profile,scratch/'conversion/kui-literal-proposal.txt')
    stem='20260927-kui-grammar-isolated'
    settings=dict(file=stem+'.csv',defaults=dict(transcription=dict(profile='kui-literal-proposal',input='form',preserve_hyphens=True),
        identity=dict(dedupe_by_entry_key=True),forms=dict(split_alternates=False),gloss=dict(grammar_tags=False)),
        sources=dict(grierson1906kui=dict(forms=dict(split_alternates=False),gloss=dict(grammar_tags=False))))
    (target/(stem+'.yaml')).write_text(yaml.safe_dump(settings,sort_keys=False))
    os.chdir(scratch)
    sys.path[:0]=[str(scratch),str(DATA)]
    import make_cldf
    results={}
    for norm in ('NFC','NFD'):
        inputs=[[ud.normalize(norm,v) if i==2 else v for i,v in enumerate(r)] for r in rows]
        csvpath=target/(stem+'.csv')
        with csvpath.open('w',newline='') as f:csv.writer(f).writerows(inputs)
        errors=io.StringIO()
        parsed,stats=make_cldf.parse_file(str(csvpath),errors,name=stem)
        assert not errors.getvalue(),errors.getvalue()
        assert len(parsed)==stats['converted']==413
        assert [r.entry_key for r in parsed]==[r[10] for r in inputs]
        for a,r in zip(parsed,inputs):
            assert (a.lang,a.old_form,a.gloss,a.native,a.ipa,a.notes,a.source,a.tags)==(r[0],r[2],r[3],r[4],r[5],r[6],r[7],r[14]),a.entry_key
            assert ud.normalize('NFC',a.form)==ud.normalize('NFC',r[2].strip('1234⁴5⁵67⁷,;.'))
        results[norm]=dict(rows=413,converted=413,errors=0,no_splits=True,ordered_keys_and_all_fields=True)
    out=HERE/'metadata-profile-proposal'
    (out/'grammar-parser-accounting.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    report=dict(status='all resolved nonblank grammar candidates passed isolated actual parser',
        input_sha256=sha(source),mapping_sha256=sha(mapping_path),profile_sha256=sha(profile),
        code_sha256={n:sha(DATA/n) for n in ('make_cldf.py','source_meta.py','tags.py')},
        counts=dict(counts),language_counts=dict(collections.Counter(r[0] for r in rows)),checks=results,
        verified_single_tag_delta=delta,previous_accounting_sha256=sha(prior),
        held_keys=[r['source_unit_key'] for r in audit if r['disposition']=='held_unresolved_route'],
        limitations=['Proposed dialect IDs stay in keyed accounting until canonical registration, not fabricated registered tags.',
            'This is a parser/identity-route fixture, not authorization to install comparator rows or adjudication of unresolved glyphs.',
            'Full build/database/browser not run.'],canonical_changes=False)
    (out/'grammar-actual-parser-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
