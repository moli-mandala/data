"""New metadata revision; historical literal integration remains immutable."""
import collections
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE/'literal-integration'
OUT=HERE/'literal-integration-metadata-r2'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def wordlist_decision(item,gloss):
    tags=[]; clean=gloss
    reason='Ordinary lexical prompt; no explicit grammatical label or safe narrower grammatical claim added.'
    if 151<=item<=164:
        tags=['num'];reason='Printed numeral prompt identifies a cardinal numeral.'
    elif 165<=item<=170:
        tags=['interr'];reason='Printed interrogative prompt, including question mark.'
    elif 171<=item<=174:
        tags=['demonstrative','sg' if item<173 else 'pl']
        reason='Printed this/that/these/those identifies demonstrative function and reference number; no inferred internal morphology or POS subclass.'
    elif 202<=item<=210:
        tags=['pron','personal']
        if item==202:tags+=['1sg'];clean='I'
        elif item in [203,204,209]:tags+=['second-person'];clean='you'
        elif item in [205,206]:tags+=['3sg']
        elif item in [207,208]:tags+=['1pl'];clean='we'
        else:tags+=['3pl']
        reason='Printed pronoun prompt establishes person and, except ambiguous you, number. Original pp97/107 do not print the old formal/informal/inclusive/exclusive expansions; those unsupported distinctions are not exported.'
    return dict(item=item,previous_gloss=gloss,gloss=clean,tags=tags,reason=reason)


def main():
    frozen=json.loads((OLD/'review-manifest.json').read_text())
    for group,folder in [('outputs_sha256',OLD),('inputs_sha256',HERE),('scripts_sha256',HERE)]:
        for name,digest in frozen[group].items():assert sha(folder/name)==digest,(group,name)
    frozen_hash=sha(OLD/'review-manifest.json')
    OUT.mkdir(exist_ok=True)
    scope=json.loads((HERE/'narrative-scope-review-20260927.json').read_text())
    by_scope={r['source_unit_key']:r for r in scope['decisions']}
    assert len(by_scope)==205
    assert collections.Counter(r['scope_tag'] for r in by_scope.values())=={'sentential':165,'multiword-expression':40}
    with (HERE.parent/'prompts.tsv').open(newline='') as f:
        concepts=[wordlist_decision(int(r['Item']),r['Gloss']) for r in csv.DictReader(f,delimiter='\t')]
    concept_map={r['item']:r for r in concepts};assert len(concepts)==210
    audit=[json.loads(s) for s in (OLD/'proposed-whole-audit.jsonl').read_text().splitlines()]
    by_entry={k:a for a in audit for k in a.get('entry_keys',a.get('proposed_entry_keys',[]))}
    changes=[];counts={};coverage=[]
    for kind in ['wordlist','narrative','comparative']:
        with (OLD/f'proposed-{kind}.csv').open(newline='') as f:oldrows=list(csv.reader(f))
        rows=copy.deepcopy(oldrows)
        for old,row in zip(oldrows,rows):
            key=row[10];unit=by_entry[key];tags=row[14].split();decision={}
            if kind=='wordlist':
                decision=concept_map[int(unit['legacy_audit']['Item'])]
                assert row[3]==decision['previous_gloss'],key
                row[3]=decision['gloss'];tags+=decision['tags']
                if ' ' in row[2].strip():tags+=['multiword-expression']
                if row[11]:tags+=['alternate']
            elif kind=='narrative':
                if key in by_scope:
                    decision=by_scope[key]
                    assert row[2]==decision['original'] and row[3]==decision['source_translation'],key
                    tags+=[decision['scope_tag']]
                    if decision['retain_interr']:tags+=['interr']
                else:
                    assert key.endswith(('-definition','-context')),key
                    decision={'reason':'One of five explicit named terms; parsi has a printed definition, other four remain blank-gloss with exact context in Notes. No guessed morphology.'}
                    if key.endswith('-context'):assert row[3]==''
                coverage.append(dict(entry_key=key,kind=kind,decision=decision))
            else:
                decision={'reason':'Original printed comparison and lect scope retained; no etymological edges or inferred donor relationship.'}
                if row[3]=='six':tags+=['num']
                if ' ' in row[2].strip():tags+=['multiword-expression']
                coverage.append(dict(entry_key=key,kind=kind,decision=decision))
            row[14]=' '.join(dict.fromkeys(tags))
            unit.setdefault('metadata_revision',[]).append(dict(entry_key=key,decision=decision,exported_gloss=row[3],exported_tags=row[14]))
            if row!=old:
                changes.append(dict(entry_key=key,kind=kind,before={'gloss':old[3],'tags':old[14]},after={'gloss':row[3],'tags':row[14]},reason=decision))
            assert all(row[i]==old[i] for i in range(15) if i not in [3,14]),key
        with (OUT/f'proposed-{kind}.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
        counts[kind]=len(rows)
    assert counts=={'wordlist':1437,'narrative':210,'comparative':18}
    assert len(coverage)==228
    for name in ['proposed-wordlist.yaml','proposed-narrative.yaml','proposed-comparative.yaml',
                 'proposed-sil-koya-literal.txt','proposed-sil-koya-narrative.txt','proposed-sil-koya-comparative.txt',
                 'proposed-entry-key-aliases.json','bibliography-update-proposal.json']:
        shutil.copyfile(OLD/name,OUT/name)
    (OUT/'proposed-whole-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
    (OUT/'proposed-wordlist-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit if a['source_unit_key'].startswith('wordlist:')))
    review=dict(status='metadata revision prepared; independent final audit pending',
                root_cause='Historical integration copied the old narrative/comparative CSVs and did not consume the already reviewed narrative-scope-review file. Wordlist builder emitted only dialect/uncertainty metadata.',
                narrative_counts=scope['counts'],narrative_and_comparative_coverage=coverage,wordlist_concept_coverage=concepts,
                source_evidence='All205 source-backed narrative decisions inspected against full preserved original/translation units; existing original-page visual review reused, with fresh original pages18/20/24/26/30 and comparative6–7 inspected during independent pass1. Original pronoun tables97/107 freshly inspected at2x: no formal/informal/inclusive/exclusive labels are printed. Complete210 prompt census reviewed; no blanket noun gender/verb morphology inferred.',
                ambiguity_policy='Keep40 clipped/continuing narrative units multiword-expression. This/that number describes the elicited referent, not an inferred morphology. Unqualified you remains second-person without number/register. All glyph uncertainties unchanged.',
                changes=changes,change_counts=dict(collections.Counter(x['kind'] for x in changes)),
                old_manifest_sha256=frozen_hash,
                input_sha256={'narrative-scope-review-20260927.json':sha(HERE/'narrative-scope-review-20260927.json'),'../prompts.tsv':sha(HERE.parent/'prompts.tsv'),'prepare_metadata_revision.py':sha(Path(__file__))},
                rows=counts,total_rows=1665,source_keys_and_all_literal_fields_unchanged=True,canonical_modified=False)
    (OUT/'metadata-repair-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    assert sha(OLD/'review-manifest.json')==frozen_hash
    for name,digest in frozen['outputs_sha256'].items():assert sha(OLD/name)==digest,name
    print(json.dumps({'rows':counts,'changes':review['change_counts'],'old_freeze_unchanged':True}))


if __name__=='__main__':main()
