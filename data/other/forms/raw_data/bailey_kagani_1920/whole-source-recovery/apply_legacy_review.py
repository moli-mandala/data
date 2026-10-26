"""Apply independently reviewed, stable-key deltas to an immutable legacy copy."""
import copy,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def apply(rows,audit):
    report=json.loads((HERE/'independent-glossary-metadata-review-20260927.json').read_text())
    assert hashlib.sha256((HERE/'legacy-before-whole-recovery.csv').read_bytes()).hexdigest()==report['input_hashes']['data/data/other/forms/20260925-bailey-kagani.csv']
    before=copy.deepcopy(rows);keyed={r[10]:r for r in rows};reasons={}
    def add(key,tags,basis):
        row=keyed[key];row[14]=' '.join(dict.fromkeys(row[14].split()+tags));reasons.setdefault(key,[]).append(basis)
    for d in report['tag_deltas']:
        r=keyed[d['entry_key']]
        assert (r[2],r[3],r[14])==(d['old_form'],d['old_gloss'],d['old_tags'])
        add(d['entry_key'],d['add_tags'],d['basis'])
    for finding in report['findings']:
        if finding['class']=='literal incidental correction':
            row=keyed[finding['entry_key']];assert row[2]==finding['old_form'];row[2]=finding['form']
            reasons.setdefault(row[10],[]).append(finding['evidence'])
    b='bailey1920kagani:'
    # The reviewed p93 paradigm explicitly identifies plural tŭsī̃ / tŭs dā;
    # English you alone would not establish this number.
    for key,tags,basis in [
      ('p107:right:line:30',['pron','first-person','sg','gen'],'Gloss my identifies first-person possessor; source93 explicitly gives first-person genitive paradigm. No gender of possessed noun inferred.'),
      ('p107:right:line:30:answer2',['pron','first-person','sg','gen'],'Exact măṛā in reviewed93 first:gen-sg; retain alternate glossary spelling separately.'),
      ('p109:left:line:18',['pron','second-person','sg','gen'],'Explicit thy gloss; possessive analysis corroborated by93 second-person genitive, without harmonizing distinct tērā spelling.'),
      ('p109:right:line:28',['pron','second-person','pl','nom'],'Exact tŭsī̃ in reviewed93 second:nom-pl.'),
      ('p109:right:line:29',['pron','second-person','pl','gen','multiword-expression'],'Exact tŭs dā in reviewed93 second:gen-pl; number belongs to possessor.'),
      ('p108:left:line:11',['prep','gen'],'Exact dā of in reviewed89 noun-prepositions:of.'),
      ('p107:left:line:10:answer3',['suffix','abl'],'Explicit bound -o with source gloss from, corroborating ablative function; no stem expansion.'),
      ('p108:left:line:31:answer2',['verb'],'Explicit cause to be put on plus existing causative label; no derivation edge.')]:add(b+key,tags,basis)
    first=keyed[b+'p108:right:line:34'];second=keyed[b+'p108:right:line:34:answer2']
    assert 'k' in first[6] and not second[6]
    second[6]=first[6];first[6]=''
    reasons.setdefault(first[10],[]).append('Move printed grouping explanation from unrelated first alternative to affected second phrase.')
    reasons.setdefault(second[10],[]).append('Preserve exact khărāb, kărnā grouping explanation on its phrase answer, retaining raw punctuation in original audit.')
    bycell={u.get('source_cell_key'):u for u in audit}
    for ref in report['cross_reference_accounting']:
        unit=bycell[ref['source_cell_key']]
        unit['resolved_target_entry_keys']=ref['candidate_target_keys']
        unit['cross_reference_decision']=ref['decision']
        if ':line:10' in ref['source_cell_key']:
            row=keyed[ref['candidate_target_keys'][0]]
            row[7]=row[7][:-1]+', cross-reference p. 108, right column, line 10]'
            unit['entry_keys']=[row[10]];unit['status']='resolved_exact_sense_cross_reference'
            reasons.setdefault(row[10],[]).append(ref['decision'])
        else:unit['status']='resolved_cross_reference_without_semantic_equation'
    for row in rows:
        if ' ' in row[2].strip() and not {'multiword-expression','sentential'} & set(row[14].split()):
            add(row[10],['multiword-expression'],'The literal source response contains multiple space-separated words; no internal lexical segmentation or sentence claim.')
    changes=[]
    for old,new in zip(before,rows):
        if old!=new:
            changes.append(dict(entry_key=new[10],fields={str(i):{'old':a,'new':b} for i,(a,b) in enumerate(zip(old,new)) if a!=b},evidence=reasons[new[10]]))
    assert [r[10] for r in before]==[r[10] for r in rows]
    (HERE/'legacy-reviewed-delta-20260927.json').write_text(json.dumps({'status':'applied to proposal only','review_report':'independent-glossary-metadata-review-20260927.json','changes':changes,'all329_keys_preserved':True},ensure_ascii=False,indent=2)+'\n')
    return rows,audit
