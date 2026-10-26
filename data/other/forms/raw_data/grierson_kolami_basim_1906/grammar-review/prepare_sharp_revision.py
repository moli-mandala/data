"""Guarded sharper-witness derivative; all earlier readings remain immutable."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = HERE.parents[6]
OUT = HERE/'sharp-reviewed-revision'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def load(p): return [json.loads(s) for s in p.read_text().splitlines()]
def dump(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
    paths={
        HERE/'reviewed-child-witnesses.jsonl':'ce78a459c1102d2b6e6376ea953f23dae4ebf9a5888daa63c9c72eae38c59ed2',
        PACKAGE/'grammar-native-review/full224-sharp-review-proposal.jsonl':'47801d590eba805cff65682759806a159ad25f97dab04e79100b4ed8ec5820a1',
        PACKAGE/'grammar-native-review/quantity-holds-root-adjudication.json':'7c5ad7cf7cfa23554277d625aa710b6daa324ddb55a1b7e2165c6601b1bbf50f',
    }
    for p,pin in paths.items(): assert sha(p)==pin,(p,sha(p))
    raw=load(HERE/'reviewed-child-witnesses.jsonl')
    reviews=load(PACKAGE/'grammar-native-review/full224-sharp-review-proposal.jsonl')
    assert len(raw)==len(reviews)==224
    assert [r['source_unit_key'] for r in raw]==[r['source_unit_key'] for r in reviews]
    holds=json.loads((PACKAGE/'grammar-native-review/quantity-holds-root-adjudication.json').read_text())
    quantity={r['source_unit_key']:r for r in holds['cases']}
    assert len(quantity)==6
    negative_source=ROOT/'tmp/jambu-next-source/kolami-grammar-originals/alternate-leaf588-printed567.jpg'
    negative_crop=negative_source.parent/'owner-native-p567-negative-controls.png'
    rect=(220,1760,4320,2290)
    if not negative_crop.exists(): Image.open(negative_source).crop(rect).save(negative_crop)
    crop_map={
        (562,'genitive-locative'):'p562-genitive-locative-native.png',
        (562,'personal-pronouns'):'p562-pronouns-native.png',
        (562,'present-copula'):'p562-present-forms-native.png',
        (562,'past-copula'):'p562-past-full-native.png',
        (563,'negative'):'p563-negatives-and-conjunctive-native.png',
        (566,'pronunciation-l-n'):'p566-pronunciation-native.png',
        (566,'pronunciation-n-nd-d'):'p566-pronunciation-native.png',
        (566,'nouns'):'p566-nouns-native.png',
        (566,'personal-pronouns'):'p566-pronouns-native.png',
        (566,'possessive-agreement'):'p566-possessives-other-native.png',
        (566,'other-pronouns'):'p566-possessives-other-native.png',
        (566,'present-copula'):'p566-copulas-native.png',
        (566,'numerals'):'p566-numerals-native.png',
        (567,'beat-paradigm'):'p567-paradigm-native.png',
        (567,'negative'):'owner-native-p567-negative-controls.png',
        (567,'conjunctive-participle'):'p567-conjunctive-native.png',
    }
    result=[]; decisions=[]; changed=Counter()
    for original,review in zip(raw,reviews):
        assert review['old_form']==original['form']
        assert review['gloss']==original['gloss']
        assert review['language_label']==original['language_label']
        assert review['printed_label_before']==original['printed_label']
        assert review['disposition']==original['proposed_disposition']
        source=(ROOT/Path(review['original_page']['path'])).resolve()
        assert sha(source)==review['original_page']['sha256']
        r=copy.deepcopy(original)
        change=review['proposed_form']!=original['form'] or review['printed_label_proposed']!=original['printed_label']
        if change or review['typed_review_uncertainty']:
            section=original['physical_parent_key'].split(':')[2]
            crop=source.parent/crop_map[(original['printed_page'],section)]
            evidence={'crop':str(crop.relative_to(ROOT)),'crop_sha256':sha(crop),
                      'native_source':str(source.relative_to(ROOT)),'native_source_sha256':sha(source)}
            d={'source_unit_key':r['source_unit_key'],'old_form':original['form'],'reviewed_form':review['proposed_form'],
               'gloss_guard':original['gloss'],'language_label_guard':original['language_label'],
               'old_printed_label':original['printed_label'],'reviewed_printed_label':review['printed_label_proposed'],
               'owner_original_corroboration':'Sharper native crop independently compared; occurrence-specific reading, no analogical normalization.',
               'evidence':evidence,'typed_uncertainty':review['typed_review_uncertainty']}
            if r['source_unit_key'] in quantity:
                q=quantity[r['source_unit_key']]
                assert q['before']==original['form'] and q['provisional_literal']==review['proposed_form']
                d['quantity_preservation']=q
                r['typed_uncertainties'].append({'type':q['uncertainty_type'],'detail':q['independent_observation'],
                                                'literal_alternatives':q['literal_alternatives'],'phonological_claim':None,
                                                'root_editorial_status':q['status']})
            elif review['typed_review_uncertainty']:
                r['typed_uncertainties'].append({'type':'source-mark-shape','detail':review['typed_review_uncertainty']})
            decisions.append(d)
            r['sharp_native_review']=d
        if review['proposed_form']!=original['form']:
            r['form']=review['proposed_form']; changed['form']+=1
        if review['printed_label_proposed']!=original['printed_label']:
            assert r['source_unit_key']=='grierson1906kolamibasim:p566:present-copula:3:independent:w1'
            r['printed_label']=review['printed_label_proposed']; changed['printed_label']+=1
            r['qualification']='Native sharper witness clearly prints 3.m.; earlier degraded 2.m. reading superseded and preserved in source_record.'
            r['typed_uncertainties']=[u for u in r['typed_uncertainties'] if u.get('type')!='source-numbering']
        r['review_status']='Owner corroborated sharper grammar review; final combined-output acceptance pending'
        result.append(r)
    assert changed=={'form':35,'printed_label':1},changed
    assert len(result)==224 and len({r['source_unit_key'] for r in result})==224
    assert Counter(r['proposed_disposition'] for r in result)==Counter(r['proposed_disposition'] for r in raw)
    assert all(a['source_record']==b['source_record'] for a,b in zip(raw,result))
    assert all(a['gloss']==b['gloss'] and a['language_label']==b['language_label'] for a,b in zip(raw,result))
    OUT.mkdir(exist_ok=True)
    (OUT/'reviewed-child-witnesses.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    dump(OUT/'owner-corroboration.json',{'reviewer':'/root/rohru_finish','input_pins':{str(p.relative_to(ROOT)):v for p,v in paths.items()},
        'counts':{'all_children_guarded':224,'form_repairs':35,'printed_label_repairs':1,'scoped_uncertainties':7,'quantity_alternative_cases':6},
        'decisions':decisions,'negative_crop':{'rectangle':list(rect),'source_sha256':sha(negative_source),'crop_sha256':sha(negative_crop)},
        'unchanged_all_glosses_and_language_labels':True,'canonical_mutations':False,
        'historical_rejections':['p562 īn→ïm is genuine diaeresis with two same-level dots; no phonemic reinterpretation.',
            'p563 sūlt-nā and p566 ū retain flat macrons; earlier resized ü impressions rejected by native controls.'],
        'acceptance_limit':'Guarded source-local derivative; final integrated output audit and installation approval still pending.'})
    dump(OUT/'status.json',{'inputs':{str(p.relative_to(ROOT)):v for p,v in paths.items()},
        'counts':{'children':224,'dispositions':dict(Counter(r['proposed_disposition'] for r in result)),'changed_fields':dict(changed)},
        'outputs':{p.name:sha(p) for p in [OUT/'reviewed-child-witnesses.jsonl',OUT/'owner-corroboration.json']},
        'accepted_for_installation':False,'pending':['Refresh exact tag and graph guards','Final combined candidate parser and independent output audit']})
    print(json.dumps({'counts':dict(changed),'reviewed_sha256':sha(OUT/'reviewed-child-witnesses.jsonl')}))

if __name__=='__main__': main()
