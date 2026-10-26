"""Independent complete documentary wording review, preserving owner records."""
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
ROOT=next(p for p in HERE.parents if (p/'data/make_cldf.py').exists())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]

original=PACKAGE/'specimen38/documentary-first-reading.jsonl'
intro_path=PACKAGE/'scope/p233-specimen-introduction-first-reading.json'
comment_path=PACKAGE/'specimen38/root-documentary-review/linguistic-commentary-reviewed-derivative.jsonl'
blocks=read(original)
assert len(blocks)==4
intro=json.loads(intro_path.read_text())
comment=read(comment_path)[0]
controls=json.loads((HERE/'control-provenance.json').read_text())
by_page={r['printed_page']:r for r in controls}
for c in controls:
    assert sha(ROOT/c['source'])==c['source_sha256']
    assert sha(ROOT/c['crop'])==c['crop_sha256']
decisions=[]
intro_new=copy.deepcopy(intro)
assert 'Dēvanāgarī' in intro['raw_physical_line_reading'][0]
intro_new['raw_physical_line_reading'][0]=intro_new['raw_physical_line_reading'][0].replace('Dēvanāgarī','Dēva-nāgarī')
decisions.append(dict(source_unit_key=intro['source_unit_key'],old_record=intro,proposed_record=intro_new,
                     changed_fields=['raw_physical_line_reading[0]'],source_evidence=by_page[233],
                     result='secure source-literal hyphen correction; documentary only',
                     reason='Original first physical line explicitly prints Dēva-nāgarī; ver-/sion physical continuation stays unchanged. Context summary is an editorial summary, not labelled verbatim.'))
for old in blocks:
    new=copy.deepcopy(old)
    page=old['printed_page']
    reason='Complete source wording agrees; whitespace/typographic quotation marks are transcription conventions.'
    changed=[]
    if 'mixed-Ahirwati' in old['source_unit_key']:
        assert old['raw_documentary_first_reading'].count('bērē-mai')==1
        new['raw_documentary_first_reading']=old['raw_documentary_first_reading'].replace('bērē-mai','bērē-nai')
        assert new['raw_documentary_first_reading']==comment['raw_documentary_first_reading']
        changed=['raw_documentary_first_reading']
        reason='Corroborate already separately reviewed one-hump n reading bērē-nai in the italic documentary quotation. History remains in root derivative. Other quoted quantities and source labels agree.'
    if page==239:
        assert old['raw_documentary_first_reading'].count('Jai-kaḷi')==2
        new['raw_documentary_first_reading']=old['raw_documentary_first_reading'].replace('Jai-kaḷi','Jai-kali')
        changed=['raw_documentary_first_reading']
        reason='Both English free-translation mentions print plain l in Jai-kali; no independent lower dot. This source witness does not alter target Roman Jai-kaḷi. Its is printed without an apostrophe; Hukamlō retains printed final quantity.'
    new['review_status']='independent full original wording reviewed; source-local documentary derivative proposal'
    new['typed_uncertainties']=[]
    decisions.append(dict(source_unit_key=old['source_unit_key'],old_record=old,proposed_record=new,
                          changed_fields=changed,source_evidence=by_page[page],reason=reason,
                          result='wording agrees' if not changed else 'guarded documentary-only literal repair'))
headings=[dict(printed_page=p,specimen_number=number,raw_headings=['INDO-ARYAN FAMILY.','CENTRAL GROUP.','RĀJASTHĀNĪ.','AHĪRWĀṬĪ.','DISTRICT '+district+'.'],
               note='Printed historical classification/lect labels; uppercase layout and small-cap variants represented as heading text, not modern-language taxonomy or new lexical rows.',source_evidence=by_page[p]) for p,number,district in [(233,37,'GURGAON'),(237,38,'ROHTAK'),(238,38,'ROHTAK')]]
for name,rows in [('all-documentary-keyed-decisions.jsonl',decisions),('documentary-reviewed-proposal.jsonl',[d['proposed_record'] for d in decisions]),('heading-coverage.jsonl',headings)]:
    (HERE/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
translation=' '.join(d['proposed_record']['raw_documentary_first_reading'] for d in decisions if d['proposed_record'].get('kind')=='whole-text-free-English-translation')
assert translation.count('Jai-kali')==2
(HERE/'complete-free-English-reviewed.txt').write_text(translation+'\n')
report=dict(status='complete independent original documentary review; guarded source-local derivative, no lexical emission',
            counts=dict(documentary_blocks=5,original_pages=4,heading_blocks=3,new_literal_repair_records=2,already_reviewed_commentary_repair_records=1,free_translation_page_blocks=2,free_translation_name_repairs=2),
            inputs={str(p.relative_to(ROOT)):sha(p) for p in [original,intro_path,comment_path,HERE/'control-provenance.json']},
            coverage=['p233 complete three physical English introduction lines, including ver-/sion; No37 headings',
                      'p237 complete provenance paragraph and mixed-language commentary, nine quoted lexical groups, No38 headings',
                      'p238 complete printed free-English opening and No38 repeated headings',
                      'p239 complete printed free-English completion and Ahirwati running heading'],
            attribution_policy='Social stereotypes/reputed avarice and mixed-language/local peculiarity claims are attributed to Grierson. Script and district prose is provenance. Free English is whole-text documentary meaning, never inferred per-word alignment or intrinsic grammar.',
            exact_English_special_cases=['Its has no printed apostrophe','soul is tightened retained, without idiomatic correction','one-and-twenty and parenthetical (joint) retained','Hukamlō final long vowel retained','Jai-kali plain l specific to English translation'],
            unresolved=[], canonical_or_owner_changes=False,
            outputs={p.name:sha(p) for p in [HERE/'all-documentary-keyed-decisions.jsonl',HERE/'documentary-reviewed-proposal.jsonl',HERE/'heading-coverage.jsonl',HERE/'complete-free-English-reviewed.txt']})
(HERE/'independent-full-documentary-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report['counts']))
