"""Apply explicit keyed own-second-reading decisions to immutable first reading."""
import hashlib
import json
from pathlib import Path

P=Path(__file__).parent
manifest=json.loads((P/'grammar-first-reading-manifest.json').read_text())
for f,sha in manifest['sha256'].items():
    assert hashlib.sha256((P/f).read_bytes()).hexdigest()==sha, f
rows=[json.loads(l) for l in (P/'grammar-first-reading.jsonl').read_text().splitlines()]
bykey={r['source_unit_key']:r for r in rows}
decisions=[]

def fix(key, old, new, reason, page):
    r=bykey['grierson1906kui:'+key]
    assert r['form']==old,(key,r['form'],old)
    r['form']=new
    decisions.append({'key':r['source_unit_key'],'field':'form','before':old,'after':new,
                      'reason':reason,'alternate_original_leaf':page})

fix('p469:chinna-negative-participle:1','ēāni suḍinā-aṅga','ēāni suḍāān-aṅga',
    'Sharper original clearly prints two consecutive macron-bearing a vowels after ḍ, followed by n; first reading misread the merged low-resolution cluster.',490)
fix('p469:kalahandi-sound-comparison:4','suḍāṅgi','saḍāṅgi','Sharper original shows plain a after initial s, not u.',490)
fix('p469:kalahandi-dative:1','ilutu','iluku','Clear k stem and arm in sharper original; retain literal cited dative, no analogy to any donor language.',490)
fix('p459:v-b-comparison:2','vesteñju','vestēñju','Sharper original has a macron over the second e.',480)
fix('p459:v-b-comparison:5','ēāñju','ēañju','This occurrence prints short a; preserve contrast with the earlier length example on the same page.',480)
fix('p459:cognate-comparison:3','ēāñju','ēañju','Repeated comparison occurrence prints short a, without harmonization to the earlier long-a example.',480)
fix('p460:smith-comparison:2','mrānu-gaṭṭañju','mrānu-gaṭṭāñju','Sharper original retains the macron after doubled ṭ; the source contrast does not license shortening all vowels.',481)
fix('p462:pronoun-table1-margin:2','mamma','māmma','Chinna Kimedi accusative margin prints a distinct macron over the first a.',483)
fix('p462:pronoun-table2-margin:5','trai','trāi','Sharper reflexive-pronoun margin has a horizontal quantity stroke over a.',483)
fix('p462:pronoun-prose:2','iañju','iāñju','Adjoining pronoun prose prints long ā, distinct from the unmarked a of ēañju.',483)
fix('p462:pronoun-prose:3','estañju','estāñju','Interrogative in pronoun prose has macron-bearing ā.',483)
fix('p462:noun-margin-postpositions:7','ṭoṭi','ṭōṭi','Narrow sharper crop shows a distinct macron over o; lower consonant-mark encoding remains under review.',483)
fix('p462:noun-margin-postpositions:8','ṭāktā','ṭākā','Narrow crop shows ṭ-ā-k-ā, without an additional t after k; lower initial mark remains under review.',483)
fix('p463:personal-1pl-incl-c2:1','pāgassu','pāgāssu','Inclusive negative cell has macron over the vowel preceding ss.',484)
fix('p463:irregular-prose:12','salla','sallā','Past conjunctive alternative prints a macron over final a.',484)
fix('p463:irregular-prose:13','sajja','sajjā','Second past conjunctive alternative also prints final ā.',484)
fix('p463:relative-adverbial-example:2','ānā-vōā','ānā-vōĩ','Final glyph has an i stem with upper nasalization mark, not an a body; source mark encoding remains qualified by the volume notation review.',484)
fix('p463:causal-example:1','ihē-igi gitāki','ihengi gitāki','Root independent close-crop review supports a continuous first word without an internal hyphen. The candidate reading remains graphemically uncertain; no morphological reconstruction.',484)
r=bykey['grierson1906kui:p463:causal-example:1']
r['uncertainty']='Source grapheme/quantity uncertainty: first word ihengi or ihēṅgi; upper nasal mark and e quantity unresolved. Following gitāki has fused medial marks. Root independently confirmed the absence of the first-reading internal hyphen in tmp/kui-grammar-causal-review.png.'
r=bykey['grierson1906kui:p463:causal-example:2']
r['uncertainty']='Source grapheme/quantity uncertainty: root independently confirms initial giw-, but the following fused vowels are not securely resolved. Retain whole printed alternative; do not reconstruct from the preceding morphology rule.'
added={
    'source_unit_key':'grierson1906kui:p462:noun-prose-ending:1',
    'printed_page':462,'physical_page':482,'block':'noun-prose-ending',
    'form':'eñju','gloss':'','route_key':'gumsur-bod-standard',
    'source_note':'Source explicitly describes masculine nouns ending in this sequence and gives lāveñju as an example; the plural uses gā.',
    'tags':['suffix','m'],'witness':'',
    'uncertainty':'Upper nasal-consonant mark encoded provisionally ñ; no phonemic interpretation is asserted.',
    'status':'own second reading; independent review pending'}
assert added['source_unit_key'] not in bykey
rows.append(added)
decisions.append({'key':added['source_unit_key'],'field':'source occurrence','before':None,'after':added,'reason':'Whole-page reread recovered an explicitly quoted noun ending omitted from the first-reading inventory.','alternate_original_leaf':483})
for suffix in ('noun-margin-shape:1','noun-margin-shape:2','noun-margin-shape:3'):
    r=bykey['grierson1906kui:p462:'+suffix]
    r['uncertainty']='The isolated ending in the noun-margin statement appears ñju in the sharper image; first-reading ĩju remains provisional pending independent grapheme review. No source-phoneme equivalence is inferred.'
for r in rows:
    if r['printed_page']==462 and r['block']=='pronoun-prose':
        r['source_note']=r['source_note'].replace('iañju/estañju','iāñju/estāñju')
    if r['source_unit_key']=='grierson1906kui:p459:cognate-comparison:4':
        r['source_note']='Repeated source comparison with the locally printed ēañju.'
    r['status']='own second reading completed; independent full review pending'
for n in (1,2,3):
    r=bykey[f'grierson1906kui:p459:letchmajee-shortening:{n}']
    old=r['source_note']
    r['source_note']=old.replace('before ĩ and h','before i and h')
    decisions.append({'key':r['source_unit_key'],'field':'source_note','before':old,'after':r['source_note'],'reason':'Original shortening statement names plain i, not nasalized ĩ.','alternate_original_leaf':480})

if __name__=='__main__':
    (P/'grammar-reviewed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    source_pages={}
    for leaf in (480,481,483,484,490):
        path=Path('tmp')/f'kui-grammar-original-leaf{leaf}.jpg'
        source_pages[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (P/'grammar-own-second-reading-decisions.json').write_text(json.dumps({
        'status':'full visual second reading completed; independent review and specified grapheme decisions pending',
        'reviewer':'asur_finish','first_reading_manifest':manifest,
        'original_page_hashes':source_pages,
        'coverage':'All 431 first-reading physical occurrences/cells reread, including all skeleton-table columns, margins, 16 empty positions, and prose comparisons. One additional explicitly quoted noun ending recovered.',
        'reviewed_units':len(rows),'blank_positions':sum(not r['form'] for r in rows),
        'notation_review':'volume-notation-review-20260927.json',
        'limits':['Source glyph uncertainties remain explicitly typed; no inferred phonemic profile.', 'No canonical installation or source-wide independent acceptance claimed.'],
        'decisions':decisions},ensure_ascii=False,indent=2)+'\n')
    print(len(rows),len(decisions))
