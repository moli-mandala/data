"""Conservative whole-article Toda leads from exact DEDR form and sense comparisons.

This is a review shortlist, not an automatic etymology assignment. The excluded
homographs and compounds are kept as explicit research holds for this checkpoint.
"""
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
candidates = json.loads((P / 'dedr-toda-form-candidates.json').read_text())
records = json.loads((P / 'survey-inventory.json').read_text())
assert len(records) == 5605
stop = {'the','a','an','to','of','in','on','at','with','for','or','and','is','as','be','one','from','by','that','this','see','which','used','some','very','other','than','its','it','his','her','their','also','etc','into','like'}
def tok(s):
    return set(re.findall(r'[a-z]{3,}', s.lower())) - stop

exclude = {
    540: 'Small bag is a phrase in the demonstrative article; the whole phrase needs ordered component analysis.',
    1091: 'Pregnant buffalo is a compound; the full expression needs ordered component analysis.',
    1320: 'Cane versus a string of cane is too broad a sense shift without source context.',
    1868: 'Bungalow/raised platform may be a contact term; the immediate donor is unresolved.',
    3027: 'Badaga village is expressly derived from Badaga haṭṭi in the comparison; resolve borrowing direction.',
    3161: 'The source says simply make, while the article specifically says make broad.',
    3295: 'The source says only S of par-; the apparent pair comparison requires the headword context.',
    4060: 'The priest garment is a phrase in a long dairy article; the lexical components need analysis.',
    4573: 'Cigar is explicitly compared with Tamil curuṭṭu; resolve the immediate donor before calling it a Toda reflex.',
    1055: 'Gorse is an introduced species, and the immediate donor of its name is unresolved.',
    681: 'The vessel expression ɨr kvāy is composite; the article comparison needs component analysis.',
    4503: 'Badaga funeral is a cultural/contact context; resolve the source and sense path.',
    5195: 'The DEDR cache has two different articles both numbered 2835 (drop/drizzle and pigeon); this parent identity must be resolved before linking.',
    1490: 'Two DEDR Toda reflexes both mean last year (articles 1619 and 5153); resolve article identity.',
    2665: 'The rice expression occurs under both DEDR 215 and 3753; resolve ordered components or article identity.',
    3570: 'The source spans foam and bubble, whose identical DEDR Toda form occurs under articles 4455 and 4463.',
    5541: 'The one-year expression occurs under both DEDR 990 and 5153; resolve article identity.',
    641: 'The source gloss is like this, but direct same-form Toda search also finds ridicule under DEDR 2043; verify the demonstrative path.',
    914: 'The source combines stained and rust, while DEDR 1385 and 1395 divide stain and rust across articles; resolve the sense split.',
}
manual_include = {14,17,213,420,3858,4727,274,2978,309,496,565,641}
# Individually screened against DEDR Toda reflex forms and glosses; conservative spelling variants.
manual_near = {374,456,500,502,536,755,806,1137,1144,1436,1490,1816,1846,1855,
               2018,2177,2281,2396,2397,2414,2462,2665,2789,2796,2995,3189,
               3353,3573,3586,3629,3927,3941,4136,4237,4243,4263,4406,
               4694,4725,5166,5183,5277,5278,5512,5519,5541,5544,5559}
selected = []
held = []
for item in candidates:
    match = item['matches'][0]
    index = item['index']
    if (match[3] != 1 and not (index in manual_near and match[3] >= 0.85)) or len(match[2]) != 1 or not match[2][0].startswith('d'):
        continue
    source_tokens = tok(item['gloss'])
    dedr_tokens = tok(match[1])
    overlap = len(source_tokens & dedr_tokens)/(min(len(source_tokens),len(dedr_tokens)) or 1)
    if index in exclude:
        held.append({'surveyIndex':index,'record':records[index-1], 'reason':exclude[index], 'candidateParent':match[2][0], 'dedrForm':match[0], 'dedrGloss':match[1]})
    elif overlap >= 0.5 or index in manual_include or index in manual_near:
        selected.append({'surveyIndex':index,'record':records[index-1], 'parent':match[2][0], 'dedrForm':match[0], 'dedrGloss':match[1], 'formScore':match[3], 'overlap':overlap})
assert not {x['surveyIndex'] for x in selected} & {x['surveyIndex'] for x in held}
assert all(records[x['surveyIndex']-1]['ID']==x['record']['ID'] for x in selected+held)
(P/'selection.json').write_text(json.dumps({'selected':selected,'held':held},ensure_ascii=False,indent=2)+'\n')
print({'selected':len(selected),'held':len(held),'unexamined':len(records)-len(selected)-len(held),'parents':len({x['parent'] for x in selected})})
