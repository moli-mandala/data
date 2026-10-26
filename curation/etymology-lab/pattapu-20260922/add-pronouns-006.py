"""Append researched proposals without modifying accepted sidecars."""
import csv, gzip, json, sys
from pathlib import Path
root = Path.cwd()
p = root / 'data/curation/etymology-lab/pattapu-20260922'
sys.path.insert(0, str(root / 'data'))
from etymology_assignments import read_assignments
z = json.load(open(p / 'proposed-decisions.json'))
seen = {a['record']['ID'] for a in z['accepted'] + z['held']} | {a['Form_ID'] for a in read_assignments()}
rules = {
 'niyi': (None, 'd3684', 'Tamil/Malayalam nī and Kodagu nī/nīnï are second-person singular pronouns. Pattapu niyi is proposed as the corresponding stem with a glide and final vowel; its short vowels are retained. The two source records specify female and male addressees but have identical forms, so both records remain distinct without inventing a morphological gender contrast.'),
 'nānu': ('i', 'd5160', 'Konda nān(u), Kui nānu and Kuvi nānu are first-person singular pronouns, alongside Tamil nāṉ and Kodagu nānï. Pattapu nānu matches this long-vowel n-initial form; the source’s lowercase i gloss is preserved.'),
 'namu': ('we (inclusive)', 'd3647', 'Tamil nām, Malayalam nām and Tulu nama occur in the inclusive we family. Pattapu namu preserves the n-m stem with short vowels and final u; its source explicitly distinguishes this inclusive response from naŋgu exclusive.'),
 'naŋgu': ('we (exclusive)', 'd5154', 'Kodagu naŋga and Tamil nāṅkaḷ occur in the DEDR5154 we family, where Tamil is explicitly exclusive. Pattapu naŋgu has the nasal-velar formation with final u and an explicit exclusive gloss. DEDR3647 also lists Tamil nāṅkaḻ exclusive: the articles overlap, and5154 is preferred here for its Kodagu naŋga comparison; this is not a claim that3647 is unrelated.'),
 'nēŋgu': ('you', 'd3688', 'Kodagu niŋga and Tamil nīṅkaḷ occur in the plural you family. Pattapu nēŋgu preserves the nasal-velar formation with a different long vowel and final u. Its source only says you, so plurality is inferred from the comparison and contrasted singular niyi responses, not silently added to the gloss.'),
 'ava': ('third-person singular, female, remote', 'd1', 'Kodagu ava explicitly means that woman, alongside Tamil/Malayalam avaḷ. Pattapu ava matches the feminine remote pronoun, including the absence of the final lateral found in Tamil; the whole article includes this inflected pronoun, not just the bare demonstrative base.'),
 'avã': ('third-person singular, male, remote', 'd1', 'Tamil/Malayalam avan and Kodagu avëⁿ mean that man. Pattapu avã retains the av- stem with a nasalized final vowel instead of a separately written final nasal. Its masculine remote meaning agrees with these specific article members.'),
 'aŋgu': ('third-person plural, animate, remote', 'd1', 'Kodagu aeŋga explicitly means those persons in the article’s remote demonstrative paradigm. Pattapu aŋgu has the same a-initial nasal-velar formation with contracted vocalism and final u. This comparison uses the animate plural pronoun, not the superficially similar Tamil aṅku there.'),
}
for r in json.load(open(p / 'inventory.json')):
    if r['ID'] in seen or r['Form'] not in rules:
        continue
    gloss, parent, evidence = rules[r['Form']]
    assert gloss is None or r['Gloss'] == gloss, r
    z['accepted'].append(dict(record=r, parent=parent, kind='reflex', citation=f'dedr[entry {parent[1:]}];' + r['Source'], evidence=f'DEDR{parent[1:]}: ' + evidence + ' Whole-article family membership is proposed; inheritance versus local transmission remains subject to review.', status='pending-review'))
(p / 'proposed-decisions.json').write_text(json.dumps(z, ensure_ascii=False, indent=2) + '\n')
used = {d for a in z['accepted'] for d in a.get('components', [a['parent']])}
articles = {r['Entry_ID']: r for r in csv.DictReader(gzip.open(root / 'data/data/dedr/entry-texts-audit.csv.gz', 'rt')) if r['Entry_ID'] in used and r['Status'] != 'duplicate-excluded'}
(p / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
