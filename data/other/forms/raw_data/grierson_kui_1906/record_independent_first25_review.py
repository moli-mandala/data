import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[5]
p=here/'table-first-reading.json'
data=json.loads(p.read_text())
rows=[r for r in data['readings'] if r['prompt_id']<=25]
assert [r['prompt_id'] for r in rows]==list(range(1,26))
def asset(rel):
 q=root/rel
 return {'path':rel,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}
items=[]
for row in rows:
 n=row['prompt_id']
 note='Independent original-image reading agrees with literal and explicitly printed alternative scope.'
 status='agrees'
 if n in (1,2):note='Printed parentheses around ṇḍi confirmed. Keep literal parenthetical notation; optional-ending expansion is not independently established by this table.'
 if n==5:note='Two alternatives in one numeral cell; separator is dot-like in retained ink. Stacked mark over a preserved; no normalized nasal/quantity inference.'
 if n in (10,12):note='Narrow original detail supports ś: upper diagonal stroke repeated in both occurrences, distinguishable from ordinary s baseline body. Resolve old plain-s alternative in favor of source ś.'
 if n==11:note='Final o has an explicit thick horizontal upper bar. Kaḍō supported; underdot of ḍ also clear.'
 if n in (15,18,21,24):
  status='literal retained with typed glyph-shape caveat'
  note='Broad curved upper mark on final i is clearly not an ordinary dot. Existing ĩ retained provisionally; compressed shape alone does not establish breve versus tilde convention. Do not add length or phonemic analysis.'
 if n in (22,25):
  status='literal retained with typed length uncertainty'
  note='Final e upper horizontal component is compressed against letter, less distinctly separated than16/19. Existing ē remains provisional; plain e alternative cannot be excluded by this scan. Do not harmonize from parallel forms.'
 items.append({'prompt_id':n,'old_literal':row['literal'],'proposed_literal':row['literal'],'forms':row['forms'],'status':status,'evidence':note})
report={'reviewer':'norton_finish','date':'2026-09-27','status':'all25 cells reviewed; no definite replacement, six typed glyph/length caveats remain',
 'review_scope':'Literal forms, separators and prompt meaning correspondence. Not complete paradigm installation approval.',
 'input_path':str(p.relative_to(root)),'input_file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'input_first25_sha256':hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
 'source':asset('tmp/kui-alternate-leaf669.jpg'),'printed_page':648,'archive_leaf':669,
 'detail_images':[asset('tmp/kui-first25-'+s+'.png') for s in ['10-12','15-19','21-25']],
 'english_source':asset('tmp/kui-649.png'),'reviewed_count':25,'items':items,
 'metadata':'Numerals1–13 and pronouns14–25 consistent with explicit English prompts; case/person analysis cannot license extra unattested forms.',
 'owner_files_changed':False}
(here/'independent-table-first25-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Saved25-cell independent review')
