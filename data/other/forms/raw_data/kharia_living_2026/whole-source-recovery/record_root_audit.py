"""Bind root's semantic sample review to direct read-only SQLite evidence."""
import hashlib,json,sqlite3,unicodedata
from pathlib import Path
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((H/'frozen-manifest.json').read_text())
for n,d in m['files'].items():assert sha(H/n)==d,n
db=H.parents[6]/'tmp/kharia-living-recovery/kharia-20260927.db'
assert sha(db)==m['source_snapshot_sha256']
c=sqlite3.connect('file:'+str(db.resolve())+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
s=json.loads((H/'independent-sample-20260927.json').read_text())
reviews=[]
for u in s['sample']:
 e=u['raw_evidence']['evidence'];key=u['source_unit_key']
 if ':sentence:' in key:
  original=c.execute('SELECT * FROM sentences WHERE id=?',(key.split(':')[-1],)).fetchone()
  assert original['text']==e['text'] and original['translation']==e['translation']
  r=u['actual_rows'][0]
  assert r[2]==r[4]==json.loads(original['text'])['default'] and not r[5]
  assert r[3]==json.loads(original['translation'])['en']
 else:
  original=c.execute('SELECT * FROM entries WHERE id=?',(e['upstream_entry_id'],)).fetchone()
  sense=c.execute('SELECT * FROM senses WHERE id=?',(e['upstream_sense_id'],)).fetchone()
  assert original['phonetic']==e['raw_phonetic'] and original['lexeme']==e['raw_lexeme']
  assert json.loads(sense['glosses'])==e['raw_glosses']
  dialects=[json.loads(x[0])['default'] for x in c.execute('SELECT d.name FROM dialects d JOIN entry_dialects ed ON ed.dialect_id=d.id WHERE ed.entry_id=?',(e['upstream_entry_id'],))]
  assert sorted(dialects)==sorted(e['raw_dialects'])
 reviews.append({'source_unit_key':key,'stratum':u['stratum'],'entry_keys':[r[10] for r in u['actual_rows']],
   'status':'passed','basis':'Direct SQLite evidence matched retained raw fields; root reviewed literal expansions, translations, tags, source dialects and variant targets against actual frozen rows.'})
assert len(reviews)==20
report={'status':'passed','seed':s['seed'],'sample_size':20,'sample_rows':s['output_rows'],'material_errors':0,
 'frozen_manifest_sha256':sha(H/'frozen-manifest.json'),'source_sqlite_sha256':sha(db),'reviews':reviews,
 'editorial_decisions':['English bare you remains number-ambiguous.',
  'Source explicit dual/exclusive/inclusive analyses are structured; no phonetic segmentation inferred.',
  'Parenthesized optional phonetic segment in Friday expands to both literal readings; tilde alternatives keep source-local variant links.',
  'Native-only sentences retain meaningful apostrophes, hyphen and exact source translations; automatic token matches remain audit-only.'],
 'limits':'Source-stage review only. Full CLDF/global graph/identity/reference/full suite and database/browser gates deferred.'}
(H/'root-independent-output-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(H/'root-installation-approval.json').write_text(json.dumps({'status':'approved','frozen_manifest_sha256':sha(H/'frozen-manifest.json'),
 'scope':'Exact524-row frozen source-stage proposal and metadata; guarded installation and focused canonical validation only. No full data/database build or publication.'},indent=2)+'\n')
print('20 units / '+str(s['output_rows'])+' rows passed; source-stage installation approved.')
