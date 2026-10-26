"""Pin a complete owner-reviewed proposal and a fresh stratified sample."""
import csv,hashlib,json,random,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[5]
OUT=HERE/'proposal'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=list(csv.reader((OUT/'20260911-kharia-living.csv').open()));indexed={r[10]:r for r in rows}
 decisions=json.loads((OUT/'metadata-decisions.json').read_text())
 census=json.loads((HERE/'snapshot-census-20260927.json').read_text())
 source=json.loads((HERE/'lexical-snapshot-20260927.json').read_text())
 owner={'status':'owner whole-source review completed; independent final output audit pending','scope':'Read all452 source lexical rows and all152 changed emitted-row decisions, all3sentences and full schema/census. Source raw phonetics, native strings and bilingual definitions retained. No speculative word segmentation, donor links or source inflection from English plural nouns.','reviewed_source_entries':452,'reviewed_output_decisions':521,'changed_rows':sum(d['old_row']!=d['new_row'] for d in decisions),'new_sentences':3,'originals':{'sqlite_sha256':source['sqlite_sha256'],'lexical_snapshot_sha256':sha(HERE/'lexical-snapshot-20260927.json')},'schema_accounting':{'included':['entries','senses','dialects','entry_dialects','sentences','senses_in_sentences'],'empty_lexical_tables':[k for k,v in census['tables'].items() if v['count']==0],'excluded_nonlexical':'438audio/438speakerlinks,69photos/69senselinks,2speakers and administrative synchronization/migration tables. No media fetched, speaker personal data not exported.'},'known_residuals':['Six legacy native-only heads plus three new native-only sentences have no source phonetic transcription.','Source spelling/shortening/glottalization distinctions preserved as before.','Leading hyphen on culture retained with typed morphological uncertainty.','Automatic sentence token matches lack interlinear glossing and split apostrophes; preserved as evidence, not accepted lexical alignment.','Bare English you/your has no number; her has no inferred possessive/object case.','Existing homographic heads and conflicting sentence glosses remain distinct source records.'],'targeted_independent_evidence':'root-sentence-evidence-review-20260927.json','focused_tests':'4 passed in0.66s; tmp/kharia-living-recovery/tests.log','deferred':['full CLDF','full suite','database','browser','canonical installation pending approval']}
 (HERE/'owner-whole-review-20260927.json').write_text(json.dumps(owner,ensure_ascii=False,indent=2)+'\n')
 summary=json.loads((OUT/'summary.json').read_text())
 summary['status']='owner reviewed complete proposal; independent audit and installation pending'
 summary['pending']=['fresh independent output audit','approval and canonical installation','full CLDF/full suite/database/browser explicitly deferred']
 summary['files']={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='summary.json'}
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 yaml='''file: 20260911-kharia-living.csv
defaults:
  transcription:
  - profile: kharia-living
    preserve_hyphens: true
  importer:
    commands:
    - [data/other/forms/raw_data/kharia_living_2026/whole-source-recovery/install_proposal.py, --install]
sources:
  living-kharia2026:
    transcription:
    - profile: kharia-living
      preserve_hyphens: true
    identity:
      dedupe_by_entry_key: true
    forms:
      split_alternates: false
    reference:
      editor: Aryaman Arora; OpenAI Codex
      ocr: false
      etymology_provenance: none
'''
 (OUT/'20260911-kharia-living.yaml').write_text(yaml)
 bib=(DATA/'cldf/sources.bib').read_text();block=re.search(r'@misc\{living-kharia2026,.*?\n\}',bib,re.S)[0]
 changes={'urldate':'2026-09-27','version':'Public SQLite snapshot modified2026-09-18, downloaded2026-09-27; historical2026-08-20 headword evidence compared','license':'Used with permission; user confirmed all selected-source reuse permissions2026-09-11 and Living Dictionaries reuse per Sarva2026-09-27','included':'Complete452 lexical entries/senses plus all3example sentences:524rows,68source variant links, one extra numbered sense. All521legacy keys and literal/Native/Phonemic fields preserved. Six legacy native-only heads and three native-only sentences retained without invented pronunciation. Automatic token matching remains audit-only; media/personal/administrative tables excluded.','provenance':'data/other/forms/20260911-kharia-living.csv; data/other/forms/raw_data/kharia_living_2026/whole-source-recovery/proposal/whole-audit.jsonl; data/other/forms/raw_data/kharia_living_2026/whole-source-recovery/snapshot-lexical-comparison-20260927.json'}
 for k,v in changes.items():block=re.sub(r'  '+k+r' = \{[^\n]*\},','  '+k+' = {'+v+'},',block)
 (OUT/'source.bib').write_text(block+'\n')
 old=json.loads((DATA/'source_checklists/audits/20260911-kharia-living-seeded-audit.json').read_text())['reviewed_upstream_ids']
 units=[json.loads(x) for x in (OUT/'whole-audit.jsonl').read_text().splitlines()]
 candidates=[u for u in units if u['status']=='emitted' and (u['source_unit_key'].split(':')[1] not in old)]
 rng=random.Random(20260927116);selected=[];used=set()
 def pick(name,predicate,n):
  available=[u for u in candidates if u['source_unit_key'] not in used and predicate(u)]
  for u in rng.sample(available,n):
   used.add(u['source_unit_key']);selected.append({'stratum':name,'source_unit_key':u['source_unit_key'],'raw_evidence':u,'actual_rows':[indexed[k] for k in u['entry_keys']]})
 def has(u,t):return any(t in indexed[k][14].split() for k in u['entry_keys'])
 pick('new native sentences',lambda u:':sentence:' in u['source_unit_key'],3)
 pick('existing whole clauses',lambda u:has(u,'sentential'),4)
 pick('numerals and ordinals',lambda u:has(u,'num'),3)
 pick('pronouns and grammatical scope',lambda u:has(u,'pron'),3)
 pick('alternatives',lambda u:any(indexed[k][11] for k in u['entry_keys']),3)
 pick('remaining lexical source',lambda u:True,4)
 sample={'seed':20260927116,'sample_size':20,'output_rows':sum(len(x['actual_rows']) for x in selected),'historical_excluded_ids':old,'sample':selected}
 (HERE/'independent-sample-20260927.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
 files=['build_proposal.py','prepare_review.py','install_proposal.py','lexical-snapshot-20260927.json','snapshot-census-20260927.json','snapshot-lexical-comparison-20260927.json','owner-whole-review-20260927.json','independent-sample-20260927.json']+[str(p.relative_to(HERE)) for p in sorted(OUT.iterdir()) if p.is_file()]
 manifest={'status':'frozen complete source-stage proposal pending independent audit','rows':524,'legacy_keys':521,'audit_units':455,'baseline_csv_sha256':sha(DATA/'data/other/forms/20260911-kharia-living.csv'),'baseline_profile_sha256':sha(DATA/'conversion/kharia-living.txt'),'source_snapshot_sha256':source['sqlite_sha256'],'files':{n:sha(HERE/n) for n in files}}
 (HERE/'frozen-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(sha(HERE/'frozen-manifest.json'),sample['output_rows'])
if __name__=='__main__':main()
