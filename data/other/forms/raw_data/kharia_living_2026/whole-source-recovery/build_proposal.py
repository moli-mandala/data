"""Complete pinned Kharia lexical snapshot, preserving all legacy identities.

The current snapshot is explicitly distinguished from the historical snapshot;
all 452 historical lexical records were compared before recovery. No installation.
"""
import csv,hashlib,json,re,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
RAW=HERE.parents[1]
DATA=HERE.parents[5]
OUT=HERE/'proposal'
SOURCE='living-kharia2026'
SNAP=HERE/'lexical-snapshot-20260927.json'
def nfc(x):return unicodedata.normalize('NFC',x)
def tags_for(gloss,form):
 """Explicit reviewed gloss classes, not arbitrary English POS assignment."""
 g=gloss.strip().lower();ts=[];basis=[]
 nums={'one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety','one hundred','four hundred'}
 if g in nums or g.startswith('fifty ('):ts+=['num'];basis+=['Explicit numerical gloss.']
 if g in {'first','second','third','fourth'}:ts+=['num','ord'];basis+=['Explicit ordinal gloss.']
 if g in {'who','whose','what'}:ts+=['pron','interr'];basis+=['Explicit interrogative gloss.']
 if g=='whose':ts+=['poss']
 if g in {'how','why','where','when','how many'}:ts+=['interr'];basis+=['Explicit interrogative gloss.']
 personal={'i':['first-person','sg','1sg'],'he':['third-person','sg','3sg'],'she':['third-person','sg','3sg'],'they':['third-person','pl','3pl'],'we':['first-person','pl','1pl'],'you':['second-person'],'we 2 (not you)':['first-person','du','exclusive'],'we 3+ (not you)':['first-person','pl','1pl','exclusive'],'they-2':['third-person','du'],'me and you':['first-person','du','inclusive']}
 if g in personal:ts+=['pron','personal']+personal[g];basis+=['Explicit personal referent and count; bare you has no guessed number and he/she no inferred grammatical gender.']
 if g in {'his','her','their','our','ours','your','their (2 people)'}:
  ts+=['pron']+([] if g=='her' else ['poss']);basis+=['Explicit pronoun meaning; bare her is ambiguous between object/possessive, so no case or possessive classification is assigned to it.']
  if g in {'his','her'}:ts+=['third-person','sg','3sg']
  elif g=='their':ts+=['third-person','pl','3pl']
  elif g=='their (2 people)':ts+=['third-person','du']
  elif g in {'our','ours'}:ts+=['first-person','pl','1pl']
  elif g=='your':ts+=['second-person']
 if g in {'this','that'}:ts+=['demonstrative'];basis+=['Explicit demonstrative gloss; no grammatical number inferred.']
 if g.startswith('to '):ts+=['verb'];basis+=['English explicitly verbal infinitival definition; no source infinitive suffix inferred.']
 if g=='fly (n)':ts+=['noun'];basis+=['Explicit source (n) label.']
 # These are full source clauses, including existing dictionary headword clauses.
 clause=bool(re.match(r'^(i |you |you two |he |she |he/she |they two |these people |he\s*\+\s*i |you\s*\+\s*i |both will)',g))
 if clause:
  ts+=['sentential','verb'];basis+=['Whole source clause and explicit translated subject/tense; no word-level alignment or morphology inferred.']
  if re.match(r'^i ',g):ts+=['first-person','sg','1sg']
  elif re.match(r'^you two ',g):ts+=['second-person','du']
  elif re.match(r'^you\s*\+\s*i ',g):ts+=['first-person','du','inclusive']
  elif re.match(r'^he\s*\+\s*i ',g):ts+=['first-person','du','exclusive']
  elif re.match(r'^you ',g):ts+=['second-person']
  elif re.match(r'^they two ',g):ts+=['third-person','du']
  elif re.match(r'^these people ',g):ts+=['third-person','pl','3pl']
  elif re.match(r'^(he|she|he/she) ',g):ts+=['third-person','sg','3sg']
  elif g.startswith('both will'):ts+=['du']
  if 'will ' in g:ts+=['fut']
  elif 'went' in g or 'were going' in g or 'was going' in g:ts+=['pret']
  elif 'had gone' in g:ts+=['pret','perfect']
  elif 'have gone' in g:ts+=['perfect']
  elif ' go' in g or 'is dead' in g:ts+=['pres']
  if ' going' in g:ts+=['progressive']
 if ' ' in form:ts+=['multiword-expression'];basis+=['Attested whitespace expression; no components fabricated.']
 return list(dict.fromkeys(ts)),basis
def main():
 snapshot=json.loads(SNAP.read_text())
 assert snapshot['sqlite_sha256']=='bc4ab203e8f6edc902cda41171c9feed602fe854ea34aee3869b610199167f93'
 comparison=json.loads((HERE/'snapshot-lexical-comparison-20260927.json').read_text())
 assert not comparison['missing_ids'] and not comparison['lexical_field_differences']
 old=[json.loads(x) for x in (RAW/'20260911-kharia-living-audit.jsonl').read_text().splitlines()]
 assert len(old)==452
 rows=[];audit=[];decisions=[]
 for record in old:
  emitted=[]
  for baseline in record['rows']:
   r=baseline.copy();ts,basis=tags_for(r[3],r[2]);oldgloss=r[3]
   if r[3].strip().lower()=='fly (n)':r[3]='fly'
   clean_pronouns={'we 2 (not you)':'we','we 3+ (not you)':'we','they-2':'they','their (2 people)':'their'}
   if r[3].strip().lower() in clean_pronouns:
    r[6]='; '.join(filter(None,[r[6],'Source pronoun analysis: '+r[3]+'.']))
    r[3]=clean_pronouns[r[3].strip().lower()]
   if r[3].startswith('fifty ('):
    r[3]='fifty';r[6]='; '.join(filter(None,[r[6],'Source literal numerical analysis: "two-twenty-ten".']))
   if r[2].startswith('-'):
    ts+=['uncertain'];r[6]='; '.join(filter(None,[r[6],'Source prints a leading boundary hyphen; its morphological interpretation is not supplied.']))
    basis+=['Leading source boundary retained without guessing suffix status.']
   r[14]=' '.join(dict.fromkeys(r[14].split()+ts))
   rows.append(r);emitted.append(r[10]);decisions.append({'entry_key':r[10],'source_gloss':oldgloss,'added_tags':ts,'basis':basis or ['No explicit structured grammatical label; no POS inferred from ordinary lexical English.'],'old_row':baseline,'new_row':r})
  audit.append({'source_unit_key':record['entry_key'],'status':'emitted','entry_keys':emitted,'evidence':record,'current_source_record':next(e for e in snapshot['entries'] if e['id']==record['upstream_entry_id'])})
 for s in snapshot['sentences']:
  native=nfc(json.loads(s['text'])['default']);translation=json.loads(s['translation']);key=f"{SOURCE}:sentence:{s['id']}"
  note='Source provides native script and free translation, but no phonetic transcription or interlinear glosses. Sentence locality is unspecified; dictionary-head dialect assignments are not transferred. Automatic token links remain unaudited lexical matching evidence.'
  r=['kh','',native,nfc(translation['en']),native,'',note,f"{SOURCE}[snapshot 2026-09-18, sentence {s['id']}]",'','',key,'','','','sentential multiword-expression']
  rows.append(r);audit.append({'source_unit_key':key,'status':'emitted','entry_keys':[key],'evidence':s,'alignment_decision':'No word rows: automatic tokens lack gloss alignment, contain ambiguous candidates and split meaningful apostrophes as ignored punctuation. Whole expression preserves every native character.','transcription_decision':'Native-only preservation, no Romanization or IPA inferred.','locality_decision':'No text/dialect relation available; parent headword dialect is not sentence attribution.'})
 assert len(rows)==524 and len({r[10] for r in rows})==524 and len(audit)==455
 for r in rows:
  assert len(r)==15 and r[2] and r[3]
  for i,x in enumerate(r):r[i]=nfc(x)
 OUT.mkdir(exist_ok=True)
 with (OUT/'20260911-kharia-living.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 (OUT/'whole-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
 (OUT/'metadata-decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2)+'\n')
 profile=(DATA/'conversion/kharia-living.txt').read_text();known={x.split('\t')[0] for x in profile.splitlines()[1:]}
 additions=sorted(set(''.join(r[2] for r in rows))-known)
 assert all('\u0900'<=x<='\u097f' for x in additions),additions
 (OUT/'kharia-living.txt').write_text(profile+''.join(f'{x}\t{x}\n' for x in additions))
 report={'status':'owner proposal pending independent review; canonical unchanged','rows':len(rows),'legacy_keys':521,'new_native_sentences':3,'audit_units':455,'metadata_changed_rows':sum(d['old_row']!=d['new_row'] for d in decisions),'profile_added_literal_native_characters':additions,'version_transition':'452 lexical records exact-match historical snapshot; omitted sentences recovered from separately pinned current2026-09-18 revision.','pending':['owner second reading of every metadata decision','independent seeded audit and all3sentence edge review','metadata/importer installation guard','focused actual parser','canonical approval'],'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}}
 (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
