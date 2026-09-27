"""Read-only duplicate-ID reconciliation; does not emit/install lexical forms."""
import collections
import hashlib
import importlib.util
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gorum_source',ROOT/'import_source.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def signature(raw):
 glosses=re.findall(r"``(.*?)''",raw,re.S)
 if glosses:
  boundary=list(re.finditer(r"``(.*?)''",raw,re.S))[-1].end()
  lexical=raw[:boundary]
 else:
  lexical=re.split(r'\bCf\.|\?\?|\s+#\d+\.',raw)[0]
 header=lexical.split('{',1)[0]
 return {'header':re.sub(r'\s+',' ',header).strip(),'headwords':re.findall(r'<([^<>]+)>',header),'labels':re.findall(r'\{([^}]+)\}',lexical),'glosses':glosses,'lexical_text':re.sub(r'\s+',' ',lexical).strip()}

SPECIAL={
'15410':('hold_id_collision','Completely different lexical records: bar=kaD forty versus kaDaman woman. Preserve occurrence-specific records/keys; never collapse by numeric ID.'),
'33670':('hold_id_collision','Unglossed kaD cross-reference to twig versus tali eave/rafter. Distinct heads and grammatical content; numeric ID cannot define one lexical identity.'),
'15440':('hold_id_collision','kaDi twig versus unglossed koNDa cross-reference to some. Distinct heads; retain separate source occurrences with different keys.'),
'11261':('preserve_lexical_extension','Second occurrence adds A-witness go?tuG and senses sari/blanket to Z go?=tuG cloth/clothing. Shared ID and shared Z form permit same lexical record, but do not discard witness, form or gloss extension. Retain both source heads, fuller sense, both raw commentaries.'),
'4172':('same_record_gloss_variant','Same head, witness, phonetic bracket and verb label; to tie_on a loincloth versus to tie a loincloth differ only on. Reuse lexical identity, retain both literal gloss readings and commentary, prefer fuller first gloss.'),
'12702':('preserve_source_form_variant','Same A witness and sixty/three-score gloss; ya=gu nika?n versus yagu nika?n changes explicit morphological segmentation. Preserve both raw spellings as source variants; do not normalize Original away.'),
'37402':('hold_headword_conflict','Same A witness and eighty/four-score gloss, but un=gi nika?n versus uGgi nika?n differs in n/G transcription and segmentation. Numeric identity may relate them, but equivalence is unproven. Keep distinct occurrence forms and mark uncertainty; do not silently collapse.'),
}

def main():
 records=module.raw_records((ROOT/'source-wayback.txt').read_bytes());groups=collections.defaultdict(list)
 for r in records:groups[r['source_id']].append(r)
 entries=[]
 for sid,rs in groups.items():
  if len(rs)<2:continue
  sigs=[signature(r['raw']) for r in rs]
  exact=len({r['raw'] for r in rs})==1
  lex_equal=len({x['lexical_text'] for x in sigs})==1
  numbered_sigs=[signature(r['raw'].splitlines()[-1]) for r in rs]
  numbered_equal=len({x['lexical_text'] for x in numbered_sigs})==1
  context=[r['raw'].rsplit('\n',1)[0] if '\n' in r['raw'] else '' for r in rs]
  if sid in SPECIAL:decision,reason=SPECIAL[sid]
  elif numbered_equal and not lex_equal:decision,reason='reuse_numbered_record_preserve_context','The numbered lexical entry itself is identical; nonidentical unnumbered parent/root headings were bundled into chunks. Preserve those headings separately as source context (and separately audit genuinely glossed unnumbered lexemes); never propagate a parent gloss onto child heads.'
  elif exact:decision,reason='reuse_exact_record','Byte-identical numbered record repeated; retain both occurrence locators in audit.'
  elif lex_equal:decision,reason='reuse_same_lexical_record','Header including witness/transcription brackets, grammatical labels and every double-quoted sense are identical. Differences are only commentary, cross-references, derivational segmentation notes or whitespace; preserve all raw/commentary variants.'
  else:decision,reason='needs_review','Unexpected lexical signature difference; no automated collapse authorized.'
  safe=decision in {'reuse_exact_record','reuse_same_lexical_record','same_record_gloss_variant','reuse_numbered_record_preserve_context'}
  fields=['header','headwords','labels','glosses']
  entries.append({'source_id':sid,'ordinals':[r['ordinal'] for r in rs],'decision':decision,'reuse_safe':safe,'reason':reason,'exact_raw_duplicate':exact,'numbered_lexical_signatures':numbered_sigs,'unnumbered_context':context,'lexical_differences':{k:[s[k] for s in sigs] for k in fields if sigs[0][k]!=sigs[1][k]},'occurrences':[{'ordinal':r['ordinal'],'raw':r['raw'],'raw_sha256':hashlib.sha256(r['raw'].encode()).hexdigest(),'lexical_signature':sg} for r,sg in zip(rs,sigs)]})
 assert len(entries)==81 and sum(e['exact_raw_duplicate'] for e in entries)==24
 counts=collections.Counter(e['decision'] for e in entries)
 (ROOT/'duplicate-reconciliation.json').write_text(json.dumps({'source_sha256':module.SOURCE_SHA256,'review_date':'2026-09-26','repeated_ids':81,'decisions':dict(counts),'policy':'Repetition never warrants arbitrary deletion. Safe reuse still preserves occurrence provenance and all distinct commentary. Nonidentical lexical attestations retain source-specific keys until intentional reconciliation. No importer/install changes performed.','entries':entries},ensure_ascii=False,indent=2)+'\n')
 print(dict(counts))
 for e in entries:
  if e['decision']=='needs_review':print(e['source_id'],e['lexical_differences'])
if __name__=='__main__':main()
