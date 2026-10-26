"""Review-only complete Naiki utterances and exact free-translation scopes.

Never installs or edits the owner token inventory. Speaker labels are source
provenance, not spoken words. Interlinear English is never called free English.
"""
import hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=HERE/'specimens-second-reading.jsonl'
 rows=[json.loads(s) for s in source.read_text().splitlines()]
 def tokenkey(r):return f'grierson1906naiki:p{r["page"]}:specimen:line{r["line"]}:word{r["word"]}'
 quote_path=HERE/'independent-quotation-census-20260927.json'
 quotes=json.loads(quote_path.read_text())
 assert quotes['input_sha256']==sha(source)
 corrections={r['source_unit_key']:r for r in quotes['corrections']}
 assert len(corrections)==30 and sum(len(r['marks']) for r in corrections.values())==31
 source_by_key={tokenkey(r):r for r in rows}
 for key,correction in corrections.items():
  assert source_by_key[key]['form']==correction['current_raw_form']
  assert correction['proposed_raw_form'].strip('‘’“”')==correction['current_raw_form']
 def label(r):return r['form'].endswith('.—')
 def form(r):return corrections.get(tokenkey(r),{}).get('proposed_raw_form',r['form'])
 def analysis(r):return r['gloss'].rstrip('.,;:?')
 def unit(block,kind,free='',paragraph=None):
  spoken=[r for r in block if not label(r)];assert spoken
  first,last=spoken[0],spoken[-1]
  return dict(source_unit_key=tokenkey(first).replace(':specimen:',f':whole-{kind}:')+'-to-'+f'p{last["page"]}l{last["line"]}w{last["word"]}',
   kind=kind,specimen='I' if first['page']<574 else 'II',printed_pages=sorted({r['page'] for r in block}),
   source_token_keys=[tokenkey(r) for r in spoken],physical_token_keys=[tokenkey(r) for r in block],
   form=' '.join(form(r) for r in spoken),raw_source_tokens=[form(r) for r in spoken],
   pre_quote_review_tokens=[r['form'] for r in spoken],quotation_review_sha256=sha(quote_path),
   interlinear_analysis=' '.join(analysis(r) for r in spoken),analysis_type='concatenated printed interlinear English, not a free or independently lexical translation',
   free_translation=free,free_translation_paragraph=paragraph,gloss_type='printed whole-block free translation' if free else 'no separately paired free translation',
   speaker_label_evidence=[dict(source_unit_key=tokenkey(r),form=r['form'],gloss=r['gloss']) for r in block if label(r)],
   source_boundary='Original terminal full stop/question mark; semicolons and physical line/page breaks do not end units.' if kind=='utterance' else 'Complete contiguous source block corresponding to the printed English paragraph; may contain several utterances.',
   review_status='Owner original boundary/alignment review; independent integration review pending',
   input_sha256=sha(source),tags=['sentential'],canonical_changed=False)
 output=[]
 for pages in [{572,573},{574,575}]:
  block=[]
  for r in rows:
   if r['page'] not in pages:continue
   block.append(r)
   if not label(r) and re.search(r'[.?][’”]*$',form(r)):
    output.append(unit(block,'utterance'));block=[]
  assert not block,'Unclosed final source utterance'
 translation=json.loads((HERE/'specimen-II-free-translation.json').read_text())
 limits=[((574,1,1),(574,11,2)),((574,12,1),(574,13,4)),((574,14,1),(574,15,5)),((575,1,1),(575,1,6)),((575,2,1),(575,3,6)),((575,4,1),(575,5,3)),((575,6,1),(575,7,5))]
 for i,((start,end),english) in enumerate(zip(limits,translation['paragraphs']),1):
  block=[r for r in rows if start<=(r['page'],r['line'],r['word'])<=end]
  record=unit(block,'translated-block',english,i)
  record['translation_source']='Grierson1906naiki printed575, free translation paragraph'+str(i)
  record['translation_input_sha256']=sha(HERE/'specimen-II-free-translation.json')
  record['source_utterance_keys']=[u['source_unit_key'] for u in output if u['kind']=='utterance' and set(u['physical_token_keys'])<=set(record['physical_token_keys'])]
  covered={k for u in output if u['source_unit_key'] in record['source_utterance_keys'] for k in u['physical_token_keys']}
  assert covered==set(record['physical_token_keys']), (i,'Free translation boundary splits a source utterance')
  output.append(record)
 utterances=[r for r in output if r['kind']=='utterance']
 physical=[k for r in utterances for k in r['physical_token_keys']]
 assert len(physical)==len(set(physical))==537
 assert set(physical)=={tokenkey(r) for r in rows}
 assert len({r['source_unit_key'] for r in output})==len(output)
 (HERE/'reviewed-whole-utterances.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
 print(json.dumps({'utterances':len(utterances),'translated_blocks':7,'records':len(output),'physical_tokens':len(physical),'speaker_labels':sum(label(r) for r in rows),'sha256':sha(HERE/'reviewed-whole-utterances.jsonl')}))
if __name__=='__main__':main()
