"""Explicit Appendix 1 grammatical metadata; no literal or IPA changes."""
import json,csv,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def decisions():
 out={}
 for n in range(179,184):out[n]=(['num'],'Source pp295 numeral prompts one through five; same analyses in p268 table4.')
 out[193]=(['num'],'Source numeric prompt half on p296.')
 for n in [176,191,192,252,253]:out[n]=(['interr'],'Explicit interrogative prompt in Appendix1.')
 out[176][0].append('adv')
 for n in [252,253]:out[n][0].append('pron')
 out[251]=(['demonstrative'],'Explicit that prompt; no grammatical gender inferred.')
 personal={227:['first-person','sg','1sg'],228:['second-person','sg','2sg'],229:['third-person','sg','3sg'],230:['third-person','sg','3sg'],231:['first-person','pl','1pl','exclusive'],232:['second-person','pl','2pl'],233:['third-person','pl','3pl'],234:['third-person','pl','3pl']}
 for n,ts in personal.items():out[n]=(['pron']+ts,'Explicit person/number prompts p296, confirmed p268 independent-pronoun table. ɑlɛ explicitly exclusive there. No gender tags: p267 states there is no grammatical gender.')
 out[235]=(['pron','refl'],'Prompt self, confirmed explicit refl analysis in examples12/20.')
 for n in [199,200,202,203,204,205,206,207,210,211,212,213,214]:out[n]=(['kinship'],'Explicit kinship relationship prompt; no referent gender transferred to grammatical gender.')
 return out
def apply(rows):
 originals=json.loads((ROOT.parent/'20260921-peterson-turi-audit.json').read_text())
 lookup={k:u for u in originals for k in u['emitted_keys']}
 rules=decisions();changes=[]
 for r in rows:
  u=lookup[r[10]];n=float(u['printed_item'].strip('.,'))
  extra,basis=rules.get(n,([],''));extra=list(extra)
  if ' ' in r[2]:extra.append('multiword-expression');basis+=' Printed multiword target expression retained as a unit.'
  prior=r[14];r[14]=' '.join(dict.fromkeys(prior.split()+extra))
  if prior!=r[14]:changes.append({'entry_key':r[10],'printed_page':u['printed_page'],'printed_item':u['printed_item'],'old_tags':prior,'tags':r[14],'basis':basis.strip()})
 return changes
if __name__=='__main__':
 base=ROOT.parent.parent.parent/'20260921-peterson-turi.csv'
 rows=list(csv.reader(base.open()));changes=apply(rows)
 report={'scope':'All275 Appendix1 source units /246 emitted rows; metadata only','source_pages':list(range(291,298)),'basis':'Complete native-page text and original-image review of metadata-bearing pp295–297. Earlier body/object lists contain no explicit person/number inflection; no blanket English-to-POS or English-plural-to-Turi-plural inference. Existing borrowing and literal-typography qualifications retained.','changes':changes,'unchanged_columns':[i for i in range(1,16) if i!=15],'baseline_sha256':hashlib.sha256(base.read_bytes()).hexdigest()}
 (HERE/'legacy-metadata-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(len(changes),'legacy rows with tag additions')
