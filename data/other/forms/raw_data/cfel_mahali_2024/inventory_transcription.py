"""Inventory raw transcription symbols without applying phonological repairs."""
import collections,json,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def inventory():
    rows=[json.loads(s) for s in (ROOT/'candidates.jsonl').read_text().splitlines()]
    counts=collections.Counter(''.join(r['raw_ipa'] for r in rows))
    symbols=[{'symbol':c,'codepoint':f'U+{ord(c):04X}','name':unicodedata.name(c,'CONTROL'),'occurrences':n} for c,n in sorted(counts.items())]
    attention='?.ɪɾʂ̘̜̠̥̹͡'
    review=[{'entry_key':r['entry_key'],'ipa':r['raw_ipa'],'gloss':r['candidate_english'],'symbols':sorted(set(r['raw_ipa'])&set(attention)),'status':'requires source review; no automatic phonological repair'} for r in rows if set(r['raw_ipa'])&set(attention)]
    return {'source_entries':len(rows),'symbols':symbols,'review_queue':review,'native_with_NUL':sum('\x00' in r['raw_native'] for r in rows),'ipa_with_NUL':sum('\x00' in r['raw_ipa'] for r in rows),'line_wrapped_ipa':sum('\n' in r['raw_ipa'] for r in rows)}

if __name__=='__main__':
    result=inventory();(ROOT/'transcription-inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f"{len(result['symbols'])} symbols; {len(result['review_queue'])} entries require targeted review")
