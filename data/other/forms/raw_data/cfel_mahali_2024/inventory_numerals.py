"""Inventory exact-IPA repetitions in the printed cardinal-number section."""
import collections, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def inventory():
    rows=[json.loads(s) for s in (ROOT/'analysis-proposals.jsonl').read_text().splitlines()]
    numbers=[r for r in rows if r['domain']=='Cardinal Numbers']
    grouped=collections.defaultdict(list)
    for row in numbers:
        grouped[row['form']].append({'entry_key':row['entry_key'],'source_gloss':row['source_gloss']})
    return {'entries':len(numbers),'distinct_exact_IPA':len(grouped),
            'scope':'Exact NFC source IPA only; no arithmetic reconstruction, phonological equivalence or proof that other entries are semantically correct.',
            'repeated_IPA':[{'source_ipa':form,'entries':entries} for form,entries in grouped.items() if len(entries)>1]}

if __name__=='__main__':
    result=inventory()
    (ROOT/'numeral-inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f"{result['entries']} cardinal records; {len(result['repeated_IPA'])} repeated exact-IPA groups")
