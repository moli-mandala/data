import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass94';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};donor='f_cem3rbh3bdmwc';pers='f_mszcztcnpsec6'
sam=json.loads((P/'pass94-samsad-research.json').read_text());url=next(x['url'] for x in sam if x['word']=='বাদাম')
rules=[dict(parent=pers,citation='Platts[p.119]',evidence='Samsad Bangala Abhidhana (Biswas 2004, p.598, bādām 1) defines Bengali bādām broadly as an edible seed with a hard covering and explicitly marks Persian bādām as its source. Platts p.119 likewise identifies Persian bādām almond/country almond. The Bengali survey peanut gloss is a specific use of this broader nut word, not evidence that Persian itself meant peanut. The existing Persian entry is linked as the source of the Bengali loan; the separate Bengali sail homonym from bādbān is excluded. Primary Bengali entry: '+url),dict(parent=donor,citation='Platts[p.119];kim-kim-sangma2012garo[wordlist item 42, site 0 Bangla]',evidence='The Bangla comparison list records badam peanut, and Samsad (Biswas 2004, p.598) documents Bengali bādām as the Persian-derived nut word. Hajong and Bishnupriya badam peanut are linked to this attested Bengali regional donor, whose Persian ancestry is established in the same pass. This is a provisional cross-Indo-Aryan transmission link: the survey site is not asserted to be the historical donor locality, and an intermediate regional language is not excluded. The exact peanut sense is preserved. Primary Bengali entry: '+url)]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Form']!='badam' or r['Gloss']!='peanut' or r['Language_ID'] not in {'B','Bishnupriya','Hajong'}:continue
 i=0 if r['Language_ID']=='B' else 1;q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='borrowed',citation=q['citation'],evidence=q['evidence']))
assert donor in {x['record']['ID'] for x in acc}
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
primary={'Platts-119-and-1094':[dict(page=1094 if x['word']=='mom' else 119,text=x.get('text',''),url=x['url']) for x in json.loads((P/'pass94-platts-research.json').read_text())],'Samsad-598-and-719':[dict(page=598 if x['word']=='বাদাম' else 719,text=x.get('text',''),url=x['url']) for x in sam if x['word']!='চীনাবাদাম']}
(P/(stem+'-primary-articles.json')).write_text(json.dumps(primary,ensure_ascii=False,indent=1)+'\n')
(P/'pass94_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem))
(P/'pass94-candle-donor-gap.md').write_text('''# Candle donor review

Platts p.1094 explicitly gives Persian mom wax and mom-battī wax candle. Samsad p.719 confirms Bengali mom from Persian and mom-bāti candle. Primary query results are cached alongside this note.

The current graph has a suitable wick family (CDIAL 11359), but no Persian or Bengali wax donor entry. Rana/Morang mom beeswax and Romani mom wax are unlinked and do not supply the required eastern donor. The Kalasha candle entry points to a generic Persian language node, which is unsuitable as a lexical donor.

No candle links were saved. Next step: add a properly sourced wax donor through the donor-ingestion workflow, or locate an existing appropriate node missed by the current search. Do not route eastern candles through the unrelated Romani or Tharu survey attestations.
''')
print({'accepted':len(acc)})
