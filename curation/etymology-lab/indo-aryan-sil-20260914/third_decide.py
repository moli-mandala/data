import json,re
from pathlib import Path
P=Path(__file__).resolve().parent;qs=json.loads((P/'third-rules.json').read_text());cs=json.loads((P/'third-candidates.json').read_text());accepted=[];held=[]
for x in cs:
 r=x['record'];i=x['families'][0];q=qs[i];l=r['Language_ID'];cl=r['clade'];w=r['Form'];reason=None
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I):reason='Source/dialect uncertainty marker: determine whether this qualifies locality attribution or lexical reading.'
 elif cl in {'Kohistani','Shinaic','Chitrali','Kunar','Pashai'}:reason='Dardic local transmission or subsection is not settled by the general plains match; in particular, CDIAL 9875 has a separate marucca branch.'
 elif i==3 and l in {'AdivasiOriya','Bote','DewasDoneDanuwar','KochariyaEastDanuwar','Majhi','hal'}:reason='The nasal roof family is plausible, but regional channa/channi and immediate contact transmission require a local comparison beyond the cited plains roofing forms.'
 elif i==5 and re.search('[tţ]',w.replace('ṭ','')):reason='Dental versus retroflex stop in the ring word is not settled; inspect the survey phonetic conventions before selecting the historical ṭh branch.'
 elif i==5 and l=='Ush':reason='Ring word has a possible regional loan pathway; the full CDIAL article does not directly establish this lect’s immediate parent.'
 elif i==8 and l in {'P','poth','awan'}:reason='CDIAL 3515 explicitly labels Punjabi kodo as a Hindi loan; donor evidence is required.'
 elif i==12 and cl not in {'Bihari','Eastern'} and l not in {'Bote','Majhi','Buksa','Rana','Dang','Kathoriya','KochilaTharu'}:reason='The bhaṇṭā eggplant family is clear, but these western forms need a regional transmission check against the primarily eastern CDIAL comparanda.'
 elif i==13 and w in {'cau','caũ','tʃəʊ'} and l not in {'jaun'}:reason='Strongly contracted rice form needs local comparison before selecting *cāmala/*cāvala.'
 if reason:held.append(dict(**x,reason=reason,passNumber=3));continue
 accepted.append(dict(record=r,family=i,parent=q['parent'],citation=q['citation'],evidence=q['evidence']))
(P/'third-decisions.json').write_text(json.dumps(dict(accepted=accepted,held=held),ensure_ascii=False,indent=1));print('accepted',len(accepted),'held',len(held))
