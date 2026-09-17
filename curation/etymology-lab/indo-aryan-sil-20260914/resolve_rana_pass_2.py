"""Explicit scope decisions after inspecting every second-pass candidate group."""
import json,re
from pathlib import Path
P=Path(__file__).resolve().parent
qs=json.loads((P/'second-rules.json').read_text());cs=json.loads((P/'resolve-rana-candidates-2.json').read_text());accepted=[];held=[]
for x in cs:
 r=x['record'];inds=x['families'];i=inds[0];q=qs[i];l=r['Language_ID'];cl=r['clade'];w=r['Form'];reason=None
 if len(inds)>1:reason='Competing shortlisted families require exact subsection review.'
 elif False: reason=None  # Locality-only flag resolved below.
 elif cl in {'Kohistani','Shinaic','Chitrali','Kunar','Pashai'} and not ((i==7 and l=='Kal') or (i==13 and l=='Kho')):reason='Dardic local development or borrowing pathway is not established by the plains comparison.'
 elif i==0 and (cl in {'Lahndic','W. Pahari','Gujaratic','Marathic','Rajasthani','Bhilic'} or l in {'Goj','kaithal','Nimadi','Khandesi','Bhilali','Bhili','Bote','Majhi','hal'}):reason='CDIAL 6328 marks several western din forms as Hindi/Sanskrit loans; establish the immediate local transmission before using a reflex edge.'
 elif i==1 and w in {'rukhwa','rukhuwa'}:reason='The -wa/-uwa extension requires a local morphology decision; do not treat the whole form as bare rukh.'
 elif i==2 and l in {'Nimadi'}:reason='Morning saber/saver may have spread through Hindi; dictionary coverage does not establish this southwestern lect’s transmission.'
 elif i==3 and l=='jaun':reason='CDIAL 5086 flags Punjabi borrowing for Jaunsari root forms; determine the immediate donor.'
 elif i==8 and cl=='Gujaratic' and w.startswith('b'):reason='CDIAL 11225 explicitly marks Gujarati b-initial forms as Hindi loans.'
 elif i==9 and w in {'ni','nĩ'}:reason='High-front nail vowel needs local paradigm evidence; do not choose nakha from short shape alone.'
 elif i==12 and l in {'N','Bote'}:reason='CDIAL 1670 marks Nepali ujjar as a Bihari loan; the Nepal-area r-form needs local transmission review.'
 elif i==18 and (re.search('[śṣ]',w) or l=='Goj'):reason='Year word requires a learned/contact or local sibilant check; CDIAL marks Punjabi baras as Central borrowing.'
 elif i==20 and l in {'Chitwan','Dang','Kathoriya','KochilaTharu','Sunha','MagahiNepal'}:reason='The Bhojpuri dāhin comparison is cross-referred to dakṣiṇā rather than dākṣiṇa; neighboring eastern right-forms need section-level and contact review.'
 elif i==22 and (cl=='Marathic' or l=='Khandesi'):reason='CDIAL 11745 labels Marathi bijlī a Hindi loan; immediate local transmission needs review.'
 elif i==23 and (cl=='Bihari' or l in {'Buksa'}):reason='CDIAL 10875 explicitly labels eastern Maithili lakṛī a Hindi loan; local transmission needs evidence.'
 elif i==24 and (cl in {'Bihari','Eastern'} or l in {'Chitwan','KochilaTharu','MagahiNepal','AdivasiOriya','Bhatri'}):reason='CDIAL 2871 labels several eastern kapṛā forms Hindi loans; identify a supported immediate donor before saving.'
 if reason:held.append(dict(**x,reason=reason,passNumber=2));continue
 accepted.append(dict(record=r,family=i,parent=q['parent'],citation=q['citation'],evidence=q['evidence']))
(P/'resolve-rana-decisions-2.json').write_text(json.dumps(dict(accepted=accepted,held=held),ensure_ascii=False,indent=1))
print('accepted',len(accepted),'held',len(held))
