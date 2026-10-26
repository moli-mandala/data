import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());d={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[]
rj={'Marw','had','bagri_fatehabad','marwari_bagra','mewari_dholpura','mewari_basad','dhundari_bamore','dhundari_badagaon','mewati_akera'}
for r in rs:
 if r['ID'] in d:continue
 f,g=r['Form'],r['Gloss'];x=None
 if not all(v.strip().split(' ')[0] in ['you','your'] for v in g.split(';')):continue
 if r['Language_ID'] in rj and f in ['tʰā̃','tʰā']:
  x=dict(parent='10511',kind='reflex',citation='CDIAL[10511]',evidence='Turner yuṣmad includes the Middle Indo-Aryan tumhe/tumhaṇa paradigm and Old Marwari thāharaü. Grierson’s historical Gujarati/Rajasthani table (Encyclopaedia Britannica 1911, XII p. 712) explicitly maps tumhe to Rajasthani thē and the oblique plural to thā̃. These survey thā/thā̃ you forms are compared with that paradigm, preserving their elicited use.')
 if f in ['tūs','tūmareh','tohare','tohar']:
  x=dict(parent='10511',kind='reflex',citation='CDIAL[10511]',evidence='Turner yuṣmad explicitly gives the Kohistani tus second-person plural and the possessive paradigm tumhāra/tohār. These corresponding survey forms retain the same historical second-person plural family, even where an oblique/possessive form supplies the elicited respectful you answer.')
 if f=='tũɛ':x=dict(parent='5889',kind='reflex',citation='CDIAL[5889 and addenda]',evidence='Turner tuvam explicitly adds western Pahari poetic tue plural you beside tu. The Jaunsari tũɛ respectful/plural responses match that extension; their plural use is not by itself evidence for an unrelated tumhe stem.')
 if f in ['tomi sʌbu','tʌmi sʌbu','t ūmsab']:
  x=dict(components=['10511','13276'],citation='CDIAL[10511];CDIAL[13276]',evidence='The whole response consists of tum/tomi you plus sab/sabu all. Turner yuṣmad explicitly contains the regional tum/tumi forms and sarva supplies sab/sabu. Both ordered lexical components are represented, preserving source spacing.')
 if f in ['tʰe log','tʰā̃ log','tʰā log','tʰẽ log','tʰe lɔg','tamlog','t ūmlog']:
  x=dict(components=['10511','11119'],citation='CDIAL[10511];CDIAL[11119];platts1884[s.v. log]',evidence='The whole response combines the regional second-person plural pronoun with log people as a pluralizer. Grierson’s historical table (1911, XII p. 712) explicitly connects Rajasthani thē/thā̃ with tumhe/tumha; Platts derives log from Prakrit logo and Sanskrit loka and explicitly records its pluralizing use. Both lexical members are represented in order.')
 if f=='tʰā̃ sārā' and r['Language_ID'] in rj:
  x=dict(components=['10511','13355'],citation='CDIAL[10511];CDIAL[13355]',evidence='The response is Rajasthani thā̃ you plus sārā all. The former belongs to the documented tumhe/tumha second-person plural paradigm; Turner sāra supplies the whole/all adjective. Both members are represented.')
 if x:a.append(dict(record=r,**x))
(P/'pass110-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n');print(len(a))
p=P/'lexical-reference-links.json';x=json.loads(p.read_text());x['platts1884[s.v. log]']='https://www.rekhta.org/urdudictionary?keyword=log';p.write_text(json.dumps(x,ensure_ascii=False,indent=1)+'\n')
(P/'rajasthani-pronoun-primary-note.md').write_text('''# Rajasthani pronoun comparison\n\nGrierson’s historical table in Encyclopaedia Britannica 1911, XII, printed p. 712 explicitly gives Apabhraṃśa tumhē → Rajasthani thē/tamē and oblique tumhahã → thā̃/tamā̃. It also gives ēho → yō and plural ēi → ē/yē. Used as supplementary primary grammatical evidence alongside CDIAL 10511 and 2530; no new lexical donor or bibliographic import is made.\n\n[Read the historical page](https://en.wikisource.org/wiki/Page:EB1911_-_Volume_12.djvu/737).\n''')
