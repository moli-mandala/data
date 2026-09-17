import json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='near-pass76';assert not (P/(stem+'-decisions.json')).exists();done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec={
'5232':'CDIAL 5232 *jījja explicitly gives Hindi jījī/jijjī elder sister. Bundeli jījhi/jīžī matches this nursery kin family with aspiration or frication retained; it is not a uniquely reconstructible inherited stem.',
'4272':'CDIAL 4272 *goḍḍa gives Nepali goṛo, Maithili goṛ and Hindi goṛā leg. Dang gora and Danuwar ghor fit this regional leg family with plain rhotic and, in ghor, aspiration retained as comparative qualifications.',
'10223':'CDIAL 10223 musala section 1 explicitly gives Lahnda mōlā and Punjabi mūhlā pestle. Goj molo/mūlo fits that contracted regional form, not the distinct muṣala branch in section 2.',
'9051':'CDIAL 9051 phala gives Lahnda/Punjabi phal fruit. Awan pal fruit is compared in that family with deaspiration retained; the exact regional transmission remains open.',
'11567':'CDIAL 11567 vārdala gives Punjabi baddal and neighboring bādal cloud. Awan badl directly fits this cloud family; the short vowel and cluster are retained.',
'3193':'CDIAL 3193 kīṭa gives Lahnda/Punjabi kīṛā insect and kīṛī ant, plus Hindi kīṛī ant. Awan kīṛa ant and Goj qiṛī are compared with this family; the ant sense and q versus k are retained as regional details.',
'11503-6':'CDIAL 11503.6 vaṅgana explicitly gives Bengali bāgun eggplant. Hajong bagon fits this contracted eastern branch with vowel rounding retained, rather than the full vātiṅgaṇa heading.',
'3275':'CDIAL 3275 *kutta gives Punjabi/Lahnda kuttā and Marwari kuto dog. Goj kottā/koto fits this t-medial dog family with vowel rounding retained.',
'3949':'CDIAL 3949 *gakṣa gives Nepali gāch and Bhojpuri gā̃ch tree. Danuwar gats/gãts fits the regional affricated tree family; deaspiration and the nasal variant are retained.',
'9198':'CDIAL 9198 *bā explicitly gives Nepali bājyā grandfather. Danuwar bajia directly matches this expanded grandfather nursery term, not the mother branch in section 2.',
'11813':'CDIAL 11813 *vibhāna gives bihān morning and addenda bhyaiṇi/byāṇi dawn. Danuwar bihani is compared with this morning family, retaining its ending; Turner also allows derivation through the related vibhāyana verbal noun.',
'4209':'CDIAL 4209 guru gives Nepali garũo and regional garuā heavy. Dotyali gəro fits this contracted heavy adjective with its vowel retained.',
'9216':'CDIAL 9216 bāla includes bālaka boy and regional bālak babe. Hadoti bāḷɛk child fits that k-extension with retroflex lateral and fronted vowel retained; learned versus vernacular regional transmission remains open.',
'2360-4':'CDIAL 2360.4 *udukk hala gives Sindhi ukhirī, Hindi ūkh lī and Gujarati ukhṛī mortar. Goj ūkhṛī and Hadoti ɔ̃khrī fit this contracted mortar branch, retaining vowel/nasal differences and the rhotic.'}
holds={
'10223':'Rana mõṭa pestle lacks the s/l sequence of musala and may involve a different pestle noun. The near match does not establish the retroflex-stop development.',
'9051':'Dang/Rana bhara fruit would require voicing ph to bh alongside l/r change. The CDIAL phala article supports the rhotic but not this voicing; verify the local reading or correspondence.',
'9216':'Rana balal child has an extra final l not documented under bāla/bālaka. Establish its suffix or whole-form history.',
'11567':'The proposed cloud match has either retroflex ḍ or initial p where the cited local bādal series has dental d and b. Verify the source spelling or local correspondence independently.',
'11503':'Torwali bāḍīgan eggplant needs an immediate-donor analysis: CDIAL 11503 explicitly identifies Iranian transmission for some nearby full forms. The IA-only borrowing preference does not settle this Iranian route.',
'3193':'Goj piṛi ant begins p rather than k/q. Distinguish a pi-/pilīla ant family from kīṭa before choosing this near match.',
'5071':'The coto short forms have dental t and no marked initial aspiration, while the chōṭṭa comparanda have retroflex ṭ. The addendum includes other child/small stems; verify the source and exact branch.',
'3949':'Maiyan gaī tree lacks the affricate defining the cited gāch family. Seek a local tree-word history rather than deleting that consonant through a near match.',
'3275':'Danuwar kuṭa dog has retroflex ṭ absent from the regional kutta forms. Verify whether this is source transcription or a distinct local development.',
'4209':'Danuwar garho heavy has h not accounted for by the garu/garuā comparanda; compare alternative heavy/dense stems and regional morphology.'}
rules=[dict(parent=p,citation='CDIAL['+p.replace('-','.')+']',evidence=e.replace('udukk hala','udukk hala'.replace(' ', '')).replace('ūkh lī','ūkhlī')) for p,e in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0];why=None
 if r['ID'] in done or len(ps)>1:continue
 if p=='11503' and r['Language_ID']=='Hajong':p='11503-6'
 if p not in set(spec)|set(holds):continue
 if p=='10223' and r['Language_ID']=='Rana':why=holds[p]
 if p=='9051' and r['Language_ID']!='awan':why=holds[p]
 if p=='9216' and r['Language_ID']=='Rana':why=holds[p]
 if p=='11567' and r['Form']!='badl':why=holds[p]
 if p=='11503':why=holds[p]
 if p=='3193' and r['Form']=='piṛi':why=holds[p]
 if p=='5071':why=holds[p]
 if p=='3949' and r['Language_ID']=='Mai':why=holds[p]
 if p=='3275' and r['Language_ID']!='Goj':why=holds[p]
 if p=='4209' and r['Language_ID']!='Dotyali':why=holds[p]
 if why:held.append(dict(record=r,families=[],reason=why,passNumber=76))
 else:
  q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'near_pass76_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})
