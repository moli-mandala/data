import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
E={
'972':'CDIAL 972 asau groups the remote-demonstrative paradigm, including Lahnda/Punjabi o, Nepali u, Hindi wah and Old Marwari vo. The survey u/o/v- singulars and ve/be plurals identify that paradigm; this is a family-level analysis of remodeled case/number forms, not a direct sound derivation of each form from nominative asau. Regional transmission remains open.',
'9691':'CDIAL 9691 ma explains replacement of the first-person nominative by oblique/instrumental forms: Prakrit maē, Apabhramsha maĩ, Hindi/Punjabi maĩ, Western Pahari me/mu, Marathi mī, and Oriya mu. It explicitly lists Chilis/Gowro ma. These survey first-person forms select ma, not aham or asme.',
'10234':'CDIAL 10234 mūtra gives Prakrit mutta, Hindi mūt, Punjabi mūtar and Bshk. mūλ “urine”. The simple t/tr and named lateral forms are supported; additional aspiration or another northern fricative needs local confirmation.',
'10203':'CDIAL 10203 mudrā gives Prakrit muddā/muddiā “seal, ring”, Oriya mudi, Konkani muddi and Lahnda mundrī. The dental mudi/mundi/mundari forms fit this signet-ring family; retroflex survey variants are kept for a separate consonant review.',
'2485':'CDIAL 2485 ekādaśa explicitly gives Torwali agāš, Palula akāš, eastern egāra, Punjabi giārā/yārā and Hindi igārah “eleven”. The survey numerical forms fit these local outcomes; Gawri yaś needs confirmation of the initial consonant against the entry’s ǰāš.',
'11616':'CDIAL 11616 viṃśati explicitly gives Kalasha bīši, Maiya bīš, Awan vī and Western Pahari bī in the addendum, alongside eastern bis. This supports the survey twenty forms, including northern loss of the final sibilant.',
'10951':'CDIAL 10951 lamba gives Prakrit laṃba, Punjabi/Lahnda lammā, Western Pahari lammo, Kumauni lāmo, Gujarati lāmbu and Maithili namā “long”. Thus Magahi namma has a documented eastern n- comparator rather than being rejected as an unexplained spelling.',
'5228':'CDIAL 5228.1 jihvā gives Prakrit jibbhā, Nepali jib(h)ro and Bhojpuri jībhi, with Maiya zīb. Additional initial y or final -ban requires a source/contamination check; the entry does not establish those changes simply from older links.',
'4287':'CDIAL 4287 godhūma explicitly lists Khowar/Gawri/Bshk. gom, Gowro gū̃, Bengali gom, eastern gahum and Western Pahari giũ. The entry notes uncertainty about retained final m in eastern forms; the comparative wheat family is still explicit.',
'4225':'CDIAL 4225.1 gūtha gives Prakrit gūha and the widespread gū/gūh/guhu “excrement” forms, including Nepali, Hindi, Bengali and Oriya. These simple feces responses fit branch 1; no Bshk. gūt form is assigned to it.',
'1605':'CDIAL 1605 iha means “here” and discusses yahā̃. It does not itself supply a whole-word etymology of interrogative kahā̃ “where”; a link inherited from an older record would skip the interrogative construction.',
'6849':'CDIAL 6849.1 dhūma gives the smoke family dhum/dum and dhua/dhuwa, including northern dūm and eastern dhuā̃. The addendum gives Gujarati/Kachchi dhũāṛo, but that and northern t- or ŋ-containing variants require specific local development or subsection review.',
'8056':'CDIAL 8056.1 pāda lists Prakrit pāya, Hindi pā/pā̃u “foot, leg”, Old Marwari pāya/pāva and Marathi/Gujarati pāy. It discusses final u/v as incorporation of inflectional endings, with pādu an explicitly less probable alternative; that qualification is retained.',
'2668':'CDIAL 2668 distinguishes kaṇṭa from 2668.2 kaṇṭaka. The latter explicitly lists Bengali kā̃ṭā, Hindi kā̃ṭā, Punjabi kaṇḍā and Nepali kā̃ṛo. Bare eastern kā̃ṭ belongs to branch 1; uncertain northern forms still require a branch decision.',
'10247':'CDIAL 10247 mūrdhan expressly says that many unaspirated muṇḍ/muṛ head forms may instead derive from, or be contaminated by, muṇḍa “shaven”. The current forms do not resolve that competing etymon.',
'5014':'CDIAL 5014 *chātti directly gives chest/breast forms chāti/chātī in Hindi, Nepali, Bihari, Bengali and Rajasthani. Simple affricate-bearing survey forms fit that family; a separate initial s needs local correspondence evidence.',
'13676':'CDIAL 13676 stabdha section 2 documents the semantic shift stiff/sluggish to cold, with Lahnda ṭhaḍḍha and Gujarati tāḍhu. It separately discusses nasal ṭhaṇḍha/thaṇḍha and probable Dravidian influence, followed by Hindi-mediated spread. Those nasal forms require their exact stored extension, not automatic use of the unsplit head.',
'135':'CDIAL 135.1 aṅguli gives Punjabi aṅgal/aṅgul/uṅglī, Nepali aũli, Bihari aṅguri and Gujarati aṅgḷī. Chilis hagvi instead needs the separately numbered aṅgūḍi branch and cannot be inferred from an unsplit older assignment.',
'4428':'CDIAL 4428 ghara gives Middle Indo-Aryan ghara and regional ghar/ghor/gar, with the addendum’s Garhwali ghor and Western Pahari ghor. Its usual connection with gṛha is expressly phonologically difficult; the link stops at the documented ghara family.',
'9758':'CDIAL 9758.1 matsya gives Prakrit maccha, Bihari/Hindi machlī/macharī and Bshk. mac “fish”; branch 2 *matsiya separately includes Khowar and Kalasha macī. These are kept distinct from the fly family despite similar modern forms.'}
qs=[];ix={};acc=[];held=[];current={r['Form_ID'] for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()) if r['Status']=='accepted' and r['Rank']=='1'}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];w=norm(r['Form']);target=k;reason=None
 if r['ID'] in current:continue
 if k in {'1605','10247'}:reason=E[k]
 elif k=='10203' and any(c in w for c in ['ḍ','ṇ','ṭ']):reason='Retroflex ring forms need local consonant/competing-family review.'
 elif k=='10234' and ('h' in w or 'ẓ' in w):reason='Additional aspiration or fricative outcome is not established by the named primary comparanda.'
 elif k=='2485' and r['Language_ID']=='Gaw':reason='Gawri initial y differs from the primary ǰāš; verify source transcription/local change.'
 elif k=='5228' and (w.startswith('y') or w.endswith('ban') or r['Language_ID']=='Gowro'):reason='Initial y, final aspiration, or -ban may involve source transcription or contamination not resolved here.'
 elif k=='6849' and any(c in w for c in ['ḍ','ṛ','ŋ']) or k=='6849' and w.startswith('t'):reason='Smoke extension or consonant development needs a specific local analysis.'
 elif k=='5014' and w.startswith('s'):reason='Initial s for ch needs local support.'
 elif k=='13676' and any(c in w for c in ['ṇ','n']):reason='Nasal cold forms require exact thaṇḍha extension review.'
 elif k=='135' and r['Language_ID']=='Chil':reason='Chilis finger form belongs to the aṅgūḍi discussion, not base aṅguli.'
 elif k=='4428' and w.startswith('kh'):reason='Initial voiceless kh requires a local correspondence check.'
 elif k=='9758' and r['Language_ID'] in {'Kho','Kal'}:target='9758-2'
 elif k=='9758' and r['Language_ID'] in {'Phal','Gowro'}:reason='Northern maci requires distinguishing matsya feminine from *matsiya branch.'
 elif k=='2668':
  if r['Language_ID'] in {'Bshk','Tor','bhatr'}:reason='The northern thorn form does not decide kaṇṭa versus kaṇṭaka here.'
  elif w.endswith(('a','o','u')):target='2668-2'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires local review.'
 if (k,target) not in ix:ix[k,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+']',evidence=E[k]))
 i=ix[k,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=17));continue
 ev=E[k]+' Joint survey comparison leaves inheritance versus transfer between Indo-Aryan languages open.'
 if 'uncertain' in r['Tags'] and r['Language_ID']=='Rana':ev+=' RNS uncertainty is locality attribution only: the source audit found zero ambiguous lexical readings; original tags are retained.'
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev))
(P/'global-first-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'global-first-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'global_first_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','global-first-decisions.json').replace('sixth','global-first'));print('accepted',len(acc),'held',len(held));print(sorted({x['parent'] for x in acc}))
