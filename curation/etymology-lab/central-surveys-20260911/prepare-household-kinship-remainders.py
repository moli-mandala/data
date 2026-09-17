from research_helpers import Batch
b=Batch(15)
def a(l,g,w,p,e,t='qualified',s=None,**kw):
 rows=[r for r in b.inv[l] if r['ID'] not in b.used and r['Form'] in w.split('|') and (r['Gloss']==g if s is None else set(r['Gloss'].split('; '))<=set(s))]
 if rows:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rows)),p,e,tier=t,source_glosses=list(dict.fromkeys(r['Gloss'] for r in rows)),**kw)
for l,w in [('Malvi','logai|lugai|logāy|lagāi|lugay'),('Nimadi','lugai|logāy')]:
 a(l,'woman; wife',w,'f_k66mptwk5okq6','The existing Jambu lugāī family is grounded in Arora’s Hindi lugāī ‘woman, wife’ record (20230403-arora.csv, sometimes pejorative). The survey meanings and consonant frame support family membership; vowel variation and local transmission remain open. This is a modern comparative family, not a claim of a reconstructed Sanskrit ancestor.',s=['woman','wife'],citation='arora',source_url='/Users/aryamanarora/Documents/Code/jambu-all/data/data/other/forms/20230403-arora.csv')
for l,w in [('Malvi','dhaṇi|dhaṇo'),('Nimadi','dhaṇi|dhāṇi')]:
 a(l,'husband',w,'6722','CDIAL 6722 gives Hindi/Gujarati dhaṇī ‘husband’ but explicitly allows dhanika under 6721 as an alternative. The latter is explained as a Sanskritization of Prakrit dhaṇia ‘praiseworthy’ from dhanya; thus master/owner > husband and praiseworthy-spouse analyses remain alternatives.',locator='6722, compare 6721')
for l,w in [('Malvi','chat'),('Nimadi','chat|chāt'),('Bagheli','ceṭ|caṭ|cheṭ')]:
 a(l,'roof',w,'4971','CDIAL 4971 *chatti gives Punjabi chatt, Hindi chat/chāt and Bihari chāt ‘roof’. It explicitly traces standard Hindi chat through Punjabi, so local inheritance versus contact is unresolved; Bagheli retroflex stops/vowels retain their source spelling.')
a('Bagheli','roof','ceni|ceniha|caṇhi|chani','4992','CDIAL 4992 *channi gives Bihari chānh/chānhī/chānhiyā ‘thatch roof’, Old Awadhi chāna and Hindi chān ‘thatch, hut’. These support the nasal family; the e-vowel and extended endings need dialectal confirmation. Stop-bearing caṇḍi/chaṇḍhi forms are excluded for further analysis.')
for l,w in [('Malvi','ṭapəri|ṭāpari|ṭāpara|ṭapəra|ṭapri|ṭaperi|ṭapiri|ṭapiro'),('Nimadi','ṭāpəro')]:
 a(l,'roof',w,'5725-3','CDIAL 5725’s -r extension specifically compares Punjabi ṭapparā/ṭapparī ‘thatch, shed’, Kumauni ṭapariyo ‘hut’ and Hindi ṭāprā ‘thatch, thatched house’. The parent is the tarpar extension, not the bare tarpa basket word; house/roof metonymy and vowel variation remain explicit.',locator='5725, -r extension')
a('Malvi','roof','jopəḍi','5403-3','CDIAL 5403.3 *jhōppa gives Hindi jhopṛī ‘hut’; the elicited roof could be the roof-for-hut metonymy. Deaspiration and the retroflex stop versus flap need checking; the nasal *jhōmpa branch is an alternative if source nasalization is missing.')
for l,w in [('Malvi','barinā|bāno|bano'),('Nimadi','banno|bārṇu|bāiṇu|bāiṇo|banna')]:
 a(l,'door',w,'6663','CDIAL 6663’s addenda explicitly give Gujarati bārṇũ and Kachchi bāyṇo ‘door’, alongside Old Marwari bāra and Gujarati bārũ. These are closer comparanda than 11553 vāraṇa ‘obstruction/doorstep’. Contracted bāno/banno and Malvi barinā still require phonological review.')
a('Bagheli','firewood','kathi','3120','CDIAL 3120 gives Hindi kāṭhī and Kachchi kāṭhī ‘wood’, with the wider kāṣṭha > kaṭṭha > kāṭh family. The survey’s dental t is preserved as a qualification; the meaning is wood used as fuel, not the unrelated kāṣṭhin ‘wooden’ adjective.')
a('Malvi','body','aŋg','114','CDIAL 114 gives Prakrit aṁga ‘limb, body’ and Hindi/Gujarati/Marathi ā̃g ‘body’. Both body and limb senses are already explicit in the primary entry.','straightforward',locator='114.1')
a('Malvi','woman','nari','7078','CDIAL 7078 gives Prakrit ṇārī and Old Marwari nāri ‘woman/wife’, supporting this family. The conservative form alone cannot distinguish inheritance from learned or Hindi-mediated use.')
a('Malvi','woman','istri','13734','CDIAL 13734 records ancient istri beside strī and the ordinary Middle Indo-Aryan itthī/thī outcomes. The Malvi retained str cluster may reflect learned or regional circulation; the exact family is supported without claiming regular local inheritance.')
a('Malvi','husband','pati','7727','CDIAL 7727 gives pati ‘master, husband’, but ordinary Prakrit paï and regional pai show medial-t loss. Malvi pati therefore likely involves a learned or contact form; the immediate route is unresolved.')
a('Bagheli','man','meney|menay|menə|mənəy','10048','CDIAL 10048 explicitly derives Bhojpuri/Awadhi manaī ‘man’ and Hindi manaī ‘man, husband’ from *mānavika. This selects that formation within the mānava entry, not unextended manuṣya. Bagheli e-vowels and reduction need local confirmation.',locator='10048, *mānavika')
b.save()
