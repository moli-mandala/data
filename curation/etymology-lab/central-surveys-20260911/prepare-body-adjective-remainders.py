from research_helpers import Batch
b=Batch(13)
def a(l,g,w,p,e,t='qualified',s=None):
 rows=[r for r in b.inv[l] if r['ID'] not in b.used and r['Form'] in w.split('|') and (r['Gloss']==g if s is None else set(r['Gloss'].split('; '))<=set(s))]
 if rows:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rows)),p,e,tier=t,source_glosses=list(dict.fromkeys(r['Gloss'] for r in rows)))
for l,w in [('Malvi','cokka|cokkā|tsokka|sokkā'),('Nimadi','coka|cokha|cokka|cokkā|cokhā|cokā')]:
 a(l,'rice',w,'4918','CDIAL 4918 gives Prakrit cokkha ‘pure, clean’, Sindhi cokho ‘grain of cleaned rice’ and Gujarati cokhā, masculine plural, ‘rice’. The semantic specialization is explicit; deaspiration and Malvi s/ts variants require local phonological review.')
for l,w in [('Malvi','loi|lui'),('Bagheli','ləhu|lohu')]:
 a(l,'blood',w,'11165','CDIAL 11165 gives Prakrit lohia, Hindi lohū/lahū and Gujarati lohī ‘blood’; contracted loi is attested in Kumauni/Garhwali and lui in other Indo-Aryan languages. '+('These distant contracted forms support a family comparison, not a demonstrated Malvi sound law.' if l=='Malvi' else 'The two vowel patterns have direct eastern/Hindi comparanda.'),'qualified' if l=='Malvi' else 'straightforward')
for l,w in [('Malvi','kaljo|kalja|kaḷjo|kāljā'),('Nimadi','kalijo|kaləjo'),('Bagheli','keleja|kereja')]:
 a(l,'heart',w,'3103','CDIAL 3103 explicitly discusses heart/liver semantic overlap, with Prakrit kāleya/kālijja, Old Awadhi kareju ‘heart’, Gujarati kāḷjũ and Hindi kalejā/karejā. These support the elicited heart meaning without changing it to liver; local inheritance versus regional transfer remains open.')
a('Bagheli','body','ḍeh|ḍehi|ḍēh','6557','CDIAL 6557 gives Prakrit deha, Awadhi dẽh and Hindi deh/dehī ‘body’. The survey’s retroflex initial is retained as a phonological question, not normalized to Hindi d.')
for l,w in [('Malvi','sarir|śarir|śərir|śarili'),('Nimadi','sarir|śərir|sari'),('Bagheli','serir|sərir')]:
 a(l,'body',w,'12335','CDIAL 12335 gives Pali/Prakrit sarīra and western Pahari sarīr, alongside contracted Old Gujarati saïra. This identifies the śarīra family, but near-Sanskrit forms may be learned or Hindi-mediated; loss or change of the final liquid in the selected extended group remains for review.')
a('Nimadi','heart','hiyo','14152','CDIAL 14152 gives Prakrit hiaa, Old Marwari hiyaü and Marwari hīyo ‘heart’. Loss of medial d/y and contraction have direct western Indo-Aryan comparanda.','straightforward')
for l,w in [('Malvi','raday|hirəda'),('Nimadi','radai'),('Bagheli','hirdey|hrədey')]:
 a(l,'heart',w,'14152','The hr̥daya family is secure semantically, but CDIAL 14152’s ordinary Middle Indo-Aryan hiaa and regional hiyā forms do not directly explain retained d/r here. Learned reintroduction or Hindi mediation, plus initial-h loss in raday/radai, requires review; the proposed historical family does not settle transmission.')
a('Bagheli','white','ujːer|ujer|ujər','1670','CDIAL 1670 compares Maithili ujjar/ujarakā and Awadhi ujar ‘white’, with Prakrit ujjala. The r outcome is directly represented; Bagheli e-vocalism remains a local correspondence to check.')
a('Bagheli','white','ujiar','1672','CDIAL 1672 *ujjvāra specifically gives Gujarati ujiyārũ ‘bright’ with influence from andhakāra. The y-bearing Bagheli form may belong here, but 1670 ujjvala with a regional extension is an alternative; neither Gujarati borrowing nor a unique reconstruction is established.')
for l,w in [('Malvi','doḷo|doḍa|doḷā|dola'),('Nimadi','dauḷa')]:
 a(l,'white',w,'6767','CDIAL 6767 gives Old Gujarati dhaülaü, Gujarati dhɔḷũ and Hindi dhaulā/dhorā ‘white’. The vowel/liquid family fits, but initial deaspiration and Malvi doḍa’s stop require local verification. The merged doro ‘thread; white’ record is excluded.')
a('Malvi','small; short','nana|nānā','12732','CDIAL 12732 connects Prakrit laṇha with Punjabi nannhā, Gujarati nānũ and Braj nānhau ‘small’. The short/small senses fit the tender/small family.','straightforward',s=['small','short'])
a('Malvi','young male; short; small','nano|nāno','12732','CDIAL 12732 includes Maithili nanuā ‘young, child’ and Nepali nāni ‘little girl’ beside Gujarati nānũ ‘small’. The merged boy/son/younger-brother and short/small senses plausibly reflect substantivization, but the combined record needs semantic review before acceptance.',s=['younger brother','boy','son','short','small'])
a('Nimadi','small','nāno|nāṇo','12732','CDIAL 12732 compares Gujarati nānũ and Braj nānhau ‘small’, through Prakrit laṇha and n-initial remodeling. The dental/retroflex nasal distinction remains as transcribed.','straightforward')
for l,w in [('Malvi','choṭo|coṭā|coṭo|choṭā|soṭo'),('Nimadi','choṭo|coṭo'),('Bagheli','choṭ|choṭa|coṭ|coṭha|coṭka|choṭka|coṭhka')]:
 a(l,'short; small',w,'5071','CDIAL 5071 compares Kashmiri ‘short, small’, Hindi choṭā, Awadhi choṭ and Gujarati choṭũ. Its addenda distinguish *chōṭa, *chōṭṭa and *cōṭa; deaspirated/sibilant variants and -ka extensions are therefore qualified, while merged short/small meanings are compatible.',s=['short','small'])
b.save()
