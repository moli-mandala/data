import json,re
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
E={
'7978':'Pavana explicitly includes Punjabi pavaṇ/pauṇ and Garhwali pɔṇ wind. These full and contracted survey forms match the documented wind family.',
'9530-7':'Section 7 *bhuṇḍa gives Gujarati bhũḍũ bad. Survey būṇḍo has lost aspiration; a local sound comparison is needed before selecting this expressive branch.',
'11398':'Varṣārātri explicitly includes Hindi/Nepali/Bengali barsāt and Gujarati barsād rainy season, with varṣartu as an alternative or influence. The source rain sense is retained as a broader use of the rainy-season family, with that deeper alternative open.',
'958':'Aṣṭhi explicitly includes Kalasha aṭhi bone, but the article identifies these Dardic forms as loans from Indo-Aryan. Establish a supported immediate donor rather than saving an inherited Sanskrit link.',
'4131':'*Gāndha explicitly gives Khowar gán wind, alongside scent and air. These gan/gān wind responses match the exact lexical comparison; the semantic connection to scent is documented.',
'9723-12':'Section 12 *māḍa explicitly gives Lahnda/Punjabi māṛā bad and WPah. maṛɔ weak/bad. These ā/a/ə-vowel flapped forms select that branch, with short-vowel *maḍḍa retained as a neighboring alternative in the article.',
'10104':'Māsa gives moon/month, including reduced Pašai māi/mä and Sindhi mā, but the exact Bashkarik mo/mõ reduction needs its own local comparison before distinguishing it from regional māh-family contact.',
'12559':'Śuṣyati explicitly lists Bashkarik šišāl and Chiliss šišēlo dry. These survey śīśāl/śīśelo forms match the cited extended adjectives directly, rather than the separate śuṣka participle.',
'11072':'*Lukka section 1 explicitly gives Bashkarik lúkuṭ small and Kashmiri lokoṭu. Survey lūkūṭ selects that branch; the article rejects a confident direct derivation from lupta.',
'2046':'Udbhūta explicitly gives Bashkarik ubuṭ light in weight, with raised to light as its stated semantic development. The survey ūbūṭ matches that direct comparison.',
'2592':'Kakṣyā explicitly gives Chiliss kaċ and Maiya kas near, tentatively through a locative kakṣye. Survey kaʦ near fits the affricate series; the proposed locative and ultimate semantic derivation remain qualified.',
'1821':'The utpātaka article explicitly gives Nepali upiũ, plural upiyā̃ flea, while questioning a connection to utpiba. Survey upiya/upiyã flea follows this exact regional comparison with the alternative deeper analysis retained.',
'6109':'*Thōba section 1 explicitly gives Kumaoni thol snout/lips, possibly through *thōvala. Dotyali thol lips matches that complete form, with the l-extension qualification preserved.',
'6438':'Durbala explicitly gives Nepali dublo and Kumaoni dubalo thin. Dotyali dublo/dubəlo matches that adjective and its documented contraction.',
'9804':'Madhya explicitly gives Nepali mā̃ and Kumaoni mu in, with irregular loss of the stop acknowledged in the article. Dotyali mā/Dangaura ma at fits this locative postposition family, with source nasal marking preserved.',
'9926':'Masta/mastaka explicitly gives Gujarati māthũ head and Kumaoni/Nepali māthi above. Both the Bhil head noun and Dotyali above forms belong to this documented first branch.',
'9123':'Baḍiśa explicitly gives spear in its WPah./Jaunsari addendum, but those barcho/baṛchā forms do not by themselves establish eastern borśa. Check the eastern spear lexeme and sibilant development.',
'9190':'Bahutva explicitly gives Bengali/Assamese bahut, Oriya bout and regional bahota many. These bohut responses fit that abundance family; influence from prabhūta is retained as discussed by Turner.',
'2330':'*Uppara gives Assamese opar upper/top and Bengali/Oriya upar/upara, but overlap with *uppari and the locative -ot in Hajong uporot requires exact base and morphology review.',
'6261':'*Dādda explicitly gives Bengali dadi grandmother and regional dada elder brother. The retroflex ḍaḍu elder-brother forms need local comparison before transferring the western grandfather forms to this sense.'}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 k='/'.join(x['parents']);r=x['record']
 if k not in E or r['ID'] in done:continue
 ev='CDIAL '+k+' '+E[k];reason=None
 if k in {'9530-7','958','10104','9123','2330'}:reason=ev
 if k=='6261' and r['Form']!='dadi':reason=ev
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Verify the source lexical reading before assigning this form.'
 if k not in ix:ix[k]=len(qs);qs.append(dict(parent=k,citation='CDIAL['+k.replace('-','.',1)+']',evidence=ev))
 i=ix[k]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=59))
 else:acc.append(dict(record=r,parent=k,family=i,citation=qs[i]['citation'],evidence=ev+' Exact form '+r['Form']+' is preserved; possible intra-Indo-Aryan transmission remains open.'))
for fn,obj in [('global-fourteenth-decisions.json',dict(accepted=acc,held=held)),('global-fourteenth-rules.json',qs)]: (P/fn).write_text(json.dumps(obj,ensure_ascii=False,indent=1))
(P/'global_fourteenth_save.py').write_text((P/'week_mother_save.py').read_text().replace('week-mother','global-fourteenth'));(P/'global_fourteenth_decide.py').write_text(Path(__file__).read_text());print(len(acc),len(held))
