import json,re
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
E={
'644':('644','ardha explicitly gives WPah. addo and Kashmiri oḍu half; Kullui adə/odʰə fit this family with local voicing/aspiration variants. A broken gloss needs separate semantic evidence.'),
'9525':('9525','bhuja arm is plausible, but the retained j and long/nasal vowel need the learned bhujā form and immediate donor verified.'),
'11465':('11465','vahya has bojh/bojha load, not an explicit heavy adjective in this article; verify the surveyed adjective or elicitation sense.'),
'114':('114','aṅga explicitly includes Bengali āṅ body and Hindi/Gujarati ãg body. These aŋ/ãg body responses match the family.'),
'4992':('4992','*channi explicitly includes chann/chānh/chān thatched roof. These deaspirated can/cani variants require local sound evidence before linking.'),
'5427':('5427-2','Section 2 ṭaṅga explicitly includes Bihari ṭā̃gā and Hindi ṭā̃gī axe/hatchet. The retroflex g-bearing central forms fit that branch; unmarked dentals and northwestern tʰoŋ need a branch/transcription check.'),
'5427-2':('5427-2','Section 2 ṭaṅga explicitly includes Bihari ṭā̃gā/ṭā̃gī and Hindi ṭā̃gī axe/hatchet. These retroflex ṭaŋi forms match that axe branch.'),
'10187':('10187-11','The article distinguishes *muṭṭa in section 1 with Dogri muṭā fat, *mōṭṭa in section 11 with Bengali moṭā fat, and aspirated branches. Survey vowel and aspiration select the branch only where supported.'),
'10187-11':('10187-11','Section 11 *mōṭṭa explicitly gives Hindi/Bengali moṭā and Nepali moṭo fat. Dangaura moṭ fat matches that branch.'),
'10187-2':('10187-2','Section 2 *muṭṭha explicitly gives Kalasha/Palula muṭh tree. The aspirated Kalasha form fits; Palula moṭ and nasalized Chiliss mū̃ṭh require their extra sound changes checked.'),
'6849-2':('6849-2','Section 2 dhūmikā/*dhūmiyā explicitly gives Bashkarik dīmī and Torwali dhimī smoke. These dīmī/dʰīmī responses select that extension, not simple dhūma.'),
'13682':('13682','stambha explicitly gives Torwali thām tree. Bashkarik is printed as quoted ṭam, requiring a dental/retroflex reading check against survey tam.'),
'2963':('2963','kavāṭa/kapāṭa explicitly includes Bihari kewāṛī and Hindi kiwāṛ door. The retained kapāṭ forms may be learned, and nasal kevãri or contracted kuaḍ need separate treatment.'),
'5021':('5021','*chādikūṭa explicitly gives Bihari/Maithili chāur ashes. The survey cʰaur/cʰāūr forms match this whole ash-heap compound.'),
'9696':('9696','makṣā/makṣikā explicitly includes Kumaoni/Nepali mākho fly and regional mākhā. Kochila masi requires a local sibilant-development check.'),
'10990-2':('10990-2','Section 2 *raśuna explicitly includes Kashmiri ruhun and Bihari rasūn garlic. The survey variants fit this r-initial family, with regional s/h and contact history left open.'),
'7025':('7025-2','Section 2 naviya explicitly includes Hindi nayā and Bengali nayā, the latter attributed to Hindi. Noya fits this y-bearing family; the exact regional route remains open. Contracted nyo remains less diagnostic.'),
'2333':('2333','*uppari explicitly includes Oriya upari and Punjabi uppar above. Survey uper/upre match the emphatically strengthened family; bare var needs its contraction independently supported.'),
'4883':('4883','cūḍa section 1 explicitly gives Bengali cul hair of the head. The Bengali and Bishnupriya cul responses fit that precise hair family.'),
'12045':('12045','The vīṭā article explicitly distinguishes Gujarati vīṭī ring under *vīṭṭa and vĩṭī ring under *viṇṭa. These Bhil viṭi/vĩṭi forms follow those Gujarati comparisons; both unnumbered subfamilies remain under the existing article node, with nasal variation preserved.'),
'2368-2':('2368-2','*ullaṭyate section 2 explicitly includes WPah. ulṭɔ left/reverse in the addendum. Nimadi/Vasavi ulṭo left thus has a directly documented directional sense.'),
'3083':('3083','kāla explicitly includes Gujarati kāḷũ, Oriya kaḷā and WPah. kāwo black. The survey variants match this colour family; Pauri kavo retains a regional liquid/glide qualification.'),
'9238':('9238-2','Section 2 *bēṭṭa explicitly gives Hindi beṭā son and beṭī daughter, also WPah. beṭṭɔ in the addendum. These kaithal terms select section 2, not *biḍḍa defective.'),
'9289':('9289','*bura includes burā/buro bad, but survey bara and Kullui bʊɽo need the vowel or retroflex development checked rather than copied from old links.'),
'7150':('7150','nikta, replaced by *nikka in Middle Indo-Aryan, explicitly gives Punjabi nikkā small and Maithili nikāh good. These small/good responses fit that documented semantic range.'),
'9092':('9092','phulla explicitly gives regional phul/phol flower. Bare phul cauliflower needs evidence for ellipsis of the vegetable compound before linking the whole survey response.'),
'5988':('5988','The traṭ article explicitly gives Hindi taṛkā dawn and Old Marwari taṛako morning, while its addendum prefers root taṭ for non-Sindhi forms. Morning responses fit this qualified expressive family; tomorrow needs its own temporal sense verified.'),
'6423':('6423','dur/duraḥ includes Kalasha dūr house and regional door forms but notes the competing *duvāra analysis. These door responses need the precise family and local semantic use resolved.'),
'549':('549','abhra explicitly gives Palula ābru cloud and Chiliss ažo rain, alongside Kandia ā̃ẓu cloud/rain. Both elicited weather senses are supported.'),
'5020':('5020','chādi explicitly gives Bashkarik čī and Palula čhī ashes. Bashkarik cī is direct; Palula unaspirated cī needs a local aspiration/transcription check.'),
'994':('994','ahi explicitly gives Khowar aī snake, analyzed through *ahika. These āī snake forms follow that explicit comparison; the k-extension is retained in evidence under the existing ahi family node.'),
'2918':('2918','kalayati is count/consider, with no interrogative when sense. Kare when must be researched independently of this misleading old assignment.'),
'603':('603','aratni explicitly gives Lahnda arak and awāṇ āruk elbow, with kk from tn. The addendum documents ṛ before k in regional elbow forms, supporting ārak/āṛak without inventing a separate root.'),
'1765':('1765','The uttama addendum explicitly derives Punjabi utte on from locative uttame. Awan/Pothwari ute above follow this documented inflected source.'),
'10896':('10896-5','laghu with kk extension explicitly gives loko/lōk light and metathesized halkā/halukā light. These k-bearing forms select the stored *laghukk branch; Chiliss lekū small needs its vowel checked.'),
'13561':('13561','sūtra explicitly gives Bihari/Bengali sutā thread. Sutːa fits; Hajong huta needs local s/h evidence before linking.'),
'13082':('13082','saṅga explicitly includes Oriya sāṅga company/companion and regional saṅg companionship. These friend responses fit that semantic family, with possible influence from saṃgha retained.'),
'10034':('10034','*mādhuka has a WPah. māū bee comparison, but that does not distinguish Hajong mau from alternatives based on madhu. An eastern primary comparison is needed.'),
'7736':('7736','pattrala meaning 2 explicitly gives Bengali pātlā and Hindi patlā thin. Hajong patla matches that thin/leaf-like branch; this meaning remains within the existing unsplit entry.'),
'5808':('5808','The tigma addendum tentatively compares Assamese ṭeṅā, phonetically tεṅa, sour/acrid. Hajong teŋa is provisionally linked through this precise comparison; Turner’s question mark and uncertain deeper derivation are retained.'),
'342':('342','anuhāra explicitly gives Nepali anuhār/anwār appearance/face. These anvar/ənuhər responses match the documented contraction and face sense.'),
'11533':('11533','vāma explicitly gives WPah. bā̃o/bauā̃ left. Kullui baʊə/bãə fit these forms; Magahi bammā needs its geminate/retained nasal analyzed.'),
'4655':('4655-2','Section 2 catvāri explicitly includes Bengali cāir and eastern cāri four. Magahi cāīr selects this branch rather than nominative masculine catvāraḥ.'),
'4655-2':('4655-2','Section 2 catvāri gives cāri/cār four; Noiri śar needs a local affricate/sibilant correspondence check.'),
'12135':('12135','vesavāra explicitly gives Nepali besār turmeric. Majhi/Jaunsari besar turmeric fit this condiment family, with regional transmission possible.'),
'2967':('2967','kaścid gives someone/anyone, not interrogative who in the article. These koi/koy who responses need local interrogative evidence or a source elicitation check.'),
'3023':('3023','kāṇḍa gives eastern kā̃ṛ arrow. Bishnupriya kar lacks the comparative nasal and needs a local correspondence or transcription check.'),
'4009':('4009-2','gati with l/ll extension explicitly includes Hindi gail/gailā and Old Marwari gailo/galo road. Survey gel/gelo selects the stored extension node rather than unextended gati.'),
'4802':('4802','*citth gives cīthaṛ/cithrũ ragged cloth. The survey cintrā/citra/cittro/sitrā cloth forms need sound and general-cloth versus rag senses verified.')}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
acc=[];held=[];rules=[];ix={}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 k='/'.join(x['parents']);r=x['record'];w=r['Form']
 if k not in E or r['ID'] in done:continue
 target,ev=E[k];reason=None
 if k in {'9525','11465','4992','9289','6423','2918','10034','2967','3023','4802','4655-2'}:reason=ev
 if k=='644' and r['Gloss']!='half':reason='The half word under broken needs explicit semantic/source evidence.'
 if k=='5427' and not w.startswith('ṭ'):reason=ev
 if k=='10187':
  if w=='muʈa':target='10187'
  elif w=='moṭhā':reason='Khandesi moṭhā big requires its aspiration distinguished from section 11 *mōṭṭa and section 12 *mōṭṭha.'
 if k=='10187-2' and r['Language_ID']!='Kal':reason=ev
 if k=='13682' and r['Language_ID']=='Bshk':reason=ev
 if k=='2963' and w!='kevari':reason=ev
 if k=='9696' and w=='masi':reason=ev
 if k=='7025' and w=='nyo':reason=ev
 if k=='2333' and w=='var':reason=ev
 if k=='9092' and r['Gloss']=='cauliflower':reason=ev
 if k=='5988' and r['Gloss']=='tomorrow':reason=ev
 if k=='5020' and r['Language_ID']=='Phal':reason=ev
 if k=='10896' and r['Language_ID']=='Chil':reason=ev
 if k=='13561' and w.startswith('h'):reason=ev
 if k=='11533' and w=='bammā':reason=ev
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires confirming the lexical reading.'
 if (k,target) not in ix:ix[k,target]=len(rules);rules.append(dict(parent=target,citation='CDIAL['+k.split('-')[0]+']',evidence=ev))
 i=ix[k,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=52))
 else:acc.append(dict(record=r,family=i,parent=target,citation=rules[i]['citation'],evidence='CDIAL '+k.split('-')[0]+' '+ev+' Exact survey form '+w+' is preserved; intra-Indo-Aryan transmission remains open.'))
(P/'global-twelfth-rules.json').write_text(json.dumps(rules,ensure_ascii=False,indent=1));(P/'global-twelfth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'global_twelfth_save.py').write_text((P/'week_mother_save.py').read_text().replace('week-mother','global-twelfth'))
print(len(acc),len(held))
