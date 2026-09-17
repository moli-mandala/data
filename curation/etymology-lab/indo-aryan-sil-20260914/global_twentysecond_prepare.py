import json
from pathlib import Path
P=Path(__file__).resolve().parent;done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec={
'4015':'CDIAL 4015 *gandhapūrikā explicitly gives Khowar gamburi and Kalasha gambhuri flower. Survey gambūrī matches this whole compound family, retaining lack of aspiration in Kalasha. The blank Proto-Burushaski node returned by search supplies no competing lexical form.',
'6849':'CDIAL 6849 dhūma explicitly gives Kalasha Rumbur thum smoke. This matches the survey aspirated voiceless onset; 6852 *dhūmara instead has an r-extension absent here.',
'7756':'CDIAL 7756 *padara explicitly gives Lahnda pēr/pêr and Punjabi pair foot, and Oriya payara leg. Awan/Goj per leg is compared with this r-extension family, preserving the broader limb gloss rather than selecting unextended pāda.',
'12815':'CDIAL 12815 sa/so explicitly gives Maiyan soh and regional so he. Maiyan/Gowro so he selects the simple pronoun, not 13607 so api he also, which adds an emphatic particle unsupported by these forms.',
'4784-3':'CDIAL 4784.3 *cikkhalla explicitly gives Bengali cikal and Hindi/Gujarati/Marathi cikhal mud. Bhatri cikəl/cikʌl fits this retained-k branch; the ungeminated cikhalla branch instead gives Hindi cihlā. The lack of aspiration is retained.',
'2368-2':'CDIAL 2368.2 *ullaṭyate gives Western Pahari ulṭo left/reverse and Old Marwari ulaṭo reversed. These ulṭa/ulṭo left forms select the second branch; the survey retroflex lateral is retained.',
'6778':'CDIAL 6778 dhānya explicitly gives Western Pahari dhān rice plant and neighboring dhān rice. Kullui dhān/dān rice matches this grain noun; lack of aspiration in dān is retained as a local transcription or phonetic qualification.',
'7563':'CDIAL 7563 nīla section 1 gives regional nīlo/nīlā blue and retroflex-lateral niḷa forms elsewhere. Kullui niḷə blue belongs to the adjective, not section 2 indigo; vowel length/stress differences are preserved.',
'7035':'CDIAL 7035 nahi addenda give naĩ/nā̃y not, alongside the discussed nahi > nahī̃ family. Kullui nai/nəi/nə̃i no is compared with this extended negative particle; the precise development and Turner’s report of disagreement over the deeper adverbial account are retained.',
'4147-3':'CDIAL 4147.3 gāvī gives regional gāi/gāy cow, including neighboring feminine forms. Survey gaiya/gəiya is compared as an expanded feminine cow form, retaining its -ya material rather than selecting the masculine gāva heading; the exact regional extension remains qualified.',
'11363':'CDIAL 11363 vartis explicitly gives Lahnda vāṭ, plural vaṭī̃ path and Indo-Aryan Romani vātī path. Goj/Noiri vaṭi path fits this regional feminine path family; its exact intra-Indo-Aryan transmission remains open.'}
holds={
'cow':'CDIAL 4093.2 *gavu and 4147.2 *gāvā (also 4147.3 gāvī in Western Pahari addenda) all provide plausible gau/go cow histories. These isolated survey forms do not identify the stem or oblique paradigm needed to choose the branch.',
'nas':'CDIAL 7031 nasta includes Pashai nās, whereas 7089 nāsā has nās directly. Goj nās alone does not establish whether a historical t was present; seek a regional paradigm or independent comparative evidence.',
'noth':'CDIAL 7031 separates nastaka, nastī and nastu branches with overlapping natho/notu forms. Chiliss/Gowro nothū/not hī cannot be assigned to the bare nasta heading without checking the gender and extended-stem history.',
'11129':'CDIAL 11129 *locis gives loi light/lustre but does not document lay fire. Verify the vowel and fire-versus-light semantic development, including competing flame words.',
'negative':'Kullui ne/neː no can reflect a reduced extended negative or the simple na particle. CDIAL 6906 versus 7035 cannot be settled from these two forms alone.'}
rules=[dict(parent=p,citation='CDIAL['+p.replace('-','.')+']',evidence=e) for p,e in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0];why=None
 if r['ID'] in done:continue
 if ps==['4015','f_2u6bliztoa2ri']:p='4015'
 elif ps==['6849','6852']:p='6849'
 elif ps==['7756','8056']:p='7756'
 elif ps==['12815','13607']:p='12815'
 elif ps==['4784','4784-3']:p='4784-3'
 elif ps==['2368','2368-2']:p='2368-2'
 elif ps==['7031','7089']:why=holds['nas']
 elif ps in [['4093','4147'],['4093','4147','4147-2']]:why=holds['cow']
 elif ps==['6906','7035']:why=holds['negative']
 elif len(ps)>1:continue
 elif p=='4147':p='4147-3'
 elif p=='7031':why=holds['noth'].replace('not hī','nothī')
 elif p=='11129':why=holds['11129']
 if why:held.append(dict(record=r,families=[],reason=why,passNumber=72))
 elif p in ix:
  q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/('global-twentysecond-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'global_twentysecond_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','global-twentysecond'))
print({'accepted':len(acc),'held':len(held)})
