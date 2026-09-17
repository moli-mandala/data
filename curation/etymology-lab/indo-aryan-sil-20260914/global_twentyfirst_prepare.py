import json
from pathlib import Path
P=Path(__file__).resolve().parent;done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec={
'959':'CDIAL 959 aṣṭhīvat explicitly gives Bengali hā̃ṭu/ā̃ṭu and Assamese ā̃ṭhu knee. The eastern survey hatu/haṭu knee forms belong here, not to *haṭṭ move. Dental t in the plain hatu spellings and absent nasalization are retained as transcription or phonetic uncertainties; the regional family is clear but those details are not a demonstrated sound law.',
'13943-3':'CDIAL 13943.3 *haṇṭ explicitly gives Bengali hā̃ṭā to walk. Nasalized hãṭa selects section 3, not aspirated *haṭṭh in section 2.',
'3144':'CDIAL 3144 kiṃcid explicitly gives Bengali kichu something/anything alongside the some/a little senses in neighboring languages. Survey kɪtʃhu some matches this indefinite pronoun.',
'9538':'CDIAL 9538 *bhull section 1 gives Bengali bhulā err/forget, with a nominal bhūl mistake in the addenda. Bengali bhul wrong is compared with that error family, retaining the survey adjectival gloss; section 2 *bhol lead astray is distinct.',
'10158':'CDIAL 10158 mukha means mouth/face; CDIAL 10174 explicitly prefers mukha with expressive doubling for several retained-kh forms rather than mukhya chief. Bagheli mukh face/mouth is linked to the mouth noun, leaving learned versus vernacular transmission open.',
'6089':'CDIAL 6089 *thaṭṭha explicitly gives Hindi ṭhāṭh frame of a roof and Old Awadhi ṭhāṭa frame on which thatch is laid. Bagheli ṭhaṭh roof is compared with that roof-frame noun, preserving the broader survey roof gloss.',
'137':'CDIAL 137 aṅguṣṭha includes Marathi ãgṭhī little finger/toe and Oriya āṅgṭhi finger/toe. Pauri aŋṭhi finger fits the shortened finger family, with medial g loss retained as a regional phonetic qualification.',
'5023':'CDIAL 5023 chādya section 1 explicitly gives Braj chajjo/chājo veranda/roof in the addenda, alongside Nepali chajjā roof/balcony. Mewari chajo roof matches this j branch, not section 2 chādiya.',
'12311':'CDIAL 12311 śamba gives Gujarati sā̃belũ/sāmelũ wooden pestle. Mewari hamela pestle is compared with that whole regional noun; initial h versus s and its ending are retained as regional comparative qualifications, with intra-Indo-Aryan transmission open.',
'5103':'CDIAL 5103 jani explicitly gives Gujarati, Punjabi and Lahnda jaṇī woman. Mewari jaṇi woman directly matches this noun.',
'9273-10':'CDIAL 9273.10 *bodda gives Punjabi boddā rotten, Hindi bodā weak and related defective senses. Mewari boda bad is compared with this specific dental-d family, retaining the broad evaluative gloss.',
'400':'CDIAL 400 anyatara explicitly gives Gujarati nerũ another/different/extraordinary/many. Mewari naro many is compared with that regional many sense, with the vowel and masculine ending retained and the precise transmission open.',
'3546':'CDIAL 3546 koṣṭha explicitly gives Lahnda/Awan koṭhā house and Punjabi koṭhā house. Pothohari koṭha house directly matches this regional house noun.',
'6906':'CDIAL 6906 na gives Lahnda and Punjabi na not, with nā̃ no in the addenda. Pothohari nã not is compared with this negative particle, retaining nasalization.',
'13355':'CDIAL 13355 sāra explicitly gives Gujarati sārũ good/whole. Goj saru good is compared with this adjectival good sense; intra-Indo-Aryan transmission remains open.',
'10555':'CDIAL 10555 *rakṣāpuṭaka explicitly gives Gujarati rākhɔṛɔ ashes/layers of ashes. Vasavi rakhaḍo ash fits this expanded ash noun; retroflex stop versus flap and the source vowel lengths are retained.',
'10188':'CDIAL 10188 *muṭṭa heart gives Nepali muṭu heart/courage. Dang moṭu heart is compared in this family with o versus u retained; the immediate Indo-Aryan transmission remains open.'}
hold={
'6909':'The nakka nose family is plausible for Bagheli nekua, but the -ua extension and front vowel require a whole-form comparison or morphological analysis.',
'5328':'CDIAL 5328 distinguishes fall from causative sweep/shake down in section 2. It does not document this jhaḍu broom noun; resolve the nominal extension rather than attaching it to the first verbal branch through an existing search link.',
'3851':'CDIAL 3851 gives khal mortar and kharal in section 2, not Bagheli kaḍi. The lateral/retroflex and suffix differences need independent evidence.',
'13673':'CDIAL 13673 stanya gives thān breast but also notes contamination from stana. Khandesi thana breast alone does not distinguish the milk-derived family from stana breast.',
'6419':'CDIAL 6419 *dumbha explicitly allows that all these NIA tail forms may be Iranian loans. Resolve the immediate donor for Dogri dumb; the preference about uncertain intra-Indo-Aryan borrowing does not settle an Iranian origin.',
'9502':'Mewari bhiñjā wet fits the broad wetting family, but its nasal consonant is not explained by the bhij comparanda under 9502. Investigate a nasal stem/participial formation.',
'9361':'Mewari baŋgi broken has a nasal absent from the bhagga/bhagā participle in 9361.2 and may instead belong to bhaṅga. Do not select the bhajyate heading solely from the existing search match.',
'664':'CDIAL 664 adhura is half-done/incomplete. Mewari adurā is glossed broken; verify whether incomplete or damaged is intended before identifying the semantic extension.',
'13382':'Nimadi honnā cloth does not directly match the sauṛ/soṛ reflexes in CDIAL 13382. The resemblance to the old Pali variant sāhunna alone is insufficient to establish the consonant developments.',
'11742':'Goj ij lightning lacks the initial consonant in neighboring bij/vij forms. CDIAL 11742 does give Konkani ij, but that distant attestation alone does not establish the Goj phonetic history.',
'9203':'CDIAL 9203 bāṇa documents arrow, not rainbow. Noiri ban rainbow needs a local semantic or compound analysis.',
'6459':'Dang ḍuar door has initial retroflex ḍ while CDIAL 6459 duvara comparanda have dental d. Verify the Dang source glyph independently; the Noira dental error cannot automatically correct another source.',
'5069':'CDIAL 5069 choka gives chori girl and Armenian Romani choki daughter. Dang čhai daughter lacks the consonants identifying that family and needs comparison with other child stems.',
'7685':'CDIAL 7685 pañjara supports ribs/side, but Dang pənǰrə beside is an adpositional use. Establish its local locative formation rather than silently treating a relational form as a bare noun.'}
rules=[dict(parent=p,citation=('CDIAL[10158];CDIAL[10174]' if p=='10158' else 'CDIAL['+p.replace('-','.')+']'),evidence=e) for p,e in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0]
 if r['ID'] in done:continue
 if ps==['13943','959']:p='959'
 elif len(ps)>1:continue
 if p=='13943' and r['Gloss']=='knee':p='959'
 if p=='13943-2':p='13943-3'
 if p=='10174':p='10158'
 if p in hold:held.append(dict(record=r,families=[],reason=hold[p],passNumber=71))
 elif p in ix:
  q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/('global-twentyfirst-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'global_twentyfirst_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','global-twentyfirst'))
print({'accepted':len(acc),'held':len(held)})
