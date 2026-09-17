import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
E={
'10511':'CDIAL 10511 yuṣmad explicitly gives northern plural tha/twa and Nepali timi alongside tumhē/tum. Only the clearly plural northern survey forms are selected here; singular thū and short oblique forms require tuvam/tumham paradigm comparison.',
'9361-2':'CDIAL 9361 distinguishes bhajyate with bhaj-/bhāj- flee from section 2 bhagna with bhag-/bhāg-. The affricate-j survey run forms belong to the first branch, correcting the candidate’s bhagna parent.',
'9828':'CDIAL 9828 manuṣya explicitly gives Bshk mīš/mĩš young man/husband with plural mānuš, and Palula mēš/mīš. These particular short-i forms are supported by the full entry; this does not resolve the separate Khowar moš ambiguity.',
'9229':'CDIAL 9229 bāhu gives Hindi bāh/bāhā arm. It explicitly marks Kalasha baza/Khowar bazu as loans from Nuristani or Iranian, so bājū/bāzā are not assigned directly to bāhu in this pass.',
'6983':'CDIAL 6983 nava gives northern nāwu/nā̃o and regional no/nūā new, including feminine naī. These simple n-vowel forms fit that family; source diphthongs and nasalization are preserved.',
'11225':'CDIAL 11225 vaḍra gives baṛā/baḍḍā and explicit Awadhi baṛkā big. These b-initial survey variants fit those forms; Palula gāḍo needs a different root comparison. Turner’s revised analysis through ēvaḍa-type demonstrative formations is retained.',
'9691':'CDIAL 9691 ma explains the singular I forms through oblique, especially instrumental, forms and expressly lists Bengali mui and Hindi maĩ. The survey mui/mũi/mæi responses fit that pronoun family.',
'5228':'CDIAL 5228 jihvā expressly lists Gawri zip, Ku. jibṛo and Nepali jib(h)ro tongue. These support the zīp and jibra variants; yīv and jiũ require local consonant evidence.',
'9982':'CDIAL 9982 māṃsa lists Kalasha mos/mõs, Bshk mā̃s, Palula mʰās and regional māsu meat. The simple survey meat responses fit these explicitly compared phonetic forms.',
'8283':'CDIAL 8283 purāṇa lists Khowar paránu, Bshk pūrən, Nepali purānu and regional purān/purāṇī old. These contracted or inflected old-thing adjectives fit that family.',
'11616':'CDIAL 11616 viṃśati explicitly lists Gawri išī and Khowar bišir twenty alongside regional bīs. The survey suffix and initial-loss forms are documented continuations, not speculative edits.',
'4428':'CDIAL 4428 ghara includes śeu. kár house. The northwestern kar/kār forms fit ghara with the regional treatment of initial voiced aspiration, with possible intra-IA transfer kept open.',
'3208':'CDIAL 3208 kukkuṭa explicitly lists Bshk kukur and L./P. kukkuṛ/kukkaṛī cock/hen, with northern kukūĩ. These chicken responses fit the bird family, distinct from superficially similar dog words.',
'13551':'CDIAL 13551 sūcī gives sūī/suĩ/sūw needle with northern diphthong and glide variants. These svī̃ responses fit the non-nasal-cluster first branch, not the distinct *sūñcī branch.',
'55':'CDIAL 55 agni lists L./P. agg/āg fire and MIA aggi/aggini. The simple survey ag/agh forms fit that family; learned agnī is identified with the Sanskrit word without asserting its local transmission history.',
'6140-3':'CDIAL 6140.3 *dita explicitly gives Nepali diyo and Hindi diyā given. The corresponding survey dija/dīyo past forms fit this branch; Gojri dina imperative needs its own paradigm and possible *dinna comparison.',
'4386':'CDIAL 4386 *grilla gives gīlā/gillā wet, but the proposed deeper *gṛdla reconstruction is very doubtful. Only the straightforward gīo continuation is selected; līlo/bīlā require distinguishing ārdraka or local sound developments.',
'11392':'CDIAL 11392 section 2 gives Nepali barsa, Bengali baris and Hindi baras year. These varṣ/borsa/boris responses fit the year sense, with possible learned or intra-IA transmission left open.',
'6914':'CDIAL 6914 nakha explicitly gives nahu/naũh and Palula nōṅg nail under its first branch; retained velar clusters there must not be mechanically reassigned to *nakkha. Chil nōr needs a local rhotic account.',
'4287':'CDIAL 4287 godhūma gives Kalasha ghöm, Palula ghōm, Torwali ghomū and Hindi gehū̃ wheat. The survey xūm/ghom/gāmo/gehõ variants fit the documented wheat family.',
'5090-4':'CDIAL 5090.4 *jūḍa explicitly gives Mth jūr and regional juṛa cold. The u-vowel cold forms fit this branch; a-vowel jar requires jaḍa/jaḍḍa comparison.',
'11533':'CDIAL 11533 vāma gives Nepali bāũ, Ku. bāyõ and Hindi bāyā̃ left. The survey bay-/baw- and nasalized variants fit that left-hand family.',
'9465':'CDIAL 9465 bhārika gives bhārī heavy. The i-final form fits; a/o-final bhāro needs distinguishing an adjectival inflection from bhāra load.',
'11572':'CDIAL 11572 vāla explicitly gives Bshk bāl hair and regional bār/bāḷ. These bāl/bāṛ responses fit the hair family with source glottal and rhotic details retained.',
'6624-2':'CDIAL 6624 dravati explicitly gives the -ḍ extension dauṛ-/doṛ- run, including Marwari doṛṇo. These survey imperative and verbal endings attach to that extension; aspiration may reflect dhāvati influence and remains qualified.',
'3196-2':'CDIAL 3196 identifies *kādṛk as the remodelled what interrogative behind Gujarati kaya and Marathi kāy. The kay/kaye forms fit; koī/kɔī needs distinguishing the indefinite pronoun.',
'4976':'CDIAL 4976 chattvara gives chappar/chaprā roof and hut across the region. The survey pp and retroflex-r forms fit that roof family, with initial aspiration retained as transcribed.',
'11378':'CDIAL 11378 section 2 vardhanī broom expressly lists bārni/baṛhanī and Sambalpur banneibā sweep. The r-bearing forms fit this broom family; bare banni needs distinguishing alternative broom formations.',
'3219':'CDIAL 3219 *kuccura gives Maiya kučara, Chilis kučuro and Palula kučuro dog; Torwali kujū is explicit. These kučur-/kučū dog responses fit that family.',
'12278':'CDIAL 12278 śata gives Kho šor, Mai šal and L./H. sau hundred. These r/l or diphthong-final hundred responses fit the documented numeral family.'}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']+['near-fourth-decisions.json']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for key in ('accepted','held') for x in d[key])
acc=[];held=[];qs=[];ix={}
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in E:continue
 k=x['parents'][0];r=x['record'];w=r['Form'];target=k;reason=None
 if r['ID'] in done:continue
 if k=='10511' and r['Language_ID'] not in {'Bshk','Tor','Gaw'}:reason='You form needs its singular/plural or oblique paradigm to distinguish tuvam from yuṣmad.'
 elif k=='9361-2':target='9361'
 elif k=='9229' and w!='bahay':reason='Arm form requires a supported immediate Iranian/Nuristani or Hindi donor; direct bāhu ancestry would skip that loan.'
 elif k=='11225' and r['Language_ID']=='Phal':reason='Palula gāḍo big needs a distinct root comparison.'
 elif k=='5228' and r['Language_ID'] in {'Goj','KochariyaEastDanuwar'}:reason='Tongue form needs local initial-y or final-consonant-loss evidence.'
 elif k=='6140-3' and r['Language_ID']=='Goj':reason='Dina give imperative needs the local paradigm; do not assume the *dita participle.'
 elif k=='4386' and w!='gīo':reason='Wet form needs grilla versus ārdraka or independent initial-consonant analysis.'
 elif k=='6914' and r['Language_ID']=='Chil':reason='Nōr nail needs an account of the final rhotic.'
 elif k=='5090-4' and 'a' in w or k=='5090-4' and 'ʌ' in w:reason='A-vowel cold response needs jaḍa/jaḍḍa versus jūḍa branch comparison.'
 elif k=='9465' and not w.endswith('ī'):reason='A/o-ending heavy adjective needs distinction from bhāra load.'
 elif k=='3196-2' and w in {'koī','kɔī'}:reason='What response resembles the indefinite koī; source and interrogative paradigm need review.'
 elif k=='11378' and r['Language_ID']=='Buksa':reason='Contracted banni broom needs distinction from other broom formations.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Source uncertainty requires checking the lexical reading.'
 if (k,target) not in ix:ix[k,target]=len(qs);qs.append(dict(parent=target,citation='CDIAL['+target.replace('-','.',1)+']',evidence=E[k]))
 i=ix[k,target]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=35));continue
 c=x['comparanda'][k][0]['record'];ev=E[k]+' Reviewed comparator: '+c['Language_ID']+' '+c['Form']+' “'+c['Gloss']+'” ('+c['ID']+'). Uncertain intra-IA transmission remains open; exact source text is preserved.'
 acc.append(dict(record=r,family=i,parent=target,citation=qs[i]['citation'],evidence=ev))
(P/'near-fifth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'near-fifth-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'near_fifth_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','near-fifth'))
from collections import Counter
print('accepted',len(acc),'held',len(held));print(Counter(x['parent'] for x in acc))
