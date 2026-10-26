"""Source-local prose inventory; no canonical writes or inferred transcriptions."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=[]
def add(page,label,forms,gloss,route='literal',note='',disposition='emit',native=None):
 rows.append(dict(source_unit_key=f'kurux-nepal:p{page}:prose:{label}',printed_page=page,pdf_page=page+8,forms=forms,gloss=gloss,route=route,native=native or '',note=note,disposition=disposition,status='prose inventory; final original/output reconciliation pending'))
add(4,'headman',['maijan'],'leader or headman',note='Attributed by the report to Bista2004:156, not independently consulted. Conventional Roman spelling; no IPA inferred.')
add(4,'festival-karma',['Karma Dharma','Karam'],'festival',note='Source describes a Uranw religious festival. Conventional cultural name; target-language attribution and pronunciation are not independently specified.')
add(4,'festival-jatara',['Jatara'],'new year festival',note='Attributed to Uranw and Sapkota2013. Conventional cultural name; exact language attribution uncertain.')
add(4,'karam-tree',['Karam'],'tree used in festival and wedding traditions',note='Conventional culturally attributed tree name on pp4–5; target-language pronunciation and botanical identity are not specified. Distinct from the festival sense.')
add(5,'festival-sukarati',['Sukarati'],'festival in which the Uranw worship the goddess of wealth',note='Attributed to Uranw and Sapkota2013. Conventional cultural name; exact language attribution uncertain.')
add(5,'hindu-festivals',['Holi','Chaith','Cauthcand','Ghari','Dipawali'],'Hindu festival names',note='Explicitly described as Hindu festivals rather than quoted Kurux lexical forms. Documentary controls.',disposition='documentary-only')
add(4,'clothing',['sarees','kurta suruwal','lungis'],'clothing terms in English ethnographic prose',note='Not explicitly assigned as target-language attestations.',disposition='documentary-only')
add(15,'autonym-phonetic',['kuɽukʰa','kuɽkʰa'],'internal name for the language and ethnic group','phonetic',note='Explicit Nepal autoglottonym. Two phonetic alternatives and three native alternatives are not individually paired by the source.')
add(15,'autonym-roman',['Kurukha','Kudukha','Khurkha'],'internal name for the language and ethnic group',note='Explicitly called Romanizations. Preserve literal spellings; no phonemic reading from conventional Roman letters.')
add(15,'autonym-native',['खुडुखा','कुड़्ख़ा','खुर्खा'],'internal name for the language and ethnic group','native',note='Three printed native spellings; owner/root original review confirms reph in खुर्खा. Do not force pairings to the two phonetic alternatives.')
add(15,'kurux-phonetic',['kuɽuχ'],'Kurux language or ethnic-group name','phonetic',note='Explicit bracketed phonetic quotation; native spelling explicitly paired in same sentence.',native='कुड़ुख़')
add(15,'uranw-phonetic',['uɾãũ̯'],'Uranw surname and community name','phonetic',note='Source says used while speaking Nepali and accepted in Nepal; attribution is contextual, not a claim of inherited Kurux origin.',native='उराँव')
add(15,'kurux-roman',['Kurux','Kurukh','Kudukh','Kunrukh','Kurka','Kuruk'],'conventional spellings of the language name',note='Main text and footnote14; documentary naming variants, no asserted phonological variants.',disposition='documentary-only')
add(15,'uranw-roman',['Uranw','Uraw','Uraon','Urau','Oraon','Oran','Oraw','Orau','Orang','Urang'],'conventional spellings of the community name',note='Main text and footnote15; no target phonetic interpretation.',disposition='documentary-only')
add(15,'outsider-names',['Dhangar','धागढ','Jhangar','झागाड्','Jhangad/Dhagar','Dhangad','Dhagar','Jangad','Janghard','Jhangad','Jhanger'],'outsider names for the people or language',note='Main text and footnote16; p16 explicitly identifies outsider origin and community dispreference. Preserve documentary names, not target-language inference.',disposition='documentary-only')
add(16,'india-native-autonyms',['कुड़ुख़','कुँड़ुख़'],'internal name Kurukh in India','native',note='Two native spellings explicitly described as internal names; no independently paired IPA supplied here.')
add(16,'india-outsider-names',['Oraon','उराँव','Dhanka','Dhangad'],'outside community names and Scheduled Tribe names',note='Explicit outsider attribution; documentary evidence.',disposition='documentary-only')
add(81,'quoted-religion-kode',['koɖe'],'religion label',note='Quoted respondent religion label in biodata; no explicit language attribution. Preserve as uncertain documentary evidence without declaring it target-language vocabulary.',disposition='documentary-only')
add(83,'quoted-religion-kachua',['kachua pachaba'],'religion label',note='Quoted respondent religion label in biodata; no explicit language attribution. Preserve raw label without phonemic interpretation.',disposition='documentary-only')
add(81,'quoted-occupation',['moɖʒduri'],'occupation label',note='Bracketed GJ03 occupation13 quotation; root original review identifies possible fine affricate/dental glyph uncertainty requiring crop/font reconciliation. Language attribution not explicit.',disposition='documentary-only')
add(81,'religion-gharkhebala',['Gharkhebala'],'traditional religion label',note='Parenthetical BS08 religion field. Exact language attribution not specified; documentary quoted label, not generic administrative omission.',disposition='documentary-only')
add(82,'religion-nadapuja',['Nadapuja'],'religion label',note='GS03 religion field. Exact language attribution not specified; retain documentary evidence.',disposition='documentary-only')
add(6,'script-name',['Tolong Siki'],'name of script developed for Kurux',note='Name of an orthographic system, not a source assertion of target lexical form or pronunciation.',disposition='documentary-only')
add(39,'sound-comparison-symbols',['b̤','bʰ','s','tʃ','χ','kʰ','ʈ','ɖ','ɽ','ə','ʌ','ɔ','ɑ','ɾ','ɹ','r','ɽ͡r','ã','an','ɳɖ','nd'],'illustrative phonetic segments used in similarity scoring',note='Metalinguistic symbols, not lexical words with meanings. Source explicitly says no attempt is made to identify genuine cognates; no variant/etymology edges inferred from similarity categories.',disposition='documentary-only')
(HERE/'prose-inventory-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
print(len(rows),'prose evidence groups')
