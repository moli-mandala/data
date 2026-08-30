#!/usr/bin/env python3
"""Install the complete analytical glossary in Buddruss's 1996 Shina riddles.

The copyrighted Stanford ILL scan is not redistributed. ``RAW`` is the checked-in,
manually collated editorial layer for every printed glossary headword on pp. 40--50.
The scan's embedded text was used only for navigation: all forms were checked against
300 dpi renders. Running riddle text, inflected examples, bibliography, and comparison-
only forms in other languages are excluded from the form table.

Buddruss writes long vowels as double vowels, falling tone as ``áa`` and rising tone
as ``aá``. Combining nasal marks are represented with Jambu's source-profile ``~``
notation (for example printed ``ã́ãi`` is ``áa~i``); no phonological normalization is
otherwise imposed. Run from ``data/``; ``--pdf`` optionally verifies the ILL scan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


SOURCE_ID = "buddruss-shina1996"
SNAPSHOT_DATE = "2026-08-28"
COLLATION_DATE = "2026-08-28"
PDF_SHA256 = "2247db7ef88ec280b1e91a0461001ca41b8399d153d2b4e93e2aee88859c9169"
PDF_PAGES = 31
ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
FORM_OUTPUT = ROOT / "data/other/forms/20260828-buddruss-shina-raetsel.csv"
AUDIT_OUTPUT = RAW_DIR / "20260828-buddruss-shina-raetsel-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260828-buddruss-shina-raetsel-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260828-buddruss-shina-raetsel-manifest.json"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Collation_Date", "Unit_ID", "PDF_Page", "Printed_Page",
    "Raw_Form", "Raw_Gloss_German", "English_Gloss", "Final_Status", "Final_Form",
    "Final_Parameter_ID", "Emitted_Key", "Resolution", "Review", "Material_Error",
    "Source", "Record_SHA256",
]

TAG_ALIASES = {
    "adjective": "adj", "adverb": "adv", "pronoun": "pron", "numeral": "num",
    "postposition": "postp", "conjunction": "conj", "particle": "part",
    "proper": "proper-noun", "interrogative": "interr", "causative": "caus",
}

# printed page | head-entry number | form | source gloss | English gloss | tags
# | direct unambiguous Turner/CDIAL id | editorial resolution / retained source note
RAW = r"""
40|01|ač-|eintreten, hineingehen|enter; go in|verb intr|227|direct headword; causative and inflected examples remain prose
40|02|achíi|Auge|eye|noun f|43|
40|03|achóo|Walnuß|walnut|noun m|48|
40|04|agúl|Hülle; Schwertscheide|cover; sheath|noun m||canonical member of the printed agúl = hagúl cross-reference pair
40|05|agurií|ungesponnene Wolle|unspun wool|noun f||source compares Sanskrit anigaruha-
40|06|ai|Ziege|goat|noun f||Turner 145 and 887 are alternatives, so neither is linked
40|07|áa~i|Mund|mouth|noun f|1533|printed nasal vowel normalized to source-profile ~ notation
40|08|ají|oben|above|adv spatial|274|
40|09|aáji|Mutter|mother|noun f|1351|
40|10|ajiíno|oben befindlich|situated above|adj spatial||source derives this from ají
40|11|akóo|Obliquus zu akí selbst|oblique form of self|pron obl||etymology unclear; explicitly printed as glossary headword
40|12|akhár|Schmied|smith|noun m loanword||Persian derivation is doubted by the source
40|13|alák b-|sich trennen|separate; part|verb intr loanword||loan from Hindi-Urdu alag
40|14|ariíno|inner|inside; inner|adj spatial||source derives this from arú
40|15|arú|innen|inside|adv spatial||dialect comparison and Turner 357 are tentative
40|16|ášpo|Pferd|horse|noun m|920|
40|17|ay-|hineinstecken (Eisen ins Feuer)|put into (iron into fire)|verb tr archaic||source marks the word obsolete and its exact meaning uncertain
40|18|azaló|Eingeweide|entrails|noun m|1182|
40|19|b-|werden|become|verb intr||Turner 9416 and 9552 are both printed, so no single link is imposed
40|20|baábo|Vater|father|noun m|9209|
40|21|báagan|Scheitel|parting of the hair|noun m||source etymology is explicitly tentative
40|22|báai-báari|zwölfschichtig; Name einer gelben Blume|twelve-layered; name of a yellow flower|noun f||Turner 6658 belongs to the numeral component, not the compound as a whole
41|01|báakur|Ziegenstall|goat pen|noun m||Burushaski comparison and Turner analysis are tentative
41|02|bal|das Hängen|hanging|noun f||Burushaski comparison
41|03|báali|Seil|rope|noun f|11572|
41|04|balóos|Steintopf|stone pot|noun m||Burushaski comparison
41|05|bambá|Wasserhahn|water tap|noun m loanword||ultimately Portuguese through Hindi-Urdu
41|06|báari|Falte, Schicht|fold; layer|noun f||Punjabi comparison and Turner 11547 are questioned
41|07|ban-|(Kleider) anziehen|put on (clothes)|verb tr|9139|
41|08|baš-|tönen, reden|sound; speak|verb intr|11589|
41|09|bat|Reisgericht|rice dish|noun m|9331|homonym 1 on this page
41|10|báato|offen|open|adj|12196|
41|11|bat|Stein|stone|noun m|11348|homonym 2 on this page
41|12|bay-|sitzen|sit|verb intr|2245|source explicitly rejects Turner 11435
41|13|bií|zwanzig|twenty|numeral|11616|
41|14|biléen|Arznei; Schießpulver|medicine; gunpowder|noun m|11892|
41|15|biš|Gift|poison|noun m|11968|
41|16|boč|Umarmung|embrace|noun m||source discusses a Shina comparison but gives no direct etymon
41|17|bói|Hauptbalken der Decke|main roof beam|noun m||Turner 11182 and 11403 are alternatives
41|18|buj-|gehen|go|verb intr||Turner 12225 and questioned 11208 are alternatives
41|19|búto|ganz|whole; all|adj|9568|
41|20|čak th-|hacken|hoe; chop|verb tr||Burushaski comparison
41|21|čakaán d-|die Beine spreizen; große Schritte machen|spread the legs; take large steps|verb intr||
41|22|čamraáto|biegsam, elastisch|flexible; elastic|adj||Burushaski comparison
41|23|čar-|weiden|graze|verb intr|4686|
41|24|čáar|vier|four|numeral|4655|
41|25|čičóoro|Span|wood shaving|noun||first printed alternate
41|25|čučóoro|Span|wood shaving|noun alternate||second printed alternate; Burushaski comparison
41|26|čilím|Wasserpfeife|water pipe; hookah|noun m loanword||Hindi-Persian loan
41|27|čímar|Eisen|iron|noun m|14496|
41|28|čhal|Zicklein|kid; young goat|noun m|4963|
41|29|čháar|Bergklippe|mountain cliff|noun m||Burushaski and Khowar comparisons
42|01|čhijóoṭ|Schatten|shadow; shade|noun f|5027|
42|02|čhii~ṣ|Berg|mountain|noun f||printed nasal vowel normalized to source-profile ~ notation; Burushaski comparison
42|03|c̣akáai|Waage|scales; balance|noun f|5714|Burushaski comparison also printed
42|04|c̣aloó|Lampe, Fackel|lamp; torch|noun m||Turner 8766 and 8711 are alternatives
42|05|c̣iín|Hirse|millet|noun f|14708|source explicitly rejects Turner 4842
42|06|c̣uk|Stickerei|embroidery|noun f||Burushaski comparison
42|07|c̣hawáaṭi|Kürbiskalebasse als Melkgefäß|gourd used as a milking vessel|noun||derivational comparison only
42|08|c̣heéc|Feld|field|noun m|3735|
42|09|c̣hile|Kleider|clothes|noun pl||source explicitly rejects apparent Sanskrit/Tamil resemblance
42|10|d-|geben; schlagen, werfen|give; strike; throw|verb tr||Turner 6141/45 is not a single resolvable CDIAL id
42|11|dadií|Großmutter|grandmother|noun f|6261|
42|12|dáado|Großvater|grandfather|noun m|6261|
42|13|dai|zehn|ten|numeral|6227|
42|14|dáai|Bart|beard|noun f|6250|
42|15|dar|Tür|door|noun m|6651|
42|16|darbáti|Schwelle|threshold|noun||source analysis is disputed
42|17|dar dar b-|zittern, klopfen (Herz)|tremble; beat (heart)|verb intr||
42|18|darú|draußen|outside|adv spatial|6651|
42|19|dauloók|Unterwelt; hell|underworld; bright|noun m||Burushaski comparison and Turner 6540 are indirect
42|20|dií|Tochter|daughter|noun f|6481|
42|21|dii~ẓ|Erdgrube für Getreidevorräte|underground grain-storage pit|noun m||printed nasal vowel normalized to source-profile ~ notation; Burushaski comparison
42|22|don|Zahn|tooth|noun m|6152|
42|23|dóono|Ochse|ox|noun m|6273|
42|24|doón|Kopftuch der Frauen|women's headscarf|noun f||Burushaski comparison
42|25|dúu|zwei|two|numeral|6648|
42|26|dub-|nicht können|be unable|verb intr||source analysis is explicitly doubtful
43|01|dáaki|Rücken|back|noun f|5582|
43|02|deér|Bauch, Magen|belly; stomach|noun f|5589|
43|03|dim|Körper, Leib|body|noun m|5551|
43|04|dóoko|Grube|pit|noun m||
43|05|dor|Kornbehälter in der Mühle|grain container in the mill|noun m||Turner 6740 is questioned
43|06|dudúro|Euterzitze|teat|noun||Burushaski comparison
43|07|ek|ein|one|numeral|2462|
43|08|fakír|Fakir|fakir|noun loanword||Arabic loan
43|09|ga|und; auch|and; also|conjunction|4402|
43|10|gachíi|Zweig|branch|noun f|3949|
43|11|gagúi|runder Mahlstein|round grinding stone|noun f||first printed alternate
43|11|gugúi|runder Mahlstein|round grinding stone|noun f alternate||second printed alternate; Burushaski relationship unclear
43|12|gal|Seilbrücke|rope bridge|noun f||Burushaski comparison
43|13|galaáṭi|Melone|melon|noun f||Burushaski comparison
43|14|gaan|Bein|leg|noun f||Burushaski comparison
43|15|gápi|Zügel|reins|noun f||Burushaski comparison
43|16|garáki|Regulierstab in der Mühle|mill regulating stick|noun f||Burushaski comparison
43|17|gáti b-|zusammenkommen|come together|verb intr||Turner 4353 is only a tentative comparison
43|18|gáaye|Lieder|songs|noun pl|4126|
43|19|gin-|nehmen|take|verb tr|4236|
43|20|giri|Fels|rock|noun f|4161|
43|21|góo|Kuh|cow|noun f||Turner 4093 and 4255 are alternatives
43|22|goóṭ|Haus|house|noun m|4336|
43|23|gui|Flamme|flame|noun f||
43|24|guúni|Faden|thread|noun f|4190|
43|25|gunóo|Same(nkorn)|seed|noun m||Burushaski comparison
43|26|hagaái|Himmel|sky|noun f|1009|
43|27|hagáar|Feuer|fire|noun m|125|
43|28|hagúi|Finger|finger; toe|noun f|135|
43|29|hagúl|Hülle; Schwertscheide|cover; sheath|noun m||printed cross-reference to agúl; installed as its variant
44|01|haliẓo|gelb|yellow|adj|13990|
44|02|haloól|Nest; Netz; Lager|nest; web; lair|noun m||source reconstruction is questioned
44|03|han|ist; sind|is; are|verb intr|9416|historical derivation and Turner assignment printed together
44|04|hat|Hand|hand|noun m|14024|
44|05|hin|Schnee|snow|noun m||four alternative Turner numbers are printed
44|06|Hindustáan|Hindustan|Hindustan|proper spatial||
44|07|hun|hoch|high; above|adv spatial||first printed alternate; Turner 2426 is questioned
44|07|húun|hoch|high; above|adv spatial alternate||second printed alternate
44|08|insáan|Mensch|human being|noun loanword||Arabic through Urdu
44|09|ispáawo|süß, schmackhaft|sweet; tasty|adj|13924|
44|10|jak|Leute|people|noun pl||source discusses tatsama versus Hindi loan analysis
44|11|jaláali|langes Haar|long hair|noun f||Burushaski comparison
44|12|jamáat|Familie|family|noun pl loanword||Persian/Arabic comparison
44|13|jangál|Wald|forest|noun m loanword||Hindi-Urdu loan
44|14|január|Lebewesen|living being|noun m loanword||Persian loan
44|15|jaráape|Socken|socks|noun pl loanword||Persian loan
44|16|jil b-|aufgehen (Gestirne)|rise (celestial body)|verb intr|5391|
44|17|jip|Zunge|tongue|noun f|5228|
44|18|jo|aus, von|from; out of|postposition||historical derivation only; Turner 274 is comparative
44|19|jon|Schlange|snake|noun m|5110|
44|20|joóẓi|Birkenbaum|birch tree|noun|9570|
44|21|julún b-|in reicher Fülle herabfallen; schwer mit Früchten beladen sein|fall in abundance; be heavily laden with fruit|verb intr||
44|22|justajúni|mit wirren, ungekämmten Haaren|with tangled, uncombed hair|adj f||
44|23|juúṣ|Birkenrinde|birch bark|noun m|9570|
44|24|kábur|Grab|grave|noun m loanword||Arabic through Urdu
44|25|kac̣úun|Karotte|carrot|noun m||first printed alternate
44|25|kac̣uún|Karotte|carrot|noun m alternate||second printed alternate; Burushaski comparison
44|26|kaagáz|Papier|paper|noun m loanword||Urdu loan
44|27|kaáki|ältere Schwester|elder sister|noun f|2998|Burushaski comparison also printed
44|28|kaáko|älterer Bruder|elder brother|noun m|2998|Burushaski comparison also printed
45|01|kaále|Tuch|cloth|noun f loanword archaic||Urdu loan is questioned; an older form is obsolete
45|02|kaparií|Topf|pot|noun f|2876|
45|03|kašiír|Kaschmir|Kashmir|proper spatial|2968|
45|04|kíno|schwarz|black|adj m|3451|feminine kíni is an inflected example, not a separate headword
45|05|koó|wer, irgendwer|who; someone|pron interr|2574|
45|06|koi|Schote|pod|noun f|3539|
45|07|koó~i|Kamm|comb|noun f|2598|printed nasal vowel normalized to source-profile ~ notation
45|08|kon|Ohr|ear|noun m|2830|
45|09|kóori|Stiefel aus Ziegenleder|goatskin boot|noun pl||Burushaski comparison
45|10|kúi|Land|land|noun||several alternative Turner numbers are printed
45|11|kúulyo|unten|below|adv spatial||Turner 3416/17 is not a single resolvable id
45|12|kut|Wand|wall|noun f|3251|
45|13|kh-|essen|eat|verb tr|3865|
45|14|khabár|Nachricht|news|noun m loanword||Arabic-Persian loan
45|15|khakáai|Walnußkern|walnut kernel|noun f||Burushaski comparison; Turner 2817 is questioned
45|16|khanár|Schwert|sword|noun f||Burushaski, Phalura, Kalasha, and Khowar comparisons
45|17|khat|Bett|bed|noun f|3781|
45|18|khatú|Deckel; Tabaksbehälter der Wasserpfeife|lid; tobacco container of a hookah|noun m||derivational comparisons only
45|19|khir|unten; unter|below; under|adv postposition||first printed alternate
45|19|khirí|unten; unter|below; under|adv postposition alternate||second printed alternate
45|19|khíri|unten; unter|below; under|adv postposition alternate||third printed alternate
45|20|khoi|Kappe, Mütze|cap; hat|noun f|3942|
45|21|khunií|Nasenschleim, Rotz|nasal mucus; snot|noun f||Phalura comparison is questioned
45|22|khur|Mühlrinne|millrace|noun m||relationship to Burushaski is unclear
45|23|khúuro|Huf|hoof|noun m|3906|
45|24|lakaláki|federnd, schaukelnd|springy; rocking|adj f||Shina, Burushaski, and Khowar comparisons
45|25|laál|Rubin|ruby|noun m loanword||Persian comparison
45|26|laltéen|Lampe|lamp|noun m loanword||English loan through Urdu
45|27|lay-|finden|find|verb tr|10948|
46|01|lei|brennende Fackel|burning torch|noun f|710|first printed alternate
46|01|lai|brennende Fackel|burning torch|noun f alternate|710|second printed alternate
46|02|léel|Blut|blood|noun m||Turner 11165 and 11168 are alternatives
46|03|loólyo|rot|red|adj m||Turner 11165/68 is not a single resolvable id; feminine form remains prose
46|04|lonóto|Sproß|sprout|noun m||Turner 11072 is only a comparison
46|05|lup-|brennen|burn|verb intr||source proposes a complex, tentative derivation
46|06|lúẓum|Koralle|coral|noun m||Burushaski comparison
46|07|ma|ich|I|pron|9691|
46|08|máa~|Mutter|mother|noun f|10016|printed nasal vowel normalized to source-profile ~ notation
46|09|macháari|Wespe|wasp|noun f||Turner 9990 and 9699 are alternatives within a comparative discussion
46|10|machíi|Fliege|fly|noun f|9696|
46|11|majaá|mitten drin, mitten in|in the middle; amid|adv spatial|9804|
46|12|makái|Mais|maize|noun f|9879|first printed alternate
46|12|makéi|Mais|maize|noun f alternate|9879|second printed alternate
46|13|maálo|Vater|father|noun|9935|
46|14|manúẓo|Mensch|human being|noun|9827|
46|15|mar-|töten|kill|verb tr|10066|
46|16|maáruč|Pfeffer|pepper|noun f|9875|Burushaski comparison also printed
46|17|maská|Butter|butter|noun m loanword||Persian loan
46|18|Maṭúuṭi|Eigenname|personal name|proper||
46|19|mathári|Walnuß|walnut|noun f||Burushaski comparison
46|20|mei|mein|my|pron poss|9691|
46|21|miike d-|urinieren|urinate|verb intr|10337|
46|22|mor|Wort|word|noun||source gives a broader comparative discussion
46|23|muč-|losgelassen werden, aufhören|be released; stop|verb intr|10181|
46|24|muchót|nach vorn|forward|adv spatial||source analyzes this as a dative of muchó(o)
46|25|muk|Edelsteine (kollektiv)|gemstones (collective)|noun m||Burushaski comparison
46|26|mukh|Gesicht|face|noun m|10174|
46|27|múus|Schlammflut|mudflow|noun f||Burushaski comparison
46|28|nára|nach unten, hinab|downward|adv spatial||first printed alternate; Turner 7189 is questioned
46|28|náre|nach unten, hinab|downward|adv spatial alternate||second printed alternate
46|29|náro|schwierig, schwer|difficult; hard|adj m||homonym 1; feminine nári remains prose; Burushaski comparison
46|30|náro|Mühlrad|mill wheel|noun m||homonym 2; Burushaski comparison
47|01|naš-|umkommen, zugrunde gehen|perish; die|verb intr|7027|
47|02|náto|Nase|nose|noun m|7031|
47|03|náte d-|tanzen|dance|verb intr|7580|
47|04|neé|nicht|not|particle neg|7603|
47|05|nein|hier|here|adv spatial||source derives this through another form; Turner 283 is indirect
47|06|nikhal-|herausnehmen|take out|verb tr|7484|
47|07|nikhay-|hinausgehen, herauskommen|go out; come out|verb intr|7479|
47|08|niílo|blau|blue|adj m|7563|feminine niíli remains prose
47|09|nóoro|Fingernagel|fingernail|noun m|6920|
47|10|nuš|ist nicht|is not|verb neg||Turner 7607 and questioned 12605 are printed together
47|11|óo~ši|Wind|wind|noun f||printed nasal vowel normalized to source-profile ~ notation; Phalura and Maiya comparisons
47|12|páa|Fuß|foot|noun m|8056|
47|13|paruj-|hören|hear|verb tr|7848|
47|14|paš-|sehen|see|verb tr|8012|
47|15|pasoó|Turban|turban|noun m||Burushaski comparison
47|16|páti|Holzschüssel|wooden bowl|noun f||homonym 1
47|17|páṭi|Gürtel|belt|noun f|7700|homonym 2; riddle number is printed with a query
47|18|píito|eng|narrow; tight|adj m||Turner 8165 is only a comparison; feminine píiti remains prose
47|19|piy-|pressen, zusammendrücken|press; squeeze|verb tr|8226|
47|20|poi|fünf|five|numeral|7655|
47|21|pon|Weg|way; path|noun f|7785|
47|22|porií|aus Yasin stammend|from Yasin|adj f demonym||
47|23|poryoó|Mann aus Yasin; Khowar-Sprecher des oberen Gilgit-Tales|man from Yasin; Khowar speaker of the upper Gilgit valley|noun m demonym||
47|24|pyúu b-|verstreut werden|be scattered|verb intr||
47|25|phac̣oó|Schwanz|tail|noun m|8249|source explicitly rejects Turner 7627 in favor of 8249
47|26|phac̣áali|Flügel|wing|noun f|7627|
47|27|phal b-|geworfen werden|be thrown|verb intr|13834|
47|28|phapaáo~|dünnes Fladenbrot|thin flatbread|noun|7934|source prints Turner 7934.2; linked to integer CDIAL parent
47|29|phar b-|sich drehen|turn; rotate|verb intr|9050|Burushaski comparison also printed
48|01|pharpiṭ|gekreuzte Lederriemen am Bett|crossed leather straps on a bed|noun f||Burushaski comparison
48|02|phatú|hinter|behind|postposition||Turner 7732 is questioned because of unexplained ph
48|03|phii~ṣ b-|langsam herausgelassen werden; ausströmen|be let out slowly; flow out|verb intr||printed nasal vowel normalized to source-profile ~ notation; Shina and Burushaski comparisons
48|04|phoṭ|Schale (der Walnuß usw.)|shell (of a walnut, etc.)|noun m||Turner 13845 is only a questioned comparison
48|05|phupúṣ|Feuerstelle mit zwei Steinen|two-stone hearth|noun m||Burushaski analysis is tentative
48|06|phyóolo|Worfelschaufel|winnowing shovel|noun m|13839|
48|07|ráfal|Gewehr|rifle|noun m loanword||English loan through regional languages
48|08|rajoó|königlich|royal|adj|10679|
48|09|ráatyo|nachts|at night|adv|10702|
48|10|re|die eine, die andere|one; the other|pron f|1295|
48|11|róoṣ|Zorn|anger|noun f|10856|
48|12|rúi~|Hexe|witch|noun f||first printed alternate; Arabic rūḥ comparison
48|12|rúui~|Hexe|witch|noun f alternate||second printed alternate
48|13|s-|schlafen|sleep|verb intr|13902|
48|14|sa|Schwester|sister|noun|13913|
48|15|sác̣o|leicht, nicht schwierig|easy; not difficult|adj m||feminine sáči remains prose; Burushaski comparison
48|16|safár|Reise; Monat Safar|journey; month of Safar|noun loanword||Arabic loan
48|17|san-|machen, herstellen|make; produce|verb tr|13126|
48|18|san|Licht|light|noun m||Burushaski comparison
48|19|sarphaloók|Oberwelt|upper world|noun||source's older serpent-world interpretation is rejected
48|20|sat-báaro|siebenschichtiges Gebilde|seven-layered structure|noun m||Turner 13139 belongs to the numeral component
48|21|sáa~ty|zusammen mit|together with|postposition|13364|first printed alternate; nasal vowel normalized to ~ notation
48|21|sáa~ity|zusammen mit|together with|postposition alternate|13364|second printed alternate
48|21|saa~ti|zusammen mit|together with|postposition alternate|13364|third printed alternate
48|22|síi~|Armee, Heer|army|noun f|13587|printed nasal vowel normalized to source-profile ~ notation
48|23|sin|Fluß|river|noun f|13415|
48|24|sóomo|Rauchloch, Dachfenster|smoke hole; roof window|noun m||Burushaski comparison
48|25|sum|Erde, Staub|earth; dust|noun m|13493|
48|26|súuri|Sonne|sun|noun f|13574|
48|27|súuryo|tagsüber|during the day|adv||contrasted with ráatyo by the source
48|28|šal|hundert|hundred|numeral|12278|source notes an unexplained final l
48|29|šeyár|Docht|wick|noun m||Turner 12673 is questioned and belongs to a comparison
48|30|šéeyo|weiß|white|adj m|12774|plural form remains prose
48|31|šikáar|Turm|tower|noun f||Burushaski comparison and Turner 12435 are tentative
49|01|šil|steinernes Tablett zum Mahlen des Korns|stone grinding tray|noun f|12459|
49|02|šóno|Ton, Laut|sound; voice|noun m||source questions a possible misprint in Bailey
49|03|šoór b-|(in Mengen) zerstreut sein|be scattered in quantities|verb intr||Burushaski comparison
49|04|šú|Hund|dog|noun m|12528|
49|05|ṣa|sechs|six|numeral|12803|
49|06|ṣač-|sich anheften|attach oneself|verb intr|13085|
49|07|ṣak|voll|full|adj||Burushaski comparison
49|08|ṣakweéli|glatt gekämmt|smoothly combed|adj f||Burushaski analysis is tentative
49|09|ṣiṣ|Kopf|head|noun m|12497|
49|10|ṣúmal|Art Rettich|kind of radish|noun m||first printed alternate
49|10|ṣúmul|Art Rettich|kind of radish|noun m alternate||second printed alternate; Burushaski comparison
49|11|tal|Zimmerdecke|ceiling|noun m|5731|
49|12|táaro|Stern|star|noun m|5798|
49|13|tatán|glatt (Fels), unbebaut, unfruchtbar|smooth (rock); uncultivated; barren|adj||Burushaski comparison
49|14|téel|Öl|oil|noun|5958|first and usual printed alternate
49|14|teél|Öl|oil|noun alternate|5958|second, explicitly rare printed alternate
49|15|to|wenn|when|particle|5639|
49|16|to ga|wenn auch, obwohl|even if; although|conjunction||
49|17|tom|Baum|tree|noun m||Burushaski comparison
49|18|tran|halb|half|numeral||Burushaski comparison
49|19|tu(r)mák|Gewehr|gun|noun m||Burushaski comparisons
49|20|tuš-|satt werden|become full; be sated|verb intr||Turner 5895 and 5897 are alternatives
49|21|th-|machen, tun; sagen|do; make; say|verb tr|13756|
49|22|thap|dunkel|dark|adj||Burushaski comparison
49|23|tharíni|Schlauch zum Buttern|churning skin|noun f||Burushaski comparison
49|24|thúun|Säule|pillar|noun f|13774|
49|25|ṭak b-|eingebunden sein|be tied up|verb intr||Turner 5420 is only comparative
49|26|ṭar b-|zerbrechen|break; shatter|verb intr||Burushaski comparison
49|27|ṭiki|Brot|bread|noun f|5459|
49|28|ṭon b-|sich bücken|stoop; bend down|verb intr||Burushaski comparison
49|29|Ṭúuṭi|Personenname|personal name|proper||
49|30|tharii~|Poloball|polo ball|noun f||first printed alternate; nasal vowel normalized to ~ notation
49|30|thári|Poloball|polo ball|noun f alternate||second printed alternate; Burushaski comparison
49|31|urán|Lamm|lamb|noun m|2349|
49|32|uwáalo|Sommer|summer|noun m|2144|
50|01|wáaku|unverständliches Wort in Rätsel 12|unintelligible word in riddle 12|uncertain||source rejects two possible comparisons; retained as an explicitly analyzed headword
50|02|wal-|holen, bringen|fetch; bring|verb tr||Turner 14246 is questioned
50|03|way-|kommen|come|verb intr||Turner 884 and 2207 are alternatives
50|04|wei|Wasser|water|noun m|1921|
50|05|xatún|Dame|lady|noun loanword||Persian loan
50|06|yay-|gehen, sich bewegen|go; move|verb intr||Turner 10452 is explicitly very uncertain
50|07|yoóno|Winter|winter|noun m||Turner 14164 and 14334 are alternatives
50|08|yó~r|Wassermühle|water mill|noun f|10412|printed nasal vowel normalized to source-profile ~ notation
50|09|yó~r-bat|Mühlstein|millstone|noun m|11348|printed nasal vowel normalized to source-profile ~ notation
50|10|yúun|Mond|moon|noun f|5301|
50|11|ẓigo|lang|long|adj m|6368|feminine ẓigi remains prose
50|12|ẓúun|ein Vogel: chough; red-billed jackdaw|a bird: chough; red-billed jackdaw|noun f||Lorimer and Bailey identifications are retained in the gloss
50|13|ẓun|Schlucht|ravine|noun f||Bailey's form is called a misprint; Phalura and Turner 6429 are only comparisons
"""


def records() -> list[dict[str, str]]:
    result = []
    counts: Counter[tuple[str, str]] = Counter()
    for row in csv.reader(io.StringIO(RAW.strip()), delimiter="|"):
        if not row or row[0].startswith("#"):
            continue
        if len(row) != 8:
            raise ValueError(row)
        page, unit, form, raw_gloss, gloss, tags, parameter, note = (
            cell.strip() for cell in row
        )
        counts[(page, unit)] += 1
        result.append({
            "page": page, "unit": unit, "form": form, "raw_gloss": raw_gloss,
            "gloss": gloss, "tags": tags, "parameter": parameter, "note": note,
            "ordinal": str(counts[(page, unit)]),
        })
    return result


def base_key(record: dict[str, str]) -> str:
    return f"{SOURCE_ID}:p{record['page']}:e{record['unit']}"


def key(record: dict[str, str]) -> str:
    suffix = f":v{record['ordinal']}" if record["ordinal"] != "1" else ""
    return base_key(record) + suffix


def locator(record: dict[str, str]) -> str:
    return f"{SOURCE_ID}[p. {record['page']}, glossary entry {record['unit']}]"


def canonical_tags(value: str) -> str:
    return " ".join(dict.fromkeys(TAG_ALIASES.get(tag, tag) for tag in value.split()))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]], header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if header:
            writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    if args.pdf and sha256(args.pdf) != PDF_SHA256:
        raise ValueError(f"unexpected PDF SHA-256: {sha256(args.pdf)}")

    raw = records()
    forms, audit = [], []
    agul_key = f"{SOURCE_ID}:p40:e04"
    for record in raw:
        emitted = key(record)
        variant_of = base_key(record) if record["ordinal"] != "1" else ""
        if record["form"] == "hagúl":
            variant_of = agul_key
        etymology = record["note"]
        if record["parameter"]:
            claim = f"Buddruss directly assigns this headword to Turner/CDIAL {record['parameter']}."
            etymology = f"{claim} {etymology}".strip()
        forms.append(dict(zip(FORM_FIELDS, [
            "Sh", record["parameter"], record["form"], record["gloss"], "", "",
            record["raw_gloss"], locator(record), "", etymology, emitted,
            variant_of, "", "", f"{canonical_tags(record['tags'])} dialect:Sh:gil:Gilgit",
        ])))
        payload = "|".join(record.values()).encode()
        audit.append({
            "Snapshot_Date": SNAPSHOT_DATE, "Collation_Date": COLLATION_DATE,
            "Unit_ID": f"p{record['page']}:e{record['unit']}",
            "PDF_Page": str(int(record["page"]) - 23), "Printed_Page": record["page"],
            "Raw_Form": record["form"], "Raw_Gloss_German": record["raw_gloss"],
            "English_Gloss": record["gloss"], "Final_Status": "installed_form",
            "Final_Form": record["form"], "Final_Parameter_ID": record["parameter"],
            "Emitted_Key": emitted,
            "Resolution": record["note"] or "manually collated printed glossary headword",
            "Review": "full manual census against 300 dpi renders; embedded PDF text used only for navigation",
            "Material_Error": "no", "Source": locator(record),
            "Record_SHA256": hashlib.sha256(payload).hexdigest(),
        })

    assert len({row["Entry_Key"] for row in forms}) == len(forms)
    assert {int(row["page"]) for row in raw} == set(range(40, 51))
    assert len([r for r in raw if r["form"] == "agúl"]) == 1
    write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
    write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audit, header=True)
    sample = sorted(audit, key=lambda row: row["Record_SHA256"])[:25]
    write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample, header=True)
    MANIFEST_OUTPUT.write_text(json.dumps({
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "bibliography": "Buddruss, Georg. 1996. Shina-Rätsel. In Dieter B. Kapp (ed.), Nānāvidhaikatā: Festschrift für Hermann Berger, 29–54. Wiesbaden: Harrassowitz.",
        "isbn": "9783447039161",
        "acquisition": "Stanford Interlibrary Loan request 446828; lender Cornell University Library (Olin)",
        "pdf_sha256": PDF_SHA256, "pdf_pages": PDF_PAGES,
        "article_printed_pages": [29, 54], "lexical_printed_pages": [39, 50],
        "pdf_redistributed": False,
        "rights": "Copyrighted ILL scan supplied for private study, scholarship, or research; the scan is not checked in.",
        "extraction": {
            "method": "complete record-by-record manual collation against 300 dpi page renders",
            "navigation_layer": "embedded PDF text, rejected as authoritative because special-glyph mappings are inaccurate",
            "checked_in_layer": "the RAW table in data/other/forms/raw_data/buddruss_shina_1996.py",
            "glossary_headword_record_count": len(raw),
            "analytical_headword_units": len({(r['page'], r['unit']) for r in raw}),
            "transcription_uncertainties_remaining": 0,
        },
        "scope": {
            "included": "every analytical glossary headword and explicit headline alternate on pp. 40–50; the glossary preface begins on p. 39",
            "excluded": "58 running riddles and translations on pp. 31–39; inflected examples not promoted to headword status; bibliography on pp. 50–53; summary on pp. 53–54; Burushaski, Kalasha, Khowar, Phalura, Maiya, and other comparison-only forms",
            "excluded_counts": {"running_riddles": 58, "bibliography_sections": 1, "summary_sections": 1},
            "cdial_policy": "only direct, unambiguous Turner/CDIAL assignments are linked; alternatives, questioned numbers, component-only references, and comparison-only claims remain prose",
            "language_model": "Gilgit Shina is attached to canonical Shina (Sh) with the existing registered dialect tag dialect:Sh:gil:Gilgit",
        },
        "transcription": {
            "source_policy": "double vowels preserve quantity; áa is falling tone and aá rising tone; diphthongal i is preserved",
            "jambu_notation": "printed nasal marks are encoded with ~ immediately after the vowel sequence so the source sound profile can parse them",
            "headword_policy": "explicit headline alternates are rows linked by Variant_Of_Key; feminine, plural, oblique, case, finite, and participial examples remain prose",
        },
        "outputs": {
            "forms": str(FORM_OUTPUT.relative_to(ROOT)), "form_count": len(forms),
            "audit": str(AUDIT_OUTPUT.relative_to(ROOT)), "audit_count": len(audit),
            "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)), "sample_count": len(sample),
        },
        "unresolved": [
            "wáaku is explicitly called unintelligible by Buddruss; it is retained with that gloss rather than silently omitted",
            "all questioned or competing etymologies remain unlinked editorial prose",
        ],
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"installed and audited {len(forms)} Shina-Rätsel glossary headword records")


if __name__ == "__main__":
    main()
