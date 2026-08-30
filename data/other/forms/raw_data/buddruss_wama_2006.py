#!/usr/bin/env python3
"""Install the complete glossary in Buddruss's 2006 Wama text article.

The copyrighted Stanford ILL scan is not redistributed. ``RAW`` is the checked-in,
manually collated transcription layer. Every printed glossary headword and explicit
headword alternate on pp. 184--191 was checked against 400 dpi renders. Two independent
Tesseract layouts were used only for navigation and discrepancy discovery; the scan,
not OCR, is authoritative. Inflected examples remain prose unless printed as headwords.

Run from ``data/``; ``--pdf`` optionally verifies the original scan byte-for-byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


SOURCE_ID = "buddruss-wama2006"
SNAPSHOT_DATE = "2026-08-24"
COLLATION_DATE = "2026-08-24"
PDF_SHA256 = "1bd67f12c5e52463549eef0da440fca7aeade8fc7b0344d05e9c24c02f55343d"
PDF_PAGES = 29
ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
FORM_OUTPUT = ROOT / "data/other/forms/20260824-buddruss-wama.csv"
AUDIT_OUTPUT = RAW_DIR / "20260824-buddruss-wama-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260824-buddruss-wama-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260824-buddruss-wama-manifest.json"

TAG_ALIASES = {
    "adjective": "adj", "adverb": "adv", "pronoun": "pron", "numeral": "num",
    "preposition": "prep", "postposition": "postp", "conjunction": "conj",
    "particle": "part", "oblique": "obl", "possessive": "poss",
    "proper": "proper-noun", "demonym": "proper-noun", "historical": "archaic",
    "classifier": "num", "interrogative": "interr", "causative": "caus",
    "past": "pret", "imperative": "impv", "article": "determiner",
    "negative": "neg", "preverb": "prefix", "enclitic": "suffix",
}
TAG_DROPS = {"case"}

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

# printed page | head-entry number on page | form | source gloss | English gloss
# | tags | direct Turner/CDIAL id | source/editorial note
RAW = r"""
184|01|a|ein; unbestimmter Artikel|one; indefinite article|numeral article|2462|
184|02|a-|kommen|come|verb intr|1288|
184|03|ā|ja|yes|particle||
184|04|ai|ich|I|pronoun||source compares Ashkun and Dameli; deeper analysis is tentative
184|05|-ai|Postposition für, um zu|for; in order to|postposition case||source analyzes an original dative
184|06|o-|Präverb senkrecht, steil hoch|straight upward|preverb spatial|2136|first printed headword alternate
184|06|ō-|Präverb senkrecht, steil hoch|straight upward|preverb spatial alternate|2136|second printed headword alternate
184|07|-u|enklitische Fragepartikel|enclitic question particle|particle enclitic||
184|08|abō|Wasser|water|noun|407|
184|09|abōtistə́|hungrig|hungry|adjective|540|
184|10|ači|Auge|eye|noun|43|
184|11|a čit|einige|some|numeral||source presents Turner 4831 versus 3167 as alternatives
184|12|ač|eins|one|numeral||source's relation to aika is uncertain
184|13|ačəgə|kalt|cold|adjective|1078|masculine headword form
184|13|ačegi|kalt|cold|adjective alternate|1078|feminine headword form
184|14|ačəwə́|Schatten|shade; shadow|noun||
184|15|ūdr-|fliegen|fly|verb intr|2001|first printed headword alternate
184|15|undr-|fliegen|fly|verb intr alternate|2001|second printed headword alternate
184|16|ūdrāwat|Flügel|wing|noun||source's comparison to T. 7733 is tentative
184|17|adusestə́|leicht von Gewicht|light in weight|adjective||etymology and analysis are tentative
184|18|oga-|hochnehmen|lift up|verb tr|1967|first printed headword alternate
184|18|õga-|hochnehmen|lift up|verb tr alternate|1967|second printed headword alternate
184|19|augān|Afghane, Pashtune|Afghan; Pashtun|noun demonym||
184|20|okas-|senkrecht hochschauen|look straight upward|verb intr||directional compound of o- and kas-
184|21|amə́|Haus|house|noun|560|first printed headword alternate
184|21|amā́|Haus|house|noun alternate|560|second printed headword alternate
184|22|imā́|mein|my|pronoun possessive||
184|23|imə́|wir|we|pronoun|986|
184|24|aŋā́|Feuer|fire|noun|125|
185|01|onis-|aufgehen Sonne|rise; come out|verb intr||first printed headword alternate
185|01|unis-|aufgehen Sonne|rise; come out|verb intr alternate||second printed headword alternate
185|02|istā́|Stern|star|noun|13713|first printed headword alternate
185|02|istə̄́|Stern|star|noun alternate|13713|second printed headword alternate
185|03|ōst-|aufstehen|stand up|verb intr|1900|
185|04|istrī́|Ehefrau|wife|noun|13734|
185|05|istrṓ|Sichel|sickle|noun||source treats etymology as uncertain
185|06|aṣai|versammelt|assembled|adjective|1468|
185|07|uṣṭ|Lippe|lip|noun|2503|
185|08|ut-|sich hinstellen|stand oneself up|verb intr|1907|
185|09|utiə|innen|inside|spatial||
185|10|otādi-|hineingehen|go inside|verb intr||
185|11|oteipei|innen|inside|spatial||
185|12|autra|sohnlos|sonless|adjective||first printed headword alternate
185|12|autre|sohnlos|sonless|adjective alternate||second printed headword alternate
185|13|utiē-(a)w-|(eilig) hereinkommen|come in quickly|verb intr||directional compound
185|14|aṭalə́|Bergabschnitt, Dorfteil|mountain section; village quarter|noun||
185|15|aw-|sich rasch bewegen, springen|move quickly; jump|verb intr||source marks Turner 1193 comparison with a query
185|16|b-|werden|become|verb intr|9416|
185|17|ba-|können|be able|verb intr||causative to b-; Ashkun comparison
185|18|-ba|Partikel beim Konjunktiv|conjunctive particle|particle suffix||
185|19|biē-|Richtungspräfix hinaus|outward|preverb spatial||first printed headword alternate
185|19|byē-|Richtungspräfix hinaus|outward|preverb spatial alternate||second printed headword alternate
185|20|baḍi-|talab gehen|go downstream|verb intr||directional compound
185|21|baḍa|Bestechung|bribe|noun loanword||from Pashto
185|22|beikə́|Tür|door|noun||
185|23|bal-|sprechen, sagen|speak; say|verb tr|9406|
185|24|beipā|außerhalb von|outside of|postposition spatial||
185|25|brā|Bruder|brother|noun|9661|
185|26|bərā|Präteritum zu gut- forttragen|carried away|verb tr past stem|9588|first printed headword alternate
185|26|birā|Präteritum zu gut- forttragen|carried away|verb tr past stem alternate|9588|second printed headword alternate
185|27|bās|Wind|wind|noun|11592|
185|28|bat-|denken, meinen|think; suppose|verb intr|9276|
185|29|bata-|denken lassen|cause to think; misinform|verb tr causative||causative of bat-
186|01|baūtiə|innen|inside|spatial||used specifically of building interiors
186|02|biē(a)w-|hinausspringen|jump out|verb intr||first printed headword alternate
186|02|byē(a)w-|hinausspringen|jump out|verb intr alternate||second printed headword alternate
186|03|bēyo|draußen|outside|spatial||first printed headword alternate
186|03|bēyu|draußen|outside|spatial alternate||second printed headword alternate
186|04|ce|Präteritum zu ko- machen, sagen|did; said|verb tr past stem||first printed gender form
186|04|ci|Präteritum zu ko- machen, sagen|did; said|verb tr past stem alternate||second printed gender form
186|05|cima-karā|Schmied|blacksmith|noun|14496|
186|06|cən|versteckt|hidden|adjective|5046|
186|07|cuniye|in Richtung auf|toward; in the direction of|spatial||
186|08|cestə|gemacht|made|participle||ce plus -stə
186|09|citra-|schreiben|write|verb tr|4810|
186|10|čital|Magen, Bauch|stomach; belly|noun|3157|
186|11|dai|Vater|father|noun|6261|
186|12|di-|gehen|go|verb intr|6365|
186|13|du|zwei|two|numeral|6648|
186|14|dikatə́|Himmel|sky|noun|6331|
186|15|dam-|ergreifen, bringen|seize; bring|verb tr|6284|
186|16|dōt|Zahn|tooth|noun|6152|
186|17|dar-|waschen|wash|verb tr||
186|18|dōst|Hand|hand|noun|14024|
186|19|dya|und|and|conjunction||first printed headword alternate
186|19|dye|und|and|conjunction alternate||second printed headword alternate
186|20|ḍob-|begegnen|meet|verb intr||source says the proposed Turner comparison is semantically unsuitable
186|21|ḍuḍū ko-|donnern|thunder|verb intr||
186|22|ḍaŋ-|hingelangen, ankommen|reach; arrive|verb intr||
186|23|ḍer|erstaunt|astonished|adjective||
186|24|g-|Präteritum zu di- gehen|went|verb intr past stem|4008|
186|25|ga|Kuh|cow|noun|4147|
186|26|gu|Kot|dung|noun|4225|
186|27|gui|Imperativ zu pr- gib mir/uns|give me; give us|verb tr imperative|4236|
186|28|gulabṓ|Fluss|river|noun|4456|
186|29|gandalə(stə)|schwer, schwanger|heavy; pregnant|adjective|14468|
186|30|goṇṭ-|binden|bind|verb tr|4205|first printed headword alternate; source also cites T. 14447
186|30|guṇṭ-|binden|bind|verb tr alternate|4205|second printed headword alternate; source also cites T. 14447
186|31|gəras|Tag|day|noun|4440|
186|32|grām|Dorf|village|noun|4368|first printed headword alternate
186|32|grə̄m|Dorf|village|noun alternate|4368|second printed headword alternate
187|01|gut(y)-|mitnehmen, wegtragen|take along; carry away|verb tr|4236|
187|02|gawār|Feld auf der Talsohle|field on the valley floor|noun|4376|first printed headword alternate
187|02|gəwar|Feld auf der Talsohle|field on the valley floor|noun alternate|4376|second printed headword alternate
187|02|gawā́r|Feld auf der Talsohle|field on the valley floor|noun alternate|4376|third printed headword alternate
187|03|jan|Numerativ für Menschen|human classifier|numeral classifier|5098|first printed headword alternate
187|03|jən|Numerativ für Menschen|human classifier|numeral classifier alternate|5098|second printed headword alternate
187|04|jit|Körper|body|noun|5244|
187|05|jūk-|gesund werden; sich versöhnen|recover; reconcile|verb intr|10481|
187|06|jowār|Mais|maize|noun loanword|10437|borrowed from Pashto
187|07|ka|was, irgendwas|what; anything|pronoun|2574|homonym 1
187|08|ka|unter|under|postposition spatial||homonym 2
187|09|kai-kai|ob oder|whether or|conjunction||source marks Turner 2967 comparison with a query
187|10|kaū|warum|why|adverb||
187|11|ko-|machen, tun|do; make|verb tr|2814|
187|12|kukūr|Hahn, Huhn|rooster; chicken|noun|3208|
187|13|-kal|Zeit, als, wenn|time; when|suffix conjunction|3084|
187|14|kulī|Familienangehöriger|family member|noun|3342|
187|15|kulāl|Töpfer|potter|noun loanword||source allows Persian origin or Turner 3341, so no single link is imposed
187|16|kam|Mann aus Kamgal|man from Kamgal|noun demonym||
187|17|Kamgāl|Ortsname|Kamgal|noun proper spatial||place name
187|18|kandə́|Baum|tree|noun|13627|
187|19|kuniək|Band|band; strap|noun||cross-reference to ṣatu-kuniək
187|20|kar|Ohr|ear|noun|3056|
187|21|karāk|Handwerkermeister|master craftsman|noun loanword||translation of Pashto kasbgar
187|22|kareik|Beratung|consultation|noun||source notes a possible homonym
187|23|keriē|Rache|revenge|noun||etymology unresolved
187|24|krōm|Dach|roof|noun|3415|
187|25|karanik|taubenähnlicher Vogel|pigeon-like bird|noun||first printed headword alternate
187|25|karanyēk|taubenähnlicher Vogel|pigeon-like bird|noun alternate||second printed headword alternate
187|26|kurī|Hund|dog|noun||
187|27|kūri|woher|from where|adverb|3384|first printed headword alternate
187|27|kuṇi|woher|from where|adverb alternate|3384|second printed headword alternate
187|28|kas-|schauen|look|verb intr|3114|first printed headword alternate
187|28|kəs-|schauen|look|verb intr alternate|3114|second printed headword alternate
187|29|kastə|welch|which; what kind of|pronoun||
187|30|kaṭək|Kampfgruppe, Heer|war party; army|noun||
187|31|kaṭəki|Kämpfer|fighter|noun||
188|01|katāri|Maisstroh|maize stalks|noun|2630|
188|02|kawarə́|Rabe|raven|noun||
188|03|la-|schlagen, werfen, töten|strike; throw; kill|verb tr|11004|
188|04|lōu|Blut|blood|noun|11165|first printed headword alternate
188|04|lou|Blut|blood|noun alternate|11165|second printed headword alternate
188|05|law-|finden, suchen|find; seek|verb tr|10948|
188|06|mac|Ehemann|husband|noun|9888|
188|07|mič|Lehm|clay|noun|10287|
188|08|muč-|fliehen|flee|verb intr|10181|
188|09|mōdə́|Kleidung|clothing|noun|9740|
188|10|muk|Gesicht|face|noun|10174|
188|11|malə́|viel, sehr|much; very|quantifier adverb||first printed headword alternate; source questions Turner 9935
188|11|mallə́|viel, sehr|much; very|quantifier adverb alternate||second printed headword alternate; source questions Turner 9935
188|12|mandə́|Hals|neck|noun|9732|
188|13|minik|Schlaf|sleep|noun||source tentatively compares T. 7200
188|14|mutruk|Urin|urine|noun|10234|
188|15|maz|Mitte|middle|noun spatial|9804|
188|16|ne|nicht|not|particle negative|7605|
188|17|nai|oder nicht|or not|particle interrogative||
188|18|nij-|Kleider waschen|wash clothes|verb tr|7185|
188|19|nalī|Wolke, Regen|cloud; rain|noun|6955|
188|20|niŋa-|erkennen, wissen|recognize; know|verb tr|7165|
188|21|niŋasə́|Vogel|bird|noun|10265|
188|22|niš-|sich setzen|sit down|verb intr|7467|source also cites T. 7464
188|23|nas|Schnabel, Regenrinne am Dach|beak; roof waterspout|noun||
188|24|nāṭ|Tanz|dance|noun|7580|
188|25|pə|zu, an, auf|to; at; on|postposition|8540|first printed headword alternate
188|25|po|zu, an, auf|to; at; on|postposition alternate|8540|second printed headword alternate
188|26|pā-|Präverb hin, weg vom Sprecher|away from the speaker|preverb spatial||
188|27|paū|dort schräg unten, hinab|down there obliquely|spatial||
188|28|pakār|Nutzen|use; benefit|noun loanword||from Persian or Pashto
188|29|pakasa-|(Hals) vorstrecken; (Zigarette) anbieten|extend the neck; offer|verb tr|8440|
188|30|pala|hinwerfen|throw away|verb tr||
188|31|palaŋ-|drehen|turn|verb tr|8591|first printed headword alternate
188|31|polaŋ-|drehen|turn|verb tr alternate|8591|second printed headword alternate
188|32|pā-las-|freilassen, entkommen lassen|release; let escape|verb tr|10994|
188|33|pəlisa-|reinigen|clean|verb tr||source's comparison to T. 10993 is tentative
189|01|palyaŋ b-|glänzen, scheinen|shine|verb intr||relation to Ashkun lightning term is unclear
189|02|pambiri|sie wurde fortgetragen|she was carried away|verb tr past||analyzed as pā- plus bərə
189|03|p-amə́|nach Hause|homeward|spatial||cross-reference to pə
189|04|panilei|oberer rechter Ortsteil von Wama|upper-right quarter of Wama|noun proper spatial||
189|05|par|-mal|time; occurrence|numeral||
189|06|par-|füllen|fill|verb tr|8107|
189|07|pr-|geben|give|verb tr|8655|
189|08|patar-|überqueren|cross|verb tr|8536|first printed headword alternate
189|08|pater-|überqueren|cross|verb tr alternate|8536|second printed headword alternate
189|08|pātar-|überqueren|cross|verb tr alternate|8536|third printed headword alternate
189|09|pa-wi-|hinschlagen|strike away|verb tr||directional compound of pa- and wi-
189|10|s-|sein|be|verb intr||
189|11|sa|jener, er|that one; he|demonstrative pronoun|12815|
189|12|seu|Brücke|bridge|noun|13585|
189|13|so|Sonne|sun|noun|13574|homonym 1
189|14|so|Wildziege|wild goat|noun||homonym 2; source's Sanskrit comparison is tentative
189|15|sakī|dort, an jener Stelle|there; at that place|spatial||
189|16|sakə́|jener, er|that one; he|demonstrative pronoun|12815|
189|17|sal|Stall|stable|noun|12414|
189|18|soṇ|Gold|gold|noun|13519|
189|19|ṣə̄rü|Bewohner von Wama|resident of Wama|noun demonym||
189|20|-stə|Morphem|participial or nominalizing suffix|suffix||source refers to grammar section 29
189|21|strimali|Frau|woman|noun|13734|
189|22|sawāk|alle|all|quantifier|13276|
189|23|sawuli|Freund, Freundin|friend|noun|13895|
189|24|šāk|zuerst|first|adverb||
189|25|šukul-|graben|dig|verb tr||first printed headword alternate
189|25|šikul-|graben|dig|verb tr alternate||second printed headword alternate
189|26|šamə́|einheimischer Name für Wama|native name for Wama|noun proper spatial||first printed headword alternate
189|26|šamə̄|einheimischer Name für Wama|native name for Wama|noun proper spatial alternate||second printed headword alternate
189|27|šip-|zerbrechen|break intransitive|verb intr|3687|
189|28|šipasu|Schwiegervater|father-in-law|noun|12753|
189|29|ṣa|Kopf|head|noun|12694|
189|30|ṣai|um willen|for the sake of|postposition|12702|
189|31|ṣiŋ|Horn|horn|noun|12583|
189|32|ṣor|Topf|pot|noun||first printed headword alternate
189|32|šor|Topf|pot|noun alternate||second printed headword alternate
190|01|ṣorə́|Sand|sand|noun|13386|first printed headword alternate
190|01|šorə|Sand|sand|noun alternate|13386|second printed headword alternate
190|02|ṣatū|Baumwollhose für Männer und Frauen|cotton trousers|noun|13468|first printed headword alternate
190|02|ṣatú|Baumwollhose für Männer und Frauen|cotton trousers|noun alternate|13468|second printed headword alternate
190|03|ṣa-topalū|Turban|turban|noun|5481|
190|04|ṣuti|Asche|ashes|noun|3709|
190|05|ta|zu, an, in|to; at; in|postposition|13760|
190|06|tai|aus|out of; from|postposition||first printed headword alternate; cross-referenced from ta
190|06|taī|aus|out of; from|postposition alternate||second printed headword alternate; cross-referenced from ta
190|07|taū|hin zu|toward|postposition||
190|08|to|dich|you|pronoun oblique||homonym 1
190|09|to|doch|indeed; however|particle||homonym 2; source marks Turner 5639 comparison with a query
190|10|tu|du|you|pronoun|5889|
190|11|tiek|Partikel fakultativ beim Absolutiv|optional absolutive particle|particle||
190|12|tapāl|feucht|damp|adjective||source says disputed variant and cites Turner 5929 versus 6028
190|13|tiw-|setzen, bauen|put; build|verb tr|13756|
190|14|wa|Präverb her auf gleicher Höhe|toward speaker on the same level|preverb spatial||first printed headword alternate
190|14|wā-|Präverb her auf gleicher Höhe|toward speaker on the same level|preverb spatial alternate||second printed headword alternate
190|15|wa-a-|herankommen|come toward|verb intr||directional compound
190|16|wi|nahe senkrecht unten befindlich|nearby directly below|spatial||
190|17|wi-|schlagen|strike|verb tr|12109|
190|18|wo|Präfix steil hinab|steeply downward|preverb spatial||first printed headword alternate
190|18|wō|Präfix steil hinab|steeply downward|preverb spatial alternate||second printed headword alternate
190|18|wū|Präfix steil hinab|steeply downward|preverb spatial alternate||third printed headword alternate
190|19|wo|Vokativpartikel|vocative particle|particle||
190|20|woi|unten, weiter weg als wi|below, farther away than wi|spatial||
190|21|wadū|beide|both|numeral||first printed headword alternate; source compares ubhā dvau without a Turner number
190|21|wudū|beide|both|numeral alternate||second printed headword alternate
190|22|wō-di-|hinabgehen, fallen|go down; fall|verb intr||
190|23|widišā|Gast|guest|noun|11738|
190|24|wō-la-|hinabwerfen|throw down|verb tr||
190|25|wo(a)mra-|hinablassen; regnen; schneien|let down; rain; snow|verb intr||source says no simplex mra- or amra- could be elicited
190|26|wa(n)kes(a)wa-|herziehen lassen|cause to draw toward|verb tr causative|2908|
190|27|weri|Sprache, Wort, Sache|speech; word; matter|noun|11327|
190|28|wāra-|zeigen|show|verb tr|12111|first printed headword alternate
190|28|wə̄ra-|zeigen|show|verb tr alternate|12111|second printed headword alternate
190|29|wārō-a-|schräg heraufkommen|come upward obliquely|verb intr||
190|30|wārāt b-|sichtbar werden|become visible|verb intr||derived from wāra-
190|31|was|Tag, Tagesetappe|day; day's journey|noun|11442|source also cites T. 11591
190|32|waṭ|Stein|stone|noun|11348|
190|33|watāk|Topf aus gebranntem Lehm|fired-clay pot|noun|11347|
190|34|wiaw-|schlagen lassen|cause to strike|verb tr causative||causative of wi-
190|35|wayō|ander|other|adjective||
190|36|ye|und|and|conjunction||
191|01|yu-|(fr)essen|eat|verb tr|10507|first printed headword alternate
191|01|yo-|(fr)essen|eat|verb tr alternate|10507|second printed headword alternate
191|02|yū|Obliquus zu ai ich|me|pronoun oblique||
191|03|yek|dieser|this|demonstrative pronoun||
191|04|yus|Gras|grass|noun|10436|
191|05|yaustag|ein wenig|a little|quantifier||source derives from *yaut-stə-ag and compares T. 10475
191|06|yoṣ|Riese|giant|noun|10395|
191|07|zō|Milch|milk|noun|14019|
191|08|zu|Tochter|daughter|noun|6481|
191|09|zagə́|Sohn|son|noun|14516|
191|10|zamə́|Schwiegersohn|son-in-law|noun|5198|first printed headword alternate
191|10|zəmə́|Schwiegersohn|son-in-law|noun alternate|5198|second printed headword alternate
191|11|žim|Schnee|snow|noun|14096|
191|12|žirik|Scham|shame|noun|14185|
191|13|žu-|weinen|weep|verb intr|10840|
191|14|žucaka|frühmorgens|early morning|adverb|10833|
191|15|žatr|Nacht; nachts|night; at night|noun adverb|10702|
191|16|žatə̄rə|die ganze Nacht|all night|adverb||
191|17|žaw-|mähen, schneiden|mow; cut|verb tr|10645|first printed headword alternate
191|17|žow-|mähen, schneiden|mow; cut|verb tr alternate|10645|second printed headword alternate
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
    return " ".join(dict.fromkeys(
        TAG_ALIASES.get(tag, tag) for tag in value.split() if tag not in TAG_DROPS
    ))


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
    forms = []
    audit = []
    for record in raw:
        emitted = key(record)
        etymology = record["note"]
        if record["parameter"]:
            claim = f"Buddruss directly compares or assigns this headword to Turner/CDIAL {record['parameter']}."
            etymology = f"{claim} {etymology}".strip()
        forms.append(dict(zip(FORM_FIELDS, [
            "Ash", record["parameter"], record["form"], record["gloss"], "", "",
            record["raw_gloss"], locator(record), "", etymology, emitted,
            base_key(record) if record["ordinal"] != "1" else "", "", "",
            f"{canonical_tags(record['tags'])} dialect:Ash:cdial-Ash-wama:Wama",
        ])))
        payload = "|".join(record.values()).encode()
        audit.append({
            "Snapshot_Date": SNAPSHOT_DATE, "Collation_Date": COLLATION_DATE,
            "Unit_ID": f"p{record['page']}:e{record['unit']}",
            "PDF_Page": str(int(record["page"]) - 171), "Printed_Page": record["page"],
            "Raw_Form": record["form"], "Raw_Gloss_German": record["raw_gloss"],
            "English_Gloss": record["gloss"], "Final_Status": "installed_form",
            "Final_Form": record["form"], "Final_Parameter_ID": record["parameter"],
            "Emitted_Key": emitted,
            "Resolution": record["note"] or "manually collated printed glossary headword",
            "Review": "full manual census against the 400 dpi render; two Tesseract page-layout passes compared",
            "Material_Error": "no", "Source": locator(record),
            "Record_SHA256": hashlib.sha256(payload).hexdigest(),
        })

    assert len({row["Entry_Key"] for row in forms}) == len(forms)
    assert {int(row["page"]) for row in raw} == set(range(184, 192))
    write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
    write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audit, header=True)
    sample = sorted(audit, key=lambda row: row["Record_SHA256"])[:25]
    write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample, header=True)
    MANIFEST_OUTPUT.write_text(json.dumps({
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "bibliography": "Buddruss, Georg. 2006. Drei Texte in der Wama-Sprache des afghanischen Hindukusch. In M. N. Bogoljubov (ed.), Indoiranskoe jazykoznanie i tipologija jazykovyx situacij, 177–200. St Petersburg: Nauka.",
        "acquisition": "Stanford Interlibrary Loan request 446830, delivered as a web scan",
        "pdf_sha256": PDF_SHA256,
        "pdf_pages": PDF_PAGES,
        "article_printed_pages": [177, 200],
        "lexical_printed_pages": [184, 191],
        "pdf_redistributed": False,
        "rights": "Copyrighted ILL scan supplied for private study, scholarship, or research; the scan is not checked in.",
        "extraction": {
            "method": "complete record-by-record manual collation against 400 dpi page renders",
            "ocr_reproducibility": [
                "tesseract -l deu+eng --psm 4",
                "tesseract -l deu+eng --psm 6",
            ],
            "checked_in_layer": "the RAW table in data/other/forms/raw_data/buddruss_wama_2006.py",
            "glossary_headword_record_count": len(raw),
            "transcription_uncertainties_remaining": 0,
        },
        "scope": {
            "included": "every printed headword and explicit headword alternate in the complete glossary on pp. 184–191",
            "excluded": "the three running texts and translations; inflected examples not promoted by Buddruss to headword status; Ashkun, Nuristani Kalasha, Dameli, Kati, Prasun, Pashai, and other-language comparison forms in etymological commentary",
            "cdial_policy": "direct unambiguous Turner/CDIAL assignments are linked; hedged, alternative, and secondary comparisons remain prose",
            "language_model": "Wama is modeled as a dialect of canonical Ashkun (Ash), following Buddruss's explicit classification, with the registered Wama dialect tag cdial-Ash-wama; Abdul Karim and Taj Mohammad remain speaker provenance rather than dialects",
        },
        "outputs": {
            "forms": str(FORM_OUTPUT.relative_to(ROOT)), "form_count": len(forms),
            "audit": str(AUDIT_OUTPUT.relative_to(ROOT)), "audit_count": len(audit),
            "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)), "sample_count": len(sample),
        },
        "unresolved": [
            "Buddruss explicitly reports uncertainty and inconsistency in Wama vowel quantity; the importer preserves his printed headword forms without normalizing those quantities",
        ],
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"installed and audited {len(forms)} Wama glossary headword records")


if __name__ == "__main__":
    main()
