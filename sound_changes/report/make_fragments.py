#!/usr/bin/env python3
"""Write report/fragments2.json: prose sections for template2.html.

Reuses the group cards and prose sections of the first report (template.html + fragments.json),
adds the subgrouping tree, and reads round-2 text from report/round2.html when it exists.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
v1 = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
for k, v in json.load(open(os.path.join(HERE, "fragments.json"), encoding="utf-8")).items():
    v1 = v1.replace("{{" + k + "}}", v)


def between(a, b, text=v1):
    i = text.index(a) + len(a)
    return text[i:text.index(b, i)]


groups = between('<div class="groups">', '</div>\n</section>')
novel = between('<ol class="novel">', '</ol>')
nur = between('<section id="nuristani" class="prose">', '</section>').replace("<h2>Nuristani: what OIA can predict</h2>", "<h3>Nuristani: what OIA can predict</h3>")
rom = between('<section id="romani" class="prose">', '</section>').replace("<h2>Romani and Domari</h2>", "<h3>Romani and Domari</h3>")
cur = between('<section id="curation" class="prose">', '</section>').replace("<h2>Data problems for curation</h2>", "")
meth = between('<section id="method" class="prose">', '</section>').replace("<h2>How it was built</h2>", "")
lim = between('<section id="limits" class="prose">', '</section>').replace("<h2>Limits and next steps</h2>", "<h3>Limits</h3>")
staged = between('<h3>A staged hypothesis</h3>', '</div>\n</section>')
staged = "<h3>A staged hypothesis</h3>" + staged
staged = staged.replace("Putting the pooled orders together with what each group found",
    "Putting the order evidence together with what each group found (the counts below compare each lect's own rules for a process, e.g. its own lenition rule against its own degemination rule; they do not claim the rules are the same change, and the rule-level consensus above never pools them)")
staged = staged.replace("after compensatory lengthening (40 : 2)", "after compensatory lengthening (45 : 2)")
staged = staged.replace("s &gt; h in Bhil, south-western Rajasthani, the Northwest, Assamese and Dhivehi, which in three separate groups is older than a later c &gt; s.",
    "debuccalisation of s, which is several different changes: unconditioned s &gt; h in the Bhil lects and Wagri; s &gt; h between vowels in Punjabi, Lahnda, Sindhi and Dhivehi; initially and between vowels in Mewari, Godwari and Wadiyara Koli; ś &gt; h after a vowel in Assamese and word-initially in Hajong and Bishnupriya. In three groups it is older than a later c &gt; s.")

B = {"firm": '<span class="badge b-firm">firm</span>', "mod": '<span class="badge b-mod">moderate</span>',
     "weak": '<span class="badge b-weak">weak</span>', "areal": '<span class="badge b-areal">areal</span>'}


def node(title, badge, evidence, children="", open_=False):
    return (f'<details{" open" if open_ else ""}><summary><b>{title}</b>{B[badge] if badge else ""}</summary>'
            f'<div class="ev">{evidence}</div>{children}</details>')


tree = '<details class="tree" open><summary><b>Indo-Iranian</b></summary>' + node(
    "Nuristani", "firm", "Sister branch, not descended from OIA. Internally {Kati, Kamviri} | {Waigali, Tregami} | Ashkun | Prasun: Kt+Kam share ṛ > ř, r- > ř-, n > ř̃, and (round 2) ā > o in the new final syllable and a > u in monosyllables, both rejected by every other lect; Wg+Gmb share post-vocalic *ʦ > ś; all share an initial r- fortition with different outputs.") + node(
    "Indo-Aryan", "", "", open_=True, children=
    node("Dardic (a geographic label, not a proven clade)", "areal",
         "What the lects share is mostly retention or areal: clusters simplify without gemination, and only after lenition (41 lects for, 0 against, counting each lect's own rules); deaspiration; three sibilants in places; post-nasal voicing. No innovation common to all of them was found.",
         node("Kashmiric: Kashmiri, Poguli, Dodi", "firm", "c j > ʦ z (every other lect rejects it); v- kept where all others have b-.") +
         node("Shinaic: Shina, Ushojo; Palula, Sauji, Kalkoti, Kundal Shahi; Domaaki", "mod",
              "Shina proper (Shina, Ushojo) is firm: tr > ʦ̣, nasal + stop > nasal, s + stop > stop. The southern lects share mostly retentions. Domaaki joins here: three sibilants, tr > ʦ̣, NC > N; the Shina and Kalkoti chains fit it at 0.58 vs Punjabi 0.65.") +
         node("Chitral: Khowar, Kalasha", "mod", "Prothetic i- before s + consonant; loss of vowel length. Khowar alone has rounding harmony and keeps r + C.") +
         node("Kohistani: Indus Kohistani, Gowro, Bhateri, Chilisso; Torwali, Bashkarik", "firm", "Stop + r > retroflex affricate, including pr > ʦ̣; the three Zoller lects are mutual nearest chains.") +
         node("Kunar and Pashai: Gawarbati, Shumashti, Dameli; Pashai; Tirahi, Wotapuri", "weak", "Laterals from stop + r (also found in Kohistani Bashkarik and West Pahari: areal). Dameli is archaic; Tirahi is not Kohistani; Wotapuri is transitional.")) +
    node("Kharoṣṭhī north-west MIA: Gāndhārī, Niya, Kharoṣṭhī inscriptions", "firm", "Three sibilants kept (merging costs +0.014 to +0.022), sv > sp, lenition by voicing only, graded by register. Gāndhārī + Niya share k > g/ɣ, found in no other lect.") +
    node("Insular: Sinhala, Old Sinhala, Dhivehi", "firm", "Left at a Pali/Aśokan-like stage (no Prakrit n > ṇ, aspirate > h or ya-śruti); own lenition that keeps d and b; deaspiration, c > s, j > d, degemination without lengthening.") +
    node("Migratory: Romani (European, Welsh, Balkan, Zargari), Lomavren; Domari", "mod", "Early exit from the Central area: shares ts > cʰ, kṣ > kʰ, ava > o, r̩ > i/u but not stop + r assimilation or s + stop aspiration; t > l before cluster simplification. Grafting the Hindi MIA block costs +0.054 to +0.087. North-western traits look like contact. Domari is a separate, more Central exit. Romani + Lomavren share voiced-aspirate devoicing and r̩ > u; Welsh, Balkan and Zargari add a > e in open first syllables.") +
    node("Mainland NIA", "firm", "Prakrit lenition before degemination (stop loss 120 lects for, 0 against), geminates, anusvāra > nasal vowel. The shared-prefix tree recovers it: after lenition and degemination, the mainland shares s + stop > aspirate and the assimilated clusters, which the Dardic and Romani chains lack.", open_=True, children=
         node("Northwest: Lahnda (north: Hindko, Pothwari; south: Saraiki, Khetrani, CDIAL L.), Punjabi, Sindhi + Kutchi", "firm",
              "No compensatory lengthening (adding it costs +0.03 to +0.07), post-nasal voicing, rt > ṭṭ, n > ṇ, s > h. Sindhic (Sindhi, Kutchi): l > r, implosives, late degemination. Tone splits north from south Lahnda. Dhatki and Kholosi are transitional.") +
         node("West Pahari: Bhadrawah | Satluj–Kullu | southern Mahasu | Chamba–Kangra", "firm",
              "Four blocks agreed by two methods. Stop + r > ṭ͡ḷ in Bhadrawah; ṣ > ś with s kept in 12 of 16; a > ɔ older than degemination in Satluj–Kullu. Its attachment (NW or Central) is open.") +
         node("South-western: Gujaratic (Gujarati, Old Gujarati, Wadiyara Koli) + Rajasthani + Bhil lects", "mod",
              "Masc. -o, l > ḷ before degemination, ṇ kept. Gujaratic: syncope before apocope, -vũ infinitive. Rajasthani: Marwari–Mewari–Godwari (s > h) | Dhundari–Hadauti | Mewati–Ahirwadi (Hindi-like). Bhil lects fit the Nimadi/Rajasthani chains better than Marathi. Gujari is an outlier. Round 2: medial u > a unites the 8 core Bhil lects and is rejected everywhere else.") +
         node("Marathi–Konkani", "firm", "c j > ʦ ʣ, l > ḷ, n > ṇ, cʰ > s; Konkani shares masc. -o with Gujarati.") +
         node("Central: Hindi–Haryanvi | Braj, Bundeli, Kannauji, Awadhi, Chhattisgarhi, Surgujia; Halbi, Bhatri", "mod",
              "The sharpest line is intervocalic l > r (present in the second set, absent in Hindi and Haryanvi), not West vs East Hindi. Old Awadhi is an older stage of Awadhi. Halbi and Bhatri side with Hindi and Odia. Intervocalic l > r also defines a block in the shared-prefix tree that cuts across Central, Bihari and Tharu and leaves Hindi–Urdu out.") +
         node("Central/Eastern Pahari: Kumaoni, Garhwali | Nepali, Dotyali", "firm", "ŋg/mb > ŋ/m at the MIA stage, v-loss, masc. -o. Central (Ku, Garh) vs Eastern (N, Dotyali) by the flap outcome, h-loss and n > ṇ.") +
         node("Eastern", "mod", "A continuum rather than a tree. Bengali's apocope post-dates Middle Bengali, so it is parallel to Assamese.",
              node("Bihari: Maithili–Magahi | Bhojpuri; western Tharu leans Awadhi/Bhojpuri", "mod", "Nasal + voiced stop assimilated early; compensatory lengthening before apocope; no Magadhan ś in the data.") +
              node("Odia (+ Desia)", "firm", "Keeps final -a, ṇ and nasal clusters.") +
              node("Bengali–Assamese; Kamarupi = Assamese, Kamta, Rajbanshi, Surjapuri, Hajong, Bishnupriya", "mod", "Kamarupi: ḍ > r rather than ṛ. Assamese alone: ś > x/h, c > s. Eastern Tharu, Danuwar and Majhi share the Bengali–Assamese NN outcome.")))
    ) + "</details>"

r2 = os.path.join(HERE, "round2.html")
r2_text = open(r2, encoding="utf-8").read() if os.path.exists(r2) else "<p>Round 2 is in progress.</p>"
findings = os.path.join(HERE, "findings2.html")
find_text = open(findings, encoding="utf-8").read() if os.path.exists(findings) else between('<ul class="findings">', '</ul>')

frag = {
    "DATE": "28 September 2026", "FINDINGS": find_text, "TREE": tree, "STAGED": staged, "R2_TEXT": r2_text,
    "GROUPS_HTML": groups, "NOVEL": novel, "SPECIAL": nur + rom, "CURATION": cur, "METHOD": meth + lim,

}
json.dump(frag, open(os.path.join(HERE, "fragments2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote fragments2.json")
