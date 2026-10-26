# Sound-change chains: OIA (CDIAL headwords) → every lect below, in one file.
# Format (sca.py, "Family files"): a line prefixed [A B] belongs only to those lects, [-C] to all
# but C, [fam] to a family (the %set lines; languages.tsv Group), [fam -C] to a family but C; an
# unprefixed line to every lect. A lect's chain is this file read with only its own lines, in order.
# Prefer widening a selector (or dropping it) to adding a near-duplicate line, so smaller lects and
# families do not miss shared changes.
%lects Mth Bi Bhoj Dang Rana Chitwan Morang DewasDoneDanuwar Kathoriya Majhi KochariyaEastDanuwar KochilaTharu MagahiNepal Bote Buksa Sunha Darai N Ku Garh eur as-ll Dotyali Lom RomWel RomBalk Zarg K Sh Kho Kal Phal Sv Kalk Ush Kund dod pog Mai Pas Bshk Dm Gaw Gowro Tor bhatr Shum Wot Tir Chil Or B A BH MH KS AdivasiOriya Hajong MB OB Bishnupriya H OAw Aw bundeli_atarra Brj bagheli_lakshman kaithal NDu SSr Kannauji-Central M OM Ko-Ul Bhili Bhilali RathwiBareli Bhatri mewari_bannoda Noiri PauriBareli PalyaBareli VB Vasavi hal Rathawi Khandesi DungraBhili Pk Pa Si Dhp Md NiDoc As-Ul OSi Ap KharI P S L OP awan kcch D srk thar poth khet Khol Wg Kam Kt Ash PNur Pr Gmb G OG OMarw Marw mewari_basad mewari_dholpura Goj had dhundari_badagaon Nimadi bagri_fatehabad dhundari_bamore mewati_akera marwari_bagra hadothi_kelwada mewati_jhambaus Rj WK ktg kul bhal kiuth WPah jaun bhad dog kc mand khash kvar cur kgr cam pan
%set bihari-tharu = Mth Bi Bhoj Dang Rana Chitwan Morang DewasDoneDanuwar Kathoriya Majhi KochariyaEastDanuwar KochilaTharu MagahiNepal Bote Buksa Sunha Darai
%set central-east-pahari-romani = N Ku Garh eur as-ll Dotyali Lom RomWel RomBalk Zarg
%set dardic-north = K Sh Kho Kal Phal Sv Kalk Ush Kund dod pog
%set dardic-south = Mai Pas Bshk Dm Gaw Gowro Tor bhatr Shum Wot Tir Chil
%set eastern = Or B A BH MH KS AdivasiOriya Hajong MB OB Bishnupriya
%set hindi = H OAw Aw bundeli_atarra Brj bagheli_lakshman kaithal NDu SSr Kannauji-Central
%set marathi-bhil = M OM Ko-Ul Bhili Bhilali RathwiBareli Bhatri mewari_bannoda Noiri PauriBareli PalyaBareli VB Vasavi hal Rathawi Khandesi DungraBhili
%set mia-insular = Pk Pa Si Dhp Md NiDoc As-Ul OSi Ap KharI
%set northwest = P S L OP awan kcch D srk thar poth khet Khol
%set nuristani = Wg Kam Kt Ash PNur Pr Gmb
%set rajasthani-gujarati = G OG OMarw Marw mewari_basad mewari_dholpura Goj had dhundari_badagaon Nimadi bagri_fatehabad dhundari_bamore mewati_akera marwari_bagra hadothi_kelwada mewati_jhambaus Rj WK
%set west-pahari = ktg kul bhal kiuth WPah jaun bhad dog kc mand khash kvar cur kgr cam pan

[P] # OIA (CDIAL headwords) → Punjabi (CDIAL "P.")
[L] # OIA (CDIAL headwords) → Lahnda (CDIAL "L.")
[awan] # OIA (CDIAL headwords) → Hindko (CDIAL "awan." Awankari + Rensch et al. 1992 Hindko)
[poth] # OIA (CDIAL headwords) → Pahari-Pothwari (Lothers & Lothers 2010 + CDIAL + Gill)
[srk] # OIA (CDIAL headwords) → Saraiki (CDIAL "srk." + Rensch et al. 1992 + Gill)
[S] # OIA (CDIAL headwords) → Sindhi (CDIAL "S.")
[kcch] # OIA (CDIAL headwords) → Kutchi (CDIAL "kcch.")
[OP] # OIA (CDIAL headwords) → Old Punjabi (Shackle's Guru Granth Sahib glossary; CDIAL "OP.")
[khet] # OIA (CDIAL headwords) → Khetrani (CDIAL "khet." + Elfenbein + Grierson LSI)
[thar] # OIA (CDIAL headwords) → Dhatki (Thari; "thar", Tharparkar)
[Khol] # OIA (CDIAL headwords) → Kholosi ("khol"; Sindhic lect of Hormozgan, Iran)
[D] # OIA (CDIAL headwords) → Domaaki (CDIAL "D." [Lorimer] + Backstrom 1992 + Weinreich 2008)
[Marw] # OIA (CDIAL headwords) → Marwari (CDIAL Marw. + marwari/mewari surveys)
[had] # OIA (CDIAL headwords) → Hadauti (hadothi + mewari surveys)
[dhundari_badagaon] # OIA (CDIAL headwords) → Shekhawati (dhundari + marwari surveys, Badagaon)
[bagri_fatehabad] # OIA (CDIAL headwords) → Bagri (bagri survey, Fatehabad)
[mewari_dholpura] # OIA (CDIAL headwords) → Mewari (mewari survey, Dholpura)
[marwari_bagra] # OIA (CDIAL headwords) → Godwari (marwari survey, Bagra)
[mewati_akera] # OIA (CDIAL headwords) → Mewati (mewati survey, Akera)
[mewari_basad] # OIA (CDIAL headwords) → Malvi (mewari_basad: varghese-john-samuel2009malvi + mewari survey)
[Goj] # OIA (CDIAL headwords) → Gujari (Rensch et al. 1992 + Hugoniot et al. 2023 + Noira; digraph sources via %source block)
[dhundari_bamore] # OIA (CDIAL headwords) → Dhundari (dhundari survey, Bamore)
[Nimadi] # OIA (CDIAL headwords) → Nimadi (Vunnamatla et al. 2012 + Bareli, the latter via a digraph %source block)
[hadothi_kelwada] # OIA (CDIAL headwords) → Shahawadi (hadothi survey, Kelwada)
[Rj] # OIA (CDIAL headwords) → Rajasthani (lect "Rj": in fact the mewari survey data)
[mewati_jhambaus] # OIA (CDIAL headwords) → Ahirwadi (mewati survey, Jhambaus)
[G] # OIA (CDIAL headwords) → Gujarati (CDIAL "G.")
[OMarw] # OIA (CDIAL headwords) → Old Marwari (CDIAL "OMarw.", Old Western Rajasthani)
[OG] # OIA (CDIAL headwords) → Old Gujarati (CDIAL "OG.", Old Western Rajasthani)
[WK] # OIA (CDIAL headwords) → Wadiyara Koli (zubair; Gujarati-type chain, see OIA-G.sc)
[bundeli_atarra] # Sound-change chains, family hindi: OIA (CDIAL headwords) → each lect below.
[Ash] # Sound-change chains, family nuristani: OIA (CDIAL headwords) → each lect below.
[Phal] # Sound-change chains, family dardic-north: OIA (CDIAL headwords) → each lect below.
[ktg] # Sound-change chains, family west-pahari: OIA (CDIAL headwords) → each lect below.
[PauriBareli RathwiBareli] # Sound-change chains, family marathi-bhil: OIA (CDIAL headwords) → each lect below.
[Phal bundeli_atarra PauriBareli RathwiBareli Ash ktg] # One file for the family (sound_changes/families.py; format in sca.py, "Family files"): a line
[Phal bundeli_atarra PauriBareli RathwiBareli Ash ktg] # prefixed [A B] belongs only to those lects, [-C] to all but C; an unprefixed line to every lect.
[Phal bundeli_atarra PauriBareli RathwiBareli Ash ktg] # A lect's chain is this file read with only its own lines, in order. Prefer widening a selector
[Phal bundeli_atarra PauriBareli RathwiBareli Ash ktg] # (or dropping it) to adding a near-duplicate line, so smaller lects do not miss shared changes.
[bundeli_atarra] # OIA (CDIAL headwords) → Bundeli (Aṭarrā; bundeli + kannauji survey lists)
[bagheli_lakshman] # OIA (CDIAL headwords) → Bagheli (Lakshman; koshy2022bagheli + bundeli survey lists)
[Brj] # OIA (CDIAL headwords) → Braj (CDIAL "Brj." + the bundeli survey list's Braj forms)
[Kannauji-Central] # OIA (CDIAL headwords) → Kannauji (central; kannauji survey list)
[NDu] # OIA (CDIAL headwords) → Chhattisgarhi (chattisgarhi survey list)
[SSr] # OIA (CDIAL headwords) → Surgujia (chattisgarhi survey list)
[H] # OIA (CDIAL headwords) → Hindi-Urdu (CDIAL "H.")
[Aw] # OIA (CDIAL headwords) → Awadhi (CDIAL "Aw.", mostly Saksena's Lakhīmpurī Awadhi)
[kaithal] # OIA (CDIAL headwords) → Haryanvi (Kaithal; webster2024haryanvi list + gill + CDIAL)
[OAw] # OIA (CDIAL headwords) → Old Awadhi (CDIAL "OAw.": Tulsīdās / Jāyasī spellings)
[Noiri] # OIA (CDIAL headwords) → Noiri (varghesekumar2015noira, watters2013northerndhule)
[Bhili] # OIA (CDIAL headwords) → Bhili (varkey-vunnamatla2018bareli, pattanaik-koul2003varli (Varli))
[PalyaBareli] # OIA (CDIAL headwords) → Palya Bareli (varkey-vunnamatla2018bareli)
[Bhilali] # OIA (CDIAL headwords) → Bhilali (varkey-vunnamatla2018bareli)
[Vasavi] # OIA (CDIAL headwords) → Vasavi (watters2013northerndhule)
[Rathawi] # OIA (CDIAL headwords) → Rathawi (varkey-vunnamatla2018bareli)
[RathwiBareli] # OIA (CDIAL headwords) → Rathwi Bareli (varkey-vunnamatla2018bareli, watters2013northerndhule)
[PauriBareli] # OIA (CDIAL headwords) → Pauri Bareli (varkey-vunnamatla2018bareli, watters2013northerndhule)
[Khandesi] # OIA (CDIAL headwords) → Khandesi (varkey-vunnamatla2018bareli)
[DungraBhili] # OIA (CDIAL headwords) → Dungra Bhil (varghesekumar2015noira)
[mewari_bannoda] # OIA (CDIAL headwords) → Wagri (mewari_bannoda) (mewari)
[M] # OIA (CDIAL headwords) → Marathi (CDIAL "M.")
[Ko-Ul] # OIA (CDIAL headwords) → Konkani (CDIAL "Ko.", file stem Ko-Ul)
[VB] # OIA (CDIAL headwords) → Vaagri Boli (srinivasa)
[hal] # OIA (CDIAL headwords) → Halbi (beine2017bhatri, woods2019halbi, CDIAL)
[Bhatri] # OIA (CDIAL headwords) → Bhatri (beine2017bhatri)
[OM] # OIA (CDIAL headwords, plus Tulpule's √roots) → Old Marathi (tulpule1999 + CDIAL "OM.")
[kiuth] # OIA (CDIAL headwords) → Kiunthali (West Pahari)
[kul] # OIA (CDIAL headwords) → Kullui (West Pahari)
[jaun] # OIA (CDIAL headwords) → Jaunsari (West Pahari)
[bhad] # OIA (CDIAL headwords) → Bhadrawahi (West Pahari)
[kc] # OIA (CDIAL headwords) → Kochi (West Pahari)
[ktg] # OIA (CDIAL headwords) → Kotgarhi (West Pahari; CDIAL "kgh.", Hendriksen's transcription)
[WPah] # OIA (CDIAL headwords) → West Pahari (unspecified) (West Pahari)
[khash] # OIA (CDIAL headwords) → Khashali (West Pahari)
[bhal] # OIA (CDIAL headwords) → Bhalesi (West Pahari)
[cur] # OIA (CDIAL headwords) → Churahi (West Pahari)
[mand] # OIA (CDIAL headwords) → Mandeali (West Pahari)
[dog] # OIA (CDIAL headwords) → Dogri (West Pahari)
[pan] # OIA (CDIAL headwords) → Pangwali (West Pahari)
[kgr] # OIA (CDIAL headwords) → Kangri (West Pahari)
[cam] # OIA (CDIAL headwords) → Chambeali (West Pahari)
[kvar] # OIA (CDIAL headwords) → Kvari (West Pahari)
[Dang] # OIA (CDIAL headwords) → Dangaura Tharu (Nepal survey wordlists: boehm, webster, mitchell-eichentopf2013tharu; per-source notation blocks at the end)
[Kathoriya] # OIA (CDIAL headwords) → Kathoriya Tharu (Nepal survey wordlists: webster, mitchell-eichentopf2013tharu)
[Rana] # OIA (CDIAL headwords) → Rana Tharu (Nepal survey wordlists: boehm, webster)
[Buksa] # OIA (CDIAL headwords) → Bhuksa Tharu (webster)
[Chitwan] # OIA (CDIAL headwords) → Chitwania Tharu (Nepal survey wordlists: boehm, webster)
[Sunha] # OIA (CDIAL headwords) → Sunha Tharu (webster)
[Morang] # OIA (CDIAL headwords) → Morangia Tharu (Nepal survey wordlist: boehm)
[Bi] # OIA (CDIAL headwords) → Bihari (CDIAL "Bi.", Grierson Bihar Peasant Life)
[Mth] # OIA (CDIAL headwords) → Maithili (CDIAL "Mth.")
[Bhoj] # OIA (CDIAL headwords) → Bhojpuri (CDIAL "Bhoj.")
[DewasDoneDanuwar] # OIA (CDIAL headwords) → Dewas-Done Danuwar (shackelford2019dewas-rai; regmi-thakur2016danuwar via a %source notation block)
[KochariyaEastDanuwar] # OIA (CDIAL headwords) → Kochariya-East Danuwar (shackelford2019dewas-rai; regmi-thakur2016danuwar via a %source notation block)
[MagahiNepal] # OIA (CDIAL headwords) → Magahi, Nepal (thakur-thakur2016magahi)
[KochilaTharu] # OIA (CDIAL headwords) → Kochila Tharu (eichentopf-mitchell2020kochila)
[Majhi] # OIA (CDIAL headwords) → Majhi (page2024majhi-bote; chalise2014majhi via a %source notation block)
[Bote] # OIA (CDIAL headwords) → Bote (page2024majhi-bote)
[Darai] # OIA (CDIAL headwords) → Darai (dhakal2011darai)
[KS] # OIA (CDIAL headwords) → Surjapuri (toulmin "KS")
[MH] # OIA (CDIAL headwords) → Rajbanshi (toulmin "MH")
[BH] # OIA (CDIAL headwords) → Kamtapuri (toulmin "BH")
[A] # OIA (CDIAL headwords) → Assamese (CDIAL "A.")
[Hajong] # OIA (CDIAL headwords) → Hajong (kim-ahmad-kim-sangma2011hajong)
[B] # OIA (CDIAL headwords) → Bengali (CDIAL "B.")
[Bishnupriya] # OIA (CDIAL headwords) → Bishnupriya Manipuri (kim-kim2008bishnupriya)
[Or] # OIA (CDIAL headwords) → Odia (CDIAL "Or.")
[AdivasiOriya] # OIA (CDIAL headwords) → Adivasi Oriya / Desia (behera2021desia, beine2017bhatri)
[MB] # OIA (CDIAL headwords) → Middle Bengali (CDIAL "MB.")
[OB] # OIA (CDIAL headwords) → Old Bengali (CDIAL "OB.", Caryāpada etc.)
[Pk] # OIA (CDIAL headwords) → Prakrit (CDIAL "Pk.", mostly Mahārāṣṭrī / Ardhamāgadhī; + Sheth 1923)
[Ap] # OIA (CDIAL headwords) → Apabhraṁśa (CDIAL "Ap." + Sheth 1923)
[Ap] # Copy of the Prakrit chain (OIA-Pk.sc); rules with negative leave-one-out value on Ap. removed
[Ap] # (no ya-śruti, no r̩ > a before dentals). Only 157 etyma: treat per-rule values as noise.
[Pa] # OIA (CDIAL headwords) → Pali (CDIAL "Pa.")
[As-Ul] # OIA (CDIAL headwords) → Ashokan Prakrit (CDIAL "As."; file stem As-Ul)
[Md] # OIA (CDIAL headwords) → Dhivehi / Maldivian (CDIAL "Md." + Fritz)
[Si] # OIA (CDIAL headwords) → Sinhala (CDIAL "Si.")
[OSi] # OIA (CDIAL headwords) → Old Sinhala (CDIAL "OSi." + Paranavitana, Sigiri graffiti glossary)
[Dhp] # OIA (CDIAL headwords) → Gāndhārī (Dhp: gandhari dictionary + CDIAL "Dhp.")
[NiDoc] # OIA (CDIAL headwords) → Niya Prakrit (CDIAL "NiDoc.", Kharoṣṭhī documents from Niya)
[NiDoc] # Copy of the Gāndhārī chain (OIA-Dhp.sc) with Niya deviations: t not voiced (-ti), r + C kept,
[NiDoc] # no tʰ > s, no c/j > y, Stop + y/v/l/r kept; Niya's special letters for lenited stops written with @notation rules at the end.
[KharI] # OIA (CDIAL headwords) → Kharoṣṭhī inscriptions (CDIAL "KharI.")
[KharI] # The Gāndhārī chain (OIA-Dhp.sc) minus everything the inscriptions do not show: no intervocalic
[KharI] # lenition, r + C and ṇ kept, sv not > sp. What is left is the NW MIA cluster core + Kharoṣṭhī spelling.
[Kt] # OIA (CDIAL headwords) → Kt (specialisation of the OIA-PNur chain)
[Kam] # OIA (CDIAL headwords) → Kam (specialisation of the OIA-PNur chain)
[Wg] # OIA (CDIAL headwords) → Wg (specialisation of the OIA-PNur chain)
[Ash] # OIA (CDIAL headwords) → Ash (specialisation of the OIA-PNur chain)
[PNur] # OIA (CDIAL headwords) → Proto-Nuristani (Strand's reconstructions, "PNur")
[Pr] # OIA (CDIAL headwords) → Pr (specialisation of the OIA-PNur chain)
[Gmb] # OIA (CDIAL headwords) → Gmb (specialisation of the OIA-PNur chain)
[bihari-tharu eastern hindi marathi-bhil mia-insular northwest nuristani rajasthani-gujarati west-pahari] #
[P] # Northwest (NW) group base chain; the other NW files copy it. MIA stage close to OIA-H.sc, then
[P] # the NW NIA changes: geminates kept after short vowels (no compensatory lengthening), post-nasal
[P] # voicing, intervocalic n > ṇ and s > h, rt > ṭṭ, -ṇā/-nā infinitives. Notes: notes/northwest.md.
[P] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-P.sc --loo
[P] %exclude-source bagri                        ; Bagri (Rajasthani, with compensatory lengthening: dugdʰa > dūdʰ) mis-filed under P.
[L] # Copy of the NW base chain (OIA-P.sc). Lahnda departs from P in ava > o (not au), no initial v > b,
[L] # -aṇ infinitive. Implosives (28% of initial g j ḍ d b) and dental+r retention are too sporadic in
[L] # CDIAL L. to model. Notes: notes/northwest.md.
[L] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-L.sc --loo
[awan] # Copy of the NW base chain (via OIA-L.sc). Hindko: -uṇ infinitive, s > h > ∅ and h > ∅ (tone),
[awan] # ō written with macron. Rensch et al.'s phonetic forms get a %source block (no written gemination,
[awan] # tonal devoicing). Notes: notes/northwest.md.
[awan] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-awan.sc --loo
[S] # Copy of the NW base chain. Sindhi: final short vowels kept (-u, -o, -ī, fem. -a), dental + r kept
[S] # (> ṭr ḍr), initial and geminate voiced stops > implosives, intervocalic l > r, m > nasalisation,
[S] # and full degemination without lengthening, AFTER those lenitions. Notes: notes/northwest.md.
[S] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-S.sc --loo
[rajasthani-gujarati] # Group chain for rajasthani-gujarati: an MIA block (as OIA-H.sc), then the NIA block
[rajasthani-gujarati] # shared with Old Gujarati / Old Marwari / the Rajasthani survey lects.
[rajasthani-gujarati] # Scoring keeps ɔ ɛ distinct from o e (round 7): final -ɔ and the G hiatus ɔ/ɛ are derived by rule.
[Phal S awan Marw OMarw] %verb-menu +aka                            ; root + -aka verbal noun (coordinator placebo: −0.003/−0.005); round 6
[poth] # Copy of the NW base chain (via OIA-L.sc). Pothwari: initial voiced aspirates devoiced (tone),
[poth] # s > h > ∅, h > ∅, ava > o, v > b, m > nasalisation. Lothers/Gill/Liljegren %source blocks:
[poth] # no written gemination; Lothers writes high vowels long. Notes: notes/northwest.md.
[poth] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-poth.sc --loo
[srk] # Copy of the NW base chain (via OIA-L.sc). Saraiki: initial d > ḍ, m > nasalisation (as in S);
[srk] # Rensch/Gill forms get a no-gemination %source block. Notes: notes/northwest.md.
[srk] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-srk.sc --loo
[kcch] # Copy of the NW base chain. Kutchi: -ṇū infinitive, masc. -o, full degemination (like Sindhi),
[kcch] # l > r, initial d > ḍ, post-nasal aspiration, u- > au-; no s > h, aspirates not > h.
[OP] # Copy of the NW base chain. Old Punjabi of the Guru Granth Sahib (Shackle): final -u/-i kept, verbs
[OP] # cited as stem + -i, anusvāra ṁ written, no s > h. `shackle-auto` (auto-linked, noisier) is kept.
[OP kcch] # Notes: notes/northwest.md.
[kcch] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-kcch.sc --loo
[OP] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-OP.sc --loo
[khet] # Copy of the NW base chain (via OIA-L.sc). Khetrani: dental + r kept, dv > b, ava > o;
[khet] # Elfenbein's forms record initial implosives (%source block). Notes: notes/northwest.md.
[khet] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-khet.sc --loo
[thar] # Built from the NW base chain but behaves like a Rajasthani/Gujarati lect: no s > h, no l > r,
[thar] # anusvāra-type nasal clusters, full degemination without lengthening, masc. -o. Notes: notes/northwest.md.
[thar] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-thar.sc --loo
[Khol] # Built from the NW base chain. Kholosi (Iran): -en infinitive, masc. -o, geminates kept, plus
[Khol] # Iranian-contact changes (kʰ > x, voiced aspirates > voiceless aspirates, de-retroflexion, s > z,
[Khol] # cʰ > ś, i u > e o). Notes: notes/northwest.md.
[Khol] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Khol.sc --loo
[D] # Built from the NW base chain but Domaaki patterns with Shina: all three sibilants kept, tr dr >
[D] # ʦ̣ ʣ̣, NC > N, no ṇ, voiced aspirates deaspirated, final vowels kept. %source blocks for
[D] # Weinreich (doubled long vowels) and Backstrom (length marks). Notes: notes/northwest.md.
[D] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-D.sc --loo
[D] %verb-menu +iyanaka                             ; passive -iya- + -anaka (round 7: replaces +iyanakam once -akaṁ > -o; with -akaṁ > -o in place, verb −0.0051 / 0.0000 vs placebo +iranaka +0.0157 / +0.0278)
[rajasthani-gujarati -OG -WK -mewati_jhambaus -G -OMarw] %ext-menu +akā ; masc. pl. -akāḥ > -ā (survey plural citations kāndā, āmbā); beats placebo +arā on train and test
[bundeli_atarra] # Group chain + l > r; survey verbs cited as bare stems. Both survey lists write ṛ as ḍ and N+C for
[bundeli_atarra] # nasal vowel + stop; their notation is in %source blocks. CL, m > ṽ and syncope prune away (LOO < 0).
[bagheli_lakshman] # Group chain; Koshy's list writes N+C clusters and rarely marks nasality, so the anusvāra rules prune
[bagheli_lakshman] # away. Koshy notation (dentals with the retroflex dot, schwa as e, length unmarked) and bundeli
[bagheli_lakshman] # notation are in %source blocks.
[Brj] # Group chain + l > r (Brj. kāro, pīro), -aka > -o (@hiatus), flap r ordered before degemination,
[Brj] # verbs cited in -no (verb menu). `%source bundeli` holds that list's notation.
[Kannauji-Central] # Group chain; roots cited as imperatives in -o. m > ṽ, ava > o and syncope prune away.
[Kannauji-Central] # Notation (kh/ch digraphs for voiceless aspirates, ṛ written ḍ, long u, N+C) is in a %source block.
[NDu SSr] # Group chain + l > r; flap ṛ. The list writes high vowels long and N+C clusters (%source block).
[H] # Group template for the hindi group (see sound_changes/notes/hindi.md). MIA block, then NIA block.
[H] # Standard H keeps intervocalic l (no `l > r`) and flaps single ḍ to ṛ after degemination.
[H] # @morph rules: -ati > -nā infinitive and in-stems > -ī (the other citation rules were removed in round 7).
[Aw] # Group chain with the eastern/central choices (l > r before degemination, flap r); no syncope.
[Aw] # Verbs cited as -ab infinitives (verb menu; the @morph -ab rules were removed in round 7, neutral); CL also before -ab (várdʰatē > bāṛʰab).
[kaithal] # Group chain without l > r, ava > o, m > ṽ; adds intervocalic n > ṇ and final h loss (tone).
[kaithal] # Webster notation (ṛ written ḍ, high vowels short) is in a %source block.
[OAw] # Group chain with the eastern/central choices (l > r before degemination, flap r).
[OAw] # OAw spelling keeps final short vowels, vowels in hiatus and medial a (no final-vowel loss, no hiatus
[OAw] # contraction, no syncope); CL and anusvāra-lengthening only in the stressed penult (-ai counts as one
[OAw] # syllable); nasal vowels written V+ṁ (@notation). Verbs cited in 3sg -ai (verb menu; the @morph -ai and in-stem rules were removed in round 7, neutral).
[hindi] # Tags: sound_changes/CATALOG.md; @x-hindi-* tags are defined in sound_changes/notes/hindi.md.
[bundeli_atarra] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-bundeli_atarra.sc --loo
[bagheli_lakshman] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-bagheli_lakshman.sc --loo
[Brj] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Brj.sc --loo
[eur] # OIA (CDIAL headwords) → European Romani (CDIAL "Eur." + boretzky1994romani)
[RomBalk] # OIA (CDIAL headwords) → Balkan Romani (CDIAL "Rom. Balk.")
[Zarg] # OIA (CDIAL headwords) → Zargari Romani (rezaibaghbidi2003zargari)
[eur RomBalk Zarg] # Group chain for the migratory lects (Romani, Domari, Lomavren, Zargari): a reduced MIA block
[RomWel] # OIA (CDIAL headwords) → Welsh Romani (CDIAL "Wel.")
[RomWel RomBalk] %verb-menu +āpaya   ; per-lect verb formation: bare causative stem -āpaya- > -av (ulav, ladav, pusav); beats the placebo -āraya on train (.3033 vs .3097) and test (.4057 vs .4390)
[as-ll] # OIA (CDIAL headwords) → Domari (CDIAL "Pal. Gy." / herin2012domari)
[as-ll] %exclude-source herin2012domari   ; inflected paradigm forms with morpheme hyphens (ǧān-d-ōm, py-əme …), 40 of 65 rows are two verbs
[Lom] # OIA (CDIAL headwords) → Lomavren (CDIAL "Arm." Gypsy)
[as-ll RomWel Lom] # Group chain for the migratory lects (Romani, Domari, Lomavren, Zargari): a reduced MIA block
[central-east-pahari-romani -Ku -Garh -N -Dotyali] # (dental + r clusters kept, s + stop kept, t d > l, ṭ ḍ > r, aspiration shift), then
[eur] # Romani-specific changes. No gemination stage: clusters simplify directly.
[RomBalk] # Balkan Romani-specific changes. No gemination stage: clusters simplify directly.
[Zarg] # Zargari-specific changes. No gemination stage: clusters simplify directly.
[RomWel] # Welsh Romani-specific changes. No gemination stage: clusters simplify directly.
[as-ll] # Domari-specific changes. No gemination stage: clusters simplify directly.
[Lom] # Lomavren-specific changes. No gemination stage: clusters simplify directly.
[Ku] # OIA (CDIAL headwords) → Kumaoni (CDIAL "Ku.")
[N] # OIA (CDIAL headwords) → Nepali (CDIAL "N.")
[Garh] # OIA (CDIAL headwords) → Garhwali (CDIAL "Garh.")
[Dotyali] # OIA (CDIAL headwords) → Dotyali (eichentopf-tupper2019dotyali)
[Dotyali Kannauji-Central rajasthani-gujarati -OG -WK -Nimadi -hadothi_kelwada -mewati_jhambaus -mewati_akera -G -OMarw] %verb-menu +itaka                            ; past participle -ita + -aka > -iyo/-yo (māryo, dekʰyɔ); beats placebo +iraka
[OG G] %ext-menu +iyakam                            ; neuter -iyakam > -iyũ (G) / -iyauṁ (OG); beats placebo +irakam on train and test
[OG G] %ext-menu +aḍakam                            ; neuter -aḍakam > -ṛũ (jʰupṛũ, cavṛũ); beats placebo +arakam on train and test
[OG G] %ext-menu +allakam                           ; neuter -allakam > -lũ (bokʰlũ, dʰāplũ); beats placebo +arakam on train and test
[Kannauji-Central] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Kannauji-Central.sc --loo
[NDu] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-NDu.sc --loo
[SSr] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-SSr.sc --loo
[NDu bagheli_lakshman SSr] %verb-menu +itasya                           ; participle -ita + enclitic -sya > the E. Hindi 3sg past -is (dekʰīs, marīs, jarīs); round 6c, placebo +itapya has no effect
[H] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-H.sc --loo
[H] %exclude-source kannauji        ; Kannauji-dialect survey forms filed under H (digraph kh/ch, -o endings)
[Bshk] # OIA (CDIAL headwords) → Bashkarik / Gawri (Bshk.; CDIAL, Gawri dictionary, RDH 1992)
[Mai] # OIA (CDIAL headwords) → Indus Kohistani (Mai.; 80% Zoller 2005, also CDIAL and RDH 1992)
[Gowro] # OIA (CDIAL headwords) → Gowro (80% Zoller 2005 with tone marks; RDH 1992, CDIAL)
[bhatr] # OIA (CDIAL headwords) → Bhateri (Zoller 2005, Strand, RDH 1992)
[Tor] # OIA (CDIAL headwords) → Torwali (Tor.; CDIAL, RDH 1992)
[Tir] # OIA (CDIAL headwords) → Tirahi (Tir.; CDIAL only)
[Pas] # OIA (CDIAL headwords) → Pashai (Paš.; CDIAL dialect forms, Strand, Lehr)
[Ku Garh N Dotyali Pas] %ext-menu +āka ; per-lect noun extension -āka (MIA -āa > -ā: the -ā nouns kaṛā, kʰājā beside -o < -aka); beats the placebo -āra on train (ext .2913 vs .2951, base .2958) and test (.2802 vs .2873, base .2873); round 7
[Ku Garh N] %exclude-source zoller2023   ; digraph aspirates (kh, bh, ph) and ə — a different notation from the CDIAL majority; few rows
[Ku N] %verb-menu +iyanakam +aka   ; per-lect verb formations: passive -iyanakam (N -inu, Ku -iṇo) and root + -aka verbal noun; both beat same-shape placebos
[Garh H] %verb-menu +aka                              ; root + -aka verbal noun (coordinator item). Round 6b gates: verb-menu NED 0.1665 → 0.1624 train, 0.1631 → 0.1579 test; ext 0.2314 → 0.2306 / 0.2358 → 0.2348; placebo +ara only −0.0003 / −0.0002
[Aw] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Aw.sc --loo
[kaithal] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-kaithal.sc --loo
[OAw] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-OAw.sc --loo
[marathi-bhil -Bhatri -hal -M -OM -Ko-Ul] # Group chain (marathi-bhil): the MIA block of OIA-M.sc (lect-pruned), then a Bhil NIA block
[marathi-bhil -Bhatri -hal -M -OM -Ko-Ul] # (template: tmp/sound_changes/marathi-bhil/bhil.sc) pruned/extended on this lect's data.
[Noiri] # Bhil type: -o (< -aka, via the extension menu; round 7: citation -o @morph rule removed), s > h, c > s, ṇ > n; varghese writes dental t d with the underdot and no long i.
[Bhili] # Bhil type: Gujarati-like masc. -o, s > h, ṇ > n, NC kept with a > ā; varkey writes ī ū for short i u.
[PalyaBareli] # Bhil type: -o (< -aka, via the extension menu; round 7: citation -o @morph rule removed), s > h; varkey writes ī ū for short i u.
[Bhilali] # Bhil type but masc. -ā (Nimadi-like), s kept, cʰ > s.
[Vasavi] # Bhil type: -o (< -aka, via the extension menu; round 7: citation -o @morph rule removed), s > h, v > b; watters writes no i/u length.
[Rathawi] # Bhil type: -o, s > h, ṇ > n.
[RathwiBareli] # Bhil type: masc. -o (from -aka via the extension menu; round 7: the -o citation @morph rule removed), s kept; mixed varkey + watters notation.
[PauriBareli] # Bhil type: -o (< -aka, via the extension menu; round 7: citation -o @morph rule removed), s > h, Marathi-like ts/dz affrication before non-front vowels.
[Khandesi] # Khandesi: no -o (masc. -ā forms are not predicted), s kept, no ḷ; varkey writes ī ū for short i u.
[DungraBhili] # Bhil type: -o (< -aka, via the extension menu; round 7: citation -o @morph rule removed), s > h, c > s, voiced aspirates lost; varghese notation (dental underdot, no long i).
[mewari_bannoda] # Wagri (mewari source): -o > -ā, s > h, c/cʰ > s, ṇ > n.
[M] # Group chain (marathi-bhil): MIA block shared with OIA-H.sc, then the Marathi NIA block.
[VB] # Vaagri Boli (srinivasa): noisy; epenthetic final ɨ after consonants (Dravidian-contact area).
[Bhatri hal] # Group chain (marathi-bhil): the MIA block of OIA-M.sc (lect-pruned), then a Halbic NIA block
[Bhatri hal] # (eastern-type: ḍ > ṛ, v > b everywhere, ṇ > n, no ḷ, s kept, no -o/-ṇẽ morphology).
[hal] # Woods's Halbi forms show compensatory lengthening (Beine writes a for ā and ə for short a, handled
[hal] # by a %source block); aspiration retained.
[Bhatri] # Short a > ə (round 7); Beine's a for ā and unmarked i/u length are a %source notation block; aspiration is
[Bhatri] # largely lost (@deaspiration) and intervocalic h dropped.
[west-pahari -ktg] # Specialised copy of the West Pahari base chain in OIA-ktg.sc (see notes/west-pahari.md).
[kiuth] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-kiuth.sc --loo
[kul] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-kul.sc --loo
[kul] # Sources: kullui-org (main notation) + blair2021kullu (village survey; mapped with the
[kul] # %source block at the end of the file).
[jaun] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-jaun.sc --loo
[jaun] # Sources: CDIAL (main notation) + john2008jaunsari (digraph aspirates, ç, no length; mapped with
[jaun] # the %source block at the end of the file).
[bhad] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-bhad.sc --loo
[kc] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-kc.sc --loo
[kc bhad] %verb-menu +iyanakam                        ; passive -iya- + -anakam verbal noun (coordinator placebo test vs -iranakam; round 6)
[ktg] # West Pahari base chain: MIA block (shared with OIA-H.sc) then the NIA / West Pahari block.
[Bishnupriya BH Hajong MH KS] # Eastern (Magadhan) chain, copied from OIA-B.sc and specialised.
[KS] # Specialised: ḍ > r (as A.), no nasality written, word-final deaspiration; -i -u lost as well as -a.
[MH] # Specialised: ḍ > r (as A.), no nasality written; unlike BH no c > s and no a > ɔ in toulmin's MH notation.
[BH] # Specialised: A.-type NC treatment is NOT used (B.-type: nasal vowel, ŋg mb > ŋ m); ḍ > r, c j > s z
[BH] # and ɔ-rounding as in the Kamarupi lects; toulmin writes open /a/ as ā and no vowel length, writes ś, and
[BH] # marks little nasality; initial l > n; non-initial deaspiration. Toulmin data are noisy (phrases, suffixes -ʸæ).
[A] # Eastern (Magadhan) chain, copied from OIA-B.sc and specialised: ś > x/h, c j > s z,
[A] # ḍ > r, verbs cited as -iba infinitives.
[Hajong] # Specialised: no vowel length (a/ā merged), A.-type NC clusters, initial ś > h (variable in the survey),
[Hajong] # ḍ > r, a > ɔ before apocope, final deaspiration, loss of retroflexion (Tibeto-Burman contact).
[B] # Eastern (Magadhan) chain. MIA block as in OIA-H.sc but with the Magadhan ś-merger, then the
[B] # NIA changes shared by the eastern group, then Bengali specifics.
[B] # The kim-* sociolinguistic-survey word lists (colloquial Bengali elicited in the Garo and Tripura
[B] # surveys) are phonetic, not Turner's orthographic transliteration: they get %source @notation blocks
[B] # at the end. Excluded: zoller2023 (Bangani forms mislinked to B.).
[Bishnupriya] # Specialised: no vowel length, initial s > h, ḍ > r, a > ɔ before apocope, final deaspiration, ṭ > t;
[Bishnupriya] # the kim-kim survey writes /i/ as ī (@notation).
[Or] # Eastern (Magadhan) chain, copied from OIA-B.sc and specialised: nasal + stop clusters kept,
[Or] # final -a kept, intervocalic n l > ṇ ḷ, verbs cited as -ibā infinitives.
[Or] # Excluded: beine2017bhatri (Bhatri survey list; ə/ɔ letters, syllable dots `no.kʰo`, r written ṛ).
[AdivasiOriya] # Eastern (Magadhan) chain, copied from OIA-Or.sc and specialised.
[AdivasiOriya] # Specialised from OIA-Or.sc: Desia drops final -a (unlike standard Odia), no ḷ, a > ɔ, ā > a (no length);
[AdivasiOriya] # behera writes plain i u as ī ū (@notation).
[AdivasiOriya] # Excluded: beine2017bhatri (Bhatri survey list with a different notation: o for ɔ, plain i u,
[AdivasiOriya] # syllable dots); the file models behera2021desia (Desia), whose ɔ / a / ī ū letters are handled
[AdivasiOriya] # by the final @notation and @a-rounding rules.
[MB OB] # The Bengali chain (OIA-B.sc) stopped at an earlier stage: final vowels and ṇ still there,
[MB OB] # verbs cited in the 3sg (-ai).
[MB] # Specialised: like OB (no hiatus contraction, no apocope, NC clusters kept, ṇ kept) but 3sg -ai > -e (its @morph rule was removed in round 7; verbs take the verb-menu forms).
[OB] # Specialised (earlier stage): no -a apocope, ṇ kept, NC clusters kept with a > ā, hiatus not yet contracted.
[eastern marathi-bhil -OM -Ko-Ul ktg] # Score with
[Noiri] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-Noiri.sc --loo
[Bhili] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-Bhili.sc --loo
[PalyaBareli] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-PalyaBareli.sc --loo
[Bhilali] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-Bhilali.sc --loo
[Vasavi] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-Vasavi.sc --loo
[Rathawi] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-Rathawi.sc --loo
[RathwiBareli] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-RathwiBareli.sc --loo
[PauriBareli] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-PauriBareli.sc --loo
[Khandesi] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-Khandesi.sc --loo
[DungraBhili] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-DungraBhili.sc --loo
[mewari_bannoda] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-mewari_bannoda.sc --loo
[M] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-M.sc --loo
[Ko-Ul] # Group chain (marathi-bhil): the OIA-M.sc chain with Konkani deviations (verbs cited in the
[Ko-Ul] # present -tā, final vowels kept but shortened). Score with
[Ko-Ul] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-Ko-Ul.sc --loo
[VB] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-VB.sc --loo
[hal] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-hal.sc --loo
[Bhatri] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-Bhatri.sc --loo
[OM] # Group chain (marathi-bhil): the OIA-M.sc chain, then @notation rules mapping into Tulpule's
[OM] # Devanagari transliteration (digraph aspirates, written inherent -a, anusvara ṁ, -aṇē infinitive,
[OM] # macron ē ō). Score with
[OM] #     python3 sound_changes/evaluate.py score sound_changes/rules/OIA-OM.sc --loo
[ktg] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-ktg.sc --loo
[kc ktg] %ext-menu +ṭṭaka +ṭṭikā ; round 7: MIA -ṭṭa- noun extension, as in Kotgarhi (nāḍī́ > naḷṭo, dvā́ra > daṭṭo, tumba > tumṭe); placebo +ppaka +ppikā: Δext train −.0044 vs −.0014, test −.0033 vs .0000
[WPah] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-WPah.sc --loo
[khash] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-khash.sc --loo
[bhal] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-bhal.sc --loo
[cur] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-cur.sc --loo
[mand] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-mand.sc --loo
[dog] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-dog.sc --loo
[pan] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-pan.sc --loo
[kgr] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-kgr.sc --loo
[cam] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-cam.sc --loo
[kvar] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-kvar.sc --loo
[Ku Garh N Dotyali] # Group base chain: central-east-pahari-romani (MIA block shared with OIA-H.sc), then Pahari NIA.
[bihari-tharu] # Bihari-Tharu group base chain (see sound_changes/notes/bihari-tharu.md).
[bihari-tharu] # MIA block as in OIA-H.sc, then the eastern NIA block.
[Bhoj] %verb-menu +alla                             ; Bhojpuri -al verbal noun/infinitive < MIA -alla- (placebo: -akka/-atta/-anna/-ala ±0; -alla verb 0.3399 → 0.2176 train)
[Darai] %verb-menu +ikke                             ; Darai -ike infinitive < MIA -ikke (locative of -ikka-); placebo: -ikke verb 0.2282 → 0.0814 train, -itte/-ippe 0.2232, -ikkam ±0
[KS] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-KS.sc --loo
[MH] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-MH.sc --loo
[BH] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-BH.sc --loo
[A] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-A.sc --loo
[B] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-B.sc --loo
[Or] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Or.sc --loo
[AdivasiOriya] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-AdivasiOriya.sc --loo
[Or AdivasiOriya] %exclude-source beine2017bhatri
[MB] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-MB.sc --loo
[Pk Ap] # Group base chain for mia-insular (see sound_changes/notes/mia-insular.md). Score with
[Pk Ap] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Pk.sc --loo
[Pk Ap] # Syntax is documented at the top of sound_changes/sca.py.
[Pk Ap] # Notation: CDIAL writes anusvāra before stops (paṁca); Sheth 1923 writes the homorganic nasal
[Pk Ap] # (pañca). Both sources are kept; the chain predicts the CDIAL spelling and a %source block at the
[Pk Ap] # end converts to Sheth's.
[Pa] # Copy of the mia-insular base chain (OIA-Pk.sc) specialised for Pali: no intervocalic lenition,
[Pa] # no n > ṇ, no ya-śruti, no y- > j- (each hurts train NED when added; see notes/mia-insular.md);
[Pa] # r̩ > a; jñ ny ṇy > ññ; hy > yh; rdʰ > ḍḍʰ; ḍ > ḷ; vv > bb; homorganic nasals instead of ṁ.
[Pk Pa] %ext-menu +ī +ā +ika +am                     ; feminine -ī / -ā of a-stems (kandará > kaṁdarī, karkaṭa > kakkaḍī, kaŋkaṇa > kaŋkaṇī): placebo +ū; Δext train/test +ī −0.0058/−0.0049, +ā −0.0028/−0.0034 vs +ū −0.0008/−0.0007; jointly −0.0085/−0.0082, net +607; +ika: −0.0033/−0.0030 vs placebo +ira −0.0012/−0.0012 (masc./neut. -ika > -ia: avalagna > ōlaggia, uddʰava > uddʰavia; the menu had only -ikā, -ikka, -iyaka); +am (adverbial/neuter -aṁ: ihá > ihaṁ, kṣiprá > kʰippaṁ): −0.0005/−0.0004 vs placebo +um −0.0001/−0.0001; round 7
[Ap] %ext-menu +ī                                 ; feminine -ī on the stem (nidrā́ > ṇiddaḍī-type): placebo +ū; Δext train/test +ī −0.0105/−0.0005 vs +ū 0.0000/0.0000 (+ā −0.0005/0.0000 does not beat it on test, not added); round 7
[As-Ul] # The Pali chain (OIA-Pa.sc: no intervocalic lenition, no aya > ē) plus Ashokan features: anusvāra
[As-Ul] # before stops, rt > ṭṭ, (eastern) ṇ > n, Brāhmī spelling without geminates. The CDIAL As. forms mix
[As-Ul] # the eastern (Kalsi, Dhauli, pillar edicts: r > l, -e), western (Girnar) and north-western
[As-Ul] # (Kharoṣṭhī: ś ṣ kept) versions; r > l was tried and dropped (net −0.001 after rt > ṭṭ; r kept
[As-Ul] # in 143 vs l in 87 train pairs).
[As-Ul] %ext-menu +ā +am                             ; feminine / nominative -ā on the stem (the -ā nouns of the edicts): placebo +ū; Δext train/test +ā −0.0106/−0.0140 vs +ū −0.0022/−0.0045 (+ī −0.0043/−0.0031 loses to the placebo on test, not added); +am (neuter nom./acc. and adverbial -aṁ: kā́ma > kāmaṁ, kr̩tyá > kacaṁ): −0.0090/−0.0026 vs placebo +um −0.0004/−0.0004; round 7
[Md] # Copy of the Sinhala chain (OIA-Si.sc), with the mia-insular MIA block, then the Insular block.
[Md] # Dhivehi deviations: s > h (initial after c > s; intervocalic late, after degemination and
[Md] # apocope), y- > d-, ṭṭ > r̆, p > f, final -a lost (-u after r), final -ā > -i, ṇ > n, -anī verb
[Md] # citation. Fritz's notation (ṣ for r̆) is handled in a %source block. See notes/mia-insular.md.
[Md] %ext-menu +ī +ika                            ; -ī on the stem (Dhivehi -ī nouns): placebo +ū; Δext train/test +ī −0.0047/−0.0050 vs +ū −0.0020/−0.0018 (+ā −0.0014/−0.0008 loses to the placebo, not added); +ika (-ika > -i: the Si-type -ia contraction): −0.0036/−0.0057 vs placebo +ira −0.0001/−0.0015; round 7
[Si] # mia-insular base chain (as in OIA-Pk.sc) for the Sinhala-Prakrit stage, then the Insular
[Si] # changes (deaspiration, c > s, j > d, lenition, degemination without lengthening,
[Si] # prenasalisation, umlaut, vowel shortening). s > h is NOT in the chain: CDIAL Si. keeps s in
[Si] # ~2/3 of the forms (learned stratum), so s > h hurts train NED. See notes/mia-insular.md.
[nuristani] # Nuristani is a sister branch of Indo-Aryan within Indo-Iranian, so these are correspondence
[nuristani] # rules read from the OIA side, not a historical derivation. Where OIA has merged a PIIr
[nuristani] # contrast that Nuristani keeps (the palatal series), the rule is tagged @nur-archaism and the
[nuristani] # unpredictable residue is discussed in notes/nuristani.md.
[nuristani] %exclude-source nured                        ; NuRED forms: own notation (ċ = ʦ, ë = ə, n̆, capital archiphonemes) and mostly compounds
[Kt] %gold-drop caron grave                       ; round 7 scoring declaration (coordinator): every caron in this lect is Strand's ǐ, his letter for the non-syllabic i glide (gǐu, ʦǐe, ǐuṣ), not a rising tone or an accent. The one other caron, CDIAL yákr̩t > yɔ̌k, is the house rendering of Turner's yɔ̈̄k (long centralised ɔ), not a tone either. Stopgap (coordinator): `grave` drops the Vedic grave of CDIAL śikyà (f_f2ekngihsqnya), the OIA headword parsed as the Kt reflex from the "GM 22" citation, which otherwise makes every CDIAL Kt pair tone-scored; remove grave when the record is fixed.
[Or A MB B Si Ash] %verb-menu +aka                              ; root + -aka action noun (B ā̃cā, cālā; placebo -aṭaka/-ara checked by the coordinator)
[Hajong] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Hajong.sc --loo
[Bishnupriya] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Bishnupriya.sc --loo
[OB] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-OB.sc --loo
[Si NiDoc] %ext-menu +ī +ā                              ; feminine -ī / animate -ā on the stem (garbʰíṇī > gæbinī, kā́ka > kākā): placebo +ū; Δext train/test +ī −0.0035/−0.0045, +ā −0.0029/−0.0051 vs +ū −0.0011/−0.0004; jointly net +100; round 7
[OSi] # Copy of the Sinhala chain (OIA-Si.sc). Old Sinhala deviations: NO loss of intervocalic stops
[OSi] # (−0.015 train NED when present) but p > v and ṭ ḍ > ḷ; r̩ > a; all long vowels shortened;
[OSi] # prenasalisation and anusvāra not written. Paranavitana's ț ņ ļ in a %source block.
[NiDoc Dhp KharI] # mia-insular base chain (see OIA-Pk.sc), specialised for the Kharoṣṭhī-written North-western
[NiDoc Dhp KharI] # MIA: three sibilants kept, many clusters with r kept, kṣ kept; intervocalic stops voiced /
[NiDoc Dhp KharI] # spirantised rather than lost. Kharoṣṭhī spelling marks neither vowel length nor gemination:
[NiDoc Dhp KharI] # the chain first derives the (phonological) MIA form, then applies @notation rules at the end.
[Dhp] %ext-menu +ā                                 ; -ā on the stem (feminine and consonant-stem nominatives): placebo +ū; Δext train/test +ā −0.0031/−0.0047 vs +ū −0.0025/−0.0011 (+ī −0.0014/−0.0010 loses to the placebo, not added); round 7
[Phal] # OIA (CDIAL headwords) → Palula (CDIAL "Phal.", Liljegren's dictionary, Strand, Decker 1992)
[Sh] %verb-menu +āpayikka   ; causative -āpaya- + -ikka (partial source of Sh -oiki); round 6 placebo: verb −0.0122/−0.0117 vs +āparikka −0.0079/−0.0075
[Sh] # OIA (CDIAL headwords) → Shina (CDIAL "Sh." plus Schmidt, Degener, Backstrom, Rajapurohit, Buddruss, LSI)
[Kal] # OIA (CDIAL headwords) → Kalasha (CDIAL "Kal.", Trail & Cooper 1999, Decker 1992)
[Sv] # OIA (CDIAL headwords) → Sauji (CDIAL "Sv.", Knobloch 2020, Decker 1992, Liljegren)
[Kalk] # OIA (CDIAL headwords) → Kalkoti (Kalkoti dictionary, Hultman 2023, Rensch et al. 1992, Liljegren)
[Ush] # OIA (CDIAL headwords) → Ushojo (Rensch et al. 1992, Liljegren)
[Kho] %verb-menu +ayikka +āpayikka   ; causative -aya-/-āpaya- + -ikka infinitive (Kho -eik); placebo-tested round 6: beats -arikka/-āparikka by 0.010/0.011 train, 0.007/0.015 test
[Kho] # OIA (CDIAL headwords) → Khowar (CDIAL "Kho." plus Bashir 2023, Strand, Decker 1992)
[K] # OIA (CDIAL headwords) → Kashmiri (CDIAL "K.", Grierson's dictionary transcription)
[Kund] # OIA (CDIAL headwords) → Kundal Shahi (Kundal Shahi wordlist, Liljegren)
[dod] # OIA (CDIAL headwords) → Dodi (CDIAL "dod.", i.e. Siraji-type forms)
[pog] # OIA (CDIAL headwords) → Poguli (CDIAL "Pog.", Schmidt)
[dardic-north] #
[Phal] # Dardic (Shinaic, southern). Unlike Shina proper, Palula keeps initial voiced aspirates (and
[Phal] # shifts medial breathiness onto them), keeps tr/dr, st/ṣṭ and intervocalic k, voices stops after
[Phal] # nasals instead of losing them, and moves r into the first syllable (after lenition). Verbs
[Phal] # are cited in -e (Liljegren); the stem-citing sources (CDIAL, Strand, Decker) drop it in a %source block.
[Phal] # Liljegren's tone marks are approximated by a %source block (all long vowels falling).
[Sh] # Dardic (Shinaic, Shina proper). Clusters simplify without MIA gemination; the Shina-proper
[Sh] # hallmarks are tr/dr > ʦ̣/ʣ̣, nasal + stop > nasal and s + stop > stop (after lenition);
[Sh] # voiced aspirates deaspirate, intervocalic stops, v and h vanish, v- > b-. Verbs are cited
[Sh] # with the CDIAL (Kohistani) -oiki infinitive, cut to -oik. The Shina data mix several
[Sh] # dialects and notations; per-source %source blocks give each source its own infinitive
[Sh] # convention and Schmidt's tone marks. The floor is high.
[Kal] # Dardic (Chitrali). Kalasha keeps initial voiced aspirates, s + C and stop + r clusters, adds
[Kal] # prothetic i- before s + C, keeps intervocalic k, has p > v only after inherited v was lost, and
[Kal] # has no length contrast. First-syllable a > o before a nasal cluster + a (weak). Verbs are cited
[Kal] # as -ik (Trail & Cooper); CDIAL's 1sg -im, Decker's length marks and Trail & Cooper's ḷ are
[Kal] # handled in %source blocks. Trail & Cooper's retroflex vowels (ạ ọ ụ, from lost ḍ/ṛ/l/ṇ) are
[Kal] # too sporadic to model.
[Sv] # Dardic (Shinaic, southern; chain copied from Palula and specialised). Sauji's own innovation
[Sv] # is the lateral tr > ɬ, dr > l; voiced aspirates are lost, intervocalic k survives, no length
[Sv] # contrast in the target notation. Verbs cited in -il.
[Kalk] # Dardic (Shinaic, southern; chain copied from Palula and specialised). tr is kept; stops are
[Kalk] # voiced after nasals and voiced dentals then lost after n (dánta > dān); ṭ/ḍ > r; r + dental >
[Kalk] # retroflex; r moves into the first syllable after lenition (as in Palula); voiced aspirates are
[Kalk] # lost; intervocalic k survives. Verbs cited in -ūn. The dictionary's æ and the surveys'
[Kalk] # length/affricate spellings are handled in %source blocks.
[Ush] # Dardic (Shinaic; chain copied from Shina). Ushojo patterns with Shina proper: tr > ʦ̣,
[Ush] # s + stop > stop after lenition, r + C > C, deaspiration; kṣ > ṣ. Verbs cited in -ōn.
[Kho] # Dardic (Chitrali): clusters largely survive (no MIA gemination; r + C and s + C stay), prothetic
[Kho] # i- before s + C, intervocalic t > r, k/g > ɣ, other stops vanish; voiced aspirates lose
[Kho] # aspiration, first-syllable a > o by harmony with a following a/u/ō, length is not phonemic in the
[Kho] # target notation. The @morph rules
[Kho] # model the -ik infinitive in which Kho. verbs are cited. Score with
[Kho] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Kho.sc --loo
[K] # Dardic (Kashmiric) chain. Clusters simplify without MIA gemination, voiced aspirates lose
[K] # aspiration, palatal stops become dental affricates, ś/ṣ → h (s is kept), short high vowels
[K] # lower in closed syllables, r + C and geminates simplify only after intervocalic lenition,
[K] # final short vowels survive only as (unscored) matras. Verbs are cited as -un infinitives.
[Kund] # Dardic (Shinaic, southern; chain copied from Palula and specialised). tr is kept, stops are
[Kund] # voiced after nasals, s + stop > stop, ṭ/ḍ > ṛ, ṇ > n, voiced aspirates lost, no length
[Kund] # contrast in the target notation. Verbs cited as bare stems.
[dod] # Dardic (Kashmiric; chain copied from Kashmiri and specialised). Dodi keeps voiced aspirates
[dod] # and ṇ, flaps ṭ/ḍ > ṛ, has v- > b-, y- > j- and
[dod] # Pahari-type citation forms (masc. -ō, infinitive -ṇō), while sharing the Kashmiri affrication c > ʦ, j > z and kṣ > cʰ.
[pog] # Dardic (Kashmiric; chain copied from Kashmiri and specialised). Shares Kashmiri deaspiration,
[pog] # affrication, ṇ > n and v retention; its own trait is sibilant + stop > h + stop (aṣṭā́ > āhṭ).
[pog] # Verbs cited in -nu (modelled as -un).
[Ash Kam] %gold-drop caron                             ; round 7 scoring declaration (coordinator): every caron in this lect is Strand's ǐ, his letter for the non-syllabic i glide (gǐu, ʦǐe, ǐuṣ), not a rising tone or an accent.
; ── classes (shared by every lect; a few lects redefine G or Nasal right after) ──
class G2 = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ś h   ; from changes.sc
class G3 = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l s h ʦ ʦʰ ʣ ʣʰ ḷ   ; from changes.sc
class Nasal2 = ŋ ñ ṇ n m ṁ   ; from changes.sc
class G_std = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l s h   ; changes.sc's G (a lect's own G may differ)
class Nasal_std = ŋ ñ ṇ n m   ; changes.sc's Nasal (a lect's own Nasal may differ)
class CnoR_std = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m y v l s h   ; changes.sc's CnoR (a lect's own CnoR may differ)
class Stop  = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ
class Nasal = ŋ ñ ṇ n m
[nuristani] class Nasal = ŋ ñ ṇ n m ṁ
class Son   = r y v l
class Sib   = ś ṣ s
class Asp   = kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ
class VAsp  = gʰ jʰ ḍʰ dʰ bʰ
class G     = Stop Nasal Son s h            ; consonants that can be geminate after MIA
[S] class G     = Stop Nasal Son s h ɠ ʄ ᶑ ɓ
[marathi-bhil] class G     = Stop Nasal Son s h ʦ ʦʰ ʣ ʣʰ ḷ
[west-pahari] class G     = Stop Nasal Son s ś h
[eastern] class G     = Stop Nasal Son ś h
class CnoR  = Stop Nasal y v l s h
class NR    = Stop Nasal y v l s h           ; consonants other than r
class NDStop = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ p pʰ b bʰ d dʰ   ; stops other than t tʰ
class Front = i ī e ē ĩ ī̃ ẽ y
class Gx    = k kʰ g gʰ c cʰ j jʰ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ Nasal Son s ś h   ; G minus ṭ (ṭṭʰ survives)
class Cx    = Stop ŋ ñ ṇ n r y v l ś ṣ s h ḣ ; consonants other than m
class Them  = k kʰ g gʰ c cʰ j jʰ ś ṣ ṭ ṭʰ ḍ ḍʰ l p pʰ b bʰ ṇ ; root-final consonants (not the inflectional stem finals t d n s r m)
[B] %exclude-source zoller2023
[Dm] # OIA (CDIAL headwords) → Dameli (Dm.; CDIAL, Perder 2013, Decker 1992, Liljegren)
[Chil] # OIA (CDIAL headwords) → Chilisso (70% RDH 1992, CDIAL)
[Shum] # OIA (CDIAL headwords) → Shumashti (Shum.; CDIAL, Buddruss 1979)
[Wot] # OIA (CDIAL headwords) → Wotapuri-Katarqalai (Wot.; CDIAL only)
[Gaw] # OIA (CDIAL headwords) → Gawarbati (Gaw.; CDIAL, Decker 1992)
[dardic-south] #
[Bshk] # dardic-south, Kohistani branch (Swat). Stop+r > laterals/affricates: tr kr > ɬ, dr gr > l,
[Bshk] # pr > c, bʰr > j; c j not affricated; ē > i; ava > ō; deaspiration; intervocalic t d k g p v y
[Bshk] # lost, ṭ ḍ > r; word-final high vowels long. Accent on the last syllable; Gawri's tone is high except
[Bshk] # rising after a voiceless aspirate (round 7);
[Bshk] # its æ for /a/ and RDH's long vowels / unaspirated ʦ̣ are %source blocks.
[Bshk] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Bshk.sc --loo
[Mai] # dardic-south, Kohistani branch. Shared chain: no MIA gemination (clusters simplify), Stop+r >
[Mai] # retroflex affricate (tr pr > ʦ̣, dr bʰr > ẓ), ṣṭ > ṭʰ, c j > ʦ z, intervocalic t d > l (a Mai
[Mai] # peculiarity; other lects lose them), ṭ ḍ > ṛ, k g lost, p > v, lenition before degemination,
[Mai] # apocope. Voiced aspirates kept (Zoller writes bʰ dʰ gʰ). Zoller's tone (â falling, ǎ rising) sits on
[Mai] # the last syllable; round 7 writes it at the end of the chain: rising before a final nasal and after a
[Mai] # breathy onset, falling elsewhere. The Vedic accent does not predict the contour (notes, Round 7).
[Mai] # %source blocks: RDH long vowels; CDIAL/RDH cite verb stems (no -ǎṽ); Zoller's release vowel (-ṇḍᵃ).
[Mai] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Mai.sc --loo
[Gowro] # dardic-south, Kohistani branch; Mai chain minus t > l (dentals lost) and minus final aspiration,
[Gowro] # with inherited vowel length neutralised. Zoller's -ō̃ infinitive comes from the verb menu (round 7: @morph citation rules removed).
[Gowro] # Rising tone dominates in Gowro (ǎ ~80% of toned vowels); round 7 writes it on the accented last syllable.
[Gowro] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Gowro.sc --loo
[bhatr] # dardic-south, Kohistani branch; Mai chain minus t > l (dentals lost), plus ē > i, ava > ō,
[bhatr] # p > v feeding v-loss, voiceless ṭʰ tʰ > ṭ t. Zoller's Bhateri uses extra letters (palatalised
[bhatr] # ď ť ľ ň ř ž, ȳ, ø) that are not mapped; its æ for /a/ and RDH's notation are %source blocks.
[bhatr] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-bhatr.sc --loo
[bhatr] %gold-drop caron @strand                   ; round 7: Strand's ǐ is a glide letter (gʰr̩tá > gʰǐū, bʰaginī > bʰǐõ, hŕ̩daya > hǐū), not a rising tone; Zoller's carons stay (a scoring declaration, not gated)
[Gowro Bshk bhatr Mai] ; (round 7: `%gold-drop tone` removed — a no-op, the tone score reads the marks and the segmental scores drop them)
[Tor] # dardic-south, Kohistani branch (Swat). tr > ʦ̣ only; c j not affricated (as Bashkarik); voiced
[Tor] # aspirates deaspirated; ava > ō; intervocalic k p voiced, dentals lost; non-initial kṣ > ṣ;
[Tor] # final r lost; word-final high vowels long. RDH's long vowels and unaspirated ʦ̣ are a %source block.
[Tor] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Tor.sc --loo
[Tir] # dardic-south; keeps Stop+r and ṣṭ clusters (as Dameli), ś > s and ṣ > x (Pashto-like), ō > ọ,
[Tir] # r̩ > a, n t > n d, deaspiration after flapping, intervocalic dentals and h lost, initial h lost.
[Tir] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Tir.sc --loo
[M] %verb-menu +aka +akkanakam                   ; root + -aka verbal noun (M. -ā nouns/verbal forms); +akkanakam = expressive -akk- stem + -anakam (ʦubakṇẽ, saḍakṇẽ, saṭakṇẽ; round 6 third pass: verb −0.0030 train / −0.0009 test vs placebo +attanakam −0.0001 / 0). Round 6: verb 0.1897 → 0.1873 train, 0.1946 → 0.1917 test; placebo +ara −0.0002/−0.0003; +āpayanakam −0.0007 vs placebo +ārayanakam ≈ −0.0002 (not adopted, margin too small)
[Ko-Ul] %verb-menu +anta                             ; present participle -ant- (Konkani present -tā: cartā, jaḷtā). Round 6 third pass: verb 0.4591 → 0.4538 train, 0.4184 → 0.4129 test; placebo +ampa ±0
[bihari-tharu Ku Garh N Dotyali eastern hindi marathi-bhil northwest rajasthani-gujarati west-pahari] ; ══ MIA ══════════════════════════════════════════════════════════════════
[Kannauji-Central OAw bagheli_lakshman Brj SSr PauriBareli OP thar] # ── Common MIA core (shared; Pali chain minus Pali-specific changes; round-7 integration) ──
[Kannauji-Central bagheli_lakshman Brj OP] # ── end of core; lect-specific chain follows ──
[bundeli_atarra bagheli_lakshman] # ── Hindi post-MIA prefix: first 2 of 3 shared changes (round-7 unification) ──
[Or BH MH] ; if:starred,!ext a > a a / _ #                    ; @morph — Turner's *reconstructed (late, non-Sanskritic) etyma usually carry the -ā (< -aka) extension (Or. 45 % -ā vs 27 % -a)   [dropped: starred-headword citation shortcut; the extension menu derives these endings]
[A M] ; ── block dental-retroflex-optional: Sporadic retroflexion of initial dentals (daśa > ḍasa, daṁśa > ḍaṁsa, dahati > ḍahai). `optional:` = sporadic: forks the derivation where it applies (best-of; b [sporadic (blocks/dental-retroflex-optional.sc)] ──
[A M] optional: {t d dʰ} > {ṭ ḍ ḍʰ} / # _          ; @retroflex — sporadic initial dental > retroflex (daśa > ḍasa, daṁśa > ḍaṁsa, *tar- > ṭar-) / (block dental-retroflex-optional)
[Rana DewasDoneDanuwar Sunha KochariyaEastDanuwar Darai Majhi Bote KochilaTharu pan] ; ── block length-two-mora-tonic: MIA law of two morae, stress-conditioned (Bihari-Tharu): a long vowel shortens before a consonant cluster unless an accented syllable follows (pretonic long vow [@two-mora @stress — two-mora law, not before an accented syllable (blocks/length-two-mora-tonic.sc; blocks round 2026-09-30)] ──
[Rana DewasDoneDanuwar Sunha KochariyaEastDanuwar Darai Majhi Bote KochilaTharu pan] {ā ī ū ē ō ā́ ī́ ū́ ḗ ṓ} > {a i u e o á í ú é ó} / _ C C // _ !#* Acc ; @two-mora @stress — long V shortens before CC unless an accent follows (mū́lya > *mulla > Mth mūl; pretonic rāṣṭrá > rāṭṭʰa keeps ā) / (block length-two-mora-tonic)
[dardic-north -Sv -Kalk -Kho -Kund] ; ── Citation morphology ─────────────────────────────────────────────────
[dardic-north -Sh -Kho -K -Kal -dod Pk Ap Pa As-Ul] ; ── Vowels ──────────────────────────────────────────────────────────────
[mia-insular -Pk -Ap -Pa -As-Ul] ; ══ MIA stage ════════════════════════════════════════════════════════════
[-Pas -Aw -Si -awan -Kam] ; ── block upa-va-optional: Sporadic reduction of the prefixes upa-/ava- to va-/ba- (initial u-/a- lost, the labial kept or strengthened), OIA stage (2026-10-01): upaccʰada > Ash vāʦā, Kam [cross-family law (blocks/upa-va-optional.sc)] ──
[-Pas -Aw -Si -awan -Kam] optional: {u ú a á} {p v} {a á ā ā́} > v $3 | b $3 / # _ ; @upa-va — upa-/ava- > va- | ba- / (block upa-va-optional)
[Kannauji-Central kaithal Brj] # ── Hindi post-MIA prefix: first 3 of 3 shared changes (round-7 unification) ──
[Vasavi] ; ── block dv-b-initial: Initial dv- > b- beside d(u)v- (dvāra > H bār ~ duār, dvitīya > bījā ~ dūjā); the chains have it in only 16 lects. Reverse-reconstruction loop, iteration 1 (202 [reverse-reconstruction loop (blocks/dv-b-initial.sc)] ──
[Vasavi] optional: d v > b / # _                      ; @dv-b — initial dv- > b- (dvinavati > H bānve, dvāra > bār, dvitīya > bījā) / (block dv-b-initial)
[-Si] ; ── block dv-b-initial: Initial dv- > b- beside d(u)v- (dvāra > H bār ~ duār, dvitīya > bījā ~ dūjā); the chains have it in only 16 lects. Reverse-reconstruction loop, iteration 1 (202 [reverse-reconstruction loop (blocks/dv-b-initial.sc)] ──
[-Si] optional: d v > b / # _                      ; @dv-b — initial dv- > b- (dvinavati > H bānve, dvāra > bār, dvitīya > bījā) / (block dv-b-initial)
[Rana Dang Kathoriya Bi N Sh Kho K BH MH kaithal NDu H M OM Pk Md Pa OP S P kcch poth awan Wg OG Marw G marwari_bagra dhundari_bamore Goj jaun mand cam kul] ; ── block st-retroflex-medial: Medial counterpart of st-retroflex-optional: st(ʰ) > ṭṭʰ (adʰástāt > heṭʰ, ásthi > aṭṭʰi). Reverse-reconstruction loop, iteration 1 (2026-10-02). Beat a same-sh [reverse-reconstruction loop (blocks/st-retroflex-medial.sc)] ──
[Rana Dang Kathoriya Bi N Sh Kho K BH MH kaithal NDu H M OM Pk Md Pa OP S P kcch poth awan Wg OG Marw G marwari_bagra dhundari_bamore Goj jaun mand cam kul] optional: s {t tʰ} > ṭ ṭʰ / V _              ; @st-retroflex — sporadic medial st(ʰ) > ṭṭʰ (adʰástāt > H heṭʰ, ásthi > Pk aṭṭʰi) / (block st-retroflex-medial)
[M] ŕ̩ > i / _ C* V                             ; @r-vocalic @stress — round 2: tonic r̩ > i (gŕ̩dʰra > gīdʰ, vŕ̩ścika > vĩʦū); unaccented/pretonic r̩ > a below (weak: −0.0003)
[-Bhoj -Dang -Kathoriya -N -eur -as-ll -Lom -RomWel -RomBalk -Zarg -Sv -dod -bhatr -Wot -Tir -Or -Aw -M -VB -Pa -Si -Ap -kcch -thar -Kam -OG -cam] ; ── block r-vocalic-r-kept: Dardic/NW keep the r of r̩ (Cr̩ > Cri/Cru/Cra): *tr̩pu > Phal truṇuʦ̣ʰīr, tr̩ṣṭá > Phal trīṣṭo, Kho treṣṭu, gŕ̩dʰya > Dm grī̃c, r̩ṣabʰá > Kho reṣu, K riśu. Reve [reverse-reconstruction loop (blocks/r-vocalic-r-kept.sc)] ──
[-Bhoj -Dang -Kathoriya -N -eur -as-ll -Lom -RomWel -RomBalk -Zarg -Sv -dod -bhatr -Wot -Tir -Or -Aw -M -VB -Pa -Si -Ap -kcch -thar -Kam -OG -cam] optional: {r̩ ŕ̩} > r a | r i | r u          ; @r-vocalic — r̩ vocalised with its r kept (tr̩ṣṭá > Phal trīṣṭo, Kho treṣṭu; r̩ṣabʰá > K riśu) / (block r-vocalic-r-kept)
[S P poth L srk awan] {ī ū} > {i u} / _ C C !#* Acc                ; @two-mora @stress — 2 lects / long ī ū shorten before a cluster when the Vedic accent follows (ūrdʰvá > ubʰo, kūrmá > kumī); open-syllable pretonic shortening is rejected (worse than no rule) / (changes.sc two-mora.iu.ccacc)
[khet] {ī ū} > {i u} / _ C C                        ; @two-mora — long ī ū shorten before a cluster (the accent adds nothing here); open-syllable pretonic shortening rejected
[OAw M] a y a > ē / _ C // _ t {i ē} #               ; @aya-e — OIA medial -aya- > ē before a consonant (nayana > nēṇa), not in the -ayati verb ending (ākalayati > M ākaḷṇẽ); unified at round-7 integration
[N M Pk G ktg] i ṁ > ī / _ {ś ṣ s}                          ; @anusvara-loss @anusvara-vowel — iṁ before a sibilant > ī without nasality (MIA pīsaï, vīsaï): piṁṣati > pisṇẽ/pīsṇo, śiṁśumāra > susvār, viṁśati > -vīs; non-numerals 22 of 27 oral in G M N Ku (hill-climb round 6)
[Ku N Dotyali kgr] # ── Common MIA core (shared; Pali chain minus Pali-specific changes; round-7 integration) ──
[kgr] ; ── block ks-cch-optional: Sporadic kṣ > ccʰ beside kkʰ (MIA doublets: akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta). `optional:` = sporadic: forks the derivation where it applies (best [sporadic (blocks/ks-cch-optional.sc)] ──
[kgr] optional: k ṣ > c cʰ                         ; @ks — sporadic kṣ > ccʰ beside kkʰ (akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta) / (block ks-cch-optional)
[kvar] # ── W. Pahari post-MIA prefix: first 5 of 5 shared changes (round-7 unification) ──
[Si kgr] ; ── block dental-retroflex-optional: Sporadic retroflexion of initial dentals (daśa > ḍasa, daṁśa > ḍaṁsa, dahati > ḍahai). `optional:` = sporadic: forks the derivation where it applies (best-of; b [sporadic (blocks/dental-retroflex-optional.sc)] ──
[Si kgr] optional: {t d dʰ} > {ṭ ḍ ḍʰ} / # _          ; @retroflex — sporadic initial dental > retroflex (daśa > ḍasa, daṁśa > ḍaṁsa, *tar- > ṭar-) / (block dental-retroflex-optional)
[kgr kvar] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[kgr kvar] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[kgr] ; ── block r-anticipation-optional: Sporadic r-anticipation (Dardic/NW 'metathesis'): the r of the first-syllable coda or of the second syllable's onset cluster is copied into the onset of the ini [sporadic (blocks/r-anticipation-optional.sc)] ──
[kgr] optional: {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} > {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} / # _ V r C | # _ V C r ; @metathesis — sporadic r-anticipation into the initial onset (tarka > trakk-, kárman > krom) / (block r-anticipation-optional)
[kgr] ; ── block cr-keep-optional: Sporadic retention of stop + r (Northwest, W. Pahari, NW Ashokan, Shina/Savi, Old Gujarati): in some words the r of stop + r escapes the MIA assimilation (putra [sporadic (blocks/cr-keep-optional.sc)] ──
[kgr] optional: {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} r > {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} ; @assim-sonorant @cr-keep — sporadic retention of stop + r (dā́tra > P dātrī ~ dātī, átra > As. atra) / (block cr-keep-optional)
[kgr] ; ── block mia-core: Common MIA core — OIA → Middle Indo-Aryan changes shared by the mainland lects. The Pali chain minus its Pali-specific changes (final-consonant loss, -ayati > - [common MIA core: blocks/mia-core.sc] ──
[kgr] t > ṭ / # p r a _ i                          ; @prati-retroflexion @mia-core — 6 lects / (changes.sc prati-retroflexion)
[kgr] {ai au} > {ē ō}                              ; @ai-au @mia-core — 97 lects / (changes.sc ai-au.aiau)
[kgr] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@mia-core — ś ṣ > s, Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[kgr] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[kgr] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[kgr] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[kgr] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[kgr] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[kgr] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[kgr] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[kgr] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[kgr] ; ── block mia-stop-clusters: MIA assimilation of stop + sonorant/nasal and stop + stop clusters (blocks theme "mia", 2026-09-30). The same two changes, in this order, are positions 18–19 of [@mia-core — stop + sonorant/nasal, stop + stop] ──
[kgr] Stop {r y v l Nasal} > $1-asp $1             ; @assim-sonorant — 83 lects / stop + r y v l nasal > geminate stop (putra > putta, śakya > sakka, agni > aggi) / (changes.sc assim-sonorant.stopryvlna)
[kgr] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[kgr] C {r y v l} > $1 / # _                       ; @initial-degem @mia-core — 15 lects / (changes.sc initial-degem.cryvl)
[kgr] C $1 > $2 / C _                              ; @cluster-simplify @mia-core — 7 lects / (changes.sc cluster-simplify.c.postconsonantal)
[kgr] C C > $2 / # _                               ; @initial-degem @mia-core — 60 lects / (changes.sc initial-degem.cc)
[Zarg Kannauji-Central SSr kgr khash kvar] ; ── block mia-vowels: Early MIA vowel changes (blocks theme "mia", 2026-09-30): the diphthongs ai au monophthongise and syllabic r̩ becomes i. Lects with other r̩ outcomes (r̩ > ri-, [inherited from the W. Pahari sisters (small-lect default, 2026-09-30)] ──
[Zarg Kannauji-Central SSr kgr khash kvar] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[Zarg Kannauji-Central SSr kgr khash kvar] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[Ku N OAw kgr] # ── end of core; lect-specific chain follows ──
[hindi -bundeli_atarra -NDu] {ā ī ū} > {a i u} / _ C* V C* V C* V C* V    ; @x-hindi-pretonic-shortening @stress — 3 lects / long V with ≥3 syllables after it shortens (pānīyāmalaka > paniyālā, ānāyayati > anānā); positional, not Vedic-accent-conditioned (round 2) / (changes.sc pretonic-shortening.aiu.cvcvcvcv)
[H] # ── Hindi post-MIA prefix: first 3 of 3 shared changes (round-7 unification) ──
[hindi -OAw] u > ū / C _ #                                ; @vowel-shift @x-hindi-final-u @stress — 7 lects / final -u > -ū except under the Vedic accent: oxytone -ú is lost like -a (áśru > ā̃sū, sáktu > sattū; bāhú > bāh, vēṇú > ben); round 2 / (changes.sc vowel-shift.u.c)
[west-pahari] V > $1-acc                                  ; @acc-loss
[kiuth kc] # ── W. Pahari post-MIA prefix: first 4 of 5 shared changes (round-7 unification) ──
[cam kul] # ── W. Pahari post-MIA prefix: first 5 of 5 shared changes (round-7 unification) ──
[bagheli_lakshman] ; ── block v-b-skip-optional: Sporadic escape of initial v- from v- > b- (2026-10-01): some words keep v- where the lect otherwise has b- (P viśvāsa > vasāh beside vāla > bāl). Mechanism: an [sporadic (blocks/v-b-skip-optional.sc)] ──
[bagheli_lakshman] optional: v > ʋ / # _                        ; @v-b — sporadic: initial v- escapes v- > b- (viśvāsa > P vasāh) / (block v-b-skip-optional)
[jaun bhad kgr] # ── W. Pahari post-MIA prefix: first 3 of 5 shared changes (round-7 unification) ──
[kaithal H kc kvar cam kul] ; ── block lar-v-b: Initial v- > b- (NIA): Hindi, Bihari, Eastern, Pahari, Rajasthani, Punjabi, Lahnda, Sindhi, Dardic, Romani … the most widespread NIA consonant change. Geminate  [@v-b — initial v- > b- (blocks/lar-v-b.sc)] ──
[kaithal H kc kvar cam kul] v > b / # _                                  ; @v-b — 88 lects / vadʰū́ > H. bahū, vā́la > H. bāl / (changes.sc v-b.v.initial)
[west-pahari -bhal -mand] V > $1+stress / # C* _                      ; @accent-placement — initial stress (first syllable), as in the sister lects ktg, dog, khash, WPah, bhad, kgr; placed right after the Vedic accent is lost (round 7: prosody-only, ext_acc better on train, test not worse; few marked forms)
[hindi -OAw -NDu -Aw -SSr jaun kiuth kc bhad kgr kvar cam kul] # ── end of shared prefix ──
[NDu Aw] {u ú} > ū / C _ #                            ; @x-hindi-final-u — 3 lects / final -u > -ū (áśru > ā̃sū); accent-blind here (excluding oxytone -ú is worse in this lect); round 2 / (changes.sc final-u)
[Aw] ī > i / C _ #                               ; @stress @final-vowel-loss — unaccented/post-tonic final -ī shortens (Aw whispered -ⁱ: bʰaginī > bahinⁱ, sapátnī > savatⁱ); tonic -ī́ keeps its length (excluding it is better by 0.0012); round 2
[Aw] ; ∅ > i / C _ t a v y a #                      ; @anaptyxis — the MIA gerundive -iavva: an i is inserted between a root consonant and -tavya (car-tavya > caritavya > cariavva), i.e. the set -itavya generalised; round 5 (Aw -ab infinitive)   [removed: -itavya is now on the verb menu; an i inserted only before -tavya is morphology]
[OM] {¹ ² ³} > ∅                                 ; @notation — homonym digits on Tulpule's √roots
[OM] ∅ > + ṇ ẽ / # √ (C) (C) (C) V (C) (C) _ #    ; @morph — Tulpule's √roots are cited as -ṇē verbs; '+' shields the root-final consonant
[OM] √ > ∅                                       ; @notation — root sign
[Ku N] # ── C/E Pahari post-MIA prefix: first 3 of 3 shared changes (round-7 unification) ──
[Pas] # dardic-south, Pashai. CDIAL cites many Pashai dialects: 44% of etyma have >1 distinct form.
[Pas] # Voiced Stop+r > l (bʰrātr- > lā-); voiceless Cr variously kept or > ɬ (not modelled); c j and
[Pas] # v kept; ṣṭ/st kept; deaspiration (voiced and voiceless) before intervocalic loss; old -v- lost
[Pas] # before p > v; n t > n d; kṣ > ʦ̣ʰ- / -ṣ-; apocope before hiatus contraction (*-aka keeps -a).
[Pas] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Pas.sc --loo
[Dm] # dardic-south, Kunar. Archaic in keeping Stop+r and ṣṭ clusters; y j > ź; initial s + aspirate
[Dm] # > aspirate; deaspiration; intervocalic v y h lost before p > v; ṭ ḍ > r̆; y-prothesis before e.
[Dm] # Morgenstierne's ɔ for ā and flap r̆ (CDIAL) vs Perder/Decker's ā and r are %source blocks.
[Dm] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Dm.sc --loo
[Dm] %gold-drop circumflex                      ; round 7: Morgenstierne's (CDIAL) Dameli â is a vowel letter (anấr, tấlu, nấya: â + stress acute), not a falling tone; a scoring declaration (not gated)
[Chil] # dardic-south, Kohistani branch. RDH marks the (positional) length of stressed / closed-syllable
[Chil] # vowels (and never aspirates ʦ̣): inherited length is removed in the chain and RDH's length re-added by a %source block.
[Chil] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Chil.sc --loo
[Shum] # dardic-south, Kunar; Gawarbati-like chain (Stop+r > ɬ/l, c j > ʦ z, apocope before hiatus
[Shum] # contraction so *-aka keeps a vowel) plus ava > ō, aya > ē, y-prothesis before e, late h loss,
[Shum] # non-initial ś > s and kʰ > x.
[Shum] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Shum.sc --loo
[Wot] # dardic-south; mixed profile: dr > l (as Bashkarik/Kunar) but gr > g and ṣṭ > ṭʰ (Kohistani-like),
[Wot] # c j > ʦ z, ṣ > ś, deaspiration, ava > ō, intervocalic dentals, p, h lost, apocope early,
[Wot] # high vowels short before C and long word-finally.
[Wot] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Wot.sc --loo
[Gaw] # dardic-south, Kunar. Voiceless Stop+r > ɬ, voiced > l, labial+r > labial+l; c j > ʦ z; y- > ź;
[Gaw] # deaspiration; intervocalic v y h lost before p > v; apocope (not -u) before hiatus contraction.
[Gaw] #     python sound_changes/evaluate.py score sound_changes/rules/OIA-Gaw.sc --loo
[dardic-south] ; ══ accent, citation morphology ═════════════════════════════════════════
[Kho] ; ── a-harmony (must see the OIA verb ending: rákṣati > roʦ̣ʰík) ─────────
[nuristani -Ash -Gmb -Pr] ; ── block finals-nuristani-accent: Nuristani accent-conditioned apocope, first step (blocks theme "finals", 2026-09-30): at the head of the chain, while the Vedic accent is still visible, postton [posttonic final -a/-i marked for apocope (blocks/finals-nuristani-accent.sc) — replaces `use final-vowel-loss.a.accc`: posttonic final -a marked for apocope (tonic -á of oxytones survives)] ──
[nuristani -Ash -Gmb -Pr] a > ă / Acc !#* C _ #                        ; @final-vowel-loss @stress — 4 lects / posttonic final -a marked (Kt. kárṇa > kor, dánta > dut, pʰála > pul) / (changes.sc final-vowel-loss.a.accc)
[DewasDoneDanuwar nuristani -Ash -Gmb -Pr] if:!verb i > ĭ / Acc !#* C _ #               ; @final-vowel-loss @stress — 4 lects / posttonic final -i of nouns marked (Kt. ṓṣadʰi > uṣe) / (changes.sc final-vowel-loss.i.accc)
[nuristani -Ash -Gmb -Pr] ; ── block upa-va-optional: Sporadic reduction of the prefixes upa-/ava- to va-/ba- (initial u-/a- lost, the labial kept or strengthened), OIA stage (2026-10-01): upaccʰada > Ash vāʦā, Kam [cross-family law (blocks/upa-va-optional.sc)] ──
[nuristani -Ash -Gmb -Pr] optional: {u ú a á} {p v} {a á ā ā́} > v $3 | b $3 / # _ ; @upa-va — upa-/ava- > va- | ba- / (block upa-va-optional)
[eur] {a á} > e / # C C* _ {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ ṇ n r l ś ṣ y} V                ; @vowel-shift — Romani a > e in an open first syllable before a non-labial (kʰer < gʰara, pʰen- < bʰánati, śel < śatá, del < dádāti), but not before s or h (hásati > asal, prahasati > prasal, bahís > avri); local copy of the shared vowel-shift.aa…, round 7
[RomWel RomBalk Zarg] {a á} > e / # C C* _ {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ ṇ n r l ś ṣ s h y} V ; @vowel-shift — 3 lects / Romani a > e in an open first syllable before a non-labial (kʰer < gʰara, pʰen- < bʰánati, śel < śatá, del < dádāti; labials keep/round a: sov, nav); absent in Lom (+0.018) and Domari (+0.010); eur ±0 (+0.0004, exact +1.1pp); not accent-conditioned / (changes.sc vowel-shift.aa.cckkhgghcchjjh)
[Kho K Gaw Wg Kt] ; ── block ml-bl-optional: Sporadic ml-/mr- > bl-/br-, OIA stage (2026-10-01): *mlāyita > Wg brē, Gaw blē, Ash vlēi. Dardic/Nuristani; few words. `optional:` = sporadic: forks the derivat [cross-family law (blocks/ml-bl-optional.sc)] ──
[Kho K Gaw Wg Kt] optional: m {l r} > b $2 / # _               ; @ml-bl — ml-/mr- > bl-/br- / (block ml-bl-optional)
[Si] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[Si] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[Si] ; ── block st-retroflex-medial: Medial counterpart of st-retroflex-optional: st(ʰ) > ṭṭʰ (adʰástāt > heṭʰ, ásthi > aṭṭʰi). Reverse-reconstruction loop, iteration 1 (2026-10-02). Beat a same-sh [reverse-reconstruction loop (blocks/st-retroflex-medial.sc)] ──
[Si] optional: s {t tʰ} > ṭ ṭʰ / V _              ; @st-retroflex — sporadic medial st(ʰ) > ṭṭʰ (adʰástāt > H heṭʰ, ásthi > Pk aṭṭʰi) / (block st-retroflex-medial)
[Buksa Rana Dang DewasDoneDanuwar Kathoriya Majhi Bote N BH A bundeli_atarra Pk Si Pa] ; ── block tsn-nh: MIA tsn > ṇh, missing in 9 groups (jyṓtsnā > Pk joṇhā > kiuth juhṇ, DewasDone jun). Reverse-reconstruction loop, iteration 3 (2026-10-02). Beat a same-shape pla [reverse-reconstruction loop (blocks/tsn-nh.sc)] ──
[Buksa Rana Dang DewasDoneDanuwar Kathoriya Majhi Bote N BH A bundeli_atarra Pk Si Pa] optional: t s n > ṇ h                        ; @tsn-nh — tsn > ṇh (jyṓtsnā > Pk joṇhā > kiuth juhṇ, Bhatri jon) / (block tsn-nh)
[MagahiNepal Bhoj Mth Bi] t a v y a > a b / C _ #                      ; @x-bihari-tharu-tavya — 4 lects / gerundive -tavya > MIA -(i)avva > -ab after a consonant root (kar-tavya > karab, dekʰ- > dekʰab) / (changes.sc tharu-tavya.c)
[MagahiNepal Mth Bi] t a v y a > a b / V _ #                      ; @x-bihari-tharu-tavya — 3 lects / after a vowel-final stem (-itavya, -āpitavya) -tavya > -avva > -ab as well / (changes.sc tharu-tavya.2)
[MagahiNepal Mth Bi] i a > e / _ b #                              ; @hiatus — 3 lects / -iab (< -itavya, MIA -iavva) > -eb (ocʰāeb, kamāeb) / (changes.sc hiatus.ia.b)
[Mth Bi] ; ── block length-two-mora-tonic: MIA law of two morae, stress-conditioned (Bihari-Tharu): a long vowel shortens before a consonant cluster unless an accented syllable follows (pretonic long vow [@two-mora @stress — two-mora law, not before an accented syllable (blocks/length-two-mora-tonic.sc; blocks round 2026-09-30)] ──
[Mth Bi] {ā ī ū ē ō ā́ ī́ ū́ ḗ ṓ} > {a i u e o á í ú é ó} / _ C C // _ !#* Acc ; @two-mora @stress — long V shortens before CC unless an accent follows (mū́lya > *mulla > Mth mūl; pretonic rāṣṭrá > rāṭṭʰa keeps ā) / (block length-two-mora-tonic)
[A B] ; if:starred a > a a / _ # // k _                    ; @morph — Turner's *reconstructed etyma usually carry the -ā (< -aka) extension; not on -ka forms (the extended variants already have it; -uka would double to -uā)   [dropped: starred-headword citation shortcut; the extension menu derives these endings]
[Kho Kal Dm nuristani -Gmb -PNur] ; ── block s-ts-initial-optional: Initial ś- > ʦ-, OIA stage (2026-10-01): Nuristani (śákr̩t > Kam ʦiṭ, Wg ʦoṭ; *ut-śiras > Wg uʦẽ) and a few Dardic lects. Beat a same-shape placebo (ś- > ṭ-) cl [cross-family law (blocks/s-ts-initial-optional.sc)] ──
[Kho Kal Dm nuristani -Gmb -PNur] optional: ś > ʦ / # _                        ; @s-ts — initial ś- > ʦ- / (block s-ts-initial-optional)
[Kho K Kal Gaw Dm Ash Kam Pr] ; ── block r-vocalic-r-kept: Dardic/NW keep the r of r̩ (Cr̩ > Cri/Cru/Cra): *tr̩pu > Phal truṇuʦ̣ʰīr, tr̩ṣṭá > Phal trīṣṭo, Kho treṣṭu, gŕ̩dʰya > Dm grī̃c, r̩ṣabʰá > Kho reṣu, K riśu. Reve [reverse-reconstruction loop (blocks/r-vocalic-r-kept.sc)] ──
[Kho K Kal Gaw Dm Ash Kam Pr] optional: {r̩ ŕ̩} > r a | r i | r u          ; @r-vocalic — r̩ vocalised with its r kept (tr̩ṣṭá > Phal trīṣṭo, Kho treṣṭu; r̩ṣabʰá > K riśu) / (block r-vocalic-r-kept)
[Md] {a á ā ā́} > e / # C* _ C* {i í ī ī́}         ; @umlaut — first-syllable a, ā > e before i (Dhivehi e where Sinhala has æ: āmiṣa > em, *abʰilāgayati > eluvanī)
[Md] {u ú} > i / _ C* {i í ī ī́}                  ; @umlaut — u fronted before i (round 2)
[Md] {ī ī́} > e / _ C* #                         ; @vowel-shift — OIA final -ī > -e (āsandī́ > eⁿde); round 2
[OSi] {ē ḗ} > ∅ / t _ #                           ; @morph @hiatus — middle -tē verbs cited without the ending (avalambatē > olambu, vándatē > vand-); round 2
[bhatr] {a á} > ā / # _ C C                          ; @degem-lengthen — word-initial a lengthened in a closed syllable (aṣṭā́ > āṭ, adyá > āz, akṣá > āʦ̣hā́); not accent-conditioned
[Kalk Phal Kal Gaw Shum Dm Tor Tir Pas VB NiDoc Dhp KharI nuristani -Gmb -Pr] ; ── block sv-sp: The NW MIA (Gāndhārī, Niya, NW Ashokan) sv/śv > sp/śp: svásr̩ > As. spasuna, NiDoc priyaśpasu, Dhp spasu; áśva > aśpa. Reverse-reconstruction loop, iteration 2  [reverse-reconstruction loop (blocks/sv-sp.sc)] ──
[Kalk Phal Kal Gaw Shum Dm Tor Tir Pas VB NiDoc Dhp KharI nuristani -Gmb -Pr] optional: {s ś ṣ} v > $1 p                   ; @sv-sp — sv/śv > sp/śp (svásr̩ > As. spasuna, Dhp spasu; áśva > NiDoc aśpa) / (block sv-sp)
[Pas] {á í ú} > {ā ī ū} / # C* _ C (V) #         ; @stress @x-dardic-south-tonic-lengthening — accented short vowel of a CV́C(V) disyllable lengthened, i.e. in the open syllable of a word that becomes a monosyllable after apocope (yáva > yā, mála > māl, kʰála > kāl, náva > nā); the same rule for unaccented vowels costs +0.0024
[Sh] V t {i ē} > o i k / _ #                     ; @morph — -ati/-atē verbs cited as Kohistani -oiki infinitives (final -i left to final-vowel noise)
[Sh] {a á} > ō / _ n {a á} k                      ; @vowel-shift — a before n + the -aka suffix becomes ō: the stressed -ánaka verbal noun > infinitive -ōno/-ōnu/-ōn (bandʰ- > bonōnu, rōd- > rōnu) (round 5)
[Phal K] {ō ṓ} > ū                                   ; @vowel-shift — inherited ō > ū (gōla > gūli, rōṣa > rūṣ), the back partner of ē > ī; it must precede ā > ō (drag chain: +0.009 if placed after) (round 2)
[Phal] ā́ > ō                                       ; @vowel-shift @stress — Vedic-accented ā > ō in any syllable (tā́raka > tōro, kā́ṇḍa > kōṇ); accent-only +0.0010 vs the positional rule below alone
[Phal] ā > ō / # C* _ C V C* #                     ; @vowel-shift @stress — and unaccented ā in the open first syllable of a disyllable, the (penultimate) stress position (śāka > śō, gʰrāṇá > gʰrōṇ) (round 2: −0.0014 together)
[Phal] á > ā / # C* _ C V                          ; @stress @vowel-shift — accented first-syllable a in an open syllable lengthens (ná > nā-, bʰárati > bʰār-); unaccented a does not (+0.0008)
[Phal] {g d b j ḍ} > $1+asp / # _ V {dʰ bʰ gʰ jʰ ḍʰ h}  ; @aspiration-shift — breathiness of a medial voiced aspirate/h moves to the initial voiced stop (gōdʰū́ma > gʰōm, dōháyati > dʰov-)
[Phal] {a á} > ā / # (C) (C) _ C (C) {a á} #          ; @x-dardic-north-monosyllable-lengthening — a lengthens in words that become monosyllables when final -a goes (hásta > hātʰ, kárṇa > kāṇ)
[Kal] {a á ā ā́} > o / # (C) (C) _ Nasal C {a á}    ; @a-rounding — first-syllable a > o before a nasal cluster + a (áŋga > oŋg, gaṇḍa > goṇ), weak (LOO +0.0003)
[Kal] V t {i ē} > i k / _ #                       ; @morph — -ati/-atē verbs cited as -ik infinitives (Trail & Cooper)
[K] V t {i ē} > u n / _ #                       ; @morph — -ati/-atē verbs cited as -un (cálati > ʦalun)
[dod] {a á} y {a á} t {i ē} > ṇ ō / _ #           ; @morph — causatives cited in -ṇō too (gaṇáyati > gaṇṇō)
[Sh K Kal dod] ; ── Vowels ──────────────────────────────────────────────────────────────
[Sv] V t {i ē} > i l / _ #                       ; @morph — verbs cited in -il (Knobloch/Decker)
[Kalk] V t {i ē} > ū n / _ #           ; @morph — verbs cited in -ūn (Kalkoti infinitive); not numerals in -ti (round 2)
[Phal] {d g b} > $1+asp / # _ r                      ; @aspiration-shift — initial voiced stop + r becomes breathy (drakk > dʰrak, durgá > dʰrūk)
[dardic-north -Kho] {a á} {s n} > ∅ / V C* _ #                   ; @final-cons-loss @final-vowel-loss — 8 lects / -as/-an consonant stems lose their final syllable (kárman > Sh kom, ātmán > K atᵘ) / (changes.sc final-cons-loss.aasn.final-polysyllabic)
[dardic-north -K] {a á} v {a á} > u / # _ {n m s ś ṣ}          ; @hiatus — 3 lects / the prefix ava- contracts (MIA ava > o) and is raised to u before a nasal or sibilant … / (changes.sc hiatus.aavaa.nmsss)
[dardic-north -K] {a á} v {a á} > o / # _                      ; @hiatus — 3 lects / … else it is o- (Phal *avaḍḍʰāla > oḍʰṓl; as in Khowar) (round 7, consistency) / (changes.sc hiatus.aavaa.initial)
[Sh] {ā ā́} > a / V C* _ #                        ; @final-vowel-loss — final -ā shortens first (and then goes with the other final vowels)
[Sh] {ā ā́} > a / _ C* {ā ā́}                     ; @vowel-shift — ā shortens before a following (non-final) ā (āntarāla-type words: dissimilation of length)
[Sh] {u ú} > o / V C* _ C* #                     ; @vowel-shift — final-syllable u of u-stems lowers to o and so survives final-vowel loss (rájju > rajo, tā́lu > tālo, sáktu > sāto); unaccented u carries it (ú alone ±0) (round 2)
[Kho] {a á} > o / # (C) (C) _ C (C) {a á u ú ō ṓ} // # (C) (C) _ C (C) {a á} C (C) {a á}   ; @a-rounding @umlaut — first-syllable a > o before a following short a or round vowel (kárman > korum, kapṓta > kovór), not before ā or front vowels, and not in CaCaCa words (śaraṇá > śarán, *śalakka > śalák: stress on the 2nd syllable)
[Kho] {i í} > e / # (C) (C) _ C (C) {i í}         ; @vowel-shift — first-syllable i > e before a following i (ríṣṭi > reṣṭ, kilima > kelel)
[Kho] ; ── Citation morphology ─────────────────────────────────────────────────
[Kho] {a á} y {a á} t {i ē} > i k / _ #           ; @morph — causatives cited as -ik infinitives (tāráyati > torík)
[Kho] V t {i ē} > i k / _ #                       ; @morph — -ati/-atē verbs cited as -ik (drámati > dromik)
[Kho] ; ── Vowels ──────────────────────────────────────────────────────────────
[Kho] {í ī́ ú ū́} > ∅ / V C* _ # // t _            ; @final-vowel-loss @stress — accented final -i/-u are lost (āṇí > aṇ, rāśí > raś, lagʰú > loʦ), unaccented ones survive (+0.0021 if all final i are dropped); -tí nouns keep -i as -ri (round 2)
[Kho] {ī ī́} > ∅ / V C* _ #                         ; @final-vowel-loss — final long -ī is lost whatever the accent (garbʰíṇī > garbin, kṣudʰā-type) (round 6)
[K] {a á} v {a á} > ɔ / # _                      ; @hiatus — the prefix ava- contracts to ɔ- (MIA ava > o), which then takes the v- prothesis (*avastarati > vɔtʰarun, *avaśapati > vɔhavun); cf. Kho/Kal/Phal ava > o/u (round 7, consistency)
[Kam] {u ú ū ū́} > ɨ / # C C* _ C* V C* #        ; @vowel-shift — Kam: u/ū of the first syllable of an OIA disyllable after a consonant > ɨ (dʰūmá > dɨm, kúla > kɨr̆, túṣa > tɨs, buddʰa > bɨt, pū́ti > pɨ). Round 2 had it for ū only; round 7 extends it to short u (ext −0.0043/−0.0045, ext_acc −0.0049/−0.0047; 25 reflexes better, 6 worse: kunta > kut, śúci > suc) now that ɨ is a vowel to the engine (the round-6 ü placeholder is gone). Word-initial u- stays (úpa > u); any first syllable (trisyllables too) is +0.0073
[Ash] {ē ḗ} > i / _ C* V                          ; @vowel-shift @nur-innovation — Ash: ē of a non-final syllable raised to i (round 2, −0.0010). Accent-blind: pretonic-only ē > i gives −0.0009; Wg, Kt, Kam, PNur reject the rule
[-N -Ku -Dotyali -Kund -Brj -RathwiBareli] V > $1-acc                                  ; @acc-loss
[northwest -D -thar -Khol -khet] ṁ Stop > {ŋ ŋ ŋ ŋ ñ ñ ñ ñ ṇ ṇ ṇ ṇ n n n n m m m m}@2 $2 ; @notation — headword anusvāra before a stop = homorganic nasal (saṁdʰi = sandʰi)
[bundeli_atarra] ; ∅ > Ø / {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ tʰ d dʰ p pʰ b bʰ l ḷ r} _ #                   ; @morph — starred verbal roots (*aṭṭ, *ḍag) are bare stems: protect the stem-final C (Ø = citation ending, spelled out at the end)   [pruned for bundeli_atarra: LOO -0.0009]
[bagheli_lakshman] ; ∅ > Ø / {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ tʰ d dʰ p pʰ b bʰ l ḷ r} _ #                   ; @morph — starred verbal roots (*aṭṭ, *ḍag) are bare stems: protect the stem-final C (Ø = citation ending, spelled out at the end)   [pruned for bagheli_lakshman: LOO -0.0004]
[bagheli_lakshman] ; {ai au} > {ē ō}                             ; @ai-au   [pruned for bagheli_lakshman: never fires]
[Brj] {ā ī ū} > {a i u} / _ C* V C* V C* V         ; @stress @x-hindi-pretonic-shortening — long V with ≥2 syllables after it shortens (dyūtakāra > juār, *grāmadāra > gãvār); positional, not Vedic-accent-conditioned (round 2)
[Brj] ; ∅ > Ø / {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ tʰ d dʰ p pʰ b bʰ l ḷ r} _ #                   ; @morph — starred verbal roots (*aṭṭ, *ḍag) are bare stems: protect the stem-final C (Ø = citation ending, spelled out at the end)   [pruned for Brj: LOO -0.0005]
[Kannauji-Central OAw Aw] ; if:starred ∅ > Ø / {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ tʰ d dʰ p pʰ b bʰ l ḷ r} _ #                   ; @morph — (round 4: if:starred — only Turner's starred roots get the citation ending) starred verbal roots (*aṭṭ, *ḍag) are bare stems: protect the stem-final C (Ø = citation ending, spelled out at the end)   [dropped: starred-headword citation shortcut; the extension menu derives these endings]
[Kannauji-Central] ; {ai au} > {ē ō}                             ; @ai-au   [pruned for Kannauji-Central: never fires]
[H] ; round 7: removed the @morph starred-root marker (∅ > Ø) with its Ø > n ā ending and the two if:verb -nā rules (-ayati, ā-roots): the verb menu derives -nā. Jointly ext −0.0004 train / −0.0019 test, verb ±0; plain +0.0136 is the bare-vs-extended artifact (bare *aṭṭ, *ḍag no longer carry a pasted -nā). Kept: in-stem i n > ī (removal ext +0.0007) and the -ati > -nā rule (removal ext +0.0005)
[kiuth mand] {a ā} > ∅ / # _ C C                          ; @aphaeresis — 2 lects / initial a lost before a cluster (akṣōṭa > kʰoṛ) / (changes.sc aphaeresis.aa.cc)
[dardic-north -Kho -Kal] s > ∅ / # _ n                               ; @cluster-simplify — initial sn- > n- (*snāru > Sh nār, Phal nōr; snāpáyati > K nāvun). The chain left sn (gold sn 0; hill-climb round 7)
[Kt] {a ā ă} > ∅ / V {p pʰ b bʰ v} _ #           ; @final-vowel-loss — R6 fourth pass: -a after a single labial is lost early, before lenition, so the labial ends up word-final and survives as v (apūpá > úv, *ēva > ev, kakubʰá > kúv, lāva > r̆óv); stems in -as keep their s at this point (śrávas > ṣu)
[Ash Wg] ; ── block lar-nur-umlaut: Nuristani (Waigali, Ashkun): ā shortened before a short high vowel in the next syllable. Laryngeals round, pass 2 (2026-09-30). Edit here, not in the lects. [@umlaut @vowel-shift — ā shortened before a short high vowel (blocks/lar-nur-umlaut.sc)] ──
[Ash Wg] ā > a / _ C* {i u}                           ; @umlaut @vowel-shift — 2 lects / bʰārika > Wg. bari, dānin > dáni / (changes.sc umlaut.ciu)
[Pr] {a ā} > i / # C* _ C* i                      ; @umlaut @nur-innovation — Pr: first-syllable a, ā raised to i before i (antiká > itī, aṣṭʰi > ici, hári > źiaræ, gaviṣá > giṣ); blocks the later a > u. Accent-blind: tonic-only costs +0.0018, non-tonic-only +0.0003; including ā −0.0009; not beyond the first syllable (+0.0041)
[nuristani] {n s t d} > ∅ / V _ #                        ; @final-cons-loss — 7 lects / OIA word-final consonants are absent (stems cited with -an/-as/-is/-t, Nur continues the vowel-final forms): nā́man > Wg nām, sáras > *sar, rōcís > *roc / (changes.sc final-cons-loss.nstd.v)
[Sh Phal K bundeli_atarra] ; ── block dental-retroflex-optional: Sporadic retroflexion of initial dentals (daśa > ḍasa, daṁśa > ḍaṁsa, dahati > ḍahai). `optional:` = sporadic: forks the derivation where it applies (best-of; b [sporadic (blocks/dental-retroflex-optional.sc)] ──
[Sh Phal K bundeli_atarra] optional: {t d dʰ} > {ṭ ḍ ḍʰ} / # _          ; @retroflex — sporadic initial dental > retroflex (daśa > ḍasa, daṁśa > ḍaṁsa, *tar- > ṭar-) / (block dental-retroflex-optional)
[Kal] V > $1+stress / _ C* V C* #                  ; @accent-placement — stress on the OIA penult, placed early (just after accent loss, before length loss and apocope) (round 3)
[Sv Kho Phal K Kal nuristani] {ai au} > {e o}                              ; @ai-au — 12 lects / (changes.sc ai-au.aiau.2)
[Garh RomWel RomBalk dardic-north -Sv -Kalk -K -Kal Shum AdivasiOriya DungraBhili OM Noiri Bhili NiDoc Wg PNur Pr jaun] ; ── block hiatus-aya: Hiatus theme — aya > ē (blocks round, 2026-09-30). The early contraction of -aya- to ē, after OIA ai/au > ē/ō and before the MIA loss of intervocalic y: one cha [@aya-e — blocks/hiatus-aya.sc; OIA -aya- > ē (tráyaḣ > trē, vayám > bē, nayana > nēṇa); unified at round-7 integration] ──
[Garh RomWel RomBalk dardic-north -Sv -Kalk -K -Kal Shum AdivasiOriya DungraBhili OM Noiri Bhili NiDoc Wg PNur Pr jaun] ; ── block hiatus-aya-medial: Hiatus theme — aya > ē before a consonant (blocks round, 2026-09-30). The early contraction of -aya- to ē inside the word, after OIA ai/au > ē/ō and before the  ──
[Bi Ku Garh as-ll RomWel RomBalk dardic-north -Sv -Kalk -K -Kal Shum AdivasiOriya Aw DungraBhili OM Noiri Bhili NiDoc OSi Wg PNur Pr jaun dog kgr] a y a > ē / _ C                              ; @aya-e — medial aya > ē (nayana > nēṇa, śayana > sēṇa) / (block hiatus-aya-medial)
[Garh RomWel RomBalk dardic-north -Sv -Kalk -K -Kal Shum AdivasiOriya DungraBhili OM Noiri Bhili NiDoc Wg PNur Pr jaun] a y a > ē                                    ; @aya-e — aya > ē, also word-finally (tráyaḣ > trē, vayám > bē, Kal krayá > kre) / (block hiatus-aya)
[kc] a > ∅ / # _ C C V C* V                    ; @aphaeresis — initial a- lost before a cluster in words of three or more syllables (aŋguṣṭʰá > guṭʰā, akṣōṭa > kʰoṛ; round 6 second pass)
[Si] ; ── block finals-ap-shortening: Apabhraṁśa shortening of final -ā (blocks theme "finals", 2026-09-30): late MIA final long -ā shortens to -a, so that it is later lost with the old short -a by  [Ap. final -ā > -a (blocks/finals-ap-shortening.sc) — replaces `use final-a-short.a`: OIA final -ā shortened early (before -aka > -ā): latā > la-, sēnā > sena] ──
[Si] ā > a / _ #                                  ; @final-a-short @final-shortening @final-vowel-loss @vowel-shift — 41 lects / final -ā > -a (Bi. ā́rā > ār, kʰáṭvā > kʰāṭ, nāvā́ > nāv) / (changes.sc final-a-short.a)
[Pk Pa] ∅ > i / # C _ l                              ; @anaptyxis — 2 lects / initial C + l opened by i (glāni > gilāṇa, klēśa > kilēsa) / (changes.sc anaptyxis.cl)
[Bshk] ō > o / _ C C                                 ; @two-mora — ō shortened before a cluster (*spʰōṭyatē > pʰoṭ, *bōcca > boc); ā ī ū ē are not (general two-mora law: +0.0036)
[Mai] a t {i ē} > a ṽ / _ #                       ; @morph — -ati verbs cited as -ǎṽ infinitive
[Chil bhatr Mai] a t {i ē} > a / _ #                         ; @morph — -ati verbs cited as bare stems: the ending is replaced by a (removed with the final vowels), so a stem-final consonant is treated as intervocalic
[Kal] ; ── block hiatus-aya-except: Hiatus theme — aya > ē outside the -ayati verb ending (blocks round, 2026-09-30). The same early aya > ē as `hiatus-aya`, in the lects whose NIA verbs drop the  [@aya-e — blocks/hiatus-aya-except.sc; OIA -aya- > ē in stems (tráyaḣ > trē, nayana > nēṇa); the -aya- of -ayati verbs is not affected (NIA drops it: ākalayati > M ākaḷṇẽ); unifie] ──
[Kal] a y a > ē // _ t {i ē} #                     ; @aya-e — aya > ē except in -ayati/-ayatē (nayana > nēṇa, tráyaḣ > trē; ākalayati > M ākaḷṇẽ) / (block hiatus-aya-except)
[Tir Pas] V > $1-stress                               ; @tone — clear the latent Vedic accent (unused in Tirahi: the accent is placed on the last syllable) so the latent tier can carry the next mark; round 7
[Tir] V > $1+stress / {bʰ dʰ gʰ ḍʰ jʰ} _ | {bʰ dʰ gʰ ḍʰ jʰ} {r y v} _   ; @tone @depressor — latent mark on a vowel after an OIA voiced aspirate: its breathy onset, later deaspirated, is what lowers the tone (below); round 7
[Pas] V > $1+stress / {bʰ dʰ gʰ ḍʰ jʰ h kʰ tʰ ṭʰ pʰ cʰ} _ | {bʰ dʰ gʰ ḍʰ jʰ h kʰ tʰ ṭʰ pʰ cʰ} {r y v} _   ; @tone @depressor — latent mark on a vowel after an OIA aspirate or h, whose breathy/aspirated onset (later lost) conditions the tone written below; round 7
[Wot Dm] if:!verb {ā ī ū ē ō} > {a i u e o} / V C* _ C* V ; @length-loss — 2 lects / long vowels shortened in medial syllables (round 2; accent-independent: keeping tonic vowels long scores the same or worse) / (changes.sc length-loss.aiueo.vccv)
[Dm] a y a t {i ē} > a / _ #                     ; @morph — causatives cited as bare stems (ending → a, removed with the final vowels)
[dardic-south] {s ḣ t n m} > ∅ / V C (V) _ #                ; @final-cons-loss — 12 lects / OIA final consonants lost in polysyllables (kárman > kam) / (changes.sc final-cons-loss.shtnm.vcv)
[dardic-south] ; ══ OIA → Dardic ══════════════════════════════════════════════════════
[Bi Ku as-ll Wot Pas Aw OSi dog kgr] ; ── block hiatus-aya-medial: Hiatus theme — aya > ē before a consonant (blocks round, 2026-09-30). The early contraction of -aya- to ē inside the word, after OIA ai/au > ē/ō and before the  [@aya-e — blocks/hiatus-aya-medial.sc; OIA medial -aya- > ē before a consonant (nayana > nēṇa, śayana > sēṇa); unified at round-7 integration] ──
[Wot Pas] a y a > ē / _ C                              ; @aya-e — medial aya > ē (nayana > nēṇa, śayana > sēṇa) / (block hiatus-aya-medial)
[eur as-ll RomBalk Lom] {s ḣ t d m n} > ∅ / _ #                      ; @final-cons-loss — 30 lects / OIA word-final consonants (nāman > nav, karman > kam); before the verb-ending morph so stems keep theirs / (changes.sc final-cons-loss.shtdmn)
[eur RomBalk Zarg] u k a > ū / _ #                              ; @morph @aka — -uka > -ū (> -u)
[eur RomWel RomBalk Zarg Lom] {i ī} k ā > ī / _ #                           ; @morph @aka — fem. -ikā > -ī (> -i after length loss; cʰuri < kṣurikā, kangli, rakli): suffixal k lost as in -aka > -o
[Lom] a t {i ē} > ∅ / _ #                          ; @morph — -ati verbs cited as bare stems (CDIAL)
[Ku N] {á í ú} > $1+long / _ C C                    ; @degem-lengthen @stress — 3 lects / Vedic-tonic vowel before a cluster lengthens even outside the penult (adds to the positional rule; replacing the penult rule by it loses) / (changes.sc degem-lengthen.aiu.cc)
[Ku N] t > ∅ / # u _ {ś ṣ s}                        ; @assim — 3 lects / prefix ut- assimilates to a following sibilant before ts > ccʰ, as in N (*ut-śvasiti > usāsṇo, *ut-stʰāti > uṭʰṇo) / (changes.sc assim)
[dardic-north] ḣ > ∅                                        ; @final-cons-loss — 11 lects / visarga lost / (changes.sc final-cons-loss.h)
[Kund] ; ── block visarga-loss: visarga-loss — end of chain: a visarga still left in the output is lost (hill-climb round 7). These chains had no rule deleting ḣ in every position (no `final-c [end of chain: leftover visarga lost (blocks/visarga-loss.sc; hill-climb round 7)] ──
[Kund] {ḣ ḥ} > ∅                                    ; @final-cons-loss — leftover visarga lost (tráyaḣ > tīn/tre, catvā́raḣ > cār, duḣkʰá > duk) / (block visarga-loss)
[Sv] # ── Shinaic post-MIA prefix: first 3 of 6 shared changes (round-7 unification) ──
[Kund] {ā ī ū ē ō} > {a i u e o}                    ; @length-loss @vowel-shift @stress — 12 lects / unaccented long vowels shorten, (Vedic-)accented ones keep length / (changes.sc length-loss.aiueo)
[Kalk Ush Kund] # ── Shinaic post-MIA prefix: first 6 of 6 shared changes (round-7 unification) ──
[Sh Phal] ; ── block nasals-anusvara-loss: nasals-anusvara-loss — leftover anusvāra lost without nasality, nasals theme, blocks round 2026-09-30 (pass 2). The anusvāra that survives to this point (before [@anusvara-vowel — anusvāra lost (nasality mostly unwritten) (blocks/nasals-anusvara-loss.sc)] ──
[Sh Phal] ṁ > ∅                                        ; @anusvara-loss @anusvara-vowel — 53 lects / pāṁśu > awan pā, māṁsá > Si mas, áṁsa > Si as / (changes.sc anusvara-loss)
[pog dod] {a á} > ∅ / V C* _ C {a á} k {a á} #         ; @syncope — 2 lects / a is syncopated before a final -Caka (bardʰ-anaka > baḍṇō, váṇṭ-anaka > baṇṭṇō) (round 5) / (changes.sc syncope.aa.vccaakaa)
[dod] n > ṇ / _ {a á} k {a á} #                    ; @n-retroflex — n of the -naka suffix is retroflex (-ṇō infinitive, as in West Pahari) (round 5)
[eur RomWel RomBalk Zarg dardic-north -Kho -pog -K -Kal] ; ── block finals-aka-dardic: The -aka ending in Shina and its neighbours (blocks theme "finals", 2026-09-30): the masculine -akaḥ contracts to -o, early in the chain (before the Dardic apoc [-aka > -o (blocks/finals-aka-dardic.sc) — replaces `use aka.aakaa.final.2`: -aka contracts to -o once the k is lost, as in Shina (round 4; retagged as a sound change in round 6)] ──
[eur RomWel RomBalk Zarg dardic-north -Kho -pog -K -Kal] {a á} k {a á} > o / _ #                      ; @aka @hiatus — 3 lects / -aka > -o (Sh. tāraká > tāro, kulaka > kulo, tōlaka > tōlo) / (changes.sc aka.aakaa.final)
[as-ll dardic-north -Phal -Kal -dod Dm Tor] ; ── block lar-deaspiration-voiced: Loss of aspiration in the voiced aspirates (bʰ dʰ gʰ jʰ ḍʰ > b d g j ḍ), everywhere in the word. Shared across families: Dardic (Shinaic, Kohistani, Kunar, Kash [@deaspiration — voiced aspirates lose aspiration (blocks/lar-deaspiration-voiced.sc)] ──
[as-ll dardic-north -Phal -Kal -dod dardic-south -Gowro -Bshk -Chil -bhatr -Mai Si OSi nuristani -Wg -Gmb] {bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ}               ; @deaspiration @tone — 26 lects / dʰūma > Sh. dūm, bʰūmi > K. bum, gʰr̩tá > Gaw. gʌd / (changes.sc deaspiration.bhdhghjhdh)
[Sv Kalk Ush Kund] ; ── block lar-v-b: Initial v- > b- (NIA): Hindi, Bihari, Eastern, Pahari, Rajasthani, Punjabi, Lahnda, Sindhi, Dardic, Romani … the most widespread NIA consonant change. Geminate  [@v-b — initial v- > b- (blocks/lar-v-b.sc)] ──
[Sv Kalk Ush Kund] v > b / # _                                  ; @v-b — 88 lects / vadʰū́ > H. bahū, vā́la > H. bāl / (changes.sc v-b.v.initial)
[Kalk Ush Kund] ; ── block nasals-dental-merger: nasals-dental-merger — NIA ṇ > n everywhere, nasals theme, blocks round 2026-09-30. Stage 2 (Hindi belt, Bihari, Tharu, Nepali …): every ṇ — the MIA intervocali [@n-dental — NIA ṇ > n (blocks/nasals-dental-merger.sc)] ──
[Kalk Ush Kund] ṇ > n                                        ; @n-dental — 66 lects / pānī́ya > H, Bhoj pānī, guṇá > H gun, vána > Bhoj ban / (changes.sc n-dental.n)
[Ku N Sv Kalk Ush Kund] # ── end of shared prefix ──
[Garh N] # (removed: numeral-specific rule) use anusvara-loss.is                         ; @anusvara-loss — MIA -iṁś- > -īs- in the decade numerals (viṁśatí > bis, triṁśát > tis; Pk. vīsaï, tīsā)
[Ku N Bshk Wot Tir hindi -bundeli_atarra -kaithal -SSr OP S P L awan bhal jaun kiuth WPah] ; ── block dental-retroflex-optional: Sporadic retroflexion of initial dentals (daśa > ḍasa, daṁśa > ḍaṁsa, dahati > ḍahai). `optional:` = sporadic: forks the derivation where it applies (best-of; b [sporadic (blocks/dental-retroflex-optional.sc)] ──
[Ku N Bshk Wot Tir hindi -bundeli_atarra -kaithal -SSr OP S P L awan bhal jaun kiuth WPah] optional: {t d dʰ} > {ṭ ḍ ḍʰ} / # _          ; @retroflex — sporadic initial dental > retroflex (daśa > ḍasa, daṁśa > ḍaṁsa, *tar- > ṭar-) / (block dental-retroflex-optional)
[OSi] optional: t > ṭ / r̩ _                       ; @rt-retroflex — 2 lects / as Si: nŕ̩tyati > nar̆an; round 6 / (changes.sc rt-retroflex.t)
[Dhp] ; ── block r-anticipation-optional: Sporadic r-anticipation (Dardic/NW 'metathesis'): the r of the first-syllable coda or of the second syllable's onset cluster is copied into the onset of the ini [sporadic (blocks/r-anticipation-optional.sc)] ──
[Dhp] optional: {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} > {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} / # _ V r C | # _ V C r ; @metathesis — sporadic r-anticipation into the initial onset (tarka > trakk-, kárman > krom) / (block r-anticipation-optional)
[Dhp] p r a t i > p a ṭ i / # _                   ; @prati-retroflexion — prati- > paḍi- (pratimā > paḍima)
[Pk Pa Dhp OSi ktg] ; ── block dental-retroflex-optional: Sporadic retroflexion of initial dentals (daśa > ḍasa, daṁśa > ḍaṁsa, dahati > ḍahai). `optional:` = sporadic: forks the derivation where it applies (best-of; b [sporadic (blocks/dental-retroflex-optional.sc)] ──
[Pk Pa Dhp OSi ktg] optional: {t d dʰ} > {ṭ ḍ ḍʰ} / # _          ; @retroflex — sporadic initial dental > retroflex (daśa > ḍasa, daṁśa > ḍaṁsa, *tar- > ṭar-) / (block dental-retroflex-optional)
[Si Md OSi] optional: t > ṭ / r̩ _                       ; @rt-retroflex — 2 lects / as Si: nŕ̩tyati > nar̆an; round 6 / (changes.sc rt-retroflex.t)
[K] ṁ > ∅ / _ {ś ṣ s h}                         ; @anusvara-vowel — anusvāra before a sibilant is lost, not turned into n (pāṁśu > pāh, māṁsá > mās) (round 6 fourth pass)
[Kho K] ṁ > n                                        ; @anusvara-vowel — 2 lects / anusvāra > n / (changes.sc anusvara-vowel.m)
[Sv Kho Kund Kal] {ā ī ū ē ō} > {a i u e o}                    ; @length-loss @vowel-shift — 12 lects / Kalasha has no phonemic length (Trail & Cooper) / (changes.sc length-loss.aiueo)
[Kalk Ush Kund] {a á} > ā / # (C) (C) _ C C {a á ā ā́} #     ; @degem-lengthen @x-dardic-north-monosyllable-lengthening — 2 lects / a lengthens before a cluster in words that become monosyllables (dánta > dān, rakta > rāt, páñca > pānj), as in Palula; accent-independent (round 2) / (changes.sc degem-lengthen.aa.ccccaaaa)
[pog] {a á} k {a á} > u / _ #                     ; @hiatus @aka — -aka contracts to -u (the Kashmiric masc. -u), giving the -nu infinitive from -anaka (round 5)
[RathwiBareli] a > ə / Acc !#* _ #                          ; @stress @morph — final -a after the Vedic accent (non-oxytone words) is marked ə (round 7: the -o citation rule it used to block is gone; the pair still pays, ext +.0025 if dropped with it)
[Kund RathwiBareli] V > $1-acc                                  ; @acc-loss
[dod] ; ── block visarga-loss: visarga-loss — end of chain: a visarga still left in the output is lost (hill-climb round 7). These chains had no rule deleting ḣ in every position (no `final-c [end of chain: leftover visarga lost (blocks/visarga-loss.sc; hill-climb round 7)] ──
[dod] {ḣ ḥ} > ∅                                    ; @final-cons-loss — leftover visarga lost (tráyaḣ > tīn/tre, catvā́raḣ > cār, duḣkʰá > duk) / (block visarga-loss)
[pog K] {ṇ ŋ} > n                                    ; @n-dental — 2 lects / (changes.sc n-dental.nng)
[Sv Ush Kund Phal] ē > ī                                        ; @vowel-shift — 2 lects / ē > ī (Liljegren/Strand) / (changes.sc vowel-shift.e.2)
[-Mth -Tor] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[-Mth -Tor] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[dardic-north] r̩ > r i / C _                               ; @r-vocalic-ri — 4 lects / r̩ > ri after a consonant (kr̩ṣṇá > kriṣṇa, dr̩ṣṭá > driṣ) / (changes.sc r-vocalic-ri.postconsonantal)
[Brj] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[Brj] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[eur] r̩ > u / {p pʰ b bʰ m v ś ṣ} _               ; @r-vocalic — 2 lects / after labials and ś (vr̩tti > buti, śr̩ṇ- > śun) / (changes.sc r-vocalic.pphbbhmvss)
[Ku Garh N OAw H Aw OP S P L awan WK] r̩ > r i / # _                               ; @r-vocalic-ri — 13 lects / r̩kṣa > riccʰ / (changes.sc r-vocalic-ri.initial)
[Tir M Ko-Ul mia-insular -Pk -NiDoc -Dhp G] ; ── block mia-vowels-a: Early MIA vowels, western/southern r̩ outcome (blocks theme "mia", pass 2, 2026-09-30): ai au > ē ō and r̩ > a (M, G, Si, Pa …), the counterpart of mia-vowels ( [@ai-au @r-vocalic — early MIA vowels: ai au > ē ō, r̩ > a (blocks/mia-vowels-a.sc; blocks round 2026-09-30, pass 2)] ──
[Tir M Ko-Ul mia-insular -Pk -NiDoc -Dhp G] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[Tir M VB Ko-Ul mia-insular -Pk -NiDoc -Dhp thar srk G] r̩ > a                                       ; @r-vocalic — 17 lects / r̩ > a (tr̩ṇa > taṇa, kr̩ta > kata, gʰr̩ta > gʰata) / (changes.sc r-vocalic.2)
[Pk Ap Pa As-Ul] ; ── Final consonants ────────────────────────────────────────────────────
[Pk] ∅ > a / C C _ #                             ; @morph — bare roots ending in a cluster (*jʰakkʰ > jʰakkʰia)
[Pk Ap Pa As-Ul] m > ṁ / _ #                                  ; @final-cons-loss — 3 lects / final m > anusvāra (kept: ahaṁ, kiṁ) / (changes.sc final-cons-loss.m)
[Ap As-Ul] ∅ > a / Them _ #                            ; @morph — bare roots / consonant stems thematised (diś > disā, *jʰal > jʰala-)
[As-Ul] ∅ > a / n t _ #                             ; @morph — -ant participles thematised (-aṁta)
[Pk Ap Pa As-Ul] repeat: Cx > ∅ / _ #                         ; @final-cons-loss — 4 lects / all final consonants (-nt, -rk …) / (changes.sc final-cons-loss.cx.repeat)
[Pk Ap Pa As-Ul] ; ── Clusters ────────────────────────────────────────────────────────────
[Md] V t {i ē} > a n ī / _ #        ; @morph — verbs (-ati, -āti, -ōti) cited as -anī participles (pūrayati > furanī, jānāti > dannanī)
[Si] V t {i ē} > a n a v ā / _ #    ; @morph — verbs (-ati, -āti, -ōti) cited as -navā infinitives (jānāti > dannavā)
[Si Md OSi] ; ── block lar-si-umlaut-front: Sinhala first-syllable fronting umlaut: ā > æ before a front vowel in the next syllable, a > æ before a cluster + i. Sinhala, Old Sinhala. Laryngeals round (202 [@umlaut — first-syllable ā/a > æ before a front vowel (blocks/lar-si-umlaut-front.sc)] ──
[Si Md OSi] ā > æ / # C* _ C* {i ī e ē}                  ; @umlaut — 2 lects / ālapati > ælavum, āpatti > ævæta / (changes.sc umlaut.cciiee)
[Si Md OSi] a > æ / # C* _ C C {i ī}                     ; @umlaut — 2 lects / aśvattʰa > æsaṭu, bandʰita > bædi / (changes.sc umlaut.cccii)
[Si] {i ī} > a / C _ #                           ; @x-mia-insular-final-i-a — final -i > -a (ṣaṣṭí > saṭa, girí > gira, bʰīti > biya)
[Si Md OSi] ; ── block finals-am-sinhala: Sinhala -am (blocks theme "finals", 2026-09-30): final -am > -ā before the loss of final consonants; the -anakam verbal noun gives the Sinhala infinitive in -na [-am > -ā (blocks/finals-am-sinhala.sc) — replaces `use final-nasal.am`: final -am > -ā (as Si): neuter -anakam > -anakā; round 5] ──
[Si Md OSi] a m > ā / _ #                                ; @final-nasal — 2 lects / -am > -ā (Si. árgʰati+anakam > aganavā, *adʰiṣṭʰāti+anakam > iṭanavā) / (changes.sc final-nasal.am)
[Ku as-ll N RomBalk Kho NiDoc Dhp S P kcch L] r̩ > u / {p pʰ b bʰ m v} _                   ; @r-vocalic — 10 lects / u after labials / (changes.sc r-vocalic.pphbbhmv)
[-Mth -Bi -A -NiDoc -Wg -Kam -Kt -Ash -PNur -Pr -bagri_fatehabad -cur -pan] ; ── block mia-vowels: Early MIA vowel changes (blocks theme "mia", 2026-09-30): the diphthongs ai au monophthongise and syllabic r̩ becomes i. Lects with other r̩ outcomes (r̩ > ri-, [@ai-au @r-vocalic — early MIA vowels: ai au > ē ō, r̩ > i (blocks/mia-vowels.sc; blocks round 2026-09-30)] ──
[-A -Wg -Kam -Kt -Ash -PNur -Pr -cur -pan] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[-Mth -Bi -NiDoc -Wg -Kam -Kt -Ash -PNur -Pr -bagri_fatehabad] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[hindi -OAw] ; ── block nasals-anusvara-sibilant: nasals-anusvara-sibilant — MIA anusvāra before a sibilant > vowel length without nasality, Hindi belt (H, Aw, Brj, NDu, Kannauji, Bagheli, Bundeli), nasals them [@anusvara-vowel — MIA ṁ before sibilant > length without nasal (viṁśatí > bīs) (blocks/nasals-anusvara-sibilant.sc)] ──
[hindi -OAw] V ṁ > $1+long / _ s                          ; @anusvara-vowel — 7 lects / viṁśatí > H bīs, triṁśat > tīs / (changes.sc anusvara-vowel.vm.2)
[NDu] ; ∅ > Ø / {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ tʰ d dʰ p pʰ b bʰ l ḷ r} _ #                   ; @morph — starred verbal roots (*aṭṭ, *ḍag) are bare stems: protect the stem-final C (Ø = citation ending, spelled out at the end)   [pruned for NDu: LOO -0.0002]
[NDu] ; {ai au} > {ē ō}                             ; @ai-au   [pruned for NDu: never fires]
[khash] {ā ī ū} > {a i u} / _ C V C* V               ; @vowel-shift — 4 lects / long vowel shortened two or more syllables before the end (NIA weight/stress; round 2) / (changes.sc vowel-shift.aiu.cvcv)
[west-pahari -dog -kgr -kvar -WPah] ; ── block nasals-anusvara-homorganic: nasals-anusvara-homorganic — anusvāra before a stop > homorganic nasal consonant (kept, not a nasal vowel), West Pahari (bhal, jaun, khash, ktg, kiuth, kc, kul) [@nasal-cluster — anusvāra before a stop = homorganic nasal (kept as a consonant) (blocks/nasals-anusvara-homorganic.sc)] ──
[west-pahari -dog -kgr -kvar -WPah] ṁ {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} > {ŋ ŋ ŋ ŋ n n n n ṇ ṇ ṇ ṇ n n n n m m m m}@2 $2 ; @nasal-cluster — 7 lects / ṁ + stop > ŋ ñ ṇ n m + stop (the cluster survives as NC) / (changes.sc nasal-cluster.mkkhgghcch)
[Mth Bi] t t > ṭ ṭ / r̩ _                             ; @x-bihari-tharu-rt — 2 lects / the same retroflexion after vocalic r, for the geminate (mr̩ttikā > māṭī, -vr̩tti > -auṭī) / (changes.sc tharu-rt.tt)
[Bhoj] ū > u / # C* _ C V                          ; @vowel-shift — ū shortens in an open first syllable (Bhoj u for Mth/Bi ū; not accent-conditioned: tonic −0.0002, pretonic +0.0001)
[Buksa Chitwan Dang Bhoj Morang Kathoriya bundeli_atarra H Brj bhal jaun] ; ── block ks-cch-optional: Sporadic kṣ > ccʰ beside kkʰ (MIA doublets: akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta). `optional:` = sporadic: forks the derivation where it applies (best [sporadic (blocks/ks-cch-optional.sc)] ──
[Buksa Chitwan Dang Bhoj Morang Kathoriya bundeli_atarra H Brj bhal jaun] optional: k ṣ > c cʰ                         ; @ks — sporadic kṣ > ccʰ beside kkʰ (akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta) / (block ks-cch-optional)
[Garh Bishnupriya Or AdivasiOriya OB B Bhatri mewari_bannoda] # ── Common MIA core (shared; Pali chain minus Pali-specific changes; round-7 integration) ──
[Dhp] u r > r u / # d _                         ; @metathesis — r metathesised into the onset in the prefix dur- (durgá > druga, durbala > drubala, durlabʰa > drulavha); round 7
[-Kho -Kal -D -Khol] ; ── block rt-retroflex-optional: Sporadic r + dental retroflexion (MIA): r + t tʰ d dʰ > ṭṭ ṭṭʰ ḍḍ ḍḍʰ in some words only (vartate > vaṭṭai beside vattai; ardʰa > aḍḍʰa beside addʰa). `optional [sporadic r + dental > ṭṭ/ḍḍ (blocks/rt-retroflex-optional.sc)] ──
[-Kho -Kal -Khol] optional: r {t tʰ d dʰ} > {ṭ ṭ ḍ ḍ}@2 {ṭ ṭʰ ḍ ḍʰ} ; @assim-r @rt-retroflex — sporadic r + dental > retroflex geminate (vartate > vaṭṭai ~ vattai, ardʰa > aḍḍʰa) / (block rt-retroflex-optional)
[D] ; @sibilant-keep — Domaaki keeps ś ṣ s apart (śaknōti > śʌk, aṣṭā > oṣṭ, ṣōḍaśa > ṣõõi); no merger rule, and sibilant+stop clusters survive
[VB] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō, as mewari_basad and the other Rajasthani lects (tailá > tel, asáu > vɔ/vā via so). The chain had no rule for the OIA diphthongs, so ai/au survived to the output (taiḷ; hill-climb round 7) / (changes.sc ai-au.aiau)
[Ush marathi-bhil -Bhatri -PauriBareli -M -mewari_bannoda -OM -RathwiBareli -VB -Ko-Ul srk rajasthani-gujarati -bagri_fatehabad -OG -WK -mewari_dholpura -G -OMarw] {ai au} > {ē ō}                              ; @ai-au — 11 lects / round 6: restored (saúvarṇa > sōn-, not *hauvno) / (changes.sc ai-au.aiau.3)
[bagri_fatehabad Nimadi Marw had mewari_dholpura mewati_akera marwari_bagra dhundari_badagaon] # ── Rajasthanic post-MIA prefix: first 4 of 4 shared changes (round-7 unification) ──
[mewati_jhambaus] # ── Rajasthanic post-MIA prefix: first 2 of 4 shared changes (round-7 unification) ──
[rajasthani-gujarati -OG -G] ; ── block lar-u-a: Medial ŭ > a (Rajasthani, Old Marwari, Gujarati-type), not in hiatus before a lost stop. Laryngeals round (2026-09-30). Edit here, not in the lects. [@vowel-shift — medial u > a (blocks/lar-u-a.sc)] ──
[rajasthani-gujarati -OG -G] u > a / V C* _ C* V // _ {k g c j t d y} V   ; @vowel-shift — 13 lects / māṇuṣa > māṇas; but bʰalluka > bʰālū (-uka > -ū) / (changes.sc vowel-shift.u.vccv.except)
[Gaw rajasthani-gujarati -OG -WK -G -OMarw] a {d y} a > e / _ C                          ; @hiatus — 12 lects / MIA -aya- (< -ada-, -aya-) contracts to e before a consonant (kadala > keḷɔ, nayana > neṇ, gata- > gelɔ) / (changes.sc hiatus.adya.preconsonantal)
[bagri_fatehabad Nimadi Marw had mewari_dholpura mewati_jhambaus mewati_akera marwari_bagra dhundari_badagaon] # ── end of shared prefix ──
[G] ū > u // # C* _ C V                         ; @vowel-shift @x-rg-weight — ū shortens except in an open first syllable (mūla > mūḷ, śūra > sūr, karpūra > kapūr vs sūtra > sutar, śūrpa > supṛũ); Gujarati spelling of u/ū is loose, so partly orthographic
[Buksa Dang Bhoj Kathoriya] ; ── block ks-h-optional: Sporadic kṣ > h after a long vowel: kṣ as a single kʰ, then kʰ > h (dākṣiṇa > Pk. dāhiṇa, H. dāhinā; lākṣā > lāh). `optional:` = sporadic: forks the derivation  [sporadic (blocks/ks-h-optional.sc)] ──
[Buksa Dang Bhoj Kathoriya] optional: k ṣ > h / Long _                   ; @ks — sporadic kṣ > h after a long vowel (single kʰ > h: dākṣiṇa > Pk. dāhiṇa, lākṣā > lāh) / (block ks-h-optional)
[mia-insular -Pk -Ap -Pa -As-Ul] repeat: C > ∅ / _ #                          ; @final-cons-loss — 6 lects / all final consonants (-nt, -rk …) / (changes.sc final-cons-loss.c.repeat)
[Bshk bhatr] ē > i                                        ; @vowel-shift — 2 lects / ē raised to i / (changes.sc vowel-shift.e)
[K] {ō ū} > u / V C* _ C* V                     ; @vowel-shift — long back vowels shorten in non-initial, non-final syllables (masū́ra > mɔsur, trayōdaśī > truvāh); accented and unaccented alike (round 2)
[K] {i u} > {ɛ ɔ} / _ C C                       ; @vowel-shift — short high vowels lower to open-mid ɛ ɔ in (original) closed syllables (níndati > nɛndun, kukṣí > kɔcʰ, kuṇḍá > kɔnḍ); CDIAL K. writes ɛ ɔ here, e o only where a later -ᵘ raised them (round 7: e o > ɛ ɔ)
[K] {i í} > ɛ / # (C) (C) _ C {a á ā ā́}         ; @vowel-shift @umlaut — and first-syllable i lowers to ɛ before a following a or ā (siddʰa > sɛd-, kiráṇa > kɛran, jinā́ti-type); ā added in round 2; e > ɛ in round 7
[K] ē > i / _ C (C) V                           ; @vowel-shift — ē > i when a vowel follows (lēpayati > livun)
[dardic-north] ; ── Clusters I ──────────────────────────────────────────────────────────
[Kho] ; ── block r-anticipation-optional: Sporadic r-anticipation (Dardic/NW 'metathesis'): the r of the first-syllable coda or of the second syllable's onset cluster is copied into the onset of the ini [sporadic (blocks/r-anticipation-optional.sc)] ──
[Kho] optional: {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} > {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} / # _ V r C | # _ V C r ; @metathesis — sporadic r-anticipation into the initial onset (tarka > trakk-, kárman > krom) / (block r-anticipation-optional)
[Sh] ; ── block cr-keep-optional: Sporadic retention of stop + r (Northwest, W. Pahari, NW Ashokan, Shina/Savi, Old Gujarati): in some words the r of stop + r escapes the MIA assimilation (putra [sporadic (blocks/cr-keep-optional.sc)] ──
[Sh] optional: {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} r > {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} ; @assim-sonorant @cr-keep — sporadic retention of stop + r (dā́tra > P dātrī ~ dātī, átra > As. atra) / (block cr-keep-optional)
[dardic-north -Sv -K] {ś s} r > ṣ                                  ; @cluster-simplify — 3 lects / śr/sr > ṣ (úpaśrayati > Kal uṣík, *avaśrayati > uṣím; támisrā > trōmiś has metathesis instead) (round 6) / (changes.sc cluster-simplify.ssr)
[Ku Garh N Dotyali Kho Phal Kal Shum Dm Mai Pas Bishnupriya Or AdivasiOriya OB B hindi -bundeli_atarra -kaithal -NDu -Aw marathi-bhil -hal -M -OM Pk Si Md Pa As-Ul OP D thar awan WK Rj jaun dog kiuth kc kgr ktg kul] m r > m b / V _                              ; @mr-mb — 5 lects / medial mr > mb with an epenthetic stop, as Pa (āmrá > aṁba, tāmrá > taṁba, āmrātaka > aṁbāḍaya); round 7 / (changes.sc mr-mb.mr.postvocalic)
[-bagheli_lakshman -Kannauji-Central -Noiri] ; ── block ks-cch-optional: Sporadic kṣ > ccʰ beside kkʰ (MIA doublets: akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta). `optional:` = sporadic: forks the derivation where it applies (best [sporadic (blocks/ks-cch-optional.sc)] ──
[-bagheli_lakshman -Kannauji-Central -Noiri] optional: k ṣ > c cʰ                         ; @ks — sporadic kṣ > ccʰ beside kkʰ (akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta) / (block ks-cch-optional)
[WK Rj] # ── Common MIA core (shared; Pali chain minus Pali-specific changes; round-7 integration) ──
[G] ; ∅ > i / C _ t a v y a                      ; @x-rg-set-i — connecting -i- of the seṭ gerundive after a consonant root (-itavya: likʰ-itavya- > MIA likʰiavva-)   [removed: -itavya is now on the verb menu; an i inserted only before -tavya is morphology]
[G] a y a > ē / _ C // _ t {i ē} #               ; @aya-e — OIA medial -aya- > ē before a consonant (nayana > nēṇa), not in the -ayati verb ending (ākalayati > M ākaḷṇẽ); unified at round-7 integration
[-KochariyaEastDanuwar -As-Ul -OG -had] ; ── block dental-retroflex-optional: Sporadic retroflexion of initial dentals (daśa > ḍasa, daṁśa > ḍaṁsa, dahati > ḍahai). `optional:` = sporadic: forks the derivation where it applies (best-of; b [sporadic (blocks/dental-retroflex-optional.sc)] ──
[-KochariyaEastDanuwar -As-Ul -OG -had] optional: {t d dʰ} > {ṭ ḍ ḍʰ} / # _          ; @retroflex — sporadic initial dental > retroflex (daśa > ḍasa, daṁśa > ḍaṁsa, *tar- > ṭar-) / (block dental-retroflex-optional)
[Goj] {ai au} > {ē ō}                              ; @ai-au — 11 lects / round 6: restored (saúvarṇa > sōn-, not *hauvno) / (changes.sc ai-au.aiau.3)
[OM] au > ō                                      ; @ai-au — round 2: au > ō (kaudi > kōḍī type; −0.0003); ai > ē does not help OM (Tulpule's tatsama ai)
[mewari_dholpura] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[mewari_dholpura] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[Nimadi] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[Nimadi] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[OM] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[OM] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[rajasthani-gujarati -OG -WK -G -OMarw] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[rajasthani-gujarati -OG -WK -G -OMarw] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[Garh dardic-north Bshk Tor bhatr Mai Tir Khandesi Rathawi Vasavi OM Noiri Bhili mia-insular -NiDoc -Md Gmb rajasthani-gujarati -OG -WK -Marw -mewati_jhambaus -Rj -G -marwari_bagra -OMarw dog kiuth khash kvar kul] r̩ > i                                       ; @r-vocalic — 118 lects / (changes.sc r-vocalic)
[Or B OG OMarw] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[Or B OG OMarw] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[-K -dod -L -awan -mewari_basad] ; ── block ks-h-optional: Sporadic kṣ > h after a long vowel: kṣ as a single kʰ, then kʰ > h (dākṣiṇa > Pk. dāhiṇa, H. dāhinā; lākṣā > lāh). `optional:` = sporadic: forks the derivation  [sporadic (blocks/ks-h-optional.sc)] ──
[-K -dod -L -awan -mewari_basad] optional: k ṣ > h / Long _                   ; @ks — sporadic kṣ > h after a long vowel (single kʰ > h: dākṣiṇa > Pk. dāhiṇa, lākṣā > lāh) / (block ks-h-optional)
[-eur -as-ll -Lom -RomWel -RomBalk -Zarg] ; ── block st-retroflex-optional: Sporadic initial st(ʰ)- > ṭʰ beside the regular tʰ (stabdʰa > Pk. ṭʰaddʰa, H. ṭʰaṇḍā; stʰēṣṭʰa > Mth. ṭʰeṭʰ; stōka > Marw. ṭʰoḍɔ). `optional:` = sporadic: forks [sporadic (blocks/st-retroflex-optional.sc)] ──
[-eur -as-ll -Lom -RomWel -RomBalk -Zarg] optional: s {t tʰ} > ṭʰ / # _                ; @retroflex — sporadic initial st(ʰ)- > ṭʰ beside tʰ (stabdʰa > Pk. ṭʰaddʰa, H. ṭʰaṇḍā; stʰēṣṭʰa > Mth. ṭʰeṭʰ; stōka > Marw. ṭʰoḍɔ) / (block st-retroflex-optional)
[Gaw Tor Pas mewati_akera G] ; ── block ks-cch-optional: Sporadic kṣ > ccʰ beside kkʰ (MIA doublets: akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta). `optional:` = sporadic: forks the derivation where it applies (best [sporadic (blocks/ks-cch-optional.sc)] ──
[Gaw Tor Pas mewati_akera G] optional: k ṣ > c cʰ                         ; @ks — sporadic kṣ > ccʰ beside kkʰ (akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta) / (block ks-cch-optional)
[bihari-tharu Ku Garh N Dotyali dardic-north -Sh -pog -Ush -Kal dardic-south -Tor -Mai -Pas eastern hindi marathi-bhil -hal -M northwest nuristani -Ash -Wg -Kt rajasthani-gujarati west-pahari -jaun -kiuth -kul] ; ── block mia-sm: MIA sibilant + m, two outcomes (2026-10-01): mh (central, eastern, southern IA: asmad > Pk. amha > H. ham, B āmi; grīṣma > gimha) and ss (Northwest, Dardic: asm [cross-family MIA clusters: sm > mh | ss (blocks/mia-sm.sc)] ──
[bihari-tharu Ku Garh N Dotyali dardic-north -Sh -pog -Ush -Kal dardic-south -Tor -Mai -Pas eastern hindi marathi-bhil -hal -M northwest nuristani -Ash -Wg -Kt rajasthani-gujarati west-pahari -jaun -kiuth -kul] {s ś ṣ} m > mʰ | s s                         ; @sm-mh — sm > mh | ss (asmad > Pk. amha, H. ham | P asī) / (block mia-sm)
[PalyaBareli hal Bhilali] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[PalyaBareli hal Bhilali] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[bihari-tharu Ku Garh N Dotyali dardic-north -Sh -pog dardic-south -Mai eastern hindi marathi-bhil -M northwest Gmb Pr rajasthani-gujarati -G west-pahari -dog -kiuth -mand -cur -cam -kul] ; ── block mia-h-clusters: MIA h + sonorant metathesis and the tm/hv assimilations, cross-family (2026-10-01; tmp/sound_changes/hclusters). Before this block most chains had no rule for t [cross-family MIA clusters (blocks/mia-h-clusters.sc)] ──
[bihari-tharu Ku Garh N Dotyali dardic-north -Sh -pog dardic-south -Mai eastern hindi marathi-bhil -M northwest Gmb Pr rajasthani-gujarati -G west-pahari -dog -kiuth -mand -cur -cam -kul] h m > mʰ                                     ; @h-metathesis — hm > mh (brāhmaṇa > Pk. bamhaṇa > H. bāmʰan) / (block mia-h-clusters)
[bihari-tharu Ku Garh N Dotyali dardic-north -Sh -pog dardic-south -Mai eastern hindi marathi-bhil -M northwest Gmb Pr rajasthani-gujarati -G west-pahari -dog -kiuth -mand -cur -cam -kul] h n > nʰ                                     ; @h-metathesis — hn > nh (cihna > cinʰa, S cinʰu) / (block mia-h-clusters)
[bihari-tharu Ku Garh N Dotyali dardic-north -Sh -pog dardic-south -Mai eastern hindi marathi-bhil -M northwest Gmb Pr rajasthani-gujarati -G west-pahari -dog -kiuth -mand -cur -cam -kul] h ṇ > ṇ h                                    ; @h-metathesis — hṇ > ṇh (pūrvāhṇa > Pk. puvvaṇha) / (block mia-h-clusters)
[bihari-tharu Ku Garh N Dotyali dardic-north -Sh -pog dardic-south -Mai eastern hindi marathi-bhil Pk northwest Gmb Pr rajasthani-gujarati -G west-pahari -dog -kiuth -mand -cur -cam -kul] h v > b bʰ                                   ; @hv-bbh — hv > bbʰ (jihvā > Pk. jibbʰā > H. jībʰ) / (block mia-h-clusters)
[bihari-tharu Ku Garh N Dotyali dardic-north -Sh -pog dardic-south -Mai eastern hindi marathi-bhil -M northwest Gmb Pr rajasthani-gujarati west-pahari -dog -kiuth -mand -cur -cam -kul] t m > p p                                    ; @tm-pp — tm > pp (ātman > Pk. appa > H. āp, apnā) / (block mia-h-clusters)
[bihari-tharu -Rana] t s {t tʰ} > ṭ ṭʰ                            ; @ks — 8 lects / round 7: ut- + stʰā- > MIA uṭṭʰ- (*ut-stʰāti > uṭʰ, *ut-stʰāna > uṭʰān), before ts > ccʰ / (changes.sc ks.tstth)
[bihari-tharu] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[bihari-tharu] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[bihari-tharu -Darai Ku N Dotyali eastern -BH -A -B -MH -KS hindi -OAw -H -SSr marathi-bhil -hal -M -OM -VB Si Md OSi As-Ul northwest -D -Khol -L rajasthani-gujarati -mewari_basad -WK -mewari_dholpura -OMarw -Goj dog kgr mand] ; ── block mia-early: Early MIA: ai au > ē ō, r̩ > i, then the sibilant merger with the palatal and s-cluster changes. Shared by 18 lects in 7 clades (Bihari, Early NIA, Lahndic, Pun [Early MIA (blocks/mia-early.sc)] ──
[bihari-tharu -Darai Ku N Dotyali eastern -BH -A -B -MH -KS hindi -OAw -H -SSr marathi-bhil -hal -M -OM -VB Si Md OSi As-Ul northwest -D -Khol -L rajasthani-gujarati -mewari_basad -WK -mewari_dholpura -OMarw -Goj dog kgr mand] ; ── block mia-vowels: Early MIA vowel changes (blocks theme "mia", 2026-09-30): the diphthongs ai au monophthongise and syllabic r̩ becomes i. Lects with other r̩ outcomes (r̩ > ri-, [@ai-au @r-vocalic — early MIA vowels: ai au > ē ō, r̩ > i (blocks/mia-vowels.sc; blocks round 2026-09-30)] ──
[bihari-tharu -Mth -Darai -Bi Ku N Dotyali eastern -BH -A -B -MH -KS hindi -OAw -H -SSr marathi-bhil -hal -M -OM -VB Si Md OSi As-Ul northwest -D -Khol -L rajasthani-gujarati -mewari_basad -bagri_fatehabad -WK -mewari_dholpura -OMarw -Goj dog kgr mand] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[bihari-tharu -Darai Ku N Dotyali eastern -BH -A -B -MH -KS hindi -OAw -H -SSr marathi-bhil -hal -M -OM -VB Si Md OSi As-Ul northwest -D -Khol -L rajasthani-gujarati -mewari_basad -WK -mewari_dholpura -OMarw -Goj dog kgr mand] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[bihari-tharu -Darai Ku N Dotyali eastern -BH -A -B -MH -KS hindi -OAw -H -SSr marathi-bhil -hal -M -OM -VB Si Md OSi As-Ul northwest -D -Khol -L rajasthani-gujarati -mewari_basad -WK -mewari_dholpura -OMarw -Goj dog kgr mand] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@sibilant-s @palatal-Cy @ks @s-stop-aspirate @assim-sonorant — ś ṣ > s, then Cy, kṣ/ts/ps, s + stop, s + sonorant (blocks/mia-sibilants.sc; blocks round 2026-09-30, pass 2)] ──
[bihari-tharu Ku Garh N Dotyali eastern -BH -A -B -MH -KS hindi -SSr marathi-bhil -M Ap Si Md OSi As-Ul northwest -D rajasthani-gujarati dog kgr mand cam] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[bihari-tharu -Darai Ku N Dotyali eastern -BH -A -B -MH -KS hindi -OAw -H -SSr marathi-bhil -hal -M -OM -VB Si Md OSi As-Ul northwest -D -Khol -L rajasthani-gujarati -mewari_basad -WK -mewari_dholpura -OMarw -Goj dog kgr mand] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[bihari-tharu -Darai Ku N Dotyali eastern -BH -A -B -MH -KS hindi -OAw -H -SSr marathi-bhil -hal -M -OM -VB Si Md OSi As-Ul northwest -D -Khol -L rajasthani-gujarati -mewari_basad -WK -mewari_dholpura -OMarw -Goj dog kgr mand] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[bihari-tharu Ku Garh N Dotyali eastern hindi marathi-bhil -OM mia-insular -KharI northwest -D rajasthani-gujarati west-pahari] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[bihari-tharu Ku Garh N Dotyali eastern -BH -A -MH -KS hindi marathi-bhil -hal -M -VB Si Md OSi As-Ul northwest -D rajasthani-gujarati dog kgr mand cam] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[bihari-tharu -Darai Ku N Dotyali eastern -BH -A -B -MH -KS hindi -OAw -H -SSr marathi-bhil -hal -M -OM -VB Si Md OSi As-Ul northwest -D -Khol -L rajasthani-gujarati -mewari_basad -WK -mewari_dholpura -OMarw -Goj dog kgr mand] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[bihari-tharu Ku Garh N Dotyali eastern -BH -A -MH -KS hindi -H marathi-bhil -hal -M -OM Ap Si Md OSi As-Ul northwest -D -Khol -L rajasthani-gujarati -mewari_basad -mewari_dholpura -OMarw dog kgr mand] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[bihari-tharu -Darai Ku Garh N Dotyali eastern -BH -A -B -MH -KS hindi marathi-bhil -M -OM Ap Si Md OSi As-Ul northwest rajasthani-gujarati -mewari_dholpura dog kgr mand] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[As-Ul Nimadi mewati_jhambaus G] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@sibilant-s @palatal-Cy @ks @s-stop-aspirate @assim-sonorant — ś ṣ > s, then Cy, kṣ/ts/ps, s + stop, s + sonorant (blocks/mia-sibilants.sc; blocks round 2026-09-30)] ──
[As-Ul Nimadi mewati_jhambaus G] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[As-Ul Nimadi mewati_jhambaus G] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[As-Ul Nimadi mewati_jhambaus G] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[As-Ul Nimadi mewati_jhambaus G] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[As-Ul Nimadi mewati_jhambaus G] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[As-Ul Nimadi mewati_jhambaus G] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[As-Ul Nimadi mewati_jhambaus G] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[As-Ul Nimadi mewati_jhambaus G] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[srk] ; ── block mia-early: Early MIA: ai au > ē ō, r̩ > i, then the sibilant merger with the palatal and s-cluster changes. Shared by 18 lects in 7 clades (Bihari, Early NIA, Lahndic, Pun [Early MIA (blocks/mia-early.sc)] ──
[srk] ; ── block mia-vowels: Early MIA vowel changes (blocks theme "mia", 2026-09-30): the diphthongs ai au monophthongise and syllabic r̩ becomes i. Lects with other r̩ outcomes (r̩ > ri-, [@ai-au @r-vocalic — early MIA vowels: ai au > ē ō, r̩ > i (blocks/mia-vowels.sc; blocks round 2026-09-30)] ──
[srk] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[srk] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[srk] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@sibilant-s @palatal-Cy @ks @s-stop-aspirate @assim-sonorant — ś ṣ > s, then Cy, kṣ/ts/ps, s + stop, s + sonorant (blocks/mia-sibilants.sc; blocks round 2026-09-30, pass 2)] ──
[srk] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[srk] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[srk] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[srk] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[srk] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[srk] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[srk] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[srk] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] ; ── block mia-early: Early MIA: ai au > ē ō, r̩ > i, then the sibilant merger with the palatal and s-cluster changes. Shared by 18 lects in 7 clades (Bihari, Early NIA, Lahndic, Pun [Early MIA (blocks/mia-early.sc)] ──
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] ; ── block mia-vowels: Early MIA vowel changes (blocks theme "mia", 2026-09-30): the diphthongs ai au monophthongise and syllabic r̩ becomes i. Lects with other r̩ outcomes (r̩ > ri-, [@ai-au @r-vocalic — early MIA vowels: ai au > ē ō, r̩ > i (blocks/mia-vowels.sc; blocks round 2026-09-30)] ──
[Marw had hadothi_kelwada mewati_jhambaus Rj mewati_akera marwari_bagra dhundari_bamore dhundari_badagaon] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@sibilant-s @palatal-Cy @ks @s-stop-aspirate @assim-sonorant — ś ṣ > s, then Cy, kṣ/ts/ps, s + stop, s + sonorant (blocks/mia-sibilants.sc; blocks round 2026-09-30, pass 2)] ──
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewari_dholpura -G -OMarw -Goj] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[Chitwan Kathoriya Bote Gowro Aw Khol rajasthani-gujarati] m {r l} > m b                                ; @mr-mb — 21 lects / mr/ml > mb with an intrusive stop (āmra > amba > ām(b)o, āmlikā > ãbalī) / (changes.sc mr-mb.mrl)
[bihari-tharu] m r > m b                                    ; @assim-sonorant — 10 lects / round 7: MIA mr > mb (āmrá > amba > ām, tāmrá > tamba > tām) / (changes.sc assim-sonorant.mr)
[A] ō > u / _ C* {i u ī ū}                  ; @umlaut — height harmony: ō raised before a following high vowel (kṓṣṭʰa > A. kuṭʰī, prōṣṭʰī > puṭʰi)
[A OMarw] ; ── block mia-vowels: Early MIA vowel changes (blocks theme "mia", 2026-09-30): the diphthongs ai au monophthongise and syllabic r̩ becomes i. Lects with other r̩ outcomes (r̩ > ri-, [@ai-au @r-vocalic — early MIA vowels: ai au > ē ō, r̩ > i (blocks/mia-vowels.sc; blocks round 2026-09-30)] ──
[A OMarw] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[OMarw] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[west-pahari] ; ── block mia-sibilants-keep: MIA sibilant/palatal clusters in lects that keep ś (W. Pahari; blocks theme "mia", 2026-09-30): ṣ > ś instead of the mainland ś ṣ > s, then Cy, kṣ/ts/ps and sib [@sibilant-keep @palatal-Cy @ks @s-stop-aspirate — ṣ > ś kept apart from s, then Cy, kṣ/ts/ps, sibilant + stop (blocks/mia-sibilants-keep.sc; blocks round 2026-09-30)] ──
[Wot west-pahari] ṣ > ś                                        ; @sibilant-keep @sibilant-ś — 17 lects / ṣ > ś, ś kept distinct from s (varṣa > variś-, śīrṣa > śiś, puṣpa > puśpa) / (changes.sc sibilant-keep)
[dog kgr mand] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[west-pahari] {k t p} {ś s} > {k c c} {kʰ cʰ cʰ}@1         ; @ks @ts-cluster — 11 lects / kṣ (now kś) ts ps > kkʰ ccʰ (akṣi > akkʰi, vatsa > vaccʰa, kṣētra > kʰētta) / (changes.sc ks.ktpss)
[west-pahari] {s ś} Stop > $2-asp $2+asp                   ; @s-stop-aspirate — 16 lects / ś/s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.ssstop)
[kiuth kc ktg WPah] {t tʰ d dʰ} r > {ṭ͡ḷ ṭ͡ḷ ḍ͡ḷ ḍ͡ḷʰ} / # _ | {ā ī ū ē ō} _ ; @tr-lateral — 3 lects / dental + r (initial, or after a long vowel) > retroflex lateral affricate (kept in Bhalesi/Khashali) / (changes.sc tr-lateral.tthddhr.aiueo)
[bhad] {t tʰ d dʰ k kʰ g gʰ p pʰ b bʰ} r > {ṭ͡ḷ ṭ͡ḷ ḍ͡ḷ ḍ͡ḷʰ ṭ͡ḷ ṭ͡ḷ ḍ͡ḷ ḍ͡ḷʰ ṭ͡ḷ ṭ͡ḷ ḍ͡ḷ ḍ͡ḷʰ} / # _ | {ā ī ū ē ō} _ ; @tr-lateral — any stop + r (initial, or after a long vowel) > retroflex lateral affricate (kept in Bhalesi/Khashali)
[eur B Brj SSr M Pk Si Pa] ; ── block ks-nasal-optional: Sporadic kṣ + nasal > nasal + h, parallel to ṣṇ > ṇh (ślakṣṇa > Pk. saṇha, tīkṣṇa > tiṇha, pakṣman > pamha). `optional:` = sporadic: forks the derivation where  [sporadic (blocks/ks-nasal-optional.sc)] ──
[eur B Brj SSr M Pk Si Pa] optional: k ṣ {ṇ n m} > {ṇ ṇ m}@3 h          ; @ks — sporadic kṣ + nasal > nasal + h, like ṣṇ > ṇh (ślakṣṇa > saṇha, tīkṣṇa > tiṇha, pakṣman > pamha) / (block ks-nasal-optional)
[OSi] k > v / a _ ā #                             ; @int-stop-loss @ya-sruti — k of final -akā > v (Si -anavā, OSi -navā: ārōcayati > arojanavā); round 6
[MagahiNepal Vasavi Noiri RathwiBareli Md OSi bagri_fatehabad dhundari_badagaon] Sib > s                                      ; @sibilant-s — 89 lects / (changes.sc sibilant-s.sib)
[OM] a > ā / # (C) _ Nasal Stop                   ; @anusvara-vowel @degem-lengthen — 2 lects / round 6: initial-syllable a lengthened before a homorganic nasal cluster (aŋkana > āŋkaṇā) / (changes.sc anusvara-vowel.a.cnasalstop)
[Noiri RathwiBareli Md OSi] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / (changes.sc ks.ktps)
[MagahiNepal Hajong MB Vasavi] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / (changes.sc palatal-Cy.tthddhy)
[MagahiNepal] {k t p} s > {c c c} {cʰ cʰ cʰ}@1            ; @ks — Magadhan kṣ > ccʰ (kṣurá > cʰuri)
[MagahiNepal Vasavi Noiri RathwiBareli VB Ap Md OSi] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. [@s-stop-aspirate @assim-sonorant — s + stop > aspirated geminate, s + sonorant > ss (blocks/mia-s-clusters.sc; blocks round 2026-09-30)] ──
[MagahiNepal Vasavi Noiri RathwiBareli Md OSi] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[MagahiNepal Vasavi Noiri RathwiBareli Md OSi] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[Bshk Pas] k ṣ > ʦ̣ʰ / # _                              ; @ks — 2 lects / initial kṣ > ʦ̣ʰ / (changes.sc ks.ks.initial.2)
[Kho kaithal] m r > m / V _                                ; @mr-mb — medial mr > m (S āmra > āmo, tāmra > ṭrāmo); unified at round-7 integration
[dardic-north -Sv -Kalk -Ush -K -dod Gaw Chil Dm bhatr Tir] ; ── block mia-dardic-ks: Dardic kṣ and ts/ds (Shina, Chitrali; blocks theme "mia", 2026-09-30): kṣ > ʦ̣ʰ and t/d + s > ʦ, the Dardic reflexes of the clusters that the mainland turns int [@ks @ts-cluster — kṣ > ʦ̣ʰ, t d + s > ʦ (blocks/mia-dardic-ks.sc; blocks round 2026-09-30)] ──
[Phal Gaw Chil Dm bhatr Tir] k ṣ > ʦ̣ʰ                                    ; @ks — 15 lects / kṣ > ʦ̣ʰ (akṣi > aʦ̣ʰi, kṣīra > ʦ̣ʰīr, rákṣati > raʦ̣ʰ-) / (changes.sc ks.ks)
[dardic-north -Sv -Kalk -Ush -K -dod Gaw Chil Dm bhatr Tir] {t d} s > ʦ                                  ; @ts-cluster — 9 lects / t d + s > ʦ (vatsa > vaʦ-, utsa > uʦ-) / (changes.sc ts-cluster.tds)
[Bshk] k ṣ > ʦ̣                                     ; @ks @nur-archaism — 10 lects / non-initial kṣ > ʦ̣ (ŕ̩kṣa > iʦ̣) / (changes.sc ks.ks.2)
[Bshk] bʰ r > j                                    ; @tr-affricate — *bʰr > j
[Bshk] p r > c                                     ; @tr-affricate — *pr > c (prahāra > …)
[Ash] {ē ō ē̃ ō̃} > {e o ẽ õ}                       ; @notation — Aw (Saksena) marks ē ō only where CL lengthened a vowel (later rule), so the macron is dropped before CL
[nuristani] ; ── Citation morphology ──────────────────────────────────────────────────
[Kt] ; round 7: legacy @morph `a t {i e} > e / _ #` (Kt verbs in -e) removed as a neutral removal after the 10:35 verb-flag fix: ext, verb, ext_acc and plain all unchanged (the -e comes from -ati through ai-au.aai.final)
[Kam] a t {i e} > a / _ #                         ; @morph — thematic presents (drámati > *drama)
[Wg] ; round 7: legacy @morph `a t {i e} > a / _ #` removed as a neutral removal: ext 0/−0.0004, verb and ext_acc unchanged; plain +0.0017/−0.0027 with full apocope (+0.0040/+0.0008 without)
[PNur] ; round 7: legacy @morph `a t {i e} > a / _ #` (thematic presents) removed as a neutral removal: ext, verb, ext_acc unchanged; plain +0.0015/+0.0017 (it only served the bare headword of nouns in -ati)
[Pr] ; round 7: legacy @morph `if:verb a t {i e} > a / _ #` removed as a neutral removal: ext, verb, ext_acc unchanged (verbs never see @morph in ext/verb scoring); plain +0.0024/+0.0010
[Ash] a y a t {i e} > i e / _ #                   ; @morph — Ash verbs: Strand cites them in -ie (gar̆̃-ie, kas-íe); the CDIAL/Buddruss tail strips it
[Ash] a t {i e} > i e / _ #                       ; @morph — as above (−0.0063 with the tail)
[nuristani] ; ── PIIr palatals (OIA ś j h) ─────────────────────────────────────────────
[BH MH KS Ko-Ul] ; ── block ks-cch-optional: Sporadic kṣ > ccʰ beside kkʰ (MIA doublets: akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta). `optional:` = sporadic: forks the derivation where it applies (best [sporadic (blocks/ks-cch-optional.sc)] ──
[BH MH KS Ko-Ul] optional: k ṣ > c cʰ                         ; @ks — sporadic kṣ > ccʰ beside kkʰ (akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta) / (block ks-cch-optional)
[Gowro Tir pan] a y a > e                                    ; @aya-e — OIA -aya- > e (tráyaḣ > tre, vayám > be); unified at round-7 integration
[Gaw Pas BH A MH KS] ; ── block ks-h-optional: Sporadic kṣ > h after a long vowel: kṣ as a single kʰ, then kʰ > h (dākṣiṇa > Pk. dāhiṇa, H. dāhinā; lākṣā > lāh). `optional:` = sporadic: forks the derivation  [sporadic (blocks/ks-h-optional.sc)] ──
[Gaw Pas BH A MH KS] optional: k ṣ > h / Long _                   ; @ks — sporadic kṣ > h after a long vowel (single kʰ > h: dākṣiṇa > Pk. dāhiṇa, lākṣā > lāh) / (block ks-h-optional)
[Tir KharI] r Sib > $2                                   ; @assim-r @cluster-simplify — 2 lects / r + sibilant: sibilant (darśana > daśana); r + stop kept / (changes.sc assim-r.rsib)
[Darai Khol L bagri_fatehabad mewari_dholpura dhundari_badagaon cam] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks [@palatal-Cy @ks @ts-cluster — dental + y > palatal geminate, kṣ ts ps > kkʰ ccʰ (blocks/mia-palatals.sc; blocks round 2026-09-30, pass 2)] ──
[bagri_fatehabad dhundari_badagaon] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[bagri_fatehabad dhundari_badagaon] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[bhad kgr cur khash cam pan] {s ś} Stop > $2-asp $2+asp                   ; @s-stop-aspirate — 16 lects / (changes.sc s-stop-aspirate.ssstop)
[west-pahari -kvar] {r y} y > j j                                ; @palatal-Cy — 8 lects / ry, yy > jj (kārya > kajj-) / (changes.sc palatal-Cy.ryy)
[kc] {s ś h} Nasal > $2 $2+asp                    ; @s-nasal — sibilant/h + nasal > nasal + aspirated nasal
[Pk] optional: ∅ > i / V r _ Sib                ; @anaptyxis — sporadic svarabhakti i in r + sibilant (Pischel §135: amarṣa > amarisa, ādarśá > ādarisa, árśas > arisa, utkarṣa > ukkarisa beside vāsa, dasaṇa; hill-climb round 5)
[Pk] ; ── block mia-early: Early MIA: ai au > ē ō, r̩ > i, then the sibilant merger with the palatal and s-cluster changes. Shared by 18 lects in 7 clades (Bihari, Early NIA, Lahndic, Pun [Early MIA (blocks/mia-early.sc)] ──
[Pk] ; ── block mia-vowels: Early MIA vowel changes (blocks theme "mia", 2026-09-30): the diphthongs ai au monophthongise and syllabic r̩ becomes i. Lects with other r̩ outcomes (r̩ > ri-, [@ai-au @r-vocalic — early MIA vowels: ai au > ē ō, r̩ > i (blocks/mia-vowels.sc; blocks round 2026-09-30)] ──
[Pk] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[Pk] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[Pk] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@sibilant-s @palatal-Cy @ks @s-stop-aspirate @assim-sonorant — ś ṣ > s, then Cy, kṣ/ts/ps, s + stop, s + sonorant (blocks/mia-sibilants.sc; blocks round 2026-09-30, pass 2)] ──
[Pk] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[Pk] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[Pk] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[Pk] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[Pk] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[Pk] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[Pk] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[S kcch khet] # ── Sindhic post-MIA prefix: first 2 of 2 shared changes (round-7 unification) ──
[L] d v > b b                                    ; @dv-b — 16 lects / dv > b (dvādaśa > bārā, dvitīya > S ɓījo) / (changes.sc dv-b)
[eur RomBalk Zarg] k ṣ > cʰ / # _                               ; @ks — 3 lects / initial kṣ > cʰ (kṣurikā > cʰuri), medial kʰ (akṣi > akʰ) / (changes.sc ks.ks.initial)
[central-east-pahari-romani -Ku -Garh -N -Dotyali] k ṣ > kʰ                                     ; @ks — 6 lects / akʰ < akṣi, drakʰ / (changes.sc ks.ks.3)
[eur RomWel RomBalk Zarg pan] ṣ > ś                                        ; @sibilant-keep @sibilant-ś — 17 lects / ś ṣ merge as ś, s stays (śero, manuś vs. sasto) / (changes.sc sibilant-keep)
[eur RomBalk Zarg Lom] {t p} s > cʰ                                 ; @ts-cluster — 5 lects / ts ps > cʰ (matsya > macʰo) / (changes.sc ts-cluster.tps)
[eur RomBalk Zarg] # ── Migratory post-MIA prefix: first 3 of 3 shared changes (round-7 unification) ──
[Tir] ; ── block mia-vowels: Early MIA vowel changes (blocks theme "mia", 2026-09-30): the diphthongs ai au monophthongise and syllabic r̩ becomes i. Lects with other r̩ outcomes (r̩ > ri-, [inherited from the Migratory sisters (small-lect default, 2026-09-30)] ──
[Tir] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[Tir] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[as-ll] ṣ > ś / _ Stop                               ; @sibilant-keep — ṣ before stops stays š (ōṣṭʰa > ōśt, śuṣka > śkʌ)
[as-ll] {ś ṣ} > s // _ Stop                          ; @sibilant-s — elsewhere Domari merges the sibilants as s (dáśa > das, śākya > sak)
[Lom] Sib > s                                      ; @sibilant-s — 89 lects / Lomavren merges the sibilants as s (manus, les) / (changes.sc sibilant-s.sib)
[central-east-pahari-romani -Ku -Garh -N -Dotyali] {t tʰ d dʰ} y > {c c j j}                    ; @palatal-Cy — 3 lects / satya > caco, tatʰya > caco / (changes.sc palatal-Cy.tthddhy.2)
[eur RomWel RomBalk Zarg] ; ── block lar-romani-aspiration: Proto-Romani laryngeal changes, in order: (1) aspiration shift — an initial voiced stop takes the aspiration of a following voiced aspirate (Grassmann-type disp [@aspiration-shift @asp-devoicing @deaspiration — Proto-Romani laryngeals: aspiration shift, initial devoicing, medial deaspiration (blocks/lar-romani-aspiration.sc)] ──
[eur RomWel RomBalk Zarg] {b d g j} > {pʰ tʰ kʰ cʰ} / # _ V (C) VAsp   ; @aspiration-shift — 3 lects / jámbʰa > cʰam, dugdʰá > tʰud / (changes.sc aspiration-shift)
[eur RomWel RomBalk Zarg] VAsp > {kʰ cʰ ṭʰ tʰ pʰ} / # _                ; @asp-devoicing — 4 lects / bʰrā́tr̩ > pʰral, dʰūmá > tʰuv, gʰarmá > kʰam / (changes.sc asp-devoicing.vasp.initial)
[eur RomWel RomBalk Zarg] ; ── block lar-deaspiration-voiced: Loss of aspiration in the voiced aspirates (bʰ dʰ gʰ jʰ ḍʰ > b d g j ḍ), everywhere in the word. Shared across families: Dardic (Shinaic, Kohistani, Kunar, Kash [@deaspiration — medial voiced aspirates (mádʰu > mol, dugdʰá > tʰud)] ──
[eur RomWel RomBalk Zarg] {bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ}               ; @deaspiration @tone — 26 lects / dʰūma > Sh. dūm, bʰūmi > K. bum, gʰr̩tá > Gaw. gʌd / (changes.sc deaspiration.bhdhghjhdh)
[eur RomBalk Zarg S kcch khet] # ── end of shared prefix ──
[khash] {t tʰ d dʰ} r > {ṭ͡ḷ ṭ͡ḷ ḍ͡ḷ ḍ͡ḷʰ} / # _                   ; @tr-lateral — initial dental + r > retroflex lateral affricate (kept)
[bhad kgr cur khash cam pan] {s ś h} Nasal > $2+asp                       ; @s-nasal — 2 lects / sibilant/h + nasal > aspirated nasal (kr̩ṣṇa > kānʰ) / (changes.sc s-nasal.sshnasal)
[west-pahari -bhal] r ṇ > n n                                    ; @assim-r — 12 lects / rṇ > nn (dental: kárṇa > kān, ū́rṇā > ūn) / (changes.sc assim-r.rn)
[eur RomBalk] {ṇ n} {ṭ ḍ} > n r                            ; @flap — 2 lects / ṇḍ > nr (maṇḍa > manro, āṇḍa > arno) / (changes.sc flap.nntd)
[as-ll] {t p} s > cʰ                                 ; @ts-cluster — 5 lects / ts ps > cʰ (matsya > macʰo) / (changes.sc ts-cluster.tps)
[as-ll] {ṇ n} {ṭ ḍ} > n d                            ; @nasal-voicing — ṇṭ ṇḍ > nd (kaṇṭa > xand; Romani nr)
[pog] {t d} s > ʦ                                  ; @ts-cluster — 9 lects / ts > ʦ (vatsá > baʦ, mátsya > maʦ-) / (changes.sc ts-cluster.tds)
[Ush] t s > ʦ                                      ; @affrication @cluster-simplify — 13 lects / ts > ʦ (útsa > ūʦ, mátsya > maʦ-); round 6 / (changes.sc affrication.ts)
[Sv] ṣ > ś / _ C                                 ; @x-dardic-north-retroflex-sibilant-loss — ṣ > ś before consonants (dr̩ṣṭá > dēś-)
[Sv] {t d} r > {ɬ l}                              ; @tr-lateral — gā́tra > gāɬ, *jāmātra > źamaɬō, drā́kṣā > lāʦ̣
[Kho] {ś s} v > $1 p                               ; @sv-sp — 3 lects / śvaitra > (i)śpetr / (changes.sc sv-sp)
[Phal] ; ── block nasals-postnasal-voicing: nasals-postnasal-voicing — NT > ND, nasals theme, blocks round 2026-09-30. A voiceless stop, aspirated or not, is voiced after a nasal (or anusvāra-turned-nasal [@nasal-voicing — NT > ND incl. aspirates (blocks/nasals-postnasal-voicing.sc)] ──
[Phal] Nasal {k kʰ c cʰ ṭ ṭʰ t tʰ p pʰ} > $1 {g gʰ j jʰ ḍ ḍʰ d dʰ b bʰ}@2 ; @nasal-voicing — 10 lects / dánta > awan dand, káṇṭaka > awan kaṇḍā, śaŋkú > Kho śoŋg, páñca > Kho ponj / (changes.sc nasal-voicing.nasalkkhcc)
[Sh Ush D] {t d} r > {ʦ̣ ʣ̣}                            ; @tr-affricate — 3 lects / kṣḗtra > ʦ̣ʰeʦ̣, putra > puʦ̣ / (changes.sc tr-affricate.tdr)
[Sh] dʰ r > ʣ̣                                     ; @tr-affricate — the aspirate dʰr joins tr/dr > ʦ̣/ʣ̣ (vīdʰrá > béʣ̣u, *vīdʰriya > biʣ̣i) (round 6)
[Sh Ush Phal] {ś s} v > $1 p / V _                         ; @sv-sp — 2 lects / áśva > aśp / (changes.sc sv-sp.postvocalic)
[central-east-pahari-romani -Ku -Garh -N -Dotyali dardic-north -Kalk -pog -Ush -dod Dm bhatr Mai Tir Pas Pk Ap NiDoc Pa As-Ul northwest -OP -thar -Khol -khet -srk nuristani -Gmb -Pr OG dog kc mand cur khash kvar cam kul] ; ── block r-anticipation-optional: Sporadic r-anticipation (Dardic/NW 'metathesis'): the r of the first-syllable coda or of the second syllable's onset cluster is copied into the onset of the ini [sporadic (blocks/r-anticipation-optional.sc)] ──
[central-east-pahari-romani -Ku -Garh -N -Dotyali dardic-north -Kalk -pog -Ush -dod Dm bhatr Mai Tir Pas Pk Ap NiDoc Pa As-Ul northwest -OP -thar -Khol -khet -srk nuristani -Gmb -Pr OG dog kc mand cur khash kvar cam kul] optional: {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} > {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} / # _ V r C | # _ V C r ; @metathesis — sporadic r-anticipation into the initial onset (tarka > trakk-, kárman > krom) / (block r-anticipation-optional)
[rajasthani-gujarati -OG -G] s k > k k                                    ; @s-stop-aspirate — 5 lects / sk > kʰ (skandʰa > kʰā̃dʰ, *pʰaskā > pʰākʰ); round 7 / (changes.sc s-stop-aspirate.sk)
[bagri_fatehabad mewari_dholpura dhundari_badagaon] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / (changes.sc s-stop-aspirate.sstop)
[eur as-ll RomWel RomBalk Zarg OM] s Stop > $2 / # _                            ; @initial-degem — 5 lects / initial s + stop > stop (stʰūla > tʰulo, stʰāna > tʰan) / (changes.sc initial-degem.sstop)
[Garh] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[Garh] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[Ush Pas nuristani -Gmb] ; ── block ks-s-optional: Sporadic kṣ > ṣ beside the lect's kkʰ/ʦ̣ʰ (drākṣā > Kt. dros, kakṣa > kuṣ, pakṣin > Tor. paśīn). `optional:` = sporadic: forks the derivation where it applies ( [sporadic (blocks/ks-s-optional.sc)] ──
[Ush Pas nuristani -Gmb] optional: k ṣ > ṣ                            ; @ks — sporadic kṣ > ṣ beside ʦ̣ʰ/kkʰ (drākṣā > dros, kakṣa > kuṣ, pakṣin > paśīn) / (block ks-s-optional)
[Or Ko-Ul] ; ── block mia-sm: MIA sibilant + m, two outcomes (2026-10-01): mh (central, eastern, southern IA: asmad > Pk. amha > H. ham, B āmi; grīṣma > gimha) and ss (Northwest, Dardic: asm [cross-family MIA clusters: sm > mh | ss (blocks/mia-sm.sc)] ──
[Or Ko-Ul] {s ś ṣ} m > mʰ | s s                         ; @sm-mh — sm > mh | ss (asmad > Pk. amha, H. ham | P asī) / (block mia-sm)
[Pk] r y > j j                                   ; @palatal-Cy @assim-r — kārya > kajja
[M] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@sibilant-s @palatal-Cy @ks @s-stop-aspirate @assim-sonorant — ś ṣ > s, then Cy, kṣ/ts/ps, s + stop, s + sonorant (blocks/mia-sibilants.sc; blocks round 2026-09-30)] ──
[M] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[M] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[M] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[M] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[M] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[M] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[M] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[Ko-Ul] ; ── block mia-early: Early MIA: ai au > ē ō, r̩ > i, then the sibilant merger with the palatal and s-cluster changes. Shared by 18 lects in 7 clades (Bihari, Early NIA, Lahndic, Pun [Early MIA (blocks/mia-early.sc)] ──
[Ko-Ul] ; ── block mia-vowels: Early MIA vowel changes (blocks theme "mia", 2026-09-30): the diphthongs ai au monophthongise and syllabic r̩ becomes i. Lects with other r̩ outcomes (r̩ > ri-, [@ai-au @r-vocalic — early MIA vowels: ai au > ē ō, r̩ > i (blocks/mia-vowels.sc; blocks round 2026-09-30)] ──
[Ko-Ul] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[Ko-Ul] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[Ko-Ul] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@sibilant-s @palatal-Cy @ks @s-stop-aspirate @assim-sonorant — ś ṣ > s, then Cy, kṣ/ts/ps, s + stop, s + sonorant (blocks/mia-sibilants.sc; blocks round 2026-09-30, pass 2)] ──
[Ko-Ul] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[Ko-Ul] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[Ko-Ul] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[Ko-Ul] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[Ko-Ul] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[Ko-Ul] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[Ko-Ul] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[Ko-Ul] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[Pa] ; ── block mia-early: Early MIA: ai au > ē ō, r̩ > i, then the sibilant merger with the palatal and s-cluster changes. Shared by 18 lects in 7 clades (Bihari, Early NIA, Lahndic, Pun [Early MIA (blocks/mia-early.sc)] ──
[Pa] ; ── block mia-vowels: Early MIA vowel changes (blocks theme "mia", 2026-09-30): the diphthongs ai au monophthongise and syllabic r̩ becomes i. Lects with other r̩ outcomes (r̩ > ri-, [@ai-au @r-vocalic — early MIA vowels: ai au > ē ō, r̩ > i (blocks/mia-vowels.sc; blocks round 2026-09-30)] ──
[Pa] {ai au} > {ē ō}                              ; @ai-au — 97 lects / ai au > ē ō (tailá > tēla, gaurá > gōra, kaulika > kōlia) / (changes.sc ai-au.aiau)
[Pa] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[Pa] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@sibilant-s @palatal-Cy @ks @s-stop-aspirate @assim-sonorant — ś ṣ > s, then Cy, kṣ/ts/ps, s + stop, s + sonorant (blocks/mia-sibilants.sc; blocks round 2026-09-30, pass 2)] ──
[Pa] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[Pa] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[Pa] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[Pa] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[Pa] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[Pa] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[Pa] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[Sv As-Ul northwest -OP -D -Khol -khet dog kiuth kc mand cur khash kul] ; ── block cr-keep-optional: Sporadic retention of stop + r (Northwest, W. Pahari, NW Ashokan, Shina/Savi, Old Gujarati): in some words the r of stop + r escapes the MIA assimilation (putra [sporadic (blocks/cr-keep-optional.sc)] ──
[Sv As-Ul northwest -OP -D -Khol -khet dog kiuth kc mand cur khash kul] optional: {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} r > {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} ; @assim-sonorant @cr-keep — sporadic retention of stop + r (dā́tra > P dātrī ~ dātī, átra > As. atra) / (block cr-keep-optional)
[hindi -OAw] ; ── block mia-sibilants-hindi: MIA sibilants in the Hindi-type lects (blocks theme "mia", pass 2, 2026-09-30): ś ṣ > s, then a vowel + anusvāra before s lengthens without nasalisation (viṁśat [@sibilant-s @anusvara-vowel @palatal-Cy @ks @s-stop-aspirate @assim-sonorant — ś ṣ > s, V ṁ > V̄ before s, then the sibilant clusters (blocks/mia-sibilants-hindi.sc; blocks round 2026-09-30, pass 2)] ──
[hindi -OAw -H] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[hindi -OAw] V ṁ > $1+long / _ s                          ; @anusvara-vowel — 7 lects / V ṁ > V̄ before s (viṁśati > vīsa > bīs, *kaṁsa > kās-) / (changes.sc anusvara-vowel.vm.2)
[hindi -OAw] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[hindi -OAw] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[hindi -OAw -H -SSr] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[hindi -OAw -H -SSr] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[hindi -OAw] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[hindi -OAw -SSr] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[hindi -OAw -H] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[eastern] ; ── block mia-h-clusters: MIA h + sonorant metathesis and the tm/hv assimilations, cross-family (2026-10-01; tmp/sound_changes/hclusters). Before this block most chains had no rule for t [cross-family MIA clusters (blocks/mia-h-clusters.sc)] ──
[eastern] h m > mʰ                                     ; @h-metathesis — hm > mh (brāhmaṇa > Pk. bamhaṇa > H. bāmʰan) / (block mia-h-clusters)
[eastern] h n > nʰ                                     ; @h-metathesis — hn > nh (cihna > cinʰa, S cinʰu) / (block mia-h-clusters)
[eastern] h ṇ > ṇ h                                    ; @h-metathesis — hṇ > ṇh (pūrvāhṇa > Pk. puvvaṇha) / (block mia-h-clusters)
[eastern] h v > b bʰ                                   ; @hv-bbh — hv > bbʰ (jihvā > Pk. jibbʰā > H. jībʰ) / (block mia-h-clusters)
[eastern] t m > p p                                    ; @tm-pp — tm > pp (ātman > Pk. appa > H. āp, apnā) / (block mia-h-clusters)
[eastern -Or -B] ; ── block mia-sibilants-sh: MIA sibilants in the Eastern (Bengali-Assamese-Oriya) lects (blocks theme "mia", 2026-09-30): all sibilants merge in ś (Māgadhī), and the sibilant clusters assi [@sibilant-ś @ks @s-stop-aspirate @assim-sonorant — Eastern: all sibilants > ś, then kś/tś/pś, ś + stop, ś + sonorant (blocks/mia-sibilants-sh.sc; blocks round 2026-09-30)] ──
[eastern -Or -B] Sib > ś                                      ; @sibilant-ś — 11 lects / ś ṣ s > ś (sapta > śatta > B śāt, dāsa > dāśa, varṣa > variśa) / (changes.sc sibilant-ś)
[eastern -Or -B] {k t p} ś > {k c c} {kʰ cʰ cʰ}@1             ; @ks — 13 lects / kś (< kṣ) tś pś > kkʰ ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps.2)
[eastern -Or -B] ś Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 11 lects / ś + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop.2)
[eastern -Or -B] ś Son > ś ś                                  ; @assim-sonorant — 11 lects / ś + sonorant > śś (aśva > aśśa, tasya > taśśa, sahasra > sahaśśa) / (changes.sc assim-sonorant.sson.2)
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] ; ── block mia-core-ri: mia-core plus general r̩ > i (blocks theme "mia", pass 2, 2026-09-30), for core lects whose r + C assimilation comes later in the chain (see mia-core-plus). [@mia-core @acc-loss @r-vocalic — mia-core, accent loss, r̩ > i (blocks/mia-core-ri.sc; blocks round 2026-09-30, pass 2)] ──
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] ; ── block mia-core: Common MIA core — OIA → Middle Indo-Aryan changes shared by the mainland lects. The Pali chain minus its Pali-specific changes (final-consonant loss, -ayati > - ──
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda mia-insular -NiDoc -Dhp -KharI OP thar WK Rj] t > ṭ / # p r a _ i                          ; @prati-retroflexion @mia-core — 6 lects / (changes.sc prati-retroflexion)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda OP thar WK Rj kgr] ∅ > i / # C _ l                              ; @anaptyxis @mia-core — 2 lects / (changes.sc anaptyxis.cl)
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] {ai au} > {ē ō}                              ; @ai-au @mia-core — 97 lects / (changes.sc ai-au.aiau)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda Pk Ap Pa As-Ul OP thar WK Rj kgr] r̩ > i / _ ś                                 ; @r-vocalic @mia-core — 2 lects / (changes.sc r-vocalic.s)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda OP thar WK Rj kgr] r̩ > u / _ C* {u ū}                          ; @r-vocalic @umlaut @mia-core — MIA common core: r̩ > u before a following u (pr̩tʰu > putʰu, mr̩du > mudu) / (changes.sc mia-core.r-vocalic.u)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda Pa OP thar WK Rj kgr] ṁ l > l l                                    ; @nasal-assim @mia-core — MIA common core: saṁlapati > sallapati / (changes.sc mia-core.nasal-assim.l)
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@mia-core — ś ṣ > s, Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[Ku N Bishnupriya Or B Kannauji-Central bagheli_lakshman Brj SSr mewari_bannoda OP Rj] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[Ku N Bishnupriya Or Kannauji-Central bagheli_lakshman Brj mewari_bannoda OP Rj] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[Ku N Bishnupriya Or Kannauji-Central bagheli_lakshman Brj mewari_bannoda OP Rj] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[Ku N Bishnupriya Or Kannauji-Central bagheli_lakshman Brj mewari_bannoda OP Rj] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[Ku N Bishnupriya Or B Kannauji-Central bagheli_lakshman Brj mewari_bannoda OP Rj] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda mia-insular -Pk -Si -Md OP thar WK Rj kgr] j ñ > ñ ñ                                    ; @assim-sonorant @mia-core — 4 lects / (changes.sc assim-sonorant.jn.2)
[Ku Garh N Dotyali eastern -BH -Hajong -MB -MH -KS Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda Pk Si Pa OSi OP thar WK Rj kgr] v y > v v                                    ; @assim-sonorant @mia-core — 4 lects / (changes.sc assim-sonorant.vy)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B hindi -bundeli_atarra -kaithal -NDu -Aw Bhatri PauriBareli M mewari_bannoda Ko-Ul Pk Si Pa OP thar S L WK Rj G kgr] Nasal Nasal > $2 $2                          ; @assim-sonorant @mia-core — 7 lects / (changes.sc assim-sonorant.nasalnasal)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda Pk Ap Si Md OSi OP thar WK Rj kgr] Nasal {y v} > $1 $1                          ; @assim-sonorant @mia-core — 7 lects / (changes.sc assim-sonorant.nasalyv)
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] ; ── block mia-stop-clusters: MIA assimilation of stop + sonorant/nasal and stop + stop clusters (blocks theme "mia", 2026-09-30). The same two changes, in this order, are positions 18–19 of [@mia-core — stop + sonorant/nasal, stop + stop] ──
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] Stop {r y v l Nasal} > $1-asp $1             ; @assim-sonorant — 83 lects / stop + r y v l nasal > geminate stop (putra > putta, śakya > sakka, agni > aggi) / (changes.sc assim-sonorant.stopryvlna)
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda Rj] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[Ku Garh N Dotyali eastern Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda mia-insular -NiDoc -Dhp -KharI -As-Ul OP thar WK Rj kgr] l {y v} > l l                                ; @assim-sonorant @mia-core — 17 lects / (changes.sc assim-sonorant.lyv)
[Ku Garh N Dotyali eastern Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda Pk Si Md OP thar WK Rj] C {r y v l} > $1 / # _                       ; @initial-degem @mia-core — 15 lects / (changes.sc initial-degem.cryvl)
[Ku N Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri mewari_bannoda Ap OP WK Rj] C $1 > $2 / C _                              ; @cluster-simplify @mia-core — 7 lects / (changes.sc cluster-simplify.c.postconsonantal)
[Ku N Bishnupriya Or Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] C C > $2 / # _                               ; @initial-degem @mia-core — 60 lects / (changes.sc initial-degem.cc)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda VB mia-insular -NiDoc -KharI -As-Ul OP thar WK Rj kgr] a v a > ō / # _                              ; @aya-e @mia-core — 9 lects / (changes.sc aya-e.ava.initial)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda Pa OP thar WK Rj kgr] ṁ > ŋ / _ {k kʰ g gʰ}                        ; @nasal-cluster @mia-core — MIA common core: anusvāra homorganic before a velar / (changes.sc mia-core.nasal-cluster.velar)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda Pa OP thar WK Rj kgr] ṁ > n / _ {t tʰ d dʰ}                        ; @nasal-cluster @mia-core — MIA common core: before a dental / (changes.sc mia-core.nasal-cluster.dental)
[Ku Garh N Dotyali Bishnupriya Or AdivasiOriya OB B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri PauriBareli mewari_bannoda Pa OP thar WK Rj kgr] ṁ > m / _ {p pʰ b bʰ}                        ; @nasal-cluster @mia-core — MIA common core: before a labial / (changes.sc mia-core.nasal-cluster.labial)
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] V > $1-acc                                   ; @acc-loss — the Vedic pitch accent is lost after the MIA core changes (áśva > assa: no accent in MIA/NIA reflexes) / (block mia-core-ri)
[Ku N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr mewari_bannoda OP Rj] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[northwest -D -thar -Khol -khet] {l n} y > $1 $1                              ; @assim-sonorant — 7 lects / kālya > kall-, anya > aṇṇ- / (changes.sc assim-sonorant.lny)
[N] ; ── block nasals-nd-nn-pahari: nasals-nd-nn-pahari — NC > anusvāra + C, then anusvāra + g/b > geminate nasal (ŋg > ŋŋ, mb > mm); Central/Eastern Pahari type (Nepali, Kumaoni, Dotyali), nasals [@nasal-cluster @nasal-assim — NC > ṁC, then ṁg ṁb > ŋŋ mm (blocks/nasals-nd-nn-pahari.sc)] ──
[N] ; ── block nasals-nc-anusvara: nasals-nc-anusvara — MIA homorganic nasal + stop > anusvāra + stop, nasals theme, blocks round 2026-09-30 (pass 2). The written MIA anusvāra before a stop (aṁta [@nasal-cluster — homorganic nasal + stop > ṁ + stop: daṇḍá > N ḍā̃ṛo, Ku ḍā̃ṛ] ──
[N] Nasal Stop > ṁ $2                            ; @nasal-cluster — 39 lects / ánta > aṁta, páñca > paṁca > Nimadi pā̃c / (changes.sc nasal-cluster.nasalstop)
[N] ṁ {g gʰ b bʰ} > {ŋ ŋ m m}@2 {ŋ ŋ m m}@2      ; @nasal-assim — 3 lects / ṁg ṁb(ʰ) > ŋŋ mm: áŋgāra > N, Ku aŋār, kumbʰakāra > N kumāle, Ku kumār / (changes.sc nasal-assim.mgghbbh)
[Garh Dotyali Bhatri thar WK] ; ── block mia-core: Common MIA core — OIA → Middle Indo-Aryan changes shared by the mainland lects. The Pali chain minus its Pali-specific changes (final-consonant loss, -ayati > - [common MIA core: blocks/mia-core.sc] ──
[Garh Dotyali Bhatri thar WK] {ai au} > {ē ō}                              ; @ai-au @mia-core — 97 lects / (changes.sc ai-au.aiau)
[Garh Dotyali Bhatri thar WK] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@mia-core — ś ṣ > s, Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[Dotyali Bhatri thar] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[Garh Dotyali Bhatri thar WK] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[Garh Dotyali Bhatri thar WK] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[Dotyali Bhatri thar] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[Dotyali Bhatri thar] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[Garh Dotyali Bhatri thar WK] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[Dotyali Bhatri thar] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[Dotyali Bhatri thar] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[Garh Dotyali Bhatri thar WK] ; ── block mia-stop-clusters: MIA assimilation of stop + sonorant/nasal and stop + stop clusters (blocks theme "mia", 2026-09-30). The same two changes, in this order, are positions 18–19 of [@mia-core — stop + sonorant/nasal, stop + stop] ──
[Garh Dotyali Bhatri thar WK] Stop {r y v l Nasal} > $1-asp $1             ; @assim-sonorant — 83 lects / stop + r y v l nasal > geminate stop (putra > putta, śakya > sakka, agni > aggi) / (changes.sc assim-sonorant.stopryvlna)
[Garh Dotyali Bhatri WK] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[Garh Dotyali thar] C $1 > $2 / C _                              ; @cluster-simplify @mia-core — 7 lects / (changes.sc cluster-simplify.c.postconsonantal)
[Dotyali Bhatri thar WK] C C > $2 / # _                               ; @initial-degem @mia-core — 60 lects / (changes.sc initial-degem.cc)
[hindi -kaithal -OAw -H -Aw] ; ── block nasals-anusvara-sibilant: nasals-anusvara-sibilant — MIA anusvāra before a sibilant > vowel length without nasality, Hindi belt (H, Aw, Brj, NDu, Kannauji, Bagheli, Bundeli), nasals them [inherited from the Hindi sisters (small-lect default, 2026-09-30)] ──
[hindi -kaithal -OAw -H -Aw] V ṁ > $1+long / _ s                          ; @anusvara-vowel — 7 lects / viṁśatí > H bīs, triṁśat > tīs / (changes.sc anusvara-vowel.vm.2)
[hindi -kaithal -OAw -H -Aw] ; ── block mia-sibilants-hindi: MIA sibilants in the Hindi-type lects (blocks theme "mia", pass 2, 2026-09-30): ś ṣ > s, then a vowel + anusvāra before s lengthens without nasalisation (viṁśat [inherited from the Hindi sisters (small-lect default, 2026-09-30)] ──
[hindi -kaithal -OAw -H -Aw] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[hindi -kaithal -OAw -H -Aw] V ṁ > $1+long / _ s                          ; @anusvara-vowel — 7 lects / V ṁ > V̄ before s (viṁśati > vīsa > bīs, *kaṁsa > kās-) / (changes.sc anusvara-vowel.vm.2)
[hindi -kaithal -OAw -H -Aw] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[hindi -kaithal -OAw -H -Aw] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[Kannauji-Central bundeli_atarra NDu bagheli_lakshman Brj] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[hindi -kaithal -OAw -H -Aw] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[hindi -kaithal -OAw -H -Aw] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[hindi -kaithal -OAw -H -Aw] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[Kannauji-Central bundeli_atarra NDu bagheli_lakshman Brj] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[Kannauji-Central] ; ── block nasals-anusvara-sibilant: nasals-anusvara-sibilant — MIA anusvāra before a sibilant > vowel length without nasality, Hindi belt (H, Aw, Brj, NDu, Kannauji, Bagheli, Bundeli), nasals them [@anusvara-vowel — MIA ṁ before sibilant > length without nasal (viṁśatí > bīs) (blocks/nasals-anusvara-sibilant.sc)] ──
[Kannauji-Central] V ṁ > $1+long / _ s                          ; @anusvara-vowel — 7 lects / viṁśatí > H bīs, triṁśat > tīs / (changes.sc anusvara-vowel.vm.2)
[Brj] ; ── block nasals-anusvara-sibilant: nasals-anusvara-sibilant — MIA anusvāra before a sibilant > vowel length without nasality, Hindi belt (H, Aw, Brj, NDu, Kannauji, Bagheli, Bundeli), nasals them [@anusvara-vowel — MIA ṁ before sibilant > length without nasal (viṁśatí > bīs) (blocks/nasals-anusvara-sibilant.sc)] ──
[Brj] V ṁ > $1+long / _ s                          ; @anusvara-vowel — 7 lects / viṁśatí > H bīs, triṁśat > tīs / (changes.sc anusvara-vowel.vm.2)
[bagheli_lakshman] ; ── block nasals-anusvara-sibilant: nasals-anusvara-sibilant — MIA anusvāra before a sibilant > vowel length without nasality, Hindi belt (H, Aw, Brj, NDu, Kannauji, Bagheli, Bundeli), nasals them [@anusvara-vowel — MIA ṁ before sibilant > length without nasal (viṁśatí > bīs) (blocks/nasals-anusvara-sibilant.sc)] ──
[bagheli_lakshman] V ṁ > $1+long / _ s                          ; @anusvara-vowel — 7 lects / viṁśatí > H bīs, triṁśat > tīs / (changes.sc anusvara-vowel.vm.2)
[NDu] # ── Hindi post-MIA prefix: first 3 of 3 shared changes (round-7 unification) ──
[NDu] # ── end of shared prefix ──
[AdivasiOriya OB PauriBareli] ; ── block mia-core-plus: mia-core plus the two MIA changes every mainland core lect adds right after it (blocks theme "mia", pass 2, 2026-09-30): general r̩ > i (mia-core only has r̩ >  [@mia-core @acc-loss @r-vocalic @assim-r — mia-core, accent loss, r̩ > i, rC > CC (blocks/mia-core-plus.sc; blocks round 2026-09-30, pass 2)] ──
[AdivasiOriya OB PauriBareli] ; ── block mia-core: Common MIA core — OIA → Middle Indo-Aryan changes shared by the mainland lects. The Pali chain minus its Pali-specific changes (final-consonant loss, -ayati > - ──
[AdivasiOriya OB PauriBareli] {ai au} > {ē ō}                              ; @ai-au @mia-core — 97 lects / (changes.sc ai-au.aiau)
[AdivasiOriya OB PauriBareli] ; ── block mia-sibilants: MIA sibilant merger and the sibilant/palatal clusters that follow it (blocks theme "mia", 2026-09-30): ś ṣ > s, then dental + y and kṣ/ts/ps (mia-palatals), the [@mia-core — ś ṣ > s, Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[AdivasiOriya OB PauriBareli] Sib > s                                      ; @sibilant-s — 89 lects / ś ṣ > s (śata > sata, dēśa > dēsa, varṣa > varsa > vassa) / (changes.sc sibilant-s.sib)
[AdivasiOriya OB PauriBareli] ; ── block mia-sibilant-clusters: MIA sibilant/palatal clusters after the sibilant merger (blocks theme "mia", pass 2, 2026-09-30): dental + y and kṣ/ts/ps (mia-palatals), then s + stop and s +  [Cy, kṣ/ts/ps, s + stop, s + sonorant] ──
[AdivasiOriya OB PauriBareli] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[AdivasiOriya OB PauriBareli] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[AdivasiOriya OB PauriBareli] {k t p} s > {k c c} {kʰ cʰ cʰ}@1             ; @ks @ts-cluster — 80 lects / ks (< kṣ) > kkʰ, ts ps > ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps)
[AdivasiOriya OB PauriBareli] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[AdivasiOriya OB PauriBareli] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[AdivasiOriya OB PauriBareli] s Son > s s                                  ; @assim-sonorant — 72 lects / s + sonorant > ss (aśva > assa, tasya > tassa, sahasra > sahassa) / (changes.sc assim-sonorant.sson)
[AdivasiOriya OB PauriBareli] ; ── block mia-stop-clusters: MIA assimilation of stop + sonorant/nasal and stop + stop clusters (blocks theme "mia", 2026-09-30). The same two changes, in this order, are positions 18–19 of [@mia-core — stop + sonorant/nasal, stop + stop] ──
[AdivasiOriya OB PauriBareli] Stop {r y v l Nasal} > $1-asp $1             ; @assim-sonorant — 83 lects / stop + r y v l nasal > geminate stop (putra > putta, śakya > sakka, agni > aggi) / (changes.sc assim-sonorant.stopryvlna)
[AdivasiOriya OB PauriBareli] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[PauriBareli] C $1 > $2 / C _                              ; @cluster-simplify @mia-core — 7 lects / (changes.sc cluster-simplify.c.postconsonantal)
[AdivasiOriya OB PauriBareli] C C > $2 / # _                               ; @initial-degem @mia-core — 60 lects / (changes.sc initial-degem.cc)
[AdivasiOriya OB PauriBareli] V > $1-acc                                   ; @acc-loss — the Vedic pitch accent is lost after the MIA core changes (áśva > assa: no accent in MIA/NIA reflexes) / (block mia-core-plus)
[AdivasiOriya OB PauriBareli] r̩ > i                                       ; @r-vocalic — 118 lects / r̩ > i (gʰr̩ta > gʰita, tr̩ṇa > tiṇa, hr̩daya > hidaya) / (changes.sc r-vocalic)
[AdivasiOriya OB PauriBareli] {r ḣ} C > $2-asp $2                          ; @assim-r — 109 lects / r (and visarga) + C > geminate (karma > kamma, sarpa > sappa, dúḥkʰa > dukkʰa) / (changes.sc assim-r.rhc)
[Garh Dotyali Bishnupriya Or AdivasiOriya OB B SSr PauriBareli thar WK Rj] # ── end of core; lect-specific chain follows ──
[Rj] u > a / V C* _ C* V // _ k {a ā} #          ; @vowel-shift — medial u > a (māṇuṣa > māṇas), but not in the -uka suffix (@aka: bʰalluka > bʰālū)
[Garh] ; ── block r-vocalic-r-kept: Dardic/NW keep the r of r̩ (Cr̩ > Cri/Cru/Cra): *tr̩pu > Phal truṇuʦ̣ʰīr, tr̩ṣṭá > Phal trīṣṭo, Kho treṣṭu, gŕ̩dʰya > Dm grī̃c, r̩ṣabʰá > Kho reṣu, K riśu. Reve [reverse-reconstruction loop (blocks/r-vocalic-r-kept.sc)] ──
[Garh] optional: {r̩ ŕ̩} > r a | r i | r u          ; @r-vocalic — r̩ vocalised with its r kept (tr̩ṣṭá > Phal trīṣṭo, Kho treṣṭu; r̩ṣabʰá > K riśu) / (block r-vocalic-r-kept)
[Dotyali] # ── C/E Pahari post-MIA prefix: first 3 of 3 shared changes (round-7 unification) ──
[Dotyali] {á í ú} > $1+long / _ C C                    ; @degem-lengthen @stress — 3 lects / Vedic-tonic vowel before a cluster lengthens even outside the penult (adds to the positional rule; replacing the penult rule by it loses) / (changes.sc degem-lengthen.aiu.cc)
[Dotyali] # ── end of shared prefix ──
[Garh Dotyali thar WK] V > $1-acc                                  ; @acc-loss
[WK] ; ∅ > i / C _ t a v y a                      ; @x-rg-set-i — seṭ -itavya after a consonant root (as G)   [removed: -itavya is now on the verb menu; an i inserted only before -tavya is morphology]
[WK] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[WK] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[H P L] m y > m m                                   ; @assim-sonorant — my > mm, as H (yamyátē > L jammaṇ, P jammṇā, S ʄamaṇu; *ramyati > P ramṇā). The chain had no m + y rule, so my survived to the output (gold my 0; hill-climb round 7)
[L] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / (changes.sc s-stop-aspirate.sstop)
[northwest -D -thar -Khol -khet] optional: r {t tʰ d} > {ṭ ṭ ḍ}@2 {ṭ ṭʰ ḍ}@2  ; @rt-retroflex — 5 lects / r + dental > retroflex geminate (vartana > P vaṭṭṇā, *karttāra > kaṭār, S *garda > gaᶑo) / (changes.sc rt-retroflex.rtthd)
[northwest -D -thar -Khol -khet] Nasal Stop {r y v} > $1 $2                   ; @assim-sonorant — 3 lects / sonorant lost after NC (yantra > janta > jand) / (changes.sc assim-sonorant.nasalstopr)
[Khol] ; ── block nasals-anusvara-nw: nasals-anusvara-nw — anusvāra before s/h and leftover anusvāra, north-western type (Lahnda, Kutchi, Khetrani, Pothwari), nasals theme, blocks round 2026-09-30.  [@anusvara-vowel @anusvara-loss — ṁ before s/h > length, other ṁ lost (blocks/nasals-anusvara-nw.sc)] ──
[Khol] V ṁ > $1+long / _ {s h}                      ; @anusvara-vowel — 6 lects / viṁśatí > khet vīh, poth bī / (changes.sc anusvara-vowel.vm.sh)
[Khol] ; ── block nasals-anusvara-loss: nasals-anusvara-loss — leftover anusvāra lost without nasality, nasals theme, blocks round 2026-09-30 (pass 2). The anusvāra that survives to this point (before [@anusvara-loss — leftover anusvāra dropped: pāṁśu > awan pā] ──
[Garh hindi -bagheli_lakshman M mia-insular -Md -Pa thar rajasthani-gujarati -mewari_basad -marwari_bagra -Goj] ; ── block nasals-nc-anusvara: nasals-nc-anusvara — MIA homorganic nasal + stop > anusvāra + stop, nasals theme, blocks round 2026-09-30 (pass 2). The written MIA anusvāra before a stop (aṁta [@nasal-cluster (blocks/nasals-nc-anusvara.sc)] ──
[Garh hindi -bagheli_lakshman M mia-insular -Md -Pa thar rajasthani-gujarati -mewari_basad -marwari_bagra -Goj] Nasal Stop > ṁ $2                            ; @nasal-cluster — 39 lects / ánta > aṁta, páñca > paṁca > Nimadi pā̃c / (changes.sc nasal-cluster.nasalstop)
[rajasthani-gujarati -OG -WK -G -OMarw] r {t tʰ} > {ṭ ṭ}@2 {ṭ ṭʰ}@2                  ; @assim-r @rt-retroflex — 7 lects / r + t(ʰ) > retroflex geminate (vartati > vaṭvũ, karttāra > kaṭār) / (changes.sc assim-r.rtth)
[bihari-tharu -Chitwan -Morang -KochariyaEastDanuwar Ku Dotyali eastern marathi-bhil -M -OM -Noiri -Ko-Ul] ; ── block nasals-anusvara-loss-i: nasals-anusvara-loss-i — anusvāra lost after i without nasalising it, Bengali/Assamese/Odia, nasals theme, blocks round 2026-09-30 (pass 2). [@anusvara-loss @vowel-shift — anusvāra lost after i without nasalising it (viṁśati > B. bis, kiṁcid > kicʰu, piṁṣati > A. pihiba) (blocks/nasals-anusvara-loss-i.sc)] ──
[bihari-tharu -Chitwan -Morang -KochariyaEastDanuwar Ku Dotyali eastern marathi-bhil -M -OM -Noiri -Ko-Ul] ṁ > ∅ / i _                                  ; @anusvara-loss @vowel-shift — 3 lects / viṁśati > B bis, kiṁcid > B kicʰu / (changes.sc anusvara-loss.i)
[Ku Dotyali eastern -Or -A -AdivasiOriya -OB] ; ── block nasals-nd-nn-east: nasals-nd-nn-east — NC > anusvāra + C except before g/b, where the cluster assimilates to a geminate nasal (ŋg > ŋŋ, mb > mm); Bengali-Assamese type (Bengali, K [@nasal-cluster @nasal-assim — NC > ṁC, but ŋg mb > ŋŋ mm (blocks/nasals-nd-nn-east.sc)] ──
[Ku Dotyali eastern -Or -A -AdivasiOriya -OB] Nasal {k kʰ c cʰ ṭ ṭʰ t tʰ p pʰ j jʰ ḍ ḍʰ d dʰ} > ṁ $2 ; @nasal-cluster — 5 lects / not before g b: paŋka > B pā̃k, daṇḍa > B ḍā̃ṛ / (changes.sc nasal-cluster.nasalkkhcc.2)
[Ku Dotyali eastern -Or -A -AdivasiOriya -OB] {ŋ m} {g gʰ b bʰ} > $1 $1                    ; @nasal-assim — 5 lects / ŋg mb(ʰ) > ŋŋ mm: kumbʰakāra > B kumār, kumbʰīra > B kumīr / (changes.sc nasal-assim.ngmgghbbh)
[A] Nasal {k kʰ c cʰ ṭ ṭʰ t tʰ p pʰ j jʰ ḍ ḍʰ} > ṁ $2 ; @nasal-cluster — 2 lects / A.: not before g d b (kaṇṭa > kā̃ṭ, piṇḍa > pĩr; aŋgāra > āŋgār) / (changes.sc nasal-cluster.nasalkkhcc.3)
[A] m {b bʰ} > m m                               ; @nasal-assim — 2 lects / mb > mm (kambala > kāmali); A. keeps ŋg, nd / (changes.sc nasal-assim.mbbh)
[Tor] k ṣ > ṣ / V _                               ; @ks — non-initial kṣ > ṣ (akṣōṭa > aṣṓ)
[Sh Kho pog Kund Kal dardic-south] k ṣ > ʦ̣ʰ                                    ; @ks — 15 lects / kṣīrá > ʦ̣ʰīr / (changes.sc ks.ks)
[Wot] {d dʰ} r > l                                ; @tr-lateral — *dr > l (drā́kṣā > lyēś); *gr > g
[dardic-south -Gaw -Shum -Dm -Tir -Pas] {ṣ s} {ṭ ṭʰ t tʰ} > {ṭʰ tʰ}                  ; @cluster-simplify @s-stop-aspirate — 7 lects / ṣṭ > ṭʰ, st > tʰ without gemination (aṣṭā́ > aṭʰ, hásta > hatʰ) / (changes.sc s-stop-aspirate.sstthtth)
[Bshk] {t tʰ k} r > ɬ                              ; @tr-lateral — mū́tra > mūɬ, *krayakāra > ɬæyar
[Gaw] {b bʰ p} r > {b b p} l                      ; @tr-lateral — labial + r > labial + l (bʰr > bl, pr > pl)
[Bshk Gaw Shum Pas] {d dʰ g gʰ} r > l                            ; @tr-lateral — 3 lects / voiced *dr, *gr > l / (changes.sc tr-lateral.ddhgghr)
[Sh Sv Ush Phal KharI] Sib Stop > $2+asp / # _                      ; @initial-degem @s-stop-aspirate — 5 lects / skandʰá > kʰaṇ, spʰā́yatē > pʰe / (changes.sc initial-degem.sibstop)
[Sh] {ñ ṇ n m} Stop > $1                          ; @cluster-simplify @nc-simplify — 2 lects / nasal + stop > nasal (dánta > don, kā́ṇḍa > kōṇ) / (changes.sc nc-simplify.nnnmstop)
[Kalk Kho] {n ṇ} {d ḍ dʰ} > n                           ; @cluster-simplify @nc-simplify — 2 lects / voiced dental/retroflex lost after n (dánta > dān, pántʰā > pān) / (changes.sc nc-simplify.nndddh)
[Kalk Bshk Shum Chil] s C > $2 / # _                               ; @initial-degem — 5 lects / initial s + C > C (skandʰá > kan) / (changes.sc initial-degem.sc)
[Pas] {b bʰ p d dʰ g gʰ} r > l                    ; @tr-lateral — *pr, *br, *bʰr, *gr, *dr > l (bʰrāturjāyā > lājā́); dialects vary (tr kept or > ɬ)
[dardic-south -Gaw -Shum -Wot -Dm] {d dʰ} y > j                                 ; @palatal-Cy — 3 lects / dy > j / (changes.sc palatal-Cy.ddhy.2)
[Tor] {t tʰ} r > ʦ̣                               ; @tr-affricate — only *tr > ʦ̣ in Torwali (dr, bʰr > d, b)
[Tor] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc) — before the step that frees cluster m (rm > mm > m, Dardic rm > m), so only original single m lenites] ──
[Tor] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
[Tor] r > ∅ / _ C                                  ; @assim-r @cluster-simplify — 16 lects / (changes.sc assim-r.r.preconsonantal)
[K dod] Sib Stop > $2+asp                            ; @cluster-simplify @s-stop-aspirate — 2 lects / gōṣṭʰa > guṭʰ / (changes.sc s-stop-aspirate.sibstop)
[K] Nasal {t ṭ p k} > $1 {d ḍ b g}              ; @nasal-voicing — dánta > dand
[pog] {ś ṣ s} Stop > h $2                         ; @s-h — sibilant before stop > h (aṣṭā́ > āhṭ, śastrá > śāhtar)
[dardic-north -Kho] ; ── Lenition (geminates protected) ──────────────────────────────────────
[Sh] # ── Shinaic post-MIA prefix: first 2 of 6 shared changes (round-7 unification) ──
[Kho] ; ── Lenition ────────────────────────────────────────────────────────────
[Bshk] ; ── block lar-deaspiration-voiced: Loss of aspiration in the voiced aspirates (bʰ dʰ gʰ jʰ ḍʰ > b d g j ḍ), everywhere in the word. Shared across families: Dardic (Shinaic, Kohistani, Kunar, Kash [@deaspiration — voiced aspirates lose aspiration (blocks/lar-deaspiration-voiced.sc)] ──
[Bshk] {bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ}               ; @deaspiration @tone — 26 lects / dʰūma > Sh. dūm, bʰūmi > K. bum, gʰr̩tá > Gaw. gʌd / (changes.sc deaspiration.bhdhghjhdh)
[Gowro Chil bhatr Mai] {t tʰ p d dʰ bʰ} r > {ʦ̣ ʦ̣ ʦ̣ ẓ ẓ ẓ}        ; @tr-affricate — 3 lects / putrá > pūʦ̣, *prēratē > ʦ̣ēlǎṽ, abʰrá > âẓ, drā́kṣā > ẓā̂ʦ̣ʰ / (changes.sc tr-affricate.tthpddhbhr)
[Sv Kalk nuristani] k ṣ > ʦ̣                                     ; @ks @nur-archaism — 10 lects / mākṣiká > *māʦ̣i / (changes.sc ks.ks.2)
[Ash] ṣ > ś / {i ī u ū} _                         ; @nur-innovation — Ash: ṣ after high vowels is ś (kaniṣṭʰá > kaṇiśt, muṣṭí > muśt, yūṣa > yɨ̄ś); not in the other lects
[nuristani -Kt -Gmb -Kam] ś v > ś p                                    ; @nur-archaism — 4 lects / *ćw > śp as in Iranian: śváśura > *iśpaśur, áśva > *uśpa / (changes.sc nur-archaism.sv)
[Shum Dm Mai Dhp KharI nuristani] ś r > ṣ                                      ; @cluster-simplify @nur-archaism — 7 lects / *ćr > ṣ: *śrāya > *ṣai, śŕ̩ŋgāra > *ṣiŋara / (changes.sc nur-archaism.sr)
[nuristani -Ash -Kam] h > ∅ / {a ā} _ V                            ; @h-loss — 4 lects / round 6: intervocalic h lost after a/ā (kalaha > kalá, prahāra > prōr, *bahira > barú, sahita > sā); after e/i it gives z/ʦ (*dēhati > dez, lihati > lazé) / (changes.sc h-loss.h.aav)
[nuristani -Ash -Wg] ; ── block lar-nur-affrication-full: Kati-Kamviri-Prasun affricates: OIA (c)cʰ (< PIE *sḱ) > ʦ, as in Iranian s, then the shared Nuristani affrication of the PIIr palatals (lar-nur-affrication). Ka [@nur-archaism @affrication — (c)cʰ > ʦ, then ś > ʦ, h > ʣ, j > ʣ (blocks/lar-nur-affrication-full.sc)] ──
[nuristani -Ash -Wg] c cʰ > ʦ                                     ; @affrication @nur-archaism — 4 lects / ā́gaccʰati > *āʦa / (changes.sc affrication.cch)
[nuristani -Ash -Wg] cʰ > ʦ                                       ; @affrication @nur-archaism — 4 lects / cʰāyā́ > Kt ʦāvá, *cʰidati > Kt ʦie / (changes.sc affrication.ch)
[nuristani -Ash -Wg] ; ── block lar-nur-affrication: Nuristani dental affricates from the PIIr palatals (*ć *ȷ́ *ȷ́h kept distinct from the OIA palatals): ś > ʦ after a vowel/anusvāra, h > ʣ, j > ʣ before a-vowels [@nur-archaism @affrication — ś > ʦ, h > ʣ, j > ʣ] ──
[nuristani] ś > ʦ / V _ | V ṁ _                          ; @affrication @nur-archaism — 6 lects / PIIr *ć: śā́kʰā > *ʦāv, dáśa > *daʦ, viṁśatí > *viʦi / (changes.sc affrication.s.vvm)
[nuristani] h > ʣ                                        ; @affrication @nur-archaism — 7 lects / PIIr *ȷ́h: himá > *ʣim, hŕ̩daya > *ʣara / (changes.sc affrication.h)
[nuristani] j > ʣ / _ {a ā}                              ; @affrication @nur-archaism — 4 lects / PIIr *ȷ́ (PIE *ĝ) > ʣ: jā́nu > *ʣā̃, jā́mātr̩ > Kam ʣā- / (changes.sc affrication.j.aa)
[Kt Kam PNur] ṣ ṭ > ś t / {i ī u ū} _                      ; @nur-innovation — 3 lects / round 6: ṣṭ after a high vowel > št, after the ś > ʦ affrication (niṣṭi > niśte, muṣṭí > miśt / mr̆üśt, *śuṣṭa > śtu, viṣṭara > Kam viśtákāno); after a and r̩ it stays ṣṭ (r̩ṣṭí > íṣṭ, pr̩ṣṭí > pṭí). Counts after i/u: Kt 24 śt vs 5 ṣṭ, Kam 17 vs 8. Cf. Ash ṣ > ś / high V, Pr ṣṭ > št / (changes.sc nur-innovation.st.iiuu)
[Ash Wg] ; ── block lar-nur-affrication: Nuristani dental affricates from the PIIr palatals (*ć *ȷ́ *ȷ́h kept distinct from the OIA palatals): ś > ʦ after a vowel/anusvāra, h > ʣ, j > ʣ before a-vowels [@nur-archaism @affrication — ś > ʦ, h > ʣ, j > ʣ (blocks/lar-nur-affrication.sc)] ──
[Ash Pr] j > ʣ                                        ; @affrication @nur-archaism — 2 lects / Ash: PIIr *ȷ́ > ʣ also before other vowels (jā́nu > *ʣā̃) / (changes.sc affrication.j)
[Pr] ṣ ṭ > ś t                                   ; @nur-innovation — Pr ṣṭ > št, after ś > ʦ (−0.0005): muṣṭí > müśt, úṣṭra > üścúr
[nuristani] ; ── r + dental → retroflex ──────────────────────────────────────────────
[Ku Garh N Bishnupriya Or B Kannauji-Central OAw bagheli_lakshman Brj SSr Bhatri P L mewati_akera kgr] {r ḣ} C > $2-asp $2                          ; @assim-r — 109 lects / (changes.sc assim-r.rhc)
[S] v y > v / # _                               ; @cluster-simplify — initial vy- > v- (vyākʰyāna > vākʰāṇu, vyātta > vātu, vyāgʰra > vāgʰu), not vv > bb > ɓ
[northwest -D -thar -Khol -khet] s m > s s                                   ; @assim-sonorant — Northwest: MIA sm (< ṣm śm sm) > ss, not mh (vismarati > vissar-/visar-, bʰasman > bʰass, ūṣman > huss, *upasmārayati > S osāraṇu); gold s(s) 37 vs m(h) 7 in S L P OP awan (hill-climb round 6)
[hal] s C > $2-asp $2                             ; @s-stop-aspirate — s + C > unaspirated geminate in the Halbi data (hasta > hāt)
[bihari-tharu Ku Garh N Dotyali Ush Chil Tor eastern hindi -H marathi-bhil -OM -Noiri Ap Si Md OSi P poth L rajasthani-gujarati -mewari_basad -OG -G -dhundari_badagaon -OMarw west-pahari -bhal -dog -WPah -cam -pan] ; ── block mia-clusters: MIA r + C and stop-cluster assimilation (blocks theme "mia", 2026-09-30): rC > CC, then stop + sonorant/nasal and stop + stop (mia-stop-clusters). For lects wit [@assim-r @assim-sonorant @assim-stop — rC > CC, stop + sonorant/nasal, stop + stop (blocks/mia-clusters.sc; blocks round 2026-09-30)] ──
[bihari-tharu Ku Garh N Dotyali Ush Chil Tor eastern hindi marathi-bhil -OM mia-insular -NiDoc -Dhp -KharI northwest rajasthani-gujarati west-pahari -pan] {r ḣ} C > $2-asp $2                          ; @assim-r — 109 lects / r (and visarga) + C > geminate (karma > kamma, sarpa > sappa, dúḥkʰa > dukkʰa) / (changes.sc assim-r.rhc)
[bihari-tharu Ku Garh N Dotyali Ush Chil Tor eastern hindi -H marathi-bhil -OM -Noiri Ap Si Md OSi P poth L rajasthani-gujarati -mewari_basad -OG -G -dhundari_badagaon -OMarw west-pahari -bhal -dog -WPah -cam -pan] ; ── block mia-stop-clusters: MIA assimilation of stop + sonorant/nasal and stop + stop clusters (blocks theme "mia", 2026-09-30). The same two changes, in this order, are positions 18–19 of ──
[bihari-tharu Ku Garh N Dotyali Ush Chil Tor eastern hindi -H marathi-bhil -OM Ap Si Md OSi northwest -thar -S -Khol -khet rajasthani-gujarati -mewari_basad -OG -G -dhundari_badagaon -OMarw west-pahari -bhal -dog -WPah -cam -pan] Stop {r y v l Nasal} > $1-asp $1             ; @assim-sonorant — 83 lects / stop + r y v l nasal > geminate stop (putra > putta, śakya > sakka, agni > aggi) / (changes.sc assim-sonorant.stopryvlna)
[bihari-tharu Ku Garh N Dotyali Ush Chil Tor eastern hindi marathi-bhil -Noiri mia-insular northwest -Khol -khet rajasthani-gujarati -G west-pahari] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[mewari_basad OMarw Goj] ; ── block mia-sibilant-clusters-sk: Rajasthani variant of mia-sibilant-clusters (blocks theme "mia", pass 2, 2026-09-30): s + k gives unaspirated kk before the general s + stop > aspirated geminat [@palatal-Cy @ks @s-stop-aspirate @assim-sonorant — Cy, kṣ/ts/ps, sk > kk, s + stop, s + sonorant (blocks/mia-sibilant-clusters-sk.sc; blocks round 2026-09-30, pass 2)] ──
[mewari_basad OMarw Goj] ; ── block mia-palatals: MIA palatalisation of dental + y and the kṣ/ts/ps clusters (blocks theme "mia", 2026-09-30). Runs after the sibilant merger in lects with ś ṣ > s (kṣ is then ks ──
[mewari_basad Goj] s k > k k                                    ; @s-stop-aspirate — 5 lects / s + k > kk, unaspirated (śuṣka > sukka, *pask- > pakk-) / (changes.sc s-stop-aspirate.sk)
[mewari_basad OMarw Goj] ; ── block mia-s-clusters: MIA s + stop and s + sonorant clusters (blocks theme "mia", 2026-09-30). Part of mia-sibilants. ──
[mewari_basad OMarw] s Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 81 lects / s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop)
[Goj] r {t tʰ} > {ṭ ṭ}@2 {ṭ ṭʰ}@2                  ; @assim-r @rt-retroflex — 7 lects / r + t(ʰ) > retroflex geminate (vartati > vaṭvũ, karttāra > kaṭār) / (changes.sc assim-r.rtth)
[SSr] ; ∅ > Ø / {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ tʰ d dʰ p pʰ b bʰ l ḷ r} _ #                   ; @morph — starred verbal roots (*aṭṭ, *ḍag) are bare stems: protect the stem-final C (Ø = citation ending, spelled out at the end)   [pruned for SSr: LOO -0.0006]
[SSr] ; {ai au} > {ē ō}                             ; @ai-au   [pruned for SSr: never fires]
[OAw] ; V ṁ > $1+long / _ s                         ; @anusvara-vowel — MIA ṁ before sibilant > length without nasal (viṁśatí > bīs)   [pruned for OAw: LOO -0.0001]
[hindi] h {y v} > {j b}@2 {jʰ bʰ}@2                  ; @x-hindi-h-cluster — 9 lects / hy > jjʰ, hv > bbʰ (guhya > gujjʰa, jihvā > jibbʰā > jībʰ) / (changes.sc h-cluster)
[bundeli_atarra NDu poth mewari_basad marwari_bagra Goj] m {r l} > m b                                ; @mr-mb — 21 lects / m + liquid > mb (āmra > amba; tag from OIA-H.sc) / (changes.sc mr-mb.mrl)
[G] optional: r {t tʰ} > {ṭ ṭ}@2 {ṭ ṭʰ}@2        ; @assim-r @rt-retroflex — 7 lects / r + t(ʰ) > retroflex geminate (vartati > vaṭvũ, karttāra > kaṭār) / (changes.sc assim-r.rtth)
[S P Khol kcch khet awan mewari_basad OG mewari_dholpura hadothi_kelwada G marwari_bagra dhundari_bamore OMarw Goj] d v > b b                                    ; @dv-b — 16 lects / dv > bb (dvāra > bār, dvādaśa > bār-) / (changes.sc dv-b)
[Pk Ap Pa] {s h} Nasal > $2 h                           ; @s-nasal — 3 lects / kr̩ṣṇa > kaṇha, brahman > bamha / (changes.sc s-nasal.shnasal)
[Pk Marw G OMarw] j ñ > ṇ ṇ                                    ; @assim-sonorant — 4 lects / jñ > ṇṇ (jñāna > OG nāṇa, abʰijñāna > G εdʰāṇ) / (changes.sc assim-sonorant.jn)
[Pk OP S P L srk OG had mewari_dholpura Rj G dhundari_badagaon] h y > j jʰ                                   ; @palatal-Cy — 12 lects / hy > jjʰ (guhya > gujjʰa) / (changes.sc palatal-Cy.hy)
[D Marw had dhundari_badagaon] h v > b bʰ                                  ; @x-hindi-h-cluster — hv > bbʰ, the hv half of H's h-cluster (jihvā́ > jibbʰā > jībʰ, gōjihvikā > gobʰī). The chain had no hv rule, so hv survived to the output (gold hv 0; hill-climb round 7)
[kcch mand] m r > m b / V _                              ; @mr-mb — 5 lects / MIA medial mr > mb with an epenthetic stop (āmrá > amba, tāmrá > tamba); unified at round-7 integration / (changes.sc mr-mb.mr.postvocalic)
[H kcch] m l > m b / V _                              ; @mr-mb — ml > mb (kept from the old m {r l} rule)
[K H S kcch kiuth ktg] m r > m / # _                                ; @mr-mb — MIA initial mr- > m- (mrakṣaṇa > makkʰaṇa, mriyátē > marai); unified at round-7 integration
[Kt Kam] a > ā / _ r {t tʰ d dʰ ṣ ś ṇ n}              ; @assim-r @nur-innovation — 2 lects / round 6: a lengthened before r + dental/sibilant/n, ahead of the r-assimilation (karta > kāṭá, kartya > kāṭí, gʰarṣa > gā́ṣ, kárṇa > kór via ā > o); not before rv/ry (sárva > suv, śárya > śúr̆) / (changes.sc assim-r.a.rtthddhssnn)
[PNur] a > ā / _ r {ṇ n}                           ; @nur-innovation @assim-r — round 6: a lengthened before rṇ, ahead of rṇ > r (kárṇa > *kār, parṇá > *pār); before r + dental stop only in Kt/Kam (Strand's PNur has *kaṭā, *gada, *kaṭala)
[OG] ; ── block cr-keep-optional: Sporadic retention of stop + r (Northwest, W. Pahari, NW Ashokan, Shina/Savi, Old Gujarati): in some words the r of stop + r escapes the MIA assimilation (putra [sporadic (blocks/cr-keep-optional.sc)] ──
[OG] optional: {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} r > {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} ; @assim-sonorant @cr-keep — sporadic retention of stop + r (dā́tra > P dātrī ~ dātī, átra > As. atra) / (block cr-keep-optional)
[rajasthani-gujarati -G] t m > p p                                    ; @assim-stop — 15 lects / tm > pp (ātman > appa > āp) / (changes.sc assim-stop.tm)
[rajasthani-gujarati -bagri_fatehabad -Marw -had -mewari_dholpura -mewati_jhambaus -mewati_akera -marwari_bagra -dhundari_badagaon dog mand WPah cam] ; ── block mia-clusters-cr: MIA r + C and stop-cluster assimilation in lects that keep stop + r (blocks theme "mia", 2026-09-30): rC > CC, then mia-stop-clusters-cr. The Cr-keeping counter [@assim-r @assim-sonorant @assim-stop — rC > CC, stop + y v l nasal, stop + stop (stop + r kept) (blocks/mia-clusters-cr.sc; blocks round 2026-09-30)] ──
[WK Nimadi hadothi_kelwada Rj dhundari_bamore Goj mand] {r ḣ} C > $2-asp $2                          ; @assim-r — 109 lects / r (and visarga) + C > geminate (karma > kamma, sarpa > sappa, dúḥkʰa > dukkʰa) / (changes.sc assim-r.rhc)
[rajasthani-gujarati -bagri_fatehabad -Marw -had -mewari_dholpura -mewati_jhambaus -mewati_akera -marwari_bagra -dhundari_badagaon dog mand WPah cam] ; ── block mia-stop-clusters-cr: MIA stop-cluster assimilation in lects that keep stop + r (Sindhi, W. Pahari; blocks theme "mia", 2026-09-30): stop + y v l nasal and stop + stop assimilate, st ──
[KharI khet rajasthani-gujarati -bagri_fatehabad -Marw -had -mewari_dholpura -mewati_jhambaus -marwari_bagra -dhundari_badagaon dog mand WPah cam pan] Stop {y v l Nasal} > $1-asp $1               ; @assim-sonorant — 22 lects / stop + y v l nasal > geminate stop (śakya > sakka, agni > aggi, pakva > pakka) / (changes.sc assim-sonorant.stopyvlnas)
[WK Nimadi hadothi_kelwada Rj G dhundari_bamore Goj mand] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[kaithal] ; {ai au} > {ē ō}                             ; @ai-au   [pruned for kaithal: never fires]
[kaithal] ; h {y v} > {j b}@2 {jʰ bʰ}@2                 ; @x-hindi-h-cluster — hy > jjʰ, hv > bbʰ (jihvā > jibbʰā > jībʰ); LOO 0 here, kept for the group order   [pruned for kaithal: never fires]
[bihari-tharu] {n ṇ m l v} y > $1 $1                        ; @assim-sonorant — 40 lects / sonorant/nasal + y > geminate (anyá > aṇṇa > ān, kalya > kalla) / (changes.sc assim-sonorant.nnmlvy)
[bihari-tharu -Bhoj -Mth -Darai -Bi] ; ── block nasals-nt-anusvara: nasals-nt-anusvara — MIA nasal + voiceless stop > anusvāra + stop, voiced clusters kept (Tharu type), nasals theme, blocks round 2026-09-30 (pass 2). Only befor [@nasal-cluster — only before voiceless stops; N + voiced stop stays a cluster in Tharu (aŋgri, bandʰā) (blocks/nasals-nt-anusvara.sc)] ──
[bihari-tharu -Bhoj -Mth -Darai -Bi] Nasal {k kʰ c cʰ ṭ ṭʰ t tʰ p pʰ} > ṁ $2      ; @nasal-cluster — 13 lects / káṇṭaka > Bote, Majhi kā̃ṭo; ND kept (Tharu aŋgri, bandʰā) / (changes.sc nasal-cluster.nasalkkhcc)
[Bi] m {b bʰ} > m {m mʰ}                         ; @nasal-assim — only mb > mm here, before NC > anusvāra (jambūna > jāmun, skambʰa > kʰāmʰ)
[Mth] {m n} {b bʰ d dʰ} > $1 {m mʰ n nʰ}          ; @nasal-assim — mb nd > mm nn before NC > anusvāra (jambūna > jāmun, skambʰa > kʰāmʰ); ŋg stays (ā̃gur)
[Chitwan Bhoj] {m ŋ n} {b bʰ g gʰ d dʰ} > $1 {m mʰ ŋ ŋ n nʰ} ; @nasal-assim — 2 lects / mb ŋg nd > mm ŋŋ nn before NC > anusvāra (tumba > tūmā, ṭaŋga > ṭāŋī, skandʰá > kānʰ) / (changes.sc nasal-assim.mngnbbhggh)
[Bhoj Mth Bi] ; ── block nasals-nc-anusvara: nasals-nc-anusvara — MIA homorganic nasal + stop > anusvāra + stop, nasals theme, blocks round 2026-09-30 (pass 2). The written MIA anusvāra before a stop (aṁta [@nasal-cluster (blocks/nasals-nc-anusvara.sc)] ──
[Bhoj Mth Bi] Nasal Stop > ṁ $2                            ; @nasal-cluster — 39 lects / ánta > aṁta, páñca > paṁca > Nimadi pā̃c / (changes.sc nasal-cluster.nasalstop)
[bundeli_atarra H] optional: r t > ṭ ṭ                          ; @assim-r @rt-retroflex — 5 lects / r + t > ṭṭ (vartate > vaṭṭai, āvartana > auṭan) / (changes.sc assim-r.rt)
[NDu] ; r t > ṭ ṭ                                   ; @assim-r @rt-retroflex — r + t > ṭṭ (vartate > vaṭṭai, āvartana > auṭan)   [pruned for NDu: never fires]
[H] d v > b b / V _                             ; @assim-sonorant @v-b — medial dv > bb (udvartman > ubaṭ, *udvāsa > ubās, údvijatē > ūbnā); round 6b
[Pk] j ñ > ñ ñ                                    ; @assim-sonorant — 4 lects / jñ > ññ (prajñā > paññā, ajñāta > aññāta) / (changes.sc assim-sonorant.jn.2)
[Pa] h y > y h                                   ; @metathesis — guhya > guyha, gr̩hyatē > gayhati
[Pk Pa] l m > m m                                   ; @assim-sonorant — lm > mm (kalmāṣa > kammāsa, valmī́ka > Pa vammika, Pk vammia, gúlma > Pk gumma). The chain left lm (gold lm 0; hill-climb round 7)
[Pk Pa] v r > v v                                   ; @assim-sonorant — vr > vv like the other C + r (tīvrá > Pk tivva, Pa tibba; pravrajyā > pavvajjā, Pa pabbajjā). The chain left vr (gold vr 0; hill-climb round 7)
[Morang Mth Bi] {t tʰ} > {ṭ ṭʰ} / r (t) _                    ; @x-bihari-tharu-rt — 3 lects / MIA retroflexion of dentals after r (vartatē > vaṭṭai: *lōrtati > loṭab, kartana > kaṭnī, *ārta > āṭā) / (changes.sc tharu-rt.tth)
[Morang H] ; ── block mia-clusters: MIA r + C and stop-cluster assimilation (blocks theme "mia", 2026-09-30): rC > CC, then stop + sonorant/nasal and stop + stop (mia-stop-clusters). For lects wit [@assim-r @assim-sonorant @assim-stop — rC > CC, stop + sonorant/nasal, stop + stop (blocks/mia-clusters.sc; blocks round 2026-09-30)] ──
[Morang] {r ḣ} C > $2-asp $2                          ; @assim-r — 109 lects / r (and visarga) + C > geminate (karma > kamma, sarpa > sappa, dúḥkʰa > dukkʰa) / (changes.sc assim-r.rhc)
[Morang H] ; ── block mia-stop-clusters: MIA assimilation of stop + sonorant/nasal and stop + stop clusters (blocks theme "mia", 2026-09-30). The same two changes, in this order, are positions 18–19 of ──
[Morang H] Stop {r y v l Nasal} > $1-asp $1             ; @assim-sonorant — 83 lects / stop + r y v l nasal > geminate stop (putra > putta, śakya > sakka, agni > aggi) / (changes.sc assim-sonorant.stopryvlna)
[Morang] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[bundeli_atarra] ; Nasal Nasal > $2 $2                         ; @assim-sonorant — unmagna > ummagga   [pruned for bundeli_atarra: never fires]
[bagheli_lakshman] ; Nasal Stop > ṁ $2                           ; @nasal-cluster   [pruned for bagheli_lakshman: LOO -0.0046]
[hindi -OAw -Aw] r t > ṭ ṭ                                    ; @assim-r @rt-retroflex — 5 lects / r + t > ṭṭ (vartate > vaṭṭai, āvartana > auṭan) / (changes.sc assim-r.rt)
[Kannauji-Central] ; r t > ṭ ṭ                                   ; @assim-r @rt-retroflex — r + t > ṭṭ (vartate > vaṭṭai, āvartana > auṭan)   [pruned for Kannauji-Central: never fires]
[bhal] {t tʰ d dʰ} r > {ṭ͡ḷ ṭ͡ḷ ḍ͡ḷ ḍ͡ḷʰ} / # _ | V _       ; @tr-lateral — dental + r > retroflex lateral affricate, initially and after any vowel (kṣḗtra > ʦʰeṭ͡ḷ, páttra > paṭ͡ḷ)
[Rana Bhoj Mth Bi BH Aw M RathwiBareli VB Ko-Ul Ap D kcch srk mewati_jhambaus bhal cur] ; ── block mia-clusters-rt: MIA clusters preceded by the sporadic r + dental retroflexion (blocks/rt-retroflex-optional.sc). [MIA clusters with the sporadic r + dental retroflexion] ──
[Rana Bhoj Mth Bi BH Aw M RathwiBareli VB Ko-Ul Ap D kcch srk mewati_jhambaus bhal cur] ; ── block rt-retroflex-optional: Sporadic r + dental retroflexion (MIA): r + t tʰ d dʰ > ṭṭ ṭṭʰ ḍḍ ḍḍʰ in some words only (vartate > vaṭṭai beside vattai; ardʰa > aḍḍʰa beside addʰa). `optional ──
[Rana Bhoj Mth Bi BH Aw M RathwiBareli VB Ko-Ul Ap kcch srk mewati_jhambaus bhal cur] optional: r {t tʰ d dʰ} > {ṭ ṭ ḍ ḍ}@2 {ṭ ṭʰ ḍ ḍʰ} ; @assim-r @rt-retroflex — sporadic r + dental > retroflex geminate (vartate > vaṭṭai ~ vattai, ardʰa > aḍḍʰa) / (block rt-retroflex-optional)
[Rana Bhoj Mth Bi BH Aw M RathwiBareli VB Ko-Ul Ap D kcch srk mewati_jhambaus bhal cur] ; ── block mia-clusters: MIA r + C and stop-cluster assimilation (blocks theme "mia", 2026-09-30): rC > CC, then stop + sonorant/nasal and stop + stop (mia-stop-clusters). For lects wit ──
[Rana Bhoj Mth Bi BH Aw M RathwiBareli VB Ko-Ul Ap mewati_jhambaus cur] {r ḣ} C > $2-asp $2                          ; @assim-r — 109 lects / r (and visarga) + C > geminate (karma > kamma, sarpa > sappa, dúḥkʰa > dukkʰa) / (changes.sc assim-r.rhc)
[Rana Bhoj Mth Bi BH Aw M RathwiBareli VB Ko-Ul Ap D kcch srk mewati_jhambaus bhal cur] ; ── block mia-stop-clusters: MIA assimilation of stop + sonorant/nasal and stop + stop clusters (blocks theme "mia", 2026-09-30). The same two changes, in this order, are positions 18–19 of ──
[Rana Bhoj Mth Bi BH Aw M RathwiBareli VB Ko-Ul Ap mewati_jhambaus bhal cur] Stop {r y v l Nasal} > $1-asp $1             ; @assim-sonorant — 83 lects / stop + r y v l nasal > geminate stop (putra > putta, śakya > sakka, agni > aggi) / (changes.sc assim-sonorant.stopryvlna)
[Rana Bhoj Mth Bi BH Aw M RathwiBareli VB Ko-Ul Ap mewati_jhambaus cur] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[Khandesi PalyaBareli Rathawi Bhilali Bhili] ; ── block mia-clusters: MIA r + C and stop-cluster assimilation (blocks theme "mia", 2026-09-30): rC > CC, then stop + sonorant/nasal and stop + stop (mia-stop-clusters). For lects wit [@assim-r @assim-sonorant @assim-stop — rC > CC, stop + sonorant/nasal, stop + stop (blocks/mia-clusters.sc; blocks round 2026-09-30)] ──
[Khandesi PalyaBareli Rathawi Bhilali Bhili] {r ḣ} C > $2-asp $2                          ; @assim-r — 109 lects / r (and visarga) + C > geminate (karma > kamma, sarpa > sappa, dúḥkʰa > dukkʰa) / (changes.sc assim-r.rhc)
[Khandesi PalyaBareli Rathawi Bhilali Bhili] ; ── block mia-stop-clusters: MIA assimilation of stop + sonorant/nasal and stop + stop clusters (blocks theme "mia", 2026-09-30). The same two changes, in this order, are positions 18–19 of ──
[Khandesi PalyaBareli Rathawi Bhilali Bhili] Stop {r y v l Nasal} > $1-asp $1             ; @assim-sonorant — 83 lects / stop + r y v l nasal > geminate stop (putra > putta, śakya > sakka, agni > aggi) / (changes.sc assim-sonorant.stopryvlna)
[Khandesi PalyaBareli Rathawi Bhilali Bhili] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[eastern] ; ── block mia-sibilants-sh: MIA sibilants in the Eastern (Bengali-Assamese-Oriya) lects (blocks theme "mia", 2026-09-30): all sibilants merge in ś (Māgadhī), and the sibilant clusters assi [@sibilant-ś @ks @s-stop-aspirate @assim-sonorant — Eastern: all sibilants > ś, then kś/tś/pś, ś + stop, ś + sonorant (blocks/mia-sibilants-sh.sc; blocks round 2026-09-30)] ──
[eastern] Sib > ś                                      ; @sibilant-ś — 11 lects / ś ṣ s > ś (sapta > śatta > B śāt, dāsa > dāśa, varṣa > variśa) / (changes.sc sibilant-ś)
[eastern pan] {k t p} ś > {k c c} {kʰ cʰ cʰ}@1             ; @ks — 13 lects / kś (< kṣ) tś pś > kkʰ ccʰ (akṣi > akkʰi, vatsa > vaccʰa, apsaras > accʰarā) / (changes.sc ks.ktps.2)
[eastern] ś Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 11 lects / ś + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.sstop.2)
[eastern] ś Son > ś ś                                  ; @assim-sonorant — 11 lects / ś + sonorant > śś (aśva > aśśa, tasya > taśśa, sahasra > sahaśśa) / (changes.sc assim-sonorant.sson.2)
[Or B] ; ── block upa-va-optional: Sporadic reduction of the prefixes upa-/ava- to va-/ba- (initial u-/a- lost, the labial kept or strengthened), OIA stage (2026-10-01): upaccʰada > Ash vāʦā, Kam [cross-family law (blocks/upa-va-optional.sc)] ──
[Or B] optional: {u ú a á} {p v} {a á ā ā́} > v $3 | b $3 / # _ ; @upa-va — upa-/ava- > va- | ba- / (block upa-va-optional)
[Or B] ; ── block dv-b-initial: Initial dv- > b- beside d(u)v- (dvāra > H bār ~ duār, dvitīya > bījā ~ dūjā); the chains have it in only 16 lects. Reverse-reconstruction loop, iteration 1 (202 [reverse-reconstruction loop (blocks/dv-b-initial.sc)] ──
[Or B] optional: d v > b / # _                      ; @dv-b — initial dv- > b- (dvinavati > H bānve, dvāra > bār, dvitīya > bījā) / (block dv-b-initial)
[SSr] ; r t > ṭ ṭ                                   ; @assim-r @rt-retroflex — r + t > ṭṭ (vartate > vaṭṭai, āvartana > auṭan)   [pruned for SSr: never fires]
[OAw] ; r t > ṭ ṭ                                   ; @assim-r @rt-retroflex — r + t > ṭṭ (vartate > vaṭṭai, āvartana > auṭan)   [pruned for OAw: LOO < 0]
[Pa] {n ṇ} y > ñ ñ                               ; @palatal-Cy @assim-sonorant — anya > añña, puṇya > puñña (Pk aṇṇa, puṇṇa)
[As-Ul] n y > ñ ñ                                   ; @palatal-Cy @assim-sonorant — anya > añña (Pk aṇṇa)
[Pa As-Ul] Nasal {y v} > $1 $1                          ; @assim-sonorant — 7 lects / anya > aṇṇa / (changes.sc assim-sonorant.nasalyv)
[NiDoc Dhp KharI] ṣ Stop > $2-asp $2+asp                       ; @s-stop-aspirate — 3 lects / only ṣ: duṣṭa > duṭʰa; s + stop kept (vistīrṇa > vistiṁna) / (changes.sc s-stop-aspirate.sstop.3)
[Dhp] ∅ > i / r _ y                                ; @anaptyxis — 2 lects / kārya > kariya (as Pali) / (changes.sc anaptyxis.ry)
[Dhp] {r ḣ} C > $2-asp $2                          ; @assim-r — 109 lects / karma > kamma / (changes.sc assim-r.rhc)
[bagheli_lakshman] ; Nasal Nasal > $2 $2                         ; @assim-sonorant — unmagna > ummagga   [pruned for bagheli_lakshman: never fires]
[bagheli_lakshman] ; Stop Stop > $2-asp $2                       ; @assim-stop — bʰakta > bʰatta   [pruned for bagheli_lakshman: LOO -0.0001]
[Brj] ; Nasal Nasal > $2 $2                         ; @assim-sonorant — unmagna > ummagga   [pruned for Brj: never fires]
[NDu] ; Nasal Nasal > $2 $2                         ; @assim-sonorant — unmagna > ummagga   [pruned for NDu: never fires]
[SSr] ; Nasal Nasal > $2 $2                         ; @assim-sonorant — unmagna > ummagga   [pruned for SSr: never fires]
[Bhatri mewari_bannoda] # ── end of core; lect-specific chain follows ──
[west-pahari -bhad -cur -khash -cam] {s ś} {y v r} > $1 $1                        ; @assim-sonorant — 10 lects / sibilant + y/v/r > geminate sibilant (śvaśura > ss-, śyāla > ss-) / (changes.sc assim-sonorant.ssyvr)
[Buksa Mth Bi] ṁ C C > ṁ $3                                 ; @assim-stop — 3 lects / the anusvāra survives the three-consonant reduction (paŋktí > paṁtti > pā̃ti) / (changes.sc assim-stop.mcc)
[bihari-tharu -DewasDoneDanuwar -Bote] C C C > $2 $3                                ; @assim-stop — 16 lects / three-consonant clusters left by the assimilations reduce to a geminate (matsya > maccʰa) / (changes.sc assim-stop.ccc)
[NiDoc Dhp] Nasal y > ṁ ñ                                ; @assim-sonorant @nasal-cluster — 2 lects / araṇya > araṁña (ṁ before the palatal nasal) / (changes.sc assim-sonorant.nasaly)
[KharI] ; ── block nasals-nc-anusvara: nasals-nc-anusvara — MIA homorganic nasal + stop > anusvāra + stop, nasals theme, blocks round 2026-09-30 (pass 2). The written MIA anusvāra before a stop (aṁta [@nasal-cluster — anusvāra before stops (blocks/nasals-nc-anusvara.sc)] ──
[KharI] Nasal Stop > ṁ $2                            ; @nasal-cluster — 39 lects / ánta > aṁta, páñca > paṁca > Nimadi pā̃c / (changes.sc nasal-cluster.nasalstop)
[Bhoj Ku Garh N Dotyali Phal K eastern -BH hindi -bundeli_atarra -kaithal -NDu -Aw Bhatri PauriBareli M mewari_bannoda mia-insular -Ap -NiDoc -KharI -OSi OP thar S WK Rj G bhad kgr] l Stop > $2-asp $2                           ; @assim-r — 15 lects / l + stop > geminate, as the mia-core lects (pʰalgú > pʰaggu > pʰāg, valgā > bāg, kálpa > kāp). The chain had no l + stop rule, so lg/lp survived to the output (gold 0; hill-climb round 7) / (changes.sc assim-r.lstop)
[bihari-tharu Ku Garh N Dotyali eastern -BH -Hajong -KS hindi marathi-bhil northwest rajasthani-gujarati -Nimadi] {n ṇ m l v} y > $1 $1                        ; @assim-sonorant — 40 lects / nasal/l/v + y > geminate (anya > aṇṇa) / (changes.sc assim-sonorant.nnmlvy)
[Aw] ; r t > ṭ ṭ                                   ; @assim-r @rt-retroflex — r + t > ṭṭ (vartate > vaṭṭai, āvartana > auṭan)   [pruned for Aw: LOO -0.0003]
[Aw] ; Nasal Nasal > $2 $2                         ; @assim-sonorant — unmagna > ummagga   [pruned for Aw: never fires]
[Pk Si Pa As-Ul] ; ── block mia-clusters: MIA r + C and stop-cluster assimilation (blocks theme "mia", 2026-09-30): rC > CC, then stop + sonorant/nasal and stop + stop (mia-stop-clusters). For lects wit [@assim-r @assim-sonorant @assim-stop — rC > CC, stop + sonorant/nasal, stop + stop (blocks/mia-clusters.sc; blocks round 2026-09-30)] ──
[Si] {r ḣ} C > $2-asp $2                          ; @assim-r — 109 lects / r (and visarga) + C > geminate (karma > kamma, sarpa > sappa, dúḥkʰa > dukkʰa) / (changes.sc assim-r.rhc)
[Pk Si Pa As-Ul] ; ── block mia-stop-clusters: MIA assimilation of stop + sonorant/nasal and stop + stop clusters (blocks theme "mia", 2026-09-30). The same two changes, in this order, are positions 18–19 of ──
[Pk Si Pa As-Ul] Stop {r y v l Nasal} > $1-asp $1             ; @assim-sonorant — 83 lects / stop + r y v l nasal > geminate stop (putra > putta, śakya > sakka, agni > aggi) / (changes.sc assim-sonorant.stopryvlna)
[Si] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[rajasthani-gujarati -G] Stop r > $1-asp $1                           ; @assim-sonorant — 16 lects / stop + r (no Gujarati-type tr retention here) / (changes.sc assim-sonorant.stopr)
[kaithal] ; Nasal Nasal > $2 $2                         ; @assim-sonorant — unmagna > ummagga   [pruned for kaithal: never fires]
[hindi -Kannauji-Central thar L] {n ṇ l v} {y v} > $1 $1                      ; @assim-sonorant — 11 lects / anya > anna, kalya > kalla / (changes.sc assim-sonorant.nnlvyv)
[Kannauji-Central] ; Nasal Nasal > $2 $2                         ; @assim-sonorant — unmagna > ummagga   [pruned for Kannauji-Central: never fires]
[Kannauji-Central] ; {n ṇ l v} {y v} > $1 $1                     ; @assim-sonorant — anya > anna, kalya > kalla   [pruned for Kannauji-Central: LOO -0.0020]
[hindi northwest -thar -L bagri_fatehabad mewati_akera] ; ── block lar-vy-geminate: MIA geminate glides harden: vv > bb (the shared lar-v-b-geminate), then vy > bb and yv yy > jj. Together identical to the former single change v-b.vyvy ({v y} { [@v-b @y-j — geminate vv vy > bb, yv yy > jj (blocks/lar-vy-geminate.sc)] ──
[hindi northwest -thar -L bagri_fatehabad mewati_akera] ; ── block lar-v-b-geminate: MIA geminate vv (< rv, vy, dv) > bb. West Pahari, Rajasthani, Eastern, Pali. Laryngeals round (2026-09-30). Edit here, not in the lects. [@v-b — sárva > sabb-, parvan > pabba] ──
[OAw khet bagri_fatehabad mewati_akera] v v > b b                                    ; @v-b — 18 lects / sárva > sabb-, parvan > pabba, dravya > dabba / (changes.sc v-b.vv)
[hindi northwest -thar -L bagri_fatehabad mewati_akera] v y > b b                                    ; @v-b — vy > bb (cavya > H. cāb, *yāvyar- > H. jābar) / (block lar-vy-geminate)
[hindi northwest -thar -L bagri_fatehabad mewati_akera] y {v y} > j j                                ; @y-j — yy > jj (*bandʰēyya > H. bandʰej) / (block lar-vy-geminate)
[northwest -kcch] C $1 {r y v l} > $1 $2                       ; @assim-sonorant — 7 lects / sonorant lost after a geminate (catuṣpattra > -patta) / (changes.sc assim-sonorant.cryvl)
[northwest -OP -D -thar -Khol -khet] ; ── block nasals-geminate-dental: nasals-geminate-dental — NIA ṇṇ > nn, nasals theme, blocks round 2026-09-30. Stage 2 (western/north-western): the MIA geminate ṇṇ (< rṇ, ṇṇ, jñ, ṇy) becomes den [@n-dental — NIA ṇṇ > nn (blocks/nasals-geminate-dental.sc)] ──
[northwest -OP -D -thar -Khol -khet] ṇ ṇ > n n                                    ; @n-dental — 21 lects / kárṇa > kaṇṇa > G kān, P kann, parṇá > S panu, cūrṇa > G cūn / (changes.sc n-dental.nn)
[northwest -D -thar -Khol -khet] ; ── block mia-stop-clusters: MIA assimilation of stop + sonorant/nasal and stop + stop clusters (blocks theme "mia", 2026-09-30). The same two changes, in this order, are positions 18–19 of [@assim-sonorant @assim-stop — stop + sonorant/nasal, stop + stop > geminate (blocks/mia-stop-clusters.sc; blocks round 2026-09-30)] ──
[S P poth L] Stop {r y v l Nasal} > $1-asp $1             ; @assim-sonorant — 83 lects / stop + r y v l nasal > geminate stop (putra > putta, śakya > sakka, agni > aggi) / (changes.sc assim-sonorant.stopryvlna)
[P poth L] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[dhundari_bamore] ; ── block hiatus-ava-u: Hiatus theme — ava/uva > o (blocks round, 2026-09-30). ava > o (`hiatus-ava`) and, in Rajasthani and its neighbours, also uva (incl. < upa) > o: the `{a u} v a  [@aya-e — blocks/hiatus-ava-u.sc; ava/apa/upa > o in any position (nava > nau/no, dʰavala > ḍɔḷɔ)] ──
[dhundari_bamore] ; ── block hiatus-ava: Hiatus theme — ava > o (blocks round, 2026-09-30). MIA contraction of a + v + a to o (ava incl. < apa where p > v precedes it). The lects wrote it as `a v a > o ──
[dhundari_bamore] u v a > o                                    ; @aya-e — uva/upa > o (*dʰauvati > Marw dʰoṇo; ava/apa/upa > o in any position) / (block hiatus-ava-u)
[Garh] ṁ {b bʰ} > m m                               ; @nasal-assim — mb(ʰ) > mm only (Garh. keeps ŋg: kaŋgaṇ)
[bihari-tharu -Chitwan -Rana -Dang -DewasDoneDanuwar -Morang -Darai -Bi Ku Garh eastern -MH OAw marathi-bhil -DungraBhili -PauriBareli -M -Vasavi -OM -Noiri -RathwiBareli -Ko-Ul khet mewari_basad bagri_fatehabad Nimadi hadothi_kelwada mewati_jhambaus mewati_akera dhundari_bamore OMarw Goj west-pahari] ; ── block lar-v-b-geminate: MIA geminate vv (< rv, vy, dv) > bb. West Pahari, Rajasthani, Eastern, Pali. Laryngeals round (2026-09-30). Edit here, not in the lects. [@v-b — geminate vv > bb (blocks/lar-v-b-geminate.sc)] ──
[bihari-tharu -Chitwan -Rana -Dang -DewasDoneDanuwar -Morang -Darai -Bi Ku Garh eastern -MH hindi marathi-bhil -DungraBhili -PauriBareli -M -Vasavi -OM -Noiri -RathwiBareli -Ko-Ul northwest -thar -L mewari_basad bagri_fatehabad Nimadi hadothi_kelwada mewati_jhambaus mewati_akera dhundari_bamore OMarw Goj west-pahari] v v > b b                                    ; @v-b — 18 lects / sárva > sabb-, parvan > pabba, dravya > dabba / (changes.sc v-b.vv)
[KharI S mewati_akera pan] ; ── block mia-stop-clusters-cr: MIA stop-cluster assimilation in lects that keep stop + r (Sindhi, W. Pahari; blocks theme "mia", 2026-09-30): stop + y v l nasal and stop + stop assimilate, st [@assim-sonorant @assim-stop — stop + y v l nasal, stop + stop > geminate (stop + r kept) (blocks/mia-stop-clusters-cr.sc; blocks round 2026-09-30)] ──
[S] Stop {y v l Nasal} > $1-asp $1               ; @assim-sonorant — 22 lects / stop + y v l nasal > geminate stop (śakya > sakka, agni > aggi, pakva > pakka) / (changes.sc assim-sonorant.stopyvlnas)
[mewati_akera] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / stop + stop > geminate of the second (bʰakta > bʰatta, sapta > satta, mudga > mugga) / (changes.sc assim-stop.stopstop)
[S khet] {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ p pʰ b bʰ} r > $1-asp $1 ; @assim-sonorant — 2 lects / Sindhi keeps dental + r (> ṭr below) / (changes.sc assim-sonorant.kkhgghcchj)
[bundeli_atarra bagheli_lakshman] ; G $1 $1 > $1-asp $2                         ; @cluster-simplify — triple geminates from 3-C clusters reduce (uṣṭra > uṭṭʰṭa > uṭṭʰa)   [pruned: LOO < 0 here]
[hindi -bundeli_atarra -kaithal] G $1 $1 > $1-asp $2                          ; @cluster-simplify — 7 lects / triple geminates from 3-C clusters reduce (uṣṭra > uṭṭʰṭa > uṭṭʰa) / (changes.sc cluster-simplify.g)
[eastern] Nasal {y v r l} > $1 $1                      ; @assim-sonorant — 11 lects / nasal + sonorant (anya > aṇṇa, āmra > amma) / (changes.sc assim-sonorant.nasalyvrl)
[Pa] C {r y v l} > $1 / # _                       ; @initial-degem — 15 lects / non-stop + sonorant clusters keep the first member (mrakṣati > mākʰiba, śrāvaṇa > sāun) / (changes.sc initial-degem.cryvl)
[mia-insular -Ap -NiDoc -Dhp -KharI] C $1 > $2 / C _                              ; @cluster-simplify — 7 lects / three-consonant clusters: ṁdd > ṁd, kāŋkṣin > kaṁkʰi / (changes.sc cluster-simplify.c.postconsonantal)
[bihari-tharu Ku Garh N Dotyali Chil eastern hindi marathi-bhil -Khandesi -OM Pk Ap Md Pa northwest -S -khet rajasthani-gujarati -OG -G -Goj jaun kc kgr mand kvar kul] C C > $2 / # _                               ; @initial-degem — 60 lects / (changes.sc initial-degem.cc)
[S khet Nimadi G] C CnoR > $2 / # _                            ; @initial-degem — 3 lects / tr/dr survive (> ṭr) / (changes.sc initial-degem.ccnor)
[central-east-pahari-romani -Ku -Garh -N -Dotyali dardic-north -Sh -Sv -Phal Bshk Dm Tir Pas mia-insular -Md -KharI -OSi northwest -OP -Khol -khet nuristani -Gmb -Kam -Pr kgr kvar cam kul] ; ── block cr-release: End of the cluster window: the onset protected by cr-keep-optional / r-anticipation-optional becomes stop + r again, and later rules (flaps, tr > ṭr, …) see an  [end of the cluster window (blocks/cr-release.sc)] ──
[central-east-pahari-romani -Ku -Garh -N -Dotyali dardic-north -Sh -Sv -Phal Bshk Dm Tir Pas mia-insular -Md -KharI -OSi northwest -OP -Khol -khet nuristani -Gmb -Kam -Pr kgr kvar cam kul] {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} > {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} r ; @cr-keep — protected stop + r released after the cluster stage / (block cr-release)
[kiuth] a > ā / # (C) (C) _ C C                     ; @degem-lengthen — a lengthened in a closed initial syllable
[kul] a > ɔ / # (C) (C) _ // _ C (C) ā            ; @a-rounding — short a > ɔ in the initial syllable (written ɔ by kullui-org: adyá > ɔʣ, aṣṭā́ > ɔṭʰ, *avalalati > ɔḷnā), not pretonic before ā, which took the stress (as in ktg/kc: *ḍakkāra > ḍakār, *kalyā-hāra > kalār, nánāndr̩ > naṇān; round 7: ext .3702→.3694, 10 better / 2 worse)
[kc] a > o / # (C) C _ {v y} V | # _ C V // _ C ā      ; @a-rounding — Kochi: a > o (not ɔ) before v/y and vowel-initially, ahead of the shared ɔ (pára > pɔri but yava-type -ov-)
[kc ktg kul] ; ── block lar-a-rounding-whp: Inner West Pahari (Kotgarhi, Kochi, Kvari) rounding of short a to ɔ in an open initial syllable, not before ā in the next syllable. Laryngeals round (2026-09-30 [@a-rounding — open-initial-syllable a > ɔ, not before ā (blocks/lar-a-rounding-whp.sc)] ──
[kc ktg kul] a > ɔ / # (C) C _ C V | # _ C V // _ C ā     ; @a-rounding — gʰaná > gʰɔṇ, gala > gɔḷo, pára > pɔri; but kaṭāha > kəṛā (pretonic before ā), kʰaṭṭa- keeps a (closed) / (block lar-a-rounding-whp)
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] ; ── block intervocalic-retroflexion: Intervocalic retroflexion of both sonorants, l > ḷ and n > ṇ, with the MIA lenition: the broad allophonic stage behind the ḷ/ṇ belt and the later ḷ > r, ṇ > n.  [Intervocalic retroflexion of both sonorants (blocks/intervocalic-retroflexion.sc)] ──
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] ; ── block liquids-lateral-retroflexion: Lateral retroflexion — single intervocalic l > ḷ (late MIA / early NIA; liquids blocks round, 2026-09-30). Stage 1 of the staged liquid history: a broad allopho [@l-retroflex] ──
[Chitwan Rana KochariyaEastDanuwar Kathoriya Garh Or hindi -Kannauji-Central -kaithal -H -bagheli_lakshman marathi-bhil -Bhatri -Khandesi -DungraBhili -PauriBareli -Vasavi -Noiri S P kcch rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] l > ḷ / V _ V                                ; @l-retroflex — 32 lects / kāla > M kāḷā, G kāḷũ, śāla > Or sāḷa / (changes.sc l-retroflex)
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] ; ── block intervocalic-n-lenition: Intervocalic weakening, MIA: n > ṇ between vowels (allophonic retroflexion; merged back to n later in the Hindi–Bihari–Nepali lects) followed by the lenition of [Intervocalic weakening (blocks/intervocalic-n-lenition.sc)] ──
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] ; ── block nasals-retroflexion: nasals-retroflexion — MIA intervocalic n > ṇ (the Prakrit "no ṇaḥ" change), nasals theme, blocks round 2026-09-30. Stage 1 of n > ṇ > n: a single dental nasal b [@n-retroflex — MIA intervocalic n > ṇ (blocks/nasals-retroflexion.sc)] ──
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] n > ṇ / V _ V                                ; @n-retroflex @nur-innovation @x-nur-n-flap — 40 lects / pānī́ya > G, P, S pāṇī, vána > P vaṇ, S vaṇu, Or baṇa / (changes.sc n-retroflex.intervocalic)
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] ; ── block mia-lenition: MIA intervocalic lenition (blocks theme "mia", 2026-09-30): the Prakrit weakening of single intervocalic stops, in the order the lects already had it. Not in mi [@int-aspirate-h @int-retroflex @int-p-v @int-stop-loss — MIA intervocalic lenition (blocks/mia-lenition.sc; blocks round 2026-09-30)] ──
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] {kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V              ; @int-aspirate-h — 90 lects / intervocalic aspirates > h (mukʰa > muha, bádʰira > bahira, lābʰa > lāha) / (changes.sc int-aspirate-h)
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] ; ── block mia-lenition-stops: MIA lenition of the unaspirated intervocalic stops (blocks theme "mia", pass 2, 2026-09-30): ṭ > ḍ, p > v, loss of k g c j t d y. Part of mia-lenition (after as [ṭ > ḍ, p > v, stop loss (split off in pass 2 for lects without aspirate > h)] ──
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] {ṭ ṭʰ} > {ḍ ḍʰ} / V _ V                      ; @int-retroflex — 102 lects / intervocalic ṭ ṭʰ > ḍ ḍʰ (gʰaṭa > gʰaḍa, kaṭuka > kaḍua, paṭʰati > paḍʰai) / (changes.sc int-retroflex)
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] p > v / V _ V                                ; @int-p-v — 119 lects / intervocalic p > v (dīpa > dīva, kūpa > kūva, tāpa > tāva) / (changes.sc int-p-v)
[Kathoriya Garh Or bundeli_atarra OAw NDu Brj SSr PalyaBareli Rathawi hal mewari_bannoda Bhilali RathwiBareli S P rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_jhambaus -Rj -OMarw -Goj kiuth kc mand ktg cur kvar cam kul] {k g c j t d y} > ∅ / V _ V                  ; @int-stop-loss — 105 lects / intervocalic k g c j t d y lost (lōka > lōa, nadī > naī, sūcī > sūī) / (changes.sc int-stop-loss.kgcjtdy)
[bihari-tharu Ku Garh Dotyali Bishnupriya Or AdivasiOriya Hajong B hindi marathi-bhil -M -OM -VB northwest -D -kcch rajasthani-gujarati -Goj west-pahari -dog -khash -WPah] ; ── block intervocalic-n-lenition: Intervocalic weakening, MIA: n > ṇ between vowels (allophonic retroflexion; merged back to n later in the Hindi–Bihari–Nepali lects) followed by the lenition of [Intervocalic weakening (blocks/intervocalic-n-lenition.sc)] ──
[bihari-tharu Ku Garh Dotyali Bishnupriya Or AdivasiOriya Hajong B hindi marathi-bhil -M -OM -VB northwest -D -kcch rajasthani-gujarati -Goj west-pahari -dog -khash -WPah] ; ── block nasals-retroflexion: nasals-retroflexion — MIA intervocalic n > ṇ (the Prakrit "no ṇaḥ" change), nasals theme, blocks round 2026-09-30. Stage 1 of n > ṇ > n: a single dental nasal b [@n-retroflex — MIA intervocalic n > ṇ (blocks/nasals-retroflexion.sc)] ──
[bihari-tharu Ku Garh N Dotyali Bishnupriya Or AdivasiOriya Hajong B hindi marathi-bhil -M -OM Pk Ap northwest -D PNur rajasthani-gujarati -Goj west-pahari -dog -khash -WPah] n > ṇ / V _ V                                ; @n-retroflex @nur-innovation @x-nur-n-flap — 40 lects / pānī́ya > G, P, S pāṇī, vána > P vaṇ, S vaṇu, Or baṇa / (changes.sc n-retroflex.intervocalic)
[bihari-tharu Ku Garh Dotyali Bishnupriya Or AdivasiOriya Hajong B hindi marathi-bhil -M -OM -VB northwest -D -kcch rajasthani-gujarati -Goj west-pahari -dog -khash -WPah] ; ── block mia-lenition: MIA intervocalic lenition (blocks theme "mia", 2026-09-30): the Prakrit weakening of single intervocalic stops, in the order the lects already had it. Not in mi [@int-aspirate-h @int-retroflex @int-p-v @int-stop-loss — MIA intervocalic lenition (blocks/mia-lenition.sc; blocks round 2026-09-30)] ──
[bihari-tharu Ku Garh N Dotyali eastern -BH hindi marathi-bhil -OM Pk northwest -kcch rajasthani-gujarati west-pahari] {kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V              ; @int-aspirate-h — 90 lects / intervocalic aspirates > h (mukʰa > muha, bádʰira > bahira, lābʰa > lāha) / (changes.sc int-aspirate-h)
[bihari-tharu Ku Garh Dotyali Bishnupriya Or AdivasiOriya Hajong B hindi marathi-bhil -M -OM -VB northwest -D -kcch rajasthani-gujarati -Goj west-pahari -dog -khash -WPah] ; ── block mia-lenition-stops: MIA lenition of the unaspirated intervocalic stops (blocks theme "mia", pass 2, 2026-09-30): ṭ > ḍ, p > v, loss of k g c j t d y. Part of mia-lenition (after as [ṭ > ḍ, p > v, stop loss (split off in pass 2 for lects without aspirate > h)] ──
[bihari-tharu Ku Garh N Dotyali eastern hindi marathi-bhil -VB Pk Ap northwest rajasthani-gujarati west-pahari] {ṭ ṭʰ} > {ḍ ḍʰ} / V _ V                      ; @int-retroflex — 102 lects / intervocalic ṭ ṭʰ > ḍ ḍʰ (gʰaṭa > gʰaḍa, kaṭuka > kaḍua, paṭʰati > paḍʰai) / (changes.sc int-retroflex)
[-N -eur -as-ll -RomWel -Kho -Kal -Kalk -Kund -dod -pog -Mai -Pas -Dm -Gaw -Tor -Shum -Wot -A -BH -MH -OM -Pk -Pa -Si -Dhp -As-Ul -KharI -Wg -Kt -Ash -Pr -dog] p > v / V _ V                                ; @int-p-v — 119 lects / intervocalic p > v (dīpa > dīva, kūpa > kūva, tāpa > tāva) / (changes.sc int-p-v)
[bihari-tharu Ku Garh N Dotyali eastern -BH -A -MB -MH -KS hindi marathi-bhil -OM northwest -D -kcch rajasthani-gujarati west-pahari] {k g c j t d y} > ∅ / V _ V                  ; @int-stop-loss — 105 lects / intervocalic k g c j t d y lost (lōka > lōa, nadī > naī, sūcī > sūī) / (changes.sc int-stop-loss.kgcjtdy)
[L] ; ── block hiatus-ava: Hiatus theme — ava > o (blocks round, 2026-09-30). MIA contraction of a + v + a to o (ava incl. < apa where p > v precedes it). The lects wrote it as `a v a > o [@aya-e — blocks/hiatus-ava.sc; ava/apa > o (L; P prefers au before a consonant)] ──
[L] a v a > o                                    ; @aya-e — ava > o (avaśāra > Bhoj osar, avatarati > otarai, avaśyā > oś) / (block hiatus-ava)
[bihari-tharu -MagahiNepal -Chitwan -Mth -Majhi -KochilaTharu RomWel RomBalk Zarg Lom Khandesi PauriBareli M RathwiBareli Ko-Ul Ap OP P L Marw had hadothi_kelwada OMarw west-pahari -bhal -dog -kiuth -cur -khash -WPah -pan] ; ── block nasals-m-v: nasals-m-v — MIA/NIA intervocalic m > v (via nasalised ṽ, nasality unwritten), nasals theme, blocks round 2026-09-30 (pass 2). Western and north-western, West P [@m-v — intervocalic m > v in the MIA block (bʰramara > P bʰãvar; nasality often unwritten) (blocks/nasals-m-v.sc)] ──
[bihari-tharu -MagahiNepal -Chitwan -Mth -Majhi -KochilaTharu RomWel RomBalk Zarg Lom Khandesi PauriBareli M RathwiBareli Ko-Ul Ap OP P L Marw had hadothi_kelwada OMarw west-pahari -bhal -dog -kiuth -cur -khash -WPah -pan] m > v / V _ V                                ; @m-v — 33 lects / grāma > gāṽ/gāv, kamala > M kavaḷ / (changes.sc m-v.m.intervocalic)
[bundeli_atarra] ; V m > $1+nas v / _ V                        ; @m-v — nā́man > nā̃v, gōdʰū́ma > gohū̃   [pruned for bundeli_atarra: LOO -0.0001]
[NDu H Aw Brj SSr] ; ── block nasals-m-nasal-v: nasals-m-nasal-v — intervocalic m > nasal vowel + v, Hindi belt (H, Aw, Brj, SSr, NDu), nasals theme, blocks round 2026-09-30 (pass 2). The m > ṽ of nasals-m-v  [@m-v — nā́man > nā̃v, gōdʰū́ma > gohū̃ (blocks/nasals-m-nasal-v.sc)] ──
[NDu H Aw Brj SSr] V m > $1+nas v / _ V                         ; @m-v — 6 lects / nā́man > H nā̃v, gōdʰū́ma > H gohū̃ / (changes.sc m-v.vm.prevocalic.2)
[jaun kvar] ; ── block length-cl-cluster: West Pahari CL before any consonant cluster in the initial syllable, applied at the cluster stage (before the MIA assimilations are complete), so it covers gemi [@degem-lengthen — W Pahari lengthening in a closed initial syllable (# (C) (C) _ C C) (blocks/length-cl-cluster.sc; blocks round 2026-09-30)] ──
[jaun kvar] {a i u} > {ā ī ū} / # (C) (C) _ C C          ; @degem-lengthen — short vowel lengthened in a closed initial syllable (jaun kárṇa > kān, dánta > dā̃d) / (block length-cl-cluster)
[bhal bhad kgr cur khash cam pan] C $1 > $2 / # _                              ; @initial-degem — 3 lects / only geminates (śv-, sn-, hr- … survive) / (changes.sc initial-degem.c)
[bihari-tharu -MagahiNepal -Chitwan -Darai -Bi central-east-pahari-romani -Garh eastern -Or -MB -KS thar poth bhal kiuth] ; ── block hiatus-ava: Hiatus theme — ava > o (blocks round, 2026-09-30). MIA contraction of a + v + a to o (ava incl. < apa where p > v precedes it). The lects wrote it as `a v a > o [@aya-e — blocks/hiatus-ava.sc; ava (incl. < apa) > o (avaśāra > osar)] ──
[bihari-tharu -MagahiNepal -Chitwan -Darai -Bi central-east-pahari-romani -Garh eastern -Or -MB -KS thar poth khet rajasthani-gujarati -OG -WK -G -OMarw bhal kiuth] a v a > o                                    ; @aya-e — ava > o (avaśāra > Bhoj osar, avatarati > otarai, avaśyā > oś) / (block hiatus-ava)
[MagahiNepal Chitwan Bi] a v a > o / # _                              ; @aya-e — 3 lects / initial ava- (incl. < apa-) > o- / (changes.sc aya-e.ava.initial.2)
[MagahiNepal Chitwan Bi] a v a > a u / _ !{y V} // _ b #              ; @aya-e — 3 lects / medial -ava- > -au- (not the -ab infinitive, still -iavva then) / (changes.sc aya-e.ava.yv.except)
[KochilaTharu poth] ; ── block intervocalic-n-lenition: Intervocalic weakening, MIA: n > ṇ between vowels (allophonic retroflexion; merged back to n later in the Hindi–Bihari–Nepali lects) followed by the lenition of [Intervocalic weakening (blocks/intervocalic-n-lenition.sc)] ──
[KochilaTharu poth] ; ── block nasals-retroflexion: nasals-retroflexion — MIA intervocalic n > ṇ (the Prakrit "no ṇaḥ" change), nasals theme, blocks round 2026-09-30. Stage 1 of n > ṇ > n: a single dental nasal b [@n-retroflex — MIA intervocalic n > ṇ (blocks/nasals-retroflexion.sc)] ──
[KochilaTharu poth] n > ṇ / V _ V                                ; @n-retroflex @nur-innovation @x-nur-n-flap — 40 lects / pānī́ya > G, P, S pāṇī, vána > P vaṇ, S vaṇu, Or baṇa / (changes.sc n-retroflex.intervocalic)
[KochilaTharu poth] ; ── block mia-lenition: MIA intervocalic lenition (blocks theme "mia", 2026-09-30): the Prakrit weakening of single intervocalic stops, in the order the lects already had it. Not in mi [@int-aspirate-h @int-retroflex @int-p-v @int-stop-loss — MIA intervocalic lenition (blocks/mia-lenition.sc; blocks round 2026-09-30)] ──
[KochilaTharu poth] {kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V              ; @int-aspirate-h — 90 lects / intervocalic aspirates > h (mukʰa > muha, bádʰira > bahira, lābʰa > lāha) / (changes.sc int-aspirate-h)
[KochilaTharu poth] ; ── block mia-lenition-stops: MIA lenition of the unaspirated intervocalic stops (blocks theme "mia", pass 2, 2026-09-30): ṭ > ḍ, p > v, loss of k g c j t d y. Part of mia-lenition (after as [ṭ > ḍ, p > v, stop loss (split off in pass 2 for lects without aspirate > h)] ──
[KochilaTharu poth] {ṭ ṭʰ} > {ḍ ḍʰ} / V _ V                      ; @int-retroflex — 102 lects / intervocalic ṭ ṭʰ > ḍ ḍʰ (gʰaṭa > gʰaḍa, kaṭuka > kaḍua, paṭʰati > paḍʰai) / (changes.sc int-retroflex)
[KochilaTharu poth] p > v / V _ V                                ; @int-p-v — 119 lects / intervocalic p > v (dīpa > dīva, kūpa > kūva, tāpa > tāva) / (changes.sc int-p-v)
[KochilaTharu poth] {k g c j t d y} > ∅ / V _ V                  ; @int-stop-loss — 105 lects / intervocalic k g c j t d y lost (lōka > lōa, nadī > naī, sūcī > sūī) / (changes.sc int-stop-loss.kgcjtdy)
[P kcch poth L srk awan] ; ── block finals-akam-lahnda: Lahnda/Punjabi neuter -akam (blocks theme "finals", 2026-09-30): -aam/-aaṁ (< -akam) > -ā̃. [-aam > -ā̃ (blocks/finals-akam-lahnda.sc) — replaces `use final-nasal.aamm.2`: neuter -akaṁ > -aã > -ā̃ (kāṇá > kāṇā̃, kā́ka > kā̃; round 7)] ──
[P kcch poth L srk awan] a a {m ṁ} > ā̃ / _ #                         ; @final-nasal @hiatus — 5 lects / -aam (< -akam) > -ā̃ (P. kāṇá+akam > kāṇā̃, dʰūmá+akam > dʰūā̃, kā́ka+akam > kā̃) / (changes.sc final-nasal.aamm.2)
[northwest -D -thar -Khol -khet] {a ā} ā̃ > ā̃ / _ #                          ; @hiatus — 5 lects / a vowel before the new final -ā̃ is absorbed (kā́ka+akaṁ > *kāā̃ > kā̃, kīdr̩śa > kihā̃; round 7) / (changes.sc hiatus.aaa.final.3)
[L] ; ── block finals-schwa-shortening-lahnda: Lahnda/Hindko reduction of final -ā and -ī (blocks theme "finals", 2026-09-30): final -ā and (Hindko) -ī after a consonant reduce to ə, later lost by finals-apo [final -ā -ī > -ə (blocks/finals-schwa-shortening-lahnda.sc) — replaces `{ā ī} > ə / C _ #`: Apabhraṁśa shortening of final -ā and (Hindko) -ī to a reduced vowel, lost with the final short vowels (dʰā́rā > dʰār, vallī > val, bʰaginī > bʰε̄̀ṇ, garbʰíṇī > gabbʰuṇ; round 2)] ──
[L] {ā ī} > ə / C _ #                            ; @final-vowel-loss @final-a-short — final -ā -ī > -ə (L. avēlā > avel, aṣṭā́ > aṭṭʰ, kalā́ > kal) / (block finals-schwa-shortening-lahnda)
[L] ; ── block finals-schwa-shortening: Punjabi-Lahnda reduction of final -ā (blocks theme "finals", 2026-09-30): the Apabhraṁśa shortening of final -ā after a consonant gives a reduced ə, which the l ──
[S] ; ── block finals-akam-pahari: Pahari neuter -akam (blocks theme "finals", 2026-09-30): the neuter/verbal-noun -aam (< -akam, -anakam) becomes -u, the Nepali and West Pahari infinitive/citati [-aam > -u (blocks/finals-akam-pahari.sc) — replaces `use final-nasal.aam`: neuter -akaṁ > Ap. -aaũ > -u: verbal noun -anakam > -aṇu, the Sindhi infinitive (aḍḍanakam > aᶑaṇu; round 5)] ──
[S] a a m > u / _ #                              ; @final-nasal @hiatus — 5 lects / -aam (< -akam) > -u (N. áraṇa+akam > arnu, *aḍḍ+iyanakam > aṛinu, adyá+akam > āju) / (changes.sc final-nasal.aam)
[PalyaBareli Noiri Bhili] # ── Bhil post-MIA prefix: first 4 of 4 shared changes (round-7 unification) ──
[Khandesi] # ── Bhil post-MIA prefix: first 3 of 4 shared changes (round-7 unification) ──
[AdivasiOriya Bhatri Khandesi PalyaBareli Rathawi hal Bhili] ; ── block nasals-nc-lengthen: nasals-nc-lengthen — a > ā before a retained homorganic nasal + stop, Bhil/Halbic and Adivasi Oriya, nasals theme, blocks round 2026-09-30 (pass 2). nasals-anus [@anusvara-vowel — a lengthened before the retained homorganic nasal (candrá > cānd) (blocks/nasals-nc-lengthen.sc)] ──
[AdivasiOriya MB OB marathi-bhil -PauriBareli -M -OM -VB -Ko-Ul] a > ā / _ Nasal Stop                         ; @anusvara-vowel @degem-lengthen — 12 lects / dánta > dānt > DungraBhili ḍāṭ, candrá > mewari_bannoda cānd / (changes.sc anusvara-vowel.a.nasalstop)
[marathi-bhil -Bhatri -DungraBhili -hal -M -mewari_bannoda -OM -VB -Ko-Ul] u > a / V C* _ C C* V                        ; @vowel-shift — 8 lects / medial (mostly penultimate) u reduced to a before a consonant (kukkuṭa > kūkaḍo type); accent-independent (round 2). Round 4: the C spares -uka > -ua (ext-aware) / (changes.sc vowel-shift.u.vcccv)
[bihari-tharu Gmb] {s t d n} > ∅ / _ #                          ; @final-cons-loss — 17 lects / stem-final s t d n (nā́man > nāũ); other final C are Turner verbal-root citations (*kʰēll) / (changes.sc final-cons-loss.stdn)
[A MB MH KS] # ── Eastern post-MIA prefix: first 4 of 4 shared changes (round-7 unification) ──
[Bishnupriya AdivasiOriya B] # ── Eastern post-MIA prefix: first 3 of 4 shared changes (round-7 unification) ──
[bihari-tharu -Bote Ku Garh N eastern hindi -Kannauji-Central marathi-bhil -Rathawi northwest -OP -S -Khol OG mewati_jhambaus G OMarw west-pahari -kc -kgr -mand -kvar] ; ── block finals-ap-shortening: Apabhraṁśa shortening of final -ā (blocks theme "finals", 2026-09-30): late MIA final long -ā shortens to -a, so that it is later lost with the old short -a by  [Ap. final -ā > -a (blocks/finals-ap-shortening.sc) — replaces `use final-a-short.a`: MIA shortening of final -ā (mū́lā > mūḷ)] ──
[bihari-tharu -Bote Ku Garh N eastern hindi -Kannauji-Central marathi-bhil -Rathawi northwest -OP -S -Khol OG mewati_jhambaus G OMarw west-pahari -kc -kgr -mand -kvar] ā > a / _ #                                  ; @final-a-short @final-shortening @final-vowel-loss @vowel-shift — 41 lects / final -ā > -a (Bi. ā́rā > ār, kʰáṭvā > kʰāṭ, nāvā́ > nāv) / (changes.sc final-a-short.a)
[Noiri] ; ── block lar-s-h: s > h everywhere: Bhil. Laryngeals round (2026-09-30). Edit here, not in the lects. [@s-h — s > h everywhere (blocks/lar-s-h.sc)] ──
[Noiri] s > h                                        ; @s-h — 8 lects / divasá > dahaḍo, sarpá > hāp, sūcī́ > hūī / (changes.sc s-h.s)
[eastern -Or] ; ── block finals-am-east: Eastern neuter -am (blocks theme "finals", 2026-09-30): the neuter/accusative ending -am nasalises to -ã before final consonants are lost, so it survives the MI [-am > -ã (blocks/finals-am-east.sc) — replaces `use final-nasal.am.2`: neuter -am > -ã before final-consonant loss (as in A.)] ──
[eastern -Or] a m > ã / _ #                                ; @final-nasal — 7 lects / -am > -ã (A. *ujjāṭī+akam > uzār, kʰánati+āpitavyakam > kʰanāiba via -aã > -a) / (changes.sc final-nasal.am.2)
[BH A Aw] ; ── block late-mergers: Late NIA mergers, order-free: initial v- > b-, ṇ > n, initial y- > j-. Shared by 20 lects in 5 clades (Bihari, E. Hindi, Early NIA, Eastern, Other); blocks roun [package (blocks/late-mergers.sc)] ──
[BH A Aw] ; ── block lar-v-b: Initial v- > b- (NIA): Hindi, Bihari, Eastern, Pahari, Rajasthani, Punjabi, Lahnda, Sindhi, Dardic, Romani … the most widespread NIA consonant change. Geminate  [@v-b — initial v- > b- (blocks/lar-v-b.sc)] ──
[BH A Aw] v > b / # _                                  ; @v-b — 88 lects / vadʰū́ > H. bahū, vā́la > H. bāl / (changes.sc v-b.v.initial)
[BH A Aw] ; ── block nasals-dental-merger: nasals-dental-merger — NIA ṇ > n everywhere, nasals theme, blocks round 2026-09-30. Stage 2 (Hindi belt, Bihari, Tharu, Nepali …): every ṇ — the MIA intervocali [@n-dental — NIA ṇ > n (blocks/nasals-dental-merger.sc)] ──
[A Aw] ṇ > n                                        ; @n-dental — 66 lects / pānī́ya > H, Bhoj pānī, guṇá > H gun, vána > Bhoj ban / (changes.sc n-dental.n)
[BH A Aw] y > j / # _                                  ; @y-j — 101 lects / (changes.sc y-j.initial)
[BH MB MH KS] ; ── block hiatus-aya-except: Hiatus theme — aya > ē outside the -ayati verb ending (blocks round, 2026-09-30). The same early aya > ē as `hiatus-aya`, in the lects whose NIA verbs drop the  [inherited from the Eastern sisters (small-lect default, 2026-09-30)] ──
[BH MB MH KS] a y a > ē // _ t {i ē} #                     ; @aya-e — aya > ē except in -ayati/-ayatē (nayana > nēṇa, tráyaḣ > trē; ākalayati > M ākaḷṇẽ) / (block hiatus-aya-except)
[Sh eastern -Or -BH -Hajong -OB Khandesi PalyaBareli Noiri Bhili] # ── end of shared prefix ──
[Khandesi DungraBhili Rathawi PauriBareli Bhilali Noiri] if:ext a a {m ṁ} > u / _ #                ; @morph — round 4b: neuter -akam > -aaṁ > -u (before final-consonant loss)
[Khandesi PalyaBareli Rathawi Bhilali Bhili] if:ext a a {m ṁ} > ū̃ / _ #                ; @morph — round 4b: neuter -akam > -aaṁ > -ū̃ (before final-consonant loss)
[Dotyali hal mewari_bannoda Vasavi VB Ko-Ul Pk Ap thar kcch PNur mewari_dholpura hadothi_kelwada Rj] ; ── block nasals-retroflexion: nasals-retroflexion — MIA intervocalic n > ṇ (the Prakrit "no ṇaḥ" change), nasals theme, blocks round 2026-09-30. Stage 1 of n > ṇ > n: a single dental nasal b [@n-retroflex — MIA intervocalic n > ṇ (blocks/nasals-retroflexion.sc)] ──
[Dotyali hal mewari_bannoda Vasavi Ko-Ul thar mewari_dholpura hadothi_kelwada Rj] n > ṇ / V _ V                                ; @n-retroflex @nur-innovation @x-nur-n-flap — 40 lects / pānī́ya > G, P, S pāṇī, vána > P vaṇ, S vaṇu, Or baṇa / (changes.sc n-retroflex.intervocalic)
[BH Hajong mewari_bannoda Ko-Ul thar kcch] ; ── block mia-lenition-stops: MIA lenition of the unaspirated intervocalic stops (blocks theme "mia", pass 2, 2026-09-30): ṭ > ḍ, p > v, loss of k g c j t d y. Part of mia-lenition (after as [@int-retroflex @int-p-v @int-stop-loss — ṭ > ḍ, p > v, intervocalic stop loss (no aspirate > h here) (blocks/mia-lenition-stops.sc; blocks round 2026-09-30, pass 2)] ──
[Hajong mewari_bannoda Ko-Ul thar] {ṭ ṭʰ} > {ḍ ḍʰ} / V _ V                      ; @int-retroflex — 102 lects / intervocalic ṭ ṭʰ > ḍ ḍʰ (gʰaṭa > gʰaḍa, kaṭuka > kaḍua, paṭʰati > paḍʰai) / (changes.sc int-retroflex)
[BH Hajong mewari_bannoda Ko-Ul thar] p > v / V _ V                                ; @int-p-v — 119 lects / intervocalic p > v (dīpa > dīva, kūpa > kūva, tāpa > tāva) / (changes.sc int-p-v)
[BH Hajong mewari_bannoda Ko-Ul thar kcch] {k g c j t d y} > ∅ / V _ V                  ; @int-stop-loss — 105 lects / intervocalic k g c j t d y lost (lōka > lōa, nadī > naī, sūcī > sūī) / (changes.sc int-stop-loss.kgcjtdy)
[bagheli_lakshman] ; V m > $1+nas v / _ V                        ; @m-v — nā́man > nā̃v, gōdʰū́ma > gohū̃   [pruned for bagheli_lakshman: LOO -0.0011]
[Brj] ; i n > ī / _ #                               ; @morph — in-stems cited in nom. -ī (hastín > hātʰī)   [pruned for Brj: never fires]
[Kannauji-Central] ; V m > $1+nas v / _ V                        ; @m-v — nā́man > nā̃v, gōdʰū́ma > gohū̃   [pruned for Kannauji-Central: LOO -0.0064]
[NDu] ; round 7: removed the @morph in-stem rule i n > ī (neutral: ext/verb unchanged, plain +0.0002)
[SSr] ; i n > ī / _ #                               ; @morph — in-stems cited in nom. -ī (hastín > hātʰī)   [pruned for SSr: never fires]
[Aw] ; ── block liquids-lateral-retroflexion: Lateral retroflexion — single intervocalic l > ḷ (late MIA / early NIA; liquids blocks round, 2026-09-30). Stage 1 of the staged liquid history: a broad allopho [@l-retroflex — stage 1 (late MIA): single intervocalic l > ḷ] ──
[kcch] a a m > ū / _ #                             ; @final-nasal @hiatus — neuter -akaṁ > -aũ > -ū: the Kutchi infinitive -ṇū from the verbal noun -anakam (kaḍḍʰanakam > kaḍʰṇū; round 6)
[Pk Ap Pa As-Ul] ; ── Law of two morae ────────────────────────────────────────────────────
[Pk] s s > s / {ī ū ē ō ā} _                    ; @degem-lengthen — long vowel + ss keeps its length (īśvara > īsara, śīrṣa > sīsa)
[As-Ul] ; ── block length-two-mora-tonic: MIA law of two morae, stress-conditioned (Bihari-Tharu): a long vowel shortens before a consonant cluster unless an accented syllable follows (pretonic long vow [@two-mora @stress — two-mora law, not before an accented syllable (blocks/length-two-mora-tonic.sc; blocks round 2026-09-30)] ──
[As-Ul] {ā ī ū ē ō ā́ ī́ ū́ ḗ ṓ} > {a i u e o á í ú é ó} / _ C C // _ !#* Acc ; @two-mora @stress — long V shortens before CC unless an accent follows (mū́lya > *mulla > Mth mūl; pretonic rāṣṭrá > rāṭṭʰa keeps ā) / (block length-two-mora-tonic)
[Ap] ; ── block length-two-mora-tonic: MIA law of two morae, stress-conditioned (Bihari-Tharu): a long vowel shortens before a consonant cluster unless an accented syllable follows (pretonic long vow [@two-mora @stress — two-mora law, not before an accented syllable (blocks/length-two-mora-tonic.sc; blocks round 2026-09-30)] ──
[Ap] {ā ī ū ē ō ā́ ī́ ū́ ḗ ṓ} > {a i u e o á í ú é ó} / _ C C // _ !#* Acc ; @two-mora @stress — long V shortens before CC unless an accent follows (mū́lya > *mulla > Mth mūl; pretonic rāṣṭrá > rāṭṭʰa keeps ā) / (block length-two-mora-tonic)
[Pk khash] ; ── block length-two-mora-tonic: MIA law of two morae, stress-conditioned (Bihari-Tharu): a long vowel shortens before a consonant cluster unless an accented syllable follows (pretonic long vow [@two-mora @stress — two-mora law, not before an accented syllable (blocks/length-two-mora-tonic.sc; blocks round 2026-09-30)] ──
[Pk khash] {ā ī ū ē ō ā́ ī́ ū́ ḗ ṓ} > {a i u e o á í ú é ó} / _ C C // _ !#* Acc ; @two-mora @stress — long V shortens before CC unless an accent follows (mū́lya > *mulla > Mth mūl; pretonic rāṣṭrá > rāṭṭʰa keeps ā) / (block length-two-mora-tonic)
[Pa] ; ── block length-two-mora-tonic: MIA law of two morae, stress-conditioned (Bihari-Tharu): a long vowel shortens before a consonant cluster unless an accented syllable follows (pretonic long vow [@two-mora @stress — two-mora law, not before an accented syllable (blocks/length-two-mora-tonic.sc; blocks round 2026-09-30)] ──
[Pa] {ā ī ū ē ō ā́ ī́ ū́ ḗ ṓ} > {a i u e o á í ú é ó} / _ C C // _ !#* Acc ; @two-mora @stress — long V shortens before CC unless an accent follows (mū́lya > *mulla > Mth mūl; pretonic rāṣṭrá > rāṭṭʰa keeps ā) / (block length-two-mora-tonic)
[Tor eastern -AdivasiOriya -OB -B KharI D] ; ── block hiatus-aya-except: Hiatus theme — aya > ē outside the -ayati verb ending (blocks round, 2026-09-30). The same early aya > ē as `hiatus-aya`, in the lects whose NIA verbs drop the  [@aya-e — blocks/hiatus-aya-except.sc; aya > e outside the -ayati verb ending (nayana > Or. neṇa)] ──
[Tor eastern -AdivasiOriya -OB -B KharI D] a y a > ē // _ t {i ē} #                     ; @aya-e — aya > ē except in -ayati/-ayatē (nayana > nēṇa, tráyaḣ > trē; ākalayati > M ākaḷṇẽ) / (block hiatus-aya-except)
[Dotyali eastern -Or -BH -AdivasiOriya -Hajong M Vasavi D Khol mewari_dholpura Goj khash WPah] ; ── block mia-lenition: MIA intervocalic lenition (blocks theme "mia", 2026-09-30): the Prakrit weakening of single intervocalic stops, in the order the lects already had it. Not in mi [@int-aspirate-h @int-retroflex @int-p-v @int-stop-loss — MIA intervocalic lenition (blocks/mia-lenition.sc; blocks round 2026-09-30)] ──
[Dotyali Bishnupriya B Vasavi Khol mewari_dholpura] {kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V              ; @int-aspirate-h — 90 lects / intervocalic aspirates > h (mukʰa > muha, bádʰira > bahira, lābʰa > lāha) / (changes.sc int-aspirate-h)
[Dotyali eastern -Or -BH -AdivasiOriya -Hajong M Vasavi D Khol mewari_dholpura Goj khash WPah] ; ── block mia-lenition-stops: MIA lenition of the unaspirated intervocalic stops (blocks theme "mia", pass 2, 2026-09-30): ṭ > ḍ, p > v, loss of k g c j t d y. Part of mia-lenition (after as [ṭ > ḍ, p > v, stop loss (split off in pass 2 for lects without aspirate > h)] ──
[Dotyali Bishnupriya B Vasavi Khol mewari_dholpura] {ṭ ṭʰ} > {ḍ ḍʰ} / V _ V                      ; @int-retroflex — 102 lects / intervocalic ṭ ṭʰ > ḍ ḍʰ (gʰaṭa > gʰaḍa, kaṭuka > kaḍua, paṭʰati > paḍʰai) / (changes.sc int-retroflex)
[Dotyali Bishnupriya A B MH Vasavi Khol mewari_dholpura] p > v / V _ V                                ; @int-p-v — 119 lects / intervocalic p > v (dīpa > dīva, kūpa > kūva, tāpa > tāva) / (changes.sc int-p-v)
[Dotyali eastern -Or -BH -AdivasiOriya -Hajong -OB Vasavi D Khol mewari_dholpura] {k g c j t d y} > ∅ / V _ V                  ; @int-stop-loss — 105 lects / intervocalic k g c j t d y lost (lōka > lōa, nadī > naī, sūcī > sūī) / (changes.sc int-stop-loss.kgcjtdy)
[Rj] {kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V              ; @int-aspirate-h — 90 lects / (changes.sc int-aspirate-h)
[hadothi_kelwada Rj] {ṭ ṭʰ} > {ḍ ḍʰ} / V _ V                      ; @int-retroflex — 102 lects / (changes.sc int-retroflex)
[mewari_basad hadothi_kelwada Goj] # ── Rajasthanic post-MIA prefix: first 2 of 4 shared changes (round-7 unification) ──
[Rj Goj] a {d y} a > e / _ C                          ; @hiatus — 12 lects / MIA -aya- (< -ada-, -aya-) contracts to e before a consonant (kadala > keḷo, gáti/gata- > gelo) / (changes.sc hiatus.adya.preconsonantal)
[Hajong] # ── Eastern post-MIA prefix: first 2 of 4 shared changes (round-7 unification) ──
[Hajong mewari_basad hadothi_kelwada Goj] # ── end of shared prefix ──
[Or khet] ; ── block intervocalic-n-lenition: Intervocalic weakening, MIA: n > ṇ between vowels (allophonic retroflexion; merged back to n later in the Hindi–Bihari–Nepali lects) followed by the lenition of [Intervocalic weakening (blocks/intervocalic-n-lenition.sc)] ──
[Or khet] ; ── block nasals-retroflexion: nasals-retroflexion — MIA intervocalic n > ṇ (the Prakrit "no ṇaḥ" change), nasals theme, blocks round 2026-09-30. Stage 1 of n > ṇ > n: a single dental nasal b [@n-retroflex — MIA intervocalic n > ṇ (blocks/nasals-retroflexion.sc)] ──
[Or khet] n > ṇ / V _ V                                ; @n-retroflex @nur-innovation @x-nur-n-flap — 40 lects / pānī́ya > G, P, S pāṇī, vána > P vaṇ, S vaṇu, Or baṇa / (changes.sc n-retroflex.intervocalic)
[Or khet] ; ── block mia-lenition: MIA intervocalic lenition (blocks theme "mia", 2026-09-30): the Prakrit weakening of single intervocalic stops, in the order the lects already had it. Not in mi [@int-aspirate-h @int-retroflex @int-p-v @int-stop-loss — MIA intervocalic lenition (blocks/mia-lenition.sc; blocks round 2026-09-30)] ──
[Or khet] {kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V              ; @int-aspirate-h — 90 lects / intervocalic aspirates > h (mukʰa > muha, bádʰira > bahira, lābʰa > lāha) / (changes.sc int-aspirate-h)
[Or khet] ; ── block mia-lenition-stops: MIA lenition of the unaspirated intervocalic stops (blocks theme "mia", pass 2, 2026-09-30): ṭ > ḍ, p > v, loss of k g c j t d y. Part of mia-lenition (after as [ṭ > ḍ, p > v, stop loss (split off in pass 2 for lects without aspirate > h)] ──
[Or khet] {ṭ ṭʰ} > {ḍ ḍʰ} / V _ V                      ; @int-retroflex — 102 lects / intervocalic ṭ ṭʰ > ḍ ḍʰ (gʰaṭa > gʰaḍa, kaṭuka > kaḍua, paṭʰati > paḍʰai) / (changes.sc int-retroflex)
[Or khet] p > v / V _ V                                ; @int-p-v — 119 lects / intervocalic p > v (dīpa > dīva, kūpa > kūva, tāpa > tāva) / (changes.sc int-p-v)
[Or khet] {k g c j t d y} > ∅ / V _ V                  ; @int-stop-loss — 105 lects / intervocalic k g c j t d y lost (lōka > lōa, nadī > naī, sūcī > sūī) / (changes.sc int-stop-loss.kgcjtdy)
[OAw] V m V > $1 v $3+nas / _ #                   ; @m-v — in the final syllable the nasality of -m- goes onto the final vowel (daśamá > dasavaṁ/dasauṁ, dāḍima > dārivaṁ); round 6
[OAw] ; ── block nasals-m-nasal-v: nasals-m-nasal-v — intervocalic m > nasal vowel + v, Hindi belt (H, Aw, Brj, SSr, NDu), nasals theme, blocks round 2026-09-30 (pass 2). The m > ṽ of nasals-m-v  [@m-v — nā́man > nā̃v, gōdʰū́ma > gohū̃ (blocks/nasals-m-nasal-v.sc)] ──
[OAw] V m > $1+nas v / _ V                         ; @m-v — 6 lects / nā́man > H nā̃v, gōdʰū́ma > H gohū̃ / (changes.sc m-v.vm.prevocalic.2)
[BH Hajong B H Aw S P kcch L khet OG WK Marw G OMarw] i n > ī / _ #                               ; @morph — in-stems cited in the nom. -ī (hastin > hattʰī)
[OP] a a {m ṁ} > u / _ #                        ; @hiatus @final-nasal — neuter -akaṁ > -aũ > -u: the Old Punjabi verbal noun -aṇu from -anakam (útkʰanati > ukkʰaṇaṇu, katʰáyati > kahaṇu; cf. Kutchi -ṇū; round 7)
[Khol] s > ∅ / V _ Stop                           ; @cluster-simplify — s lost before a stop, with no compensatory aspiration (hásta > hat, masta > mato; round 6 second pass)
[northwest -D -khet] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here [@hiatus — blocks/hiatus-medial-core.sc; medial a(y)a from stop loss contracts to e (as in OIA-H.sc)] ──
[northwest -D -khet] a a > e / C _ C                              ; @hiatus — medial a(C)a > e (kadala > kaala > Bhoj kerā, kadalī > kayalī > kelā) / (block hiatus-medial-core)
[OP thar P kcch L awan] a ā > ā                                      ; @hiatus — a-ā hiatus > ā (kumbʰakāra > kumʰār, andʰakāra > anʰār) / (block hiatus-medial-core)
[awan] ; ── block hiatus-diphthong: Hiatus theme — a + i/u before a consonant (blocks round, 2026-09-30). The secondary diphthongs left by consonant loss monophthongise before a consonant: a+i > e [@hiatus — blocks/hiatus-diphthong.sc (spread)] ──
[awan] a {u ū} > o / _ C                            ; @hiatus — a+u > o (hastakūṭa > hatʰaüḍa > hathoḍo) / (block hiatus-diphthong)
[Bhoj Kathoriya Mth Ku Garh Dotyali eastern -AdivasiOriya -OB hindi -OAw Bhatri hal northwest -D -S -Khol WK Goj west-pahari] ; ── block liquids-flap: Retroflex flapping — single post-vocalic ḍ ḍʰ > ṛ ṛʰ (NIA; liquids blocks round, 2026-09-30). The flap of the Hindi belt, Punjabi-Lahnda-Sindhi, Pahari, Bengali [@flap] ──
[Bhoj Kathoriya Mth Darai Ku Garh Dotyali eastern -AdivasiOriya -OB hindi -OAw Bhatri hal northwest -D -S -Khol WK Goj west-pahari] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ {ḍ ḍʰ}            ; @flap — 8 lects / kaṭāha > H kaṛāhā, kukkuṭá > H kukṛā, *dʰūḍi > H dʰūṛ; geminate ḍḍ stays a stop / (changes.sc flap.ddh.postvocalic.except)
[Vasavi] if:ext a a {m ṁ} > õ / _ #                ; @morph — round 4b: neuter -akam > -aaṁ > -õ (before final-consonant loss)
[D] ; ── block hiatus-vh-loss: Hiatus theme — loss of intervocalic v and h (blocks round, 2026-09-30). The NIA loss of the weak intervocalic continuants that creates a second round of hiatus: [@v-loss @h-loss — blocks/hiatus-vh-loss.sc; intervocalic h (< aspirates) lost] ──
[D] ; ── block hiatus-v-loss: Hiatus theme — loss of intervocalic v (blocks round, 2026-09-30). The NIA loss of intervocalic v (incl. < MIA p) that opens a second round of hiatus (dīpa > dīv ──
[D] v > ∅ / V _ V                                ; @v-loss — intervocalic v (incl. < p) lost (dīpa > dīva > diyā, kūpa > kuā, kapila > kailo) / (block hiatus-v-loss)
[D] h > ∅ / V _ V                                ; @h-loss — intervocalic h (< MIA aspirates) lost (lṓhita > làu, gōdʰūma > gɔū̃) / (block hiatus-vh-loss)
[D] {ḍ b} > ∅ / V _ V                           ; @int-stop-loss — also ḍ and b lost between vowels (as in Ushojo/Shina)
[D] a a {m ṁ} > o / _ #                        ; @hiatus @final-nasal — -akaṁ > -o (tāraká > tāro, *dādda > dādo; round 7)
[D] {a ā} o > o / _ #                            ; @hiatus — a vowel before the new final -o is absorbed (tāraká+akaṁ > *tārao > tāro, mádʰu+akaṁ > mo; round 7)
[Khol] {s ḣ t d m n} > ∅ / _ #                      ; @final-cons-loss — 30 lects / only OIA-type finals; CDIAL roots (*cul, *tʰar) keep theirs / (changes.sc final-cons-loss.shtdmn)
[P khet] ; ── block finals-schwa-shortening: Punjabi-Lahnda reduction of final -ā (blocks theme "finals", 2026-09-30): the Apabhraṁśa shortening of final -ā after a consonant gives a reduced ə, which the l [final -ā > -ə (blocks/finals-schwa-shortening.sc) — replaces `use final-a-short.a.c`: Apabhraṁśa shortening of final -ā to a reduced vowel (written ə here so the -ā citation rule below does not re-lengthen it; lost with the other final short vowels: jihvā > jībʰ)] ──
[P poth L khet awan] ā > ə / C _ #                                ; @final-a-short @final-vowel-loss — 4 lects / final -ā > -ə after a consonant (P. aṣṭā́ > aṭṭʰ, ā́rā > ār, kantʰā > kandʰ) / (changes.sc final-a-short.a.c)
[L] {t d n s m ḥ} > ∅ / _ #                      ; @final-cons-loss — 2 lects / only OIA-type finals; CDIAL roots (*cul, *tʰar) keep theirs / (changes.sc final-cons-loss.tdnsmh)
[Kho Kal Dm Tir] Sib {kʰ tʰ pʰ ṭʰ} > $2 / # _                 ; @initial-degem — 2 lects / s + aspirate > aspirate (stʰā́na > tʰan), before the prothesis / (changes.sc initial-degem.sibkhthpht)
[dardic-south -Dm -Tor -Pas] {d dʰ j} y > z                               ; @palatal-Cy — 5 lects / adyá > ā̌z / (changes.sc palatal-Cy.ddhjy)
[Gowro Chil bhatr] # ── Kohistani post-MIA prefix: first 3 of 3 shared changes (round-7 unification) ──
[L] ; ── block nasals-anusvara-loss: nasals-anusvara-loss — leftover anusvāra lost without nasality, nasals theme, blocks round 2026-09-30 (pass 2). The anusvāra that survives to this point (before [@anusvara-loss — remaining anusvāra (before s h v, final) unwritten in most NW reflexes (vaṁśa > P vañjʰ, triṁśat > tīh) (blocks/nasals-anusvara-loss.sc)] ──
[L] ṁ > ∅                                        ; @anusvara-loss @anusvara-vowel — 53 lects / pāṁśu > awan pā, māṁsá > Si mas, áṁsa > Si as / (changes.sc anusvara-loss)
[poth awan] ; ── block finals-schwa-shortening-lahnda: Lahnda/Hindko reduction of final -ā and -ī (blocks theme "finals", 2026-09-30): final -ā and (Hindko) -ī after a consonant reduce to ə, later lost by finals-apo [final -ā -ī > -ə (blocks/finals-schwa-shortening-lahnda.sc) — replaces `{ā ī} > ə / C _ #`: Apabhraṁśa shortening of final -ā and (Hindko) -ī to a reduced vowel, lost with the final short vowels (dʰā́rā > dʰār, vallī > val, bʰaginī > bʰε̄̀ṇ, garbʰíṇī > gabbʰuṇ; round 2)] ──
[poth awan] {ā ī} > ə / C _ #                            ; @final-vowel-loss @final-a-short — final -ā -ī > -ə (L. avēlā > avel, aṣṭā́ > aṭṭʰ, kalā́ > kal) / (block finals-schwa-shortening-lahnda)
[poth awan] ; ── block finals-schwa-shortening: Punjabi-Lahnda reduction of final -ā (blocks theme "finals", 2026-09-30): the Apabhraṁśa shortening of final -ā after a consonant gives a reduced ə, which the l ──
[northwest -OP -srk] ; ── block nasals-anusvara-nw: nasals-anusvara-nw — anusvāra before s/h and leftover anusvāra, north-western type (Lahnda, Kutchi, Khetrani, Pothwari), nasals theme, blocks round 2026-09-30.  [@anusvara-vowel @anusvara-loss — ṁ before s/h > length, other ṁ lost (blocks/nasals-anusvara-nw.sc)] ──
[northwest -OP -srk] V ṁ > $1+long / _ {s h}                      ; @anusvara-vowel — 6 lects / viṁśatí > khet vīh, poth bī / (changes.sc anusvara-vowel.vm.sh)
[northwest -OP -srk] ; ── block nasals-anusvara-loss: nasals-anusvara-loss — leftover anusvāra lost without nasality, nasals theme, blocks round 2026-09-30 (pass 2). The anusvāra that survives to this point (before [@anusvara-loss — leftover anusvāra dropped: pāṁśu > awan pā] ──
[northwest -OP -Khol -srk] ṁ > ∅                                        ; @anusvara-loss @anusvara-vowel — 53 lects / pāṁśu > awan pā, māṁsá > Si mas, áṁsa > Si as / (changes.sc anusvara-loss)
[poth] ; a a {i ē} > i / _ #                         ; @morph @hiatus — -ayati verbs likewise cited as bare stems   [round 7: @morph removed]
[poth] ; a {i ē} > i / _ #                           ; @morph @hiatus — verbs cited as bare stems (Lothers mar, kʰā): -ati > -i, lost with the final vowels   [round 7: @morph removed]
[OG G OMarw] NDStop r > $1-asp $1                         ; @assim-sonorant — 2 lects / but t + r is kept (Western tr retention) (kṣētra > kʰetar, āmantraṇa > āmãtraṇ) / (changes.sc assim-sonorant.ndstopr)
[OG] ; ── block cr-release: End of the cluster window: the onset protected by cr-keep-optional / r-anticipation-optional becomes stop + r again, and later rules (flaps, tr > ṭr, …) see an  [end of the cluster window (blocks/cr-release.sc)] ──
[OG] {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} > {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} r ; @cr-keep — protected stop + r released after the cluster stage / (block cr-release)
[OG G OMarw] y y > j j                                    ; @palatal-Cy — 3 lects / MIA -yy- (< ry, yy) > jj (kārya > kajja > kāj) / (changes.sc palatal-Cy.yy)
[kiuth] # if:verb a a {i ē} > ṇ u / _ #                   ; @morph — -ayati verbs cited as -ṇu infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[kiuth] # if:verb a {i ē} > ṇ u / _ #                       ; @morph — -ati verbs cited as -ṇu infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[kiuth] # {ā ō ī} {i ē} > $1 ṇ u / _ #                ; @morph @hiatus — athematic/long-vowel verbs (-āti, -ōti, -ītē) cited as -ṇu infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[kiuth cur kul] ; ── block finals-apocope-i: Early loss of final short -i (blocks theme "finals", 2026-09-30): in the inner West Pahari lects final -i is lost before the other final vowels (Kiunthali, Kulu [early loss of final -i (blocks/finals-apocope-i.sc) — replaces `use final-vowel-loss.i`: final short i lost (round 2)] ──
[kiuth cur kul] i > ∅ / _ #                                  ; @final-vowel-loss — 3 lects / final short -i lost (kiuth parā́ri > parār, rāṭi > rāṛ) / (changes.sc final-vowel-loss.i)
[jaun] # if:verb a {i ē} > ṇ ō̃ / _ #                       ; @morph — -ati verbs cited as -ṇō̃ infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[ktg] # ā v a a {i ē} > ā u ṇ õ / _ #          ; @morph — causative -āpayati cited as -āuṇõ  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[ktg] # if:verb a a {i ē} > ṇ õ / _ #                   ; @morph — -ayati verbs cited as -ṇõ infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[ktg] a {i ē} > ṇ õ / _ #                       ; @morph @hiatus — -ati verbs cited as -ṇõ infinitives
[ktg] # {ā ō ī} {i ē} > $1 ṇ õ / _ #                ; @morph @hiatus — athematic/long-vowel verbs (-āti, -ōti, -ītē) cited as -ṇõ infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[Chitwan] ; ── block nasals-m-v: nasals-m-v — MIA/NIA intervocalic m > v (via nasalised ṽ, nasality unwritten), nasals theme, blocks round 2026-09-30 (pass 2). Western and north-western, West P [@m-v — kamala > kavaḷ (blocks/nasals-m-v.sc)] ──
[Chitwan] m > v / V _ V                                ; @m-v — 33 lects / grāma > gāṽ/gāv, kamala > M kavaḷ / (changes.sc m-v.m.intervocalic)
[M] u v a > ō / # _                             ; @aya-e — upa- > o-
[M] {i u} > {ī ū} / _ C #                        ; @vowel-shift — i u lengthened in final closed syllables before final-C loss (hastín > hātʰī)
[Kal] {bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ} / V _         ; @deaspiration — 2 lects / only non-initial voiced aspirates (initial bʰ dʰ gʰ survive) / (changes.sc deaspiration.bhdhghjhdh.postvocalic)
[Kal] k > ∅ / {i ī u ū} _ {a ā} #                 ; @int-stop-loss @aka — k of the -ika/-ikā, -uka suffixes is lost (→ -i, -u), unlike stem k (round 4)
[Kal] optional: k > ∅ / a _ a #                   ; @int-stop-loss @aka — sporadically also the k of -aka (→ -a): Kal. has both -a and -ak nouns (átra > atra, amítra > amitra, anāhāra > anāra beside aṭṭāla > aṭalak, utpātaka > upalak; gold -a 189 vs -ak 66). gates ext −0.0175/−0.0137, verb −0.0099/−0.0047; same-shape placebos r/l > ∅ −0.0003/−0.0011 and −0.0005/−0.0008 (hill-climb round 4)
[Sv Kalk Kal] {g t d c j y v h b} > ∅ / V _ V              ; @h-loss @int-stop-loss @v-loss — 2 lects / intervocalic k survives / (changes.sc h-loss.gtdcjyvhb.intervocalic)
[Phal] ; ── block lar-deaspiration-voiced-medial: Non-initial voiced aspirates lose aspiration (initial ones survive, with tone/breathiness). Rajasthani (Bagri, Hadothi, Mewari, Dholpuri) and West Pahari (Bhale [@deaspiration — non-initial voiced aspirates (was postvocalic only) (blocks/lar-deaspiration-voiced-medial.sc)] ──
[Phal] {gʰ jʰ ḍʰ dʰ bʰ} > {g j ḍ d b} / !# _        ; @deaspiration @tone — 7 lects / andʰakāra > andāro, garbʰa > gābu / (changes.sc deaspiration.ghjhdhdhbh.cond)
[eur] p > v / V _ V                                ; @int-p-v — 119 lects / (changes.sc int-p-v)
[hal hadothi_kelwada Rj] {k g c j t d y} > ∅ / V _ V                  ; @int-stop-loss — 105 lects / (changes.sc int-stop-loss.kgcjtdy)
[khet rajasthani-gujarati -OG -WK -G -OMarw] ; ── block hiatus-ava-u: Hiatus theme — ava/uva > o (blocks round, 2026-09-30). ava > o (`hiatus-ava`) and, in Rajasthani and its neighbours, also uva (incl. < upa) > o: the `{a u} v a  [@aya-e — blocks/hiatus-ava-u.sc; ava/apa/upa > o in any position (nava > nau/no, dʰavala > ḍɔḷɔ)] ──
[khet rajasthani-gujarati -OG -WK -G -OMarw] ; ── block hiatus-ava: Hiatus theme — ava > o (blocks round, 2026-09-30). MIA contraction of a + v + a to o (ava incl. < apa where p > v precedes it). The lects wrote it as `a v a > o ──
[khet rajasthani-gujarati -OG -WK -G -OMarw] u v a > o                                    ; @aya-e — uva/upa > o (*dʰauvati > Marw dʰoṇo; ava/apa/upa > o in any position) / (block hiatus-ava-u)
[bhad khash] # if:verb a a {i ē} > ṇ ū / _ #                   ; @morph — -ayati verbs cited as -ṇū infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[bhad] a {i ē} > ṇ ū / _ #                       ; @morph @hiatus — -ati verbs cited as -ṇū infinitives
[bhad] # {ā ō ī} {i ē} > $1 ṇ ū / _ #                ; @morph @hiatus — athematic/long-vowel verbs (-āti, -ōti, -ītē) cited as -ṇū infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[kc] # if:verb ā v a a {i ē} > ā u ṇ o / _ #          ; @morph — causative -āpayati cited as -āuṇo  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[kc] # a {i ē} > ṇ o / _ #                       ; @morph @hiatus — -ati verbs cited as -ṇo infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[kc] # {ā ō ī} {i ē} > $1 ṇ o / _ #                ; @morph @hiatus — athematic/long-vowel verbs (-āti, -ōti, -ītē) cited as -ṇo infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[dod Gowro Chil bhatr Mai] ; ── block lar-dardic-affrication: Dardic affrication of the palatal stops: c cʰ j > ʦ ʦʰ z (Kashmiri, Kohistani, Kunar). Included by lar-dardic-laryngeals where the deaspiration follows directly [@affrication — c cʰ j > ʦ ʦʰ z (blocks/lar-dardic-affrication.sc)] ──
[dod Gowro Gaw Chil bhatr Mai Tir] {c cʰ j} > {ʦ ʦʰ z}                          ; @affrication — 10 lects / cáturdaśa > Chil. ʦondʌś, jihvā́ > zīb, cʰāyā́ > Gaw. ʦʰā́ilat / (changes.sc affrication.cchj)
[Mai] jʰ > z h                                    ; @affrication — the aspirate of c j > ʦ z: Zoller writes OIA jʰ as zh (*jʰal > zhǎl, *jʰakk > zhǎkʰ, *jʰāma > zhâṽ; 23 of 23 jʰ pairs, gold jʰ 0); the h then gets the depressor tone like other breathy onsets (hill-climb round 7)
[Gowro] ; ── block v-b-skip-optional: Sporadic escape of initial v- from v- > b- (2026-10-01): some words keep v- where the lect otherwise has b- (P viśvāsa > vasāh beside vāla > bāl). Mechanism: an [sporadic (blocks/v-b-skip-optional.sc)] ──
[Gowro] optional: v > ʋ / # _                        ; @v-b — sporadic: initial v- escapes v- > b- (viśvāsa > P vasāh) / (block v-b-skip-optional)
[Gaw Shum] {k t p kʰ tʰ pʰ} r > ɬ                       ; @tr-lateral — 2 lects / voiceless Stop + r > ɬ (gā́tra > gaɬ, putrá > puɬ) / (changes.sc tr-lateral.ktpkhthphr)
[Shum Wot] y > j / # _                                  ; @y-j — 101 lects / (changes.sc y-j.initial)
[Gaw Shum Wot Dm] y > ź / # _                                  ; @y-j — 2 lects / y- > ź / (changes.sc y-j.initial.2)
[Gaw Shum Wot Chil Tir] ; ── block lar-dardic-laryngeals: Dardic (Kohistani, Kunar) laryngeal/affricate stage: the palatal stops become dental affricates, then the voiced aspirates lose aspiration. The affrication prec [@affrication @deaspiration — c cʰ j > ʦ ʦʰ z, then voiced-aspirate deaspiration (blocks/lar-dardic-laryngeals.sc)] ──
[Gaw Shum Wot Chil Tir] ; ── block lar-dardic-affrication: Dardic affrication of the palatal stops: c cʰ j > ʦ ʦʰ z (Kashmiri, Kohistani, Kunar). Included by lar-dardic-laryngeals where the deaspiration follows directly [@affrication — c cʰ j > ʦ ʦʰ z (candrá > ʦan, jantú > zān)] ──
[Shum Wot Chil] {c cʰ j} > {ʦ ʦʰ z}                          ; @affrication — 10 lects / cáturdaśa > Chil. ʦondʌś, jihvā́ > zīb, cʰāyā́ > Gaw. ʦʰā́ilat / (changes.sc affrication.cchj)
[Gaw Shum Wot Chil Tir] ; ── block lar-deaspiration-voiced: Loss of aspiration in the voiced aspirates (bʰ dʰ gʰ jʰ ḍʰ > b d g j ḍ), everywhere in the word. Shared across families: Dardic (Shinaic, Kohistani, Kunar, Kash [@deaspiration — voiced aspirates; before intervocalic loss (dʰ patterns with d: gʰr̩tá > Gaw. gʌd)] ──
[Chil] {bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ}               ; @deaspiration @tone — 26 lects / dʰūma > Sh. dūm, bʰūmi > K. bum, gʰr̩tá > Gaw. gʌd / (changes.sc deaspiration.bhdhghjhdh)
[Gowro] ; ── block lar-v-b: Initial v- > b- (NIA): Hindi, Bihari, Eastern, Pahari, Rajasthani, Punjabi, Lahnda, Sindhi, Dardic, Romani … the most widespread NIA consonant change. Geminate  [@v-b — initial v- > b- (blocks/lar-v-b.sc)] ──
[Gowro] v > b / # _                                  ; @v-b — 88 lects / vadʰū́ > H. bahū, vā́la > H. bāl / (changes.sc v-b.v.initial)
[Gowro Chil bhatr] # ── end of shared prefix ──
[Bshk Shum Wot Tor bhatr cur] a v a > ō                                    ; @aya-e — 17 lects / ava > o / (changes.sc aya-e.ava.2)
[bhal mand cur pan] # ── W. Pahari post-MIA prefix: first 2 of 5 shared changes (round-7 unification) ──
[RomBalk] ; ── block finals-apocope-a: NIA apocope of -a only (blocks theme "finals", 2026-09-30): final short -a is lost while final -i -u survive (Maithili, eastern Tharu, Bishnupriya, Desia; Balka [apocope of -a only (blocks/finals-apocope-a.sc) — replaces `a > ∅ / C _ #`: bare final -a after a consonant is lost (kárman > kam, kaṇṭa > xand, kṣīrá > kir), after the loss of intervocalic k and before -aa contracts, so the -aka derivatives keep -a > -ʌ (taptá-ka > tátʌ, kāla-ka > xálʌ); round 7] ──
[RomBalk] a > ∅ / _ #                                  ; @final-vowel-loss — 20 lects / final -a lost, -i -u kept (Mth. aŋkurá > ā̃kur, adyá > āj, áŋga > ā̃g) / (changes.sc final-vowel-loss.a)
[eur RomBalk Zarg] ; ── block lar-a-rounding-v: Romani a > o before v (< p): Balkan and European Romani. Laryngeals round, pass 2 (2026-09-30). Edit here, not in the lects. [@a-rounding — a > o before v (blocks/lar-a-rounding-v.sc)] ──
[eur RomBalk Zarg] a > o / _ v                                  ; @a-rounding — 2 lects / svapiti > sov, yava > jov, śapatʰa > sovlax / (changes.sc a-rounding.v)
[eur] y > v / a _ a                                ; @glide — -aya- > -ava- (y hardened to v between a's): cāláyati > calavel, dārayati > darav, mālayati > malav
[Lom] VAsp > h / V _ V                             ; @int-aspirate-h — intervocalic voiced aspirates > h (gōdʰūma > gihu, madʰu > mahl)
[Lom Khol] ; ── block lar-asp-devoicing: Voiced aspirates devoice everywhere (bʰ dʰ gʰ jʰ ḍʰ > pʰ tʰ kʰ cʰ ṭʰ): Lomavren, Kholosi. Laryngeals round, pass 2 (2026-09-30). Edit here, not in the lects. [@asp-devoicing — voiced aspirates devoice everywhere (blocks/lar-asp-devoicing.sc)] ──
[Lom Khol] {gʰ jʰ ḍʰ dʰ bʰ} > {kʰ cʰ ṭʰ tʰ pʰ}          ; @asp-devoicing — 2 lects / randʰ- > Lom. ərantʰ, bʰaginī > Khol. pʰen, duhitr̩ > tʰīv / (changes.sc asp-devoicing.ghjhdhdhbh)
[Lom] d > l / # _ V                                ; @x-cep-lom-d-l — initial d > l (dēśa > les, dēvatā > leval, dakṣ- > lakʰ)
[Lom] s Stop > $2+asp                              ; @s-stop-aspirate — st > tʰ (hasta > atʰ, nāsti > natʰ; Romani vast)
[Ash Wg PNur] s {tʰ pʰ kʰ k} > $2 / # _                    ; @cluster-simplify @initial-degem — 3 lects / round 6: initial s is lost before an aspirate and before k (stʰāpáyati > Wg tā, stʰā́na > Ash tana, stʰālī́ > tāl, stʰīyátē > ti, skandʰá > kana); plain st- keeps it (stambʰa > śtɔ̌ma, strī́ > istri). Kt/Kam reject it on test (stʰála > śtal, stʰū́ṇā > śtiu) / (changes.sc initial-degem.sthphkhk)
[nuristani -Ash -Pr] s > ś / _ C                                  ; @nur-innovation @x-nur-s-sh — 5 lects / s > š before consonants, early (before the r-assimilation and lenition; −0.0007 to −0.004 vs. a late position): stēná > *śtana, stambʰa > *śtamb, prasupti > Gmb pruśt / (changes.sc nur-innovation.s.preconsonantal)
[nuristani -Gmb -Pr] r {t tʰ d dʰ} > {ṭ ṭ ḍ ḍ}@2 {ṭ ṭ ḍ ḍ}@2      ; @assim-r — 5 lects / r + dental → retroflex geminate (so it escapes flapping): mūrtá > *muṭṭa, kartana > *kaṭala, árdʰa > *āṭā / (changes.sc assim-r.rtthddh)
[nuristani -Pr] r {ś ṣ} > ṣ                                  ; @assim-r — 6 lects / párśu > *pāṣu, varṣá > *vaṣ / (changes.sc assim-r.rss)
[nuristani] ṇ > ∅ / r _                                  ; @assim-r @cluster-simplify — 7 lects / kárṇa > *kār, parṇá > *pār / (changes.sc assim-r.n)
[Pr] {d dʰ} > l // {n ṇ d} _ | _ {d dʰ}          ; @nur-innovation @x-nur-pr-d-l — Pr: *d, *dʰ > l before intervocalic lenition (mádʰu > mū́lu, ádʰi > āli, dva > lɨ, dánta > letəm). Round 6: not after a nasal and not in the geminate dd(ʰ), which stay d (gandʰá > gúndu, índra > indr, buddʰi > bɨdɨ, yuddʰá > ɨd, uddʰāna > udosik)
[nuristani] ; ── Aspirates ─────────────────────────────────────────────────────────────
[Pas Bhatri OSi nuristani] ; ── block lar-deaspiration-all: Loss of aspiration in every aspirate: the voiced aspirates (the shared lar-deaspiration-voiced), then the voiceless ones (kʰ cʰ ṭʰ tʰ pʰ > k c ṭ t p). Insular ( [@deaspiration — all aspirates, voiced and voiceless (blocks/lar-deaspiration-all.sc)] ──
[Pas Bhatri OSi nuristani] ; ── block lar-deaspiration-voiced: Loss of aspiration in the voiced aspirates (bʰ dʰ gʰ jʰ ḍʰ > b d g j ḍ), everywhere in the word. Shared across families: Dardic (Shinaic, Kohistani, Kunar, Kash [@deaspiration — voiced aspirates (bʰāga > Si. bā, gʰr̩tá > Si. giya)] ──
[Bhatri Wg Gmb] {bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ}               ; @deaspiration @tone — 26 lects / dʰūma > Sh. dūm, bʰūmi > K. bum, gʰr̩tá > Gaw. gʌd / (changes.sc deaspiration.bhdhghjhdh)
[Pas Bhatri Si Md OSi nuristani] {kʰ cʰ ṭʰ tʰ pʰ} > {k c ṭ t p}               ; @deaspiration — 2 lects / voiceless aspirates (hásta > Si. at-a, kʰaḍga > Si. kaga) / (changes.sc deaspiration.khchththph)
[dardic-north -Sh -Kho -Phal -K -Kal] ; ── block finals-aka-dardic: The -aka ending in Shina and its neighbours (blocks theme "finals", 2026-09-30): the masculine -akaḥ contracts to -o, early in the chain (before the Dardic apoc [-aka > -o (blocks/finals-aka-dardic.sc) — replaces `use aka.aakaa.final.2`: masc. -aka stems in -o, as in Shina (round 4); retagged as a sound change in round 7 (-aka > -ao > -o), so it now also gives the -anaka verbal noun -ano (Sv dádāti > dāno, kʰā́dati > kʰāno)] ──
[dardic-north -Sh -Kho -Phal -K -Kal] {a á} k {a á} > o / _ #                      ; @aka @hiatus — 3 lects / -aka > -o (Sh. tāraká > tāro, kulaka > kulo, tōlaka > tōlo) / (changes.sc aka.aakaa.final)
[Gmb] ; ── block liquids-flap-td: Dardic-Nuristani intervocalic flapping — single intervocalic ṭ ḍ > ṛ (liquids blocks round, 2026-09-30). Without the Prakrit voicing of ṭ to ḍ: both retroflex s [inherited from the Nuristani sisters (small-lect default, 2026-09-30)] ──
[Gmb] {ṭ ḍ} > ṛ / V _ V                            ; @flap @int-retroflex — 19 lects / maṭʰa > *maṛa, śāṭa > *śaṛa, *aŋgūḍi > Sh háguṛi / (changes.sc flap.td.intervocalic)
[nuristani] ; ── Lenition ─────────────────────────────────────────────────────────────
[dardic-north -Kho -Ush -Kal dardic-south -Dm nuristani] ; ── block liquids-flap-td: Dardic-Nuristani intervocalic flapping — single intervocalic ṭ ḍ > ṛ (liquids blocks round, 2026-09-30). Without the Prakrit voicing of ṭ to ḍ: both retroflex s [@int-retroflex @flap — kukkuṭá > kukur] ──
[dardic-north -Kho -Ush -Kal dardic-south -Dm nuristani] {ṭ ḍ} > ṛ / V _ V                            ; @flap @int-retroflex — 19 lects / maṭʰa > *maṛa, śāṭa > *śaṛa, *aŋgūḍi > Sh háguṛi / (changes.sc flap.td.intervocalic)
[Sh] {k g t d c j y v h} > ∅ / V _ V             ; @int-stop-loss @v-loss @h-loss
[Sh] V > $1+stress / _ C* Short C* #              ; @accent-placement — stress on the penult (before hiatus contraction and final-vowel loss) when the final vowel is short …
[Sh] Long > $1+stress / _ C* #                    ; @accent-placement — … else on a long final vowel (round 3)
[Kalk pog K Bshk Pr] ; ── block liquids-flap-rhotic: Flap-rhotic merger — ṛ ṛʰ > r rʰ (liquids blocks round, 2026-09-30). Stage 2 of the flap history: the retroflex flap (blocks/liquids-flap.sc, or the Dardic inte [@flap — the flap merges with r] ──
[Pas] {ṭ ḍ} > ṛ / V _ #                           ; @flap — word-final postvocalic ṭ ḍ flapped too (root-final in verb stems); round 5
[Dm] s > ś / _ t                                 ; @palatalisation — masta > maśtak, strī́ > iśtrī
[Dm] j > ź                                       ; @affrication — jīvantá > źãta
[Dm] {ṭ ḍ} > r̆ / V _ V                          ; @int-retroflex @flap — *kuḍa > kúr̆a (CDIAL r̆)
[dardic-south -Tor] ; ── block mia-dardic-stop-loss: Dardic loss of intervocalic t d and k g (blocks theme "mia", pass 2, 2026-09-30): the core of mia-dardic-lenition, included on its own by Kohistani/Kunar/Pashai [@int-stop-loss — intervocalic t d, then k g lost (blocks/mia-dardic-stop-loss.sc; blocks round 2026-09-30, pass 2)] ──
[dardic-south] {t d} > ∅ / V _ V                            ; @int-stop-loss — 10 lects / intervocalic t d lost (pitā > piā, nadī > naī, hr̥daya > hiaa) / (changes.sc int-stop-loss.td)
[dardic-south -Tor] {k g} > ∅ / V _ V                            ; @int-stop-loss — 11 lects / intervocalic k g lost (lōka > lōa, nagara > naar, sūkara > sūar) / (changes.sc int-stop-loss.kg)
[Bshk] u a > ū / _ #                                ; @aka @hiatus — 6 lects / -uka > -ua > -ū; round 4, extension-aware / (changes.sc aka.ua.final)
[Pas] u {a ā} > ū / _ #                            ; @hiatus — 3 lects / round 4: -uka > -ua > -ū / (changes.sc hiatus.uaa.final)
[Bshk] {p v y} > ∅ / V _ V                         ; @int-stop-loss @v-loss — intervocalic labials and glides lost
[Tor] k > ∅ / V _ V #                             ; @int-stop-loss — k lost in the final syllable (-aka, -ika, -uka) though voiced elsewhere; round 4, extension-aware
[Wot] {p h} > ∅ / V _ V                           ; @int-stop-loss @h-loss — intervocalic p and h lost
[Pas] ; ── block finals-aka-iu: The -ikā/-ika and -uka endings (blocks theme "finals", 2026-09-30): after intervocalic k is lost, -ia and -ua contract to long -ī and -ū, which survive apocope  [-ikā/-uka > -ī/-ū (blocks/finals-aka-iu.sc) — replaces `use aka.ua.final`: -uka > -ua > -ū; round 4, extension-aware] ──
[Pas] {i ī u ū} {a ā} > {ī ī ū ū}@1 / _ #          ; @aka @hiatus — 8 lects / -ikā/-uka > -ia/-ua > -ī/-ū (G. kárṇikā > kānī, kuñcikā > kũcī, kārpaṭika > kāpṛī) / (changes.sc aka.iiuuaa.final)
[Tor] {ṣ s} > ∅ / V _ C                           ; @cluster-simplify — postvocalic sibilant lost before a consonant, feeding intervocalic voicing (niṣkālayati > nigāl, tīkṣṇá > tīn); round 6
[Tor] {k p} > {g b} / V _ V                       ; @int-voicing — Torwali voices intervocalic k p (niṣkālayati > nigāl); dentals are lost instead
[Tir] ṣ ṭ > ṣ t                                   ; @deretroflexion — ṣṭ > ṣt
[Wot Dm Tor] ; ── block hiatus-vh-loss: Hiatus theme — loss of intervocalic v and h (blocks round, 2026-09-30). The NIA loss of the weak intervocalic continuants that creates a second round of hiatus: [@v-loss @h-loss — blocks/hiatus-vh-loss.sc; intervocalic h] ──
[Wot Dm Tor] ; ── block hiatus-v-loss: Hiatus theme — loss of intervocalic v (blocks round, 2026-09-30). The NIA loss of intervocalic v (incl. < MIA p) that opens a second round of hiatus (dīpa > dīv ──
[Wot Dm Tor] v > ∅ / V _ V                                ; @v-loss — intervocalic v (incl. < p) lost (dīpa > dīva > diyā, kūpa > kuā, kapila > kailo) / (block hiatus-v-loss)
[Wot Tor] h > ∅ / V _ V                                ; @h-loss — intervocalic h (< MIA aspirates) lost (lṓhita > làu, gōdʰūma > gɔū̃) / (block hiatus-vh-loss)
[Mai] t > ∅ / V _ {a ā}                           ; @int-stop-loss — intervocalic t lost before a-vowels (catúraḣ > ʦaur, cētáyati > ʦē̂r, -ati verbs > bare stems); round 6
[Chil] ; ── block hiatus-aya: Hiatus theme — aya > ē (blocks round, 2026-09-30). The early contraction of -aya- to ē, after OIA ai/au > ē/ō and before the MIA loss of intervocalic y: one cha [@aya-e — blocks/hiatus-aya.sc; aya > ē] ──
[Chil] ; ── block hiatus-aya-medial: Hiatus theme — aya > ē before a consonant (blocks round, 2026-09-30). The early contraction of -aya- to ē inside the word, after OIA ai/au > ē/ō and before the  ──
[Chil] a y a > ē / _ C                              ; @aya-e — medial aya > ē (nayana > nēṇa, śayana > sēṇa) / (block hiatus-aya-medial)
[Chil] a y a > ē                                    ; @aya-e — aya > ē, also word-finally (tráyaḣ > trē, vayám > bē, Kal krayá > kre) / (block hiatus-aya)
[eur RomWel RomBalk Zarg Lom Chil Mai] {t d} > l / V _ V                            ; @t-l — 7 lects / āgata > alo, yuvatī > źuli; before cluster simplification (mattá > mato, not *malo) / (changes.sc t-l)
[eur] a > e / _ l i #                             ; @umlaut — -ali (< -ati via t > l) > -eli > 3sg -el: i-umlaut before the lost final i (pʰerel, anel, kerel)
[eur] ; ── block nasals-m-v: nasals-m-v — MIA/NIA intervocalic m > v (via nasalised ṽ, nasality unwritten), nasals theme, blocks round 2026-09-30 (pass 2). Western and north-western, West P [@m-v — nāman > nav, dʰūma > tʰuv; before r C > C (gʰarma > kʰam, not *kʰav) (blocks/nasals-m-v.sc)] ──
[eur] m > v / V _ V                                ; @m-v — 33 lects / grāma > gāṽ/gāv, kamala > M kavaḷ / (changes.sc m-v.m.intervocalic)
[Gowro Chil bhatr Mai] {k g} > ∅ / V _ V                            ; @int-stop-loss — 11 lects / intervocalic velars lost / (changes.sc int-stop-loss.kg)
[Mai] n > ṽ / a _ a                               ; @x-dardic-south-n-nasal-v — intervocalic n between a-vowels > ṽ (the -anaka verbal noun > infinitive -aṽ: kālayati → kālǎṽ); round 5
[Mai] ; ── block hiatus-dardic: Hiatus theme — Kohistani / Kunar / Pashai hiatus (blocks round, 2026-09-30). The old intervocalic v and y are lost (before the new v < p), and the second vowel  [@v-loss @hiatus — blocks/hiatus-dardic.sc; old intervocalic v, y lost before p > v (the new v survives)] ──
[Mai] {v y} > ∅ / V _ V                            ; @v-loss — old intervocalic v, y lost (before p > v: the new v survives) / (block hiatus-dardic)
[Gaw Chil] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here [@hiatus — blocks/hiatus-medial-core.sc (spread)] ──
[Gaw Chil] a a > e / C _ C                              ; @hiatus — medial a(C)a > e (kadala > kaala > Bhoj kerā, kadalī > kayalī > kelā) / (block hiatus-medial-core)
[Gaw] a ā > ā                                      ; @hiatus — a-ā hiatus > ā (kumbʰakāra > kumʰār, andʰakāra > anʰār) / (block hiatus-medial-core)
[Gaw] ; ── block mia-dardic-stop-loss: Dardic loss of intervocalic t d and k g (blocks theme "mia", pass 2, 2026-09-30): the core of mia-dardic-lenition, included on its own by Kohistani/Kunar/Pashai [@int-stop-loss — intervocalic t d, then k g lost (blocks/mia-dardic-stop-loss.sc; blocks round 2026-09-30, pass 2)] ──
[Gaw] {t d} > ∅ / V _ V                            ; @int-stop-loss — 10 lects / intervocalic t d lost (pitā > piā, nadī > naī, hr̥daya > hiaa) / (changes.sc int-stop-loss.td)
[Gaw] {k g} > ∅ / V _ V                            ; @int-stop-loss — 11 lects / intervocalic k g lost (lōka > lōa, nagara > naar, sūkara > sūar) / (changes.sc int-stop-loss.kg)
[dardic-south -Gowro -Tor -Tir] {v y} > ∅ / V _ V                            ; @v-loss — 9 lects / intervocalic v, y lost / (changes.sc v-loss.vy.intervocalic)
[Dm] {v y h} > ∅ / V _ V                         ; @v-loss @h-loss — intervocalic v, y, h lost
[Kt Kam] {n ṇ} > r̆ / V _ V                           ; @nur-innovation @x-nur-n-flap — 2 lects / intervocalic n, ṇ > ř̃ (nasalised flap), before final-vowel loss (−0.002 in Kt if later): *tanī > Kt tæ̃r̆, kāṇá > Kam kār̆̃á, kṣipaṇí > Kam ʦ̣úr̆̃i / (changes.sc nur-innovation.nn.intervocalic)
[Kt] {p b} > v / V _ #                           ; @int-p-v — R6 fourth pass: word-final labial after the early apocope above > v
[RomWel Kal Dm Mai Pas nuristani -Gmb -Kam -PNur] p > v / V _ V                                ; @int-p-v — 119 lects / (changes.sc int-p-v)
[eur] r > ∅ / _ C                                  ; @assim-r @cluster-simplify — 16 lects / r + C > C (karṇa > kan, sarpa > sap) / (changes.sc assim-r.r.preconsonantal)
[as-ll] {k g c j} > ∅ / V _ V                        ; @int-stop-loss — Central-type loss of intervocalic velars/palatals in Domari
[as-ll] a > ∅ / C _ #                                ; @final-vowel-loss — bare final -a after a consonant is lost (kárman > kam, kaṇṭa > xand, kṣīrá > kir), after the loss of intervocalic k and before -aa contracts, so the -aka derivatives keep -a > -ʌ (taptá-ka > tátʌ, kāla-ka > xálʌ); round 7
[as-ll] a > ∅ / a _ #                               ; @hiatus — final -aa (from -aya-/-aka- after consonant loss) contracts before apocope; round 6
[as-ll Ush] ā > ∅ / {i u} _ #                            ; @aka @hiatus — 2 lects / -ikā > -iā > -i after the loss of intervocalic k (round 4; held back by the agent only because it also improves the accent score) / (changes.sc aka.a.iu)
[as-ll] {t d} > r / V _ V                            ; @t-l — intervocalic t d > r (Romani l); before cluster simplification
[as-ll] i > ∅ / a r _ #                             ; @final-vowel-loss — -ari (< -ati via t > r) > 3sg -ʌr: final i lost after -ar (kílʌr, dáuʌr); elsewhere -i stays
[eur as-ll RomBalk Zarg Lom] ; ── block mia-romani-r: Romani/Domari p > v and loss of preconsonantal r (blocks theme "mia", pass 2, 2026-09-30); precedes mia-romani-clusters in those chains. Examples are schematic  [@int-p-v @assim-r — p > v, preconsonantal r lost (blocks/mia-romani-r.sc; blocks round 2026-09-30, pass 2)] ──
[eur as-ll] p > v / V _ V                                ; @int-p-v — 119 lects / intervocalic p > v (dīpa > dīv-, kūpa > kūv-, tāpa > tāv-) / (changes.sc int-p-v)
[as-ll RomBalk] r > ∅ / _ C                                  ; @assim-r @cluster-simplify — 16 lects / r before a consonant lost (karma > kam-, sarpa > sap-, dīrgʰa > dīg-) / (changes.sc assim-r.r.preconsonantal)
[central-east-pahari-romani -Ku -Garh -N -Dotyali] {k g m} r > $1 / # _                         ; @initial-degem — 3 lects / kr mr > k m (krīṇāti > kin, mrakṣ- > makʰ-); tr dr pr kept / (changes.sc initial-degem.kgmr)
[as-ll] ; ── block liquids-flap-romani: Romani-Domari-Lomavren flap — ṭ ṭʰ ḍ > r after a vowel and word-finally (liquids blocks round pass 2, 2026-09-30). The shared "migratory" outcome of the retrofl [@flap — kīṭa > kir; in Domari BEFORE cluster simplification, so old ṭṭ stays a stop (pēṭ, kúrtʌ)] ──
[central-east-pahari-romani -Ku -Garh -N -Dotyali] ; ── block mia-romani-clusters: Romani/Domari cluster simplification (blocks theme "mia", 2026-09-30): stop + stop and stop + sonorant simplify without a geminate stage; s + v/y > s. Examples  [@cluster-simplify — stop + stop, stop + sonorant, s + v/y simplify (blocks/mia-romani-clusters.sc; blocks round 2026-09-30)] ──
[eur RomWel RomBalk Zarg Lom] Stop Stop > $2                               ; @cluster-simplify @degem-plain — 17 lects / stop + stop > the second (sapta > sat-, bʰakta > bʰat-, mudga > mug-) / (changes.sc degem-plain.stopstop)
[central-east-pahari-romani -Ku -Garh -N -Dotyali] Stop {y v l Nasal} > $1                      ; @cluster-simplify — 6 lects / stop + y v l nasal/s > stop (putra > put-, agni > ag-, pakva > pak-) / (changes.sc cluster-simplify.stopyvlnas)
[central-east-pahari-romani -Ku -Garh -N -Dotyali] {ś s} {v y} > $1                             ; @cluster-simplify — 4 lects / s + v/y > s (aśva > as-, tasya > tas-) / (changes.sc cluster-simplify.ssvy)
[eur RomWel RomBalk Zarg Lom] ; ── block liquids-flap-romani: Romani-Domari-Lomavren flap — ṭ ṭʰ ḍ > r after a vowel and word-finally (liquids blocks round pass 2, 2026-09-30). The shared "migratory" outcome of the retrofl [@flap — kīṭa > kir, vaḍra > baro] ──
[central-east-pahari-romani -Ku -Garh -N -Dotyali] {ṭ ṭʰ ḍ} > r / V _ | _ #                     ; @flap — 6 lects / kīṭá > eur kiri, akṣōṭa > Lom ankʰor, vaḍra > RomBalk baro / (changes.sc flap.tthd.v)
[RomBalk] G $1 > $2                                    ; @cluster-simplify @degem-plain — 83 lects / residual geminates (vaḍra > *barr > baro, culla > culo) / (changes.sc degem-plain.g)
[mand pan] ; ── block mia-sibilants-keep: MIA sibilant/palatal clusters in lects that keep ś (W. Pahari; blocks theme "mia", 2026-09-30): ṣ > ś instead of the mainland ś ṣ > s, then Cy, kṣ/ts/ps and sib [inherited from the W. Pahari sisters (small-lect default, 2026-09-30)] ──
[mand pan] ṣ > ś                                        ; @sibilant-keep @sibilant-ś — 17 lects / ṣ > ś, ś kept distinct from s (varṣa > variś-, śīrṣa > śiś, puṣpa > puśpa) / (changes.sc sibilant-keep)
[mand] {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1      ; @palatal-Cy — 104 lects / dental + y > palatal geminate (satya > sacca, madʰya > majjʰa, adya > ajja) / (changes.sc palatal-Cy.tthddhy)
[mand pan] {k t p} {ś s} > {k c c} {kʰ cʰ cʰ}@1         ; @ks @ts-cluster — 11 lects / kṣ (now kś) ts ps > kkʰ ccʰ (akṣi > akkʰi, vatsa > vaccʰa, kṣētra > kʰētta) / (changes.sc ks.ktpss)
[mand pan] {s ś} Stop > $2-asp $2+asp                   ; @s-stop-aspirate — 16 lects / ś/s + stop > aspirated geminate (hasta > hattʰa, puṣpa > puppʰa, aṣṭa > aṭṭʰa) / (changes.sc s-stop-aspirate.ssstop)
[kvar] a > ɔ / # (C) C _ C V | # _ C V // _ C ā C* #; @a-rounding — a > ɔ in an open initial syllable, not before ā in the final syllable (round 2; weight, not accent). Round 7: written ɔ, Jouanne's vowel for rounded a (pálāla > pɔraḷ, balivárda > bɔḷēd, jámbʰa > ʣɔbṛɔ); her o continues OIA ō (kōṣṭʰāgāra > koṭʰār, *gʰōraḍa > goṛēḍ). o in rasavatī > rosuigɔr, *tapaka > toi, vásati > bośo is lexical. Verb gate waived: the old verb score (vásati > bosṇo, 1 of 2 verb pairs) was an -anaka menu chance match on the final -o
[WK Goj] a m > ū̃ / _ #                               ; @aka @final-nasal — 2 lects / neuter -am > -ũ before final -m is lost (-akam > -aũ > -ū̃: gʰōṭakam > gʰoṛū̃) / (changes.sc aka.am.final)
[mewati_jhambaus] {a ā ē ō} {i ē} > ā / _ #                   ; @morph @hiatus — survey verbs cited in the past ptc. / "he did" form, masc. -ā (Western-Hindi type)
[rajasthani-gujarati -OG -G] {s t d n m r} > ∅ / _ #                      ; @final-cons-loss — 16 lects / inflectional finals (roots cited bare keep theirs) / (changes.sc final-cons-loss.stdnmr)
[G] {a u} v {a ā} > o / # _                     ; @aya-e — ava-/apa-/upa-/avā-/upā- > o- (avalipya > orpɔ, upādʰyāya > ojʰɔ)
[G] a m > ũ / _ #                             ; @final-nasal @aka — neuter -am > -ũ before final -m is lost (-akam > -aũ > -ũ: gʰōṭakam > gʰoṛũ)
[WK G] {s t n m ḣ} > ∅ / _ #                        ; @final-cons-loss — 2 lects / inflectional finals (roots cited bare keep theirs) / (changes.sc final-cons-loss.stnmh)
[kaithal] ; V m > $1+nas v / _ V                        ; @m-v — nā́man > nā̃v, gōdʰū́ma > gohū̃   [pruned for kaithal: LOO -0.0059]
[kaithal] ; i n > ī / _ #                               ; @morph — in-stems cited in nom. -ī (hastín > hātʰī)   [pruned for kaithal: never fires]
[hindi -OAw -SSr] C > ∅ / _ #                                  ; @final-cons-loss — 27 lects / (changes.sc final-cons-loss.c)
[hindi -kaithal -OAw -bagheli_lakshman] ; ── block hiatus-ava-hindi: Hiatus theme — ava/uva > o except word-finally (blocks round, 2026-09-30). The Hindi-area form of the ava > o contraction (`hiatus-ava`): Hindi, Braj, Awadhi, B [@aya-e — blocks/hiatus-ava-hindi.sc; ava/apa > o except word-finally (avatārayati > otārnā; náva > nau)] ──
[hindi -kaithal] a v a > o // _ #                             ; @aya-e — ava/apa > o except word-finally (avatārayati > otārnā; náva > nau) / (block hiatus-ava-hindi)
[hindi -kaithal -OAw -bagheli_lakshman] u v a > o // _ #                             ; @aya-e — uva/upa > o likewise (suvarṇa > sonā, upālabʰatē > olahnā, yuvatí > joī) / (block hiatus-ava-hindi)
[Aw] V G $2 > $1+long $3 / # C* _ V #             ; @degem-lengthen — 2 lects / first syllable of a disyllable only (hasta > hātʰ; longer words keep a short vowel); round 2 / (changes.sc degem-lengthen.vg.cv)
[kaithal] ; a v a > o                                   ; @aya-e — ava/apa > o (avatārayati > otārnā, aŋkapāli > ãkor)   [pruned for kaithal: LOO -0.0019]
[Garh] a v a > ɔ                                    ; @aya-e — MIA ava > ɔ in Garh. (not o as in N.): the same open vowel as the secondary au > ɔ (pavana > pɔṇ, návanīta > nɔṇi, dʰavalī > dʰɔḷi); round 7
[Ku Garh N Dotyali] ; ── block intervocalic-n-lenition: Intervocalic weakening, MIA: n > ṇ between vowels (allophonic retroflexion; merged back to n later in the Hindi–Bihari–Nepali lects) followed by the lenition of [Intervocalic weakening (blocks/intervocalic-n-lenition.sc)] ──
[Ku Garh N Dotyali] ; ── block nasals-retroflexion: nasals-retroflexion — MIA intervocalic n > ṇ (the Prakrit "no ṇaḥ" change), nasals theme, blocks round 2026-09-30. Stage 1 of n > ṇ > n: a single dental nasal b [@n-retroflex — MIA intervocalic n > ṇ (blocks/nasals-retroflexion.sc)] ──
[Ku Garh Dotyali] n > ṇ / V _ V                                ; @n-retroflex @nur-innovation @x-nur-n-flap — 40 lects / pānī́ya > G, P, S pāṇī, vána > P vaṇ, S vaṇu, Or baṇa / (changes.sc n-retroflex.intervocalic)
[Ku Garh N Dotyali] ; ── block mia-lenition: MIA intervocalic lenition (blocks theme "mia", 2026-09-30): the Prakrit weakening of single intervocalic stops, in the order the lects already had it. Not in mi [@int-aspirate-h @int-retroflex @int-p-v @int-stop-loss — MIA intervocalic lenition (blocks/mia-lenition.sc; blocks round 2026-09-30)] ──
[Ku Garh Dotyali] {kʰ gʰ tʰ dʰ pʰ bʰ} > h / V _ V              ; @int-aspirate-h — 90 lects / intervocalic aspirates > h (mukʰa > muha, bádʰira > bahira, lābʰa > lāha) / (changes.sc int-aspirate-h)
[Ku Garh N Dotyali] ; ── block mia-lenition-stops: MIA lenition of the unaspirated intervocalic stops (blocks theme "mia", pass 2, 2026-09-30): ṭ > ḍ, p > v, loss of k g c j t d y. Part of mia-lenition (after as [ṭ > ḍ, p > v, stop loss (split off in pass 2 for lects without aspirate > h)] ──
[Ku Garh Dotyali] {ṭ ṭʰ} > {ḍ ḍʰ} / V _ V                      ; @int-retroflex — 102 lects / intervocalic ṭ ṭʰ > ḍ ḍʰ (gʰaṭa > gʰaḍa, kaṭuka > kaḍua, paṭʰati > paḍʰai) / (changes.sc int-retroflex)
[Ku Garh N Dotyali] p > v / V _ V                                ; @int-p-v — 119 lects / intervocalic p > v (dīpa > dīva, kūpa > kūva, tāpa > tāva) / (changes.sc int-p-v)
[Ku Garh Dotyali] {k g c j t d y} > ∅ / V _ V                  ; @int-stop-loss — 105 lects / intervocalic k g c j t d y lost (lōka > lōa, nadī > naī, sūcī > sūī) / (changes.sc int-stop-loss.kgcjtdy)
[N] ; ── block finals-akam-pahari: Pahari neuter -akam (blocks theme "finals", 2026-09-30): the neuter/verbal-noun -aam (< -akam, -anakam) becomes -u, the Nepali and West Pahari infinitive/citati [-aam > -u (blocks/finals-akam-pahari.sc) — replaces `use final-nasal.aam`: neuter -aaṁ (< -akam) > -u: -anakam > -nu, the N. infinitive (rakʰnu, carnu)] ──
[N] a a m > u / _ #                              ; @final-nasal @hiatus — 5 lects / -aam (< -akam) > -u (N. áraṇa+akam > arnu, *aḍḍ+iyanakam > aṛinu, adyá+akam > āju) / (changes.sc final-nasal.aam)
[Ku Garh N Dotyali] {s t d m ḥ} > ∅ / _ #                        ; @final-cons-loss — 3 lects / only real OIA finals (starred roots like *cil keep theirs) / (changes.sc final-cons-loss.stdmh)
[Ku Garh Dotyali] n > ∅ / V _ #                                ; @final-cons-loss — 3 lects / the final -n of OIA an-/in-stems is lost as in the nominative (kárman > kām, ātmán > āp, gōsvāmin > guśɛ̃, *kōṭʰin > koṛī); round 7 / (changes.sc final-cons-loss.n.v)
[N] n > ∅ / a _ #                                ; @final-cons-loss — the final -n of OIA an-stems is lost as in the nominative (kárman > kām, nā́man > nāũ, rṓman > rãũ, mūrdʰán > muṛʰ); in-stems keep it until the later -in > -i; round 7
[eastern] C > ∅ / V (C) (C) V _ #                      ; @final-cons-loss — 11 lects / polysyllables only: CDIAL root etyma (*cul, *pʰaṭ) are bare citation stems, not words / (changes.sc final-cons-loss.c.vccv)
[bihari-tharu Ku Garh N Dotyali eastern hindi northwest -khet rajasthani-gujarati -OG -OMarw] ; ══ NIA ══════════════════════════════════════════════════════════════════
[P L] G $1 > $2 / _ {ā ī ū ē ō} | _ V C C          ; @degem-plain — 2 lects / pretonic: geminate simplifies before a heavy (stressed) syllable (*pēṭṭāra > piṭārī, *aṭṭakk- > aṭakṇā) / (changes.sc degem-plain.g.aiueovcc)
[P poth] G $1 > $2 / _ C                              ; @degem-plain — 4 lects / a geminate simplifies before a consonant (after syncope: *kʰiccaḍikā > kʰicṛī; round 6) / (changes.sc degem-plain.g.preconsonantal)
[OP D P L khet awan] {ā ī ū} > {a i u} / _ C (C) {ā ī ū ē ō} C    ; @pretonic-shortening — 3 lects / long vowel shortened before a stressed long vowel (ākʰyāna > akʰāṇ) / (changes.sc pretonic-shortening.aiu.ccaiueoc)
[P] ; if:verb ∅ > ṇ ā / C _ #                             ; @morph — consonant-final CDIAL roots (*dabb) are verbal: cited as -ṇā infinitives   [round 7: @morph removed]
[P] ; if:verb ā a a {i ē} > ā u ṇ ā / _ #                 ; @morph — causatives whose stem ends in ā once the consonant is lost take -āuṇā   [round 7: @morph removed]
[P] ; if:verb v a a {i ē} > u ṇ ā / _ #                   ; @morph — -āpayati causatives > -āuṇā (prāpayati > pāuṇā)   [round 7: @morph removed]
[P] ; if:verb a a {i ē} > ṇ ā / _ #                       ; @morph — -ayati > -ṇā (kalpayati > kappṇā)   [round 7: @morph removed]
[P] ; if:verb a {i ē} > ṇ ā / _ #                           ; @morph — -ati verbs cited as -ṇā infinitives   [round 7: @morph removed]
[P] ; if:!accented,!ext a > ā / C _ #                  ; @morph — masc. -ā citation forms (MIA -aka) — only for headwords without a Vedic accent: Vedic-attested a-stems mostly surface bare (round 2)
[L] ; if:verb ∅ > a ṇ / C _ #                             ; @morph — consonant-final CDIAL roots (*dabb) are verbal: cited as -aṇ infinitives   [round 7: @morph removed]
[L] ; if:verb ā a a {i ē} > ā v a ṇ / _ #                 ; @morph — causatives on long-vowel stems: -āvaṇ   [round 7: @morph removed]
[srk] ; if:verb ∅ > a ṇ / C _ #                             ; @morph — consonant-final CDIAL roots are verbal: cited as -aṇ infinitives   [round 7: @morph removed]
[S] ; if:verb ∅ > a ṇ u / C _ #                           ; @morph — consonant-final CDIAL roots (*dabb) are verbal: cited as -aṇu infinitives   [round 7: @morph removed]
[S] ; if:verb ā a a {i ē} > ā i ṇ u / _ #                 ; @morph — causatives on long-vowel stems keep -āiṇu (cʰādayati > cʰā̃iṇu, utpātayati > upāiṇu)   [round 7: @morph removed]
[S] ; if:verb v a a {i ē} > ā i ṇ u / _ #                 ; @morph — -āpayati causatives > -āiṇu   [round 7: @morph removed]
[S] ; if:verb a a {i ē} > a ṇ u / _ #                     ; @morph — -ayati > -aṇu   [round 7: @morph removed]
[S] ; if:verb a {i ē} > a ṇ u / _ #                         ; @morph — -ati verbs cited as -aṇu infinitives   [round 7: @morph removed]
[kcch] ; if:verb ∅ > ṇ ū / C _ #                             ; @morph — consonant-final CDIAL roots (*jʰal) are verbal: cited as -ṇū infinitives   [round 7: @morph removed]
[kcch] ; if:verb a {i ē} > ṇ ū / _ #                           ; @morph — infinitive -ṇū   [round 7: @morph removed]
[Khol] ∅ > e n / C _ #                             ; @morph — consonant-final CDIAL roots are verbal: cited as -en infinitives
[Khol] a a {i ē} > e n / _ #                       ; @morph @hiatus — infinitive -en
[Khol] ; a {i ē} > e n / _ #                         ; @morph @hiatus — infinitive -en (píbati > pīyen)   [round 7: @morph removed]
[mewari_dholpura marwari_bagra] ; ── block lar-s-h-prevocalic: Marwari/Gujarati-type debuccalisation of s before a vowel: intervocalic (the shared lar-s-h-intervocalic) and word-initial. Marwari of Bagra, Dholpuri, Wagdi-Ko [@s-h — s > h before a vowel, initially and intervocalically (blocks/lar-s-h-prevocalic.sc)] ──
[mewari_dholpura marwari_bagra] ; ── block lar-s-h-intervocalic: Intervocalic s > h: Lahnda, Punjabi, Sindhi, Dhivehi (late there); the Marwari-type lects add initial s- > h- (lar-s-h-prevocalic). Laryngeals round (2026-09-30 [@s-h — intervocalic (śaśa > hāho)] ──
[WK mewari_dholpura marwari_bagra] s > h / # _ V                                ; @s-h — initial (saptá > hāt, sūrya > hūraj, sūkara > huvar) / (block lar-s-h-prevocalic)
[G] ḍ > ṛ / V _ V                               ; @flap — single intervocalic ḍ (< ṭ) flaps before degemination, so ḍḍ > ḍ stays (ghaṭa > ghaṛɔ vs *aḍḍ > aḍvũ); ḍʰ is not flapped (paḍʰvũ)
[Pk Ap Pa] ; ── Intervocalic lenition ───────────────────────────────────────────────
[Pk] ā > a / {n ṇ} _ t {i ē} #                   ; @morph — -nāti (9th-class) verbs thematised (jānāti > jāṇai, gr̩hṇāti > giṇhai); round 2
[mia-insular -Si -KharI -As-Ul] a y a > ē / _ t {i ē}                        ; @aya-e — 3 lects / only in -ayati causatives (prasārayati > pasārēi); nouns keep aya (ālaya, kṣaya) / (changes.sc aya-e.aya.tie)
[Si Md OSi] ; ══ Insular ══════════════════════════════════════════════════════════════
[Khol] Stop Stop > $2-asp $2                        ; @assim-stop — 109 lects / (changes.sc assim-stop.stopstop)
[OM] {n ṇ} m > m m                               ; @assim-sonorant — unmatta > umatt-
[PalyaBareli hal M mewari_bannoda Bhili Ko-Ul mewari_dholpura G dhundari_bamore] ; ── block liquids-lateral-retroflexion: Lateral retroflexion — single intervocalic l > ḷ (late MIA / early NIA; liquids blocks round, 2026-09-30). Stage 1 of the staged liquid history: a broad allopho [@l-retroflex] ──
[PalyaBareli hal mewari_bannoda mewari_dholpura G dhundari_bamore] l > ḷ / V _ V                                ; @l-retroflex — 32 lects / kāla > M kāḷā, G kāḷũ, śāla > Or sāḷa / (changes.sc l-retroflex)
[khet rajasthani-gujarati -OG -WK -Nimadi -G -OMarw] a h {i ī} > e / _ C                          ; @h-loss @hiatus — 15 lects / a(h)i > ɛ / (changes.sc h-loss.ahii.preconsonantal)
[Tir] {ʦʰ ʣʰ} > {ʦ ʣ}                              ; @deaspiration — Kvari: the new affricates lose aspiration (was {c cʰ j jʰ} > {ʦ ʦ ʣ ʣ})
[dardic-south -Gaw -Tor] ; ── block mia-dardic-lenition: Dardic intervocalic lenition (Kohistani, Kunar, Pashai; blocks theme "mia", 2026-09-30): no aspirate > h and no general stop loss as in the mainland (mia-leniti [@flap @int-stop-loss @int-p-v — Dardic lenition: ṭ ḍ > ṛ, t d and k g lost, p > v (blocks/mia-dardic-lenition.sc; blocks round 2026-09-30)] ──
[dardic-south -Gaw -Tor] {ṭ ḍ} > ṛ / V _ V                            ; @flap @int-retroflex — 19 lects / intervocalic ṭ ḍ > ṛ (gʰaṭa > gʰaṛ, kaṭuka > kaṛu, nāḍī > nāṛ) / (changes.sc flap.td.intervocalic)
[dardic-south -Gaw -Tor] ; ── block mia-dardic-stop-loss: Dardic loss of intervocalic t d and k g (blocks theme "mia", pass 2, 2026-09-30): the core of mia-dardic-lenition, included on its own by Kohistani/Kunar/Pashai [t d, then k g lost (split off in pass 2)] ──
[dardic-south -Gaw -Tor] {t d} > ∅ / V _ V                            ; @int-stop-loss — 10 lects / intervocalic t d lost (pitā > piā, nadī > naī, hr̥daya > hiaa) / (changes.sc int-stop-loss.td)
[dardic-south -Gaw -Tor] {k g} > ∅ / V _ V                            ; @int-stop-loss — 11 lects / intervocalic k g lost (lōka > lōa, nagara > naar, sūkara > sūar) / (changes.sc int-stop-loss.kg)
[Shum Wot Dm Mai Pas] p > v / V _ V                                ; @int-p-v — 119 lects / intervocalic p > v (dīpa > dīv, kūpa > kūv, tāpa > tāv) / (changes.sc int-p-v)
[Gowro] a n a a > ō̃ / V C* _ #                     ; @hiatus @x-dardic-south-n-nasal-v — post-lenition -anaa (verbal noun -anaka) contracts to -ō̃ in polysyllables (kalpáyati → kapō̃̌, ālapati → alō̃̌); round 5
[DewasDoneDanuwar Majhi Ku Gowro Gaw Tir Pas bagheli_lakshman Rathawi hal mewari_bannoda Vasavi Bhilali thar Khol rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -Rj -dhundari_bamore -OMarw -Goj] ; ── block finals-aka-i: The -ikā/-ika ending (blocks theme "finals", 2026-09-30): after intervocalic k is lost, -ia/-iā contracts to -ī, which survives apocope (finals-apocope*); -uka  [-ikā > -ī (blocks/finals-aka-i.sc) — replaces `use aka.iaa.final`: -ika/-ikā > -ia/-iā > -ī] ──
[DewasDoneDanuwar Majhi Ku Gowro Gaw Tir Pas bagheli_lakshman Rathawi hal mewari_bannoda Vasavi Bhilali thar Khol rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -Rj -dhundari_bamore -OMarw -Goj] i {a ā} > ī / _ #                            ; @aka @hiatus — 20 lects / -ikā/-ika > -ia > -ī (mwr. kīṭá+ikā > kīḍī, kukkuṭá+ikā > kūkḍī, bʰārika > bʰārī) / (changes.sc aka.iaa.final)
[Goj] ; ── block finals-aka-iu: The -ikā/-ika and -uka endings (blocks theme "finals", 2026-09-30): after intervocalic k is lost, -ia and -ua contract to long -ī and -ū, which survive apocope  [-ikā/-uka > -ī/-ū (blocks/finals-aka-iu.sc) — replaces `use aka.iaa.final`: -ika/-ikā > -ia/-iā > -ī (kuñcikā > kũcī, kārpaṭika > kāpṛī)] ──
[Goj] {i ī u ū} {a ā} > {ī ī ū ū}@1 / _ #          ; @aka @hiatus — 8 lects / -ikā/-uka > -ia/-ua > -ī/-ū (G. kárṇikā > kānī, kuñcikā > kũcī, kārpaṭika > kāpṛī) / (changes.sc aka.iiuuaa.final)
[WK] ; ── block nasals-anusvara-west: nasals-anusvara-west — anusvāra > nasal vowel, Rajasthani type, nasals theme, blocks round 2026-09-30. NIA: the anusvāra (< MIA homorganic nasal before a stop)  [@anusvara-vowel — anusvāra > nasal vowel, long before a light syllable (blocks/nasals-anusvara-west.sc)] ──
[WK] V ṁ > $1+long+nas / _ C {a i u} | _ C #      ; @anusvara-vowel — 13 lects / lengthened before a light/final syllable: dánta > Rj, had dā̃t, páñca > had, Nimadi pā̃c / (changes.sc anusvara-vowel.vm.caiuc)
[WK] ; ── block nasals-anusvara-nasal: nasals-anusvara-nasal — NIA anusvāra > short nasal vowel (no lengthening), nasals theme, blocks round 2026-09-30 (pass 2). West Pahari (bhad, jaun, khash, ktg,  [@anusvara-vowel — short nasal vowel elsewhere (before a heavy syllable): áŋgāra > ãgār (chain; Nimadi aŋgār), bʰaṇṭākī > bʰãṭāī (chain)] ──
[WK] ; ── block lar-s-h-prevocalic: Marwari/Gujarati-type debuccalisation of s before a vowel: intervocalic (the shared lar-s-h-intervocalic) and word-initial. Marwari of Bagra, Dholpuri, Wagdi-Ko [@s-h — s > h before a vowel, initially and intervocalically (blocks/lar-s-h-prevocalic.sc)] ──
[WK] ; ── block lar-s-h-intervocalic: Intervocalic s > h: Lahnda, Punjabi, Sindhi, Dhivehi (late there); the Marwari-type lects add initial s- > h- (lar-s-h-prevocalic). Laryngeals round (2026-09-30 [@s-h — intervocalic (śaśa > hāho)] ──
[WK Goj] a ū̃ > ū̃ / _ #                              ; @aka @hiatus — 2 lects / -aū̃ (< -akam) contracts to -ū̃ / (changes.sc aka.au.final)
[OG] {a u} v a > o / # _                         ; @aya-e — ava-/apa-/upa- > o- (avalipya > orpɔ, upagīti > ovī)
[OG] a m > u ṁ / _ #                             ; @final-nasal @aka — neuter -am > Ap. -uṁ before final -m is lost (-akam > -aüṁ); a sound change, so it also runs for verbs (-itavyakam > -ivaüṁ)
[OG] {s t d n m r} > ∅ / _ #                      ; @final-cons-loss — 16 lects / inflectional finals (roots cited bare keep theirs) / (changes.sc final-cons-loss.stdnmr)
[OG OMarw] ; ══ NIA (Old Western Rajasthani) ══════════════════════════════════════════
[OG OMarw] a a > a / _ {i ē}                            ; @hiatus — 2 lects / -ayati > -aï (dʰārayati > dʰārai) / (changes.sc hiatus.aa.ie)
[OMarw] i {a ā} > i / _ #                           ; @hiatus — -ika/-ikā > -i (dʰanikā > dʰaṇi)
[N] V G $2 > $1+long $3 / # C* _ V C V #         ; @degem-lengthen — 3 lects / compensatory lengthening of a first-syllable vowel before a geminate when two syllables follow (rākʰnu < *rakkʰaṇaaṁ) / (changes.sc degem-lengthen.vg.cvcv)
[N] {ḍ ḍʰ} > {r ṛʰ} / V _ V                      ; @flap — 2 lects / single MIA -ḍ- (< -ṭ-, -ḍ-) > N. r (gʰaṭī > gʰari) / (changes.sc flap.ddh.intervocalic.2)
[Dotyali] a {i ē} > y o / _ #                          ; @morph @hiatus — verbs cited as 3sg past -yo (hityo, sunyo)
[Ku N Dotyali Chil thar S Khol kcch rajasthani-gujarati -mewari_basad -OG -dhundari_bamore bhal jaun] a a > o / _ #                                ; @hiatus — 20 lects / -aka > -ao > -o (masc. extended stems) / (changes.sc hiatus.aa.final)
[S] ; if:starred,!ext a > o / C _ #                    ; @morph — Turner's reconstructed (*) a-stems are cited in -o (extended -aka type), attested OIA a-stems in -u (round 2)   [dropped: starred-headword citation shortcut; the extension menu derives these endings]
[S] ; a > u / C _ #                               ; @morph — -a stems: nom. -u (MIA -o/-u)   [round 7: @morph removed]
[KS] ; [round 7: removed legacy @morph `V a {i ē} > ā / C _ #` + `a {i ē} > ā / C _ #` — ACCEPT: ext −0.0005 train, test =; plain +0.0009 / +0.0029 (bare headword only)]
[MH] ; [round 7: removed legacy @morph `if:verb (V) {a ā} {i ē} > ∅ / _ #` — neutral removal: ext/verb/accent unchanged on train and test; plain +0.0028 / +0.0089, the bare-headword artefact (verbs now come from the menu root)]
[BH] ; [round 7: removed legacy @morph `V a {i ē} > ā / C _ #` + `a {i ē} > ā / C _ #` — neutral removal: ext/verb/accent unchanged; plain +0.0030 / +0.0038 (bare headword only)]
[A] ; [round 7: removed legacy @morph `if:verb V a {i ē} > i b a / C _ #` + `if:verb a {i ē} > i b a / C _ #` + `if:verb ā {i ē} > i b a / _ #` (verb -iba citations) — neutral removal: ext, verb, ext_acc unchanged on train and test (verbs take -iba from the menu: -itavya > -iba); plain +0.0535 / +0.0502, the bare-headword artefact. Each alone is neutral too (plain +0.0186, +0.0327, +0.0022)]
[Hajong] ; [round 7: removed legacy @morph `V a {i ē} > ā / C _ #` + `a {i ē} > ā / C _ #` — neutral removal: ext/verb/accent unchanged; plain +0.0020 / 0.0000 (bare headword only)]
[B] ; [round 7: removed legacy @morph `V a {i ē} > ā / C _ #` + `a {i ē} > ā / C _ #` + `if:verb ā {i ē} > ā / _ #` (verbs cited as -ā verbal nouns) — ext −0.0001 train, test =, verb =, ext_acc =; plain +0.0141 / +0.0147 is the bare-vs-extended artefact: the -ā citation (cālayati > cālā, ālīyatē > ālā) now comes from the %verb-menu root + -aka form, and the rules only served the bare headword]
[eastern] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[eastern] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[eastern -Or -AdivasiOriya -OB -B] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[eastern -Or -AdivasiOriya -OB -B] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[hindi -OAw] ; ── block nasals-dental-merger: nasals-dental-merger — NIA ṇ > n everywhere, nasals theme, blocks round 2026-09-30. Stage 2 (Hindi belt, Bihari, Tharu, Nepali …): every ṇ — the MIA intervocali [@n-dental — NIA ṇ > n (blocks/nasals-dental-merger.sc)] ──
[hindi -OAw] ṇ > n                                        ; @n-dental — 66 lects / pānī́ya > H, Bhoj pānī, guṇá > H gun, vána > Bhoj ban / (changes.sc n-dental.n)
[bundeli_atarra] ; round 7: removed the @morph -ayati bare-stem rule (neutral: ext/verb/acc unchanged, plain ±0); the -ati rule below is kept: removing it costs ext +0.0007 (4 words worse)
[kaithal H] ā > a / C _ #                                ; @final-vowel-loss — 5 lects / Ap. shortening of final -ā (jihvā́ > jībʰ); feeds the loss below / (changes.sc final-vowel-loss.a.c)
[kaithal] ; round 7: removed the @morph -ayati bare-stem rule (neutral); the -ati rule below and the starred-root marker are kept: removing them costs ext +0.0004 / +0.0006 train
[Buksa Chitwan Rana Dang Sunha Kathoriya bundeli_atarra kaithal] a {i ē} > ∅ / _ #                           ; @morph @hiatus — survey verbs cited as bare stems/imperatives (-ati)
[S] ; ── block finals-ap-shortening: Apabhraṁśa shortening of final -ā (blocks theme "finals", 2026-09-30): late MIA final long -ā shortens to -a, so that it is later lost with the old short -a by  [Ap. final -ā > -a (blocks/finals-ap-shortening.sc) — replaces `use final-a-short.a`: final -ā shortened (as in Sindhi)] ──
[S] ā > a / _ #                                  ; @final-a-short @final-shortening @final-vowel-loss @vowel-shift — 41 lects / final -ā > -a (Bi. ā́rā > ār, kʰáṭvā > kʰāṭ, nāvā́ > nāv) / (changes.sc final-a-short.a)
[L srk] ; if:verb a a {i ē} > a ṇ / _ #                       ; @morph — -ayati > -aṇ   [round 7: @morph removed]
[L] ; if:verb a {i ē} > a ṇ / _ #                           ; @morph — -ati verbs cited as -aṇ infinitives   [round 7: @morph removed]
[L] ; if:starred,!ext a > ā / C _ #                    ; @morph — masc. -ā citation forms (MIA -aka) for Turner's reconstructed (*) headwords; attested a-stems mostly surface bare (if:starred 0.0002 better than if:!accented; round 2)   [dropped: starred-headword citation shortcut; the extension menu derives these endings]
[thar Khol khet] ; ── block liquids-rhotacism: Rhotacism of the retroflex lateral — ḷ > r (liquids blocks round, 2026-09-30). Stage 2 of the staged liquid history, narrower than stage 1 (blocks/liquids-later [inherited from the Sindhic sisters (small-lect default, 2026-09-30)] ──
[thar Khol khet] ḷ > r                                        ; @l-r — kāla > *kāḷa > Brj kāro, pʰála > Bhoj pʰar, galati > S ɠaraṇu / (block liquids-rhotacism)
[khet] ; ══ NIA ══════════════════════════════════════════════════════════════════
[khet] ; if:verb a a {i ē} > a n / _ #                       ; @morph — -ayati > -an   [round 7: @morph removed]
[khet] ; if:!accented,!ext a > ā / C _ #                  ; @morph — masc. -ā citation forms (MIA -aka), only for headwords without a Vedic accent (round 2)
[northwest -OP -D -thar -Khol -kcch] ; ── block lar-s-h-intervocalic: Intervocalic s > h: Lahnda, Punjabi, Sindhi, Dhivehi (late there); the Marwari-type lects add initial s- > h- (lar-s-h-prevocalic). Laryngeals round (2026-09-30 [@s-h — intervocalic s > h (blocks/lar-s-h-intervocalic.sc)] ──
[northwest -OP -D -thar -Khol -kcch WK mewari_dholpura marwari_bagra] s > h / V _ V                                ; @s-h — 8 lects / aśani > āhiṇ, divasa > P. dĩh, áśva > assa > Md. aha / (changes.sc s-h.s.intervocalic)
[poth srk awan] ū > ū̃ / _ h                                 ; @nasalisation — 3 lects / spontaneous nasalisation of ū before h (< kʰ tʰ; múkʰa > mū̃, gūtʰa > gū̃̀; round 7) / (changes.sc nasalisation)
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -G -dhundari_bamore -OMarw -Goj] ; ── block hiatus-medial-west: Hiatus theme — the Rajasthani medial contraction (blocks round, 2026-09-30). Where the final -aa has already become -o (the finals), every other a/ā + a/ā hiatu [@hiatus — blocks/hiatus-medial-west.sc] ──
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -G -dhundari_bamore] {a ā} {a ā} > ā / _ !#                       ; @hiatus — non-final a/ā hiatus > ā (andʰakāra > andʰaāra > ãdʰārɔ, ãdʰārũ) / (block hiatus-medial-west)
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -G -OMarw -Goj] a {o u} > o                                  ; @hiatus — a+o/u > o (hastakūṭa > hatʰaüḍa > hathoḍo) / (block hiatus-medial-west)
[bihari-tharu -MagahiNepal -DewasDoneDanuwar -Bhoj -Mth -Darai -Majhi -Bi -Bote] a > ∅ / V C* C _ ḍ a a #                     ; @syncope — 2 lects / the stem vowel before the MIA -aḍa- extension is lost (vatsa-aḍaka > bacʰṛā, carma-aḍaka > cāmṛā); read on the extended form, before -aa > -ā / (changes.sc syncope.a.vccdaa)
[KochariyaEastDanuwar] a {i ē} > l o / _ #                           ; @morph @hiatus — verbs cited in the -lo (l-past) form (kártati > kātlo)
[bihari-tharu -MagahiNepal -Dang -Bhoj -Mth -Darai -Bi] ; ── block finals-syncope-dra: Syncope before -ḍaa/-raa (blocks theme "finals", 2026-09-30): the stem vowel before -ḍaa/-raa (from -aḍaka, -araka) is lost after a closed syllable, read on the [syncope before -ḍaa/-raa (blocks/finals-syncope-dra.sc) — replaces `use syncope.a.vccdraa`: the stem vowel before the MIA -aḍa-/-ara- extensions is lost (vatsa-aḍaka > bacʰṛā, carma-aḍaka > cāmṛā); read on the extended form, before -aa > -ā] ──
[bihari-tharu -Dang -Bhoj -Mth] a > ∅ / V C* C _ {ḍ r} a a #                 ; @syncope — 12 lects / before -ḍaa/-raa (Bi. gargara+aka > gagrā, bárkara+aka > bakrā, *kʰuṇṭa+araka > kʰũṭrā) / (changes.sc syncope.a.vccdraa)
[Morang] a > ∅ / V C* C _ l l a a #                   ; @syncope — 7 lects / likewise before -alla- (-lā) / (changes.sc syncope.a.vccllaa)
[Mth Bi] a {i ē} > b / a _ #                         ; @morph @hiatus — -ayati verbs (after stop loss -aai) cited as -ab (katʰáyati > kahab)
[Bi] a {i ē} > a b / C _ #                       ; @morph @hiatus — -ati verbs cited as -ab infinitives (karṓti > karab)
[bihari-tharu -Buksa -Chitwan -Dang -Bhoj -KochariyaEastDanuwar -Kathoriya -Mth -KochilaTharu] ; ── block finals-syncope-suffix: Syncope before the MIA -aḍa-/-ara-/-alla- extensions (blocks theme "finals", 2026-09-30): the stem vowel before -ḍaa/-raa/-llaa (from -aḍaka, -araka, -allaka) i [syncope before -ḍaa/-raa/-llaa (blocks/finals-syncope-suffix.sc) — replaces `use syncope.a.vccdraa`; `use syncope.a.vccllaa`: the stem vowel before the MIA -aḍa-/-ara-/-alla- extensions is lost (vatsa-aḍaka > bacʰṛā, carma-aḍaka > cāmṛā); read on the extended form, before -aa > -ā / likewise before -alla- (-lā)] ──
[Rana DewasDoneDanuwar Morang Sunha Majhi Bote] a > ∅ / V C* C _ {ḍ r} a a #                 ; @syncope — 12 lects / before -ḍaa/-raa (Bi. gargara+aka > gagrā, bárkara+aka > bakrā, *kʰuṇṭa+araka > kʰũṭrā) / (changes.sc syncope.a.vccdraa)
[bihari-tharu -Buksa -Chitwan -Dang -Bhoj -KochariyaEastDanuwar -Kathoriya -Mth -KochilaTharu] a > ∅ / V C* C _ l l a a #                   ; @syncope — 7 lects / before -llaa (Bi. pā́tra+allaka > patlā, vala+allaka > ballā) / (changes.sc syncope.a.vccllaa)
[Bhoj Mth Bi] a > ∅ / V C* C _ r a a #                     ; @syncope — 2 lects / the stem vowel before the MIA -ara- extension is lost (Bhoj/Mth: only this and the -aḍa- one help) (vatsa-aḍaka > bacʰṛā, carma-aḍaka > cāmṛā); read on the extended form, before -aa > -ā / (changes.sc syncope.a.vccraa)
[bihari-tharu -Bote OAw hal cam kul] a a > ā / _ #                                ; @hiatus — 17 lects / -aka > -aa > -ā (ghōṭaka > gʰoṛā; extension-aware score 0.3725 → 0.3485 in Dang) / (changes.sc hiatus.aa.final.2)
[Buksa Rana Dang DewasDoneDanuwar mewari_dholpura] ; ── block hiatus-medial: Hiatus theme — medial a(C)a contraction (blocks round, 2026-09-30). After the MIA loss of intervocalic stops/y, a-a before a consonant contracts to e and a-ā to [@hiatus — blocks/hiatus-medial.sc] ──
[Buksa Rana Dang DewasDoneDanuwar mewari_dholpura] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here ──
[Buksa Rana Dang DewasDoneDanuwar mewari_dholpura] a a > e / C _ C                              ; @hiatus — medial a(C)a > e (kadala > kaala > Bhoj kerā, kadalī > kayalī > kelā) / (block hiatus-medial-core)
[mewari_dholpura] repeat: {a ā} {a ā} > ā                      ; @hiatus — a/ā hiatus > ā (andʰakāra > andʰaāra > ā̃dʰār, kumbʰakāra > kumʰār, rājana > rāaṇ > rāṇ) / (block hiatus-medial)
[mewati_jhambaus] ; ── block hiatus-medial-west: Hiatus theme — the Rajasthani medial contraction (blocks round, 2026-09-30). Where the final -aa has already become -o (the finals), every other a/ā + a/ā hiatu [inherited from the Rajasthanic sisters (small-lect default, 2026-09-30)] ──
[mewati_jhambaus] {a ā} {a ā} > ā / _ !#                       ; @hiatus — non-final a/ā hiatus > ā (andʰakāra > andʰaāra > ãdʰārɔ, ãdʰārũ) / (block hiatus-medial-west)
[mewati_jhambaus] a {o u} > o                                  ; @hiatus — a+o/u > o (hastakūṭa > hatʰaüḍa > hathoḍo) / (block hiatus-medial-west)
[G] i > a / _ ḷ                                 ; @vowel-shift — short i > a before retroflex ḷ (miláti > maḷvũ, giláti > gaḷvũ, hiláti > haḷvũ, *vilagyati > vaḷagvũ, *āmilati > āmaḷvũ; train 17 a vs 1 i; gates ext −0.0004/−0.0003, verb −0.0011/−0.0012, 20 better / 4 worse) (hill-climb round 4)
[G] {a ā ē ō} {i ē} > v ũ / _ #                        ; @morph @hiatus — -ati/-atē/-āti/-ēti/-ōti verbs cited as -vũ infinitives (likʰati > lakʰvũ)
[G] a h {i ī} > ɛ / _ C                         ; @hiatus @h-loss — a(h)i contracts to open ɛ (abʰijñāna > ɛdʰāṇ, grahila > gʰɛlũ, mahilā > mɛḷā)
[G] repeat: a ũ > ũ / _ #                     ; @hiatus @aka — -aũ (< -akam) contracts to -ũ, repeatedly, so -aka stems + -akam (-aaũ) also end in -ũ (āmalaka > āmḷũ, avaśīrṣaka > osīsũ)
[G] {i ī} {a ā} > e / _ C                     ; @hiatus — i + a contract to ε medially (karṇikāra > kaṇer, dīpataila > divel)
[G] {a ā} {a ā} > ā / _ !#                       ; @hiatus — 8 lects / medial a/ā + a/ā contract to ā (andʰakāra > andʰaāra > ãdʰārɔ); round 7 / (changes.sc hiatus.aaaa.cond)
[dhundari_badagaon Goj] ; ── block hiatus-vh-loss: Hiatus theme — loss of intervocalic v and h (blocks round, 2026-09-30). The NIA loss of the weak intervocalic continuants that creates a second round of hiatus: [@v-loss @h-loss — blocks/hiatus-vh-loss.sc; intervocalic h (< MIA aspirates) lost] ──
[dhundari_badagaon Goj] ; ── block hiatus-v-loss: Hiatus theme — loss of intervocalic v (blocks round, 2026-09-30). The NIA loss of intervocalic v (incl. < MIA p) that opens a second round of hiatus (dīpa > dīv ──
[dhundari_badagaon Goj] v > ∅ / V _ V                                ; @v-loss — intervocalic v (incl. < p) lost (dīpa > dīva > diyā, kūpa > kuā, kapila > kailo) / (block hiatus-v-loss)
[dhundari_badagaon Goj] h > ∅ / V _ V                                ; @h-loss — intervocalic h (< MIA aspirates) lost (lṓhita > làu, gōdʰūma > gɔū̃) / (block hiatus-vh-loss)
[mewari_dholpura] ; ── block hiatus-medial-west: Hiatus theme — the Rajasthani medial contraction (blocks round, 2026-09-30). Where the final -aa has already become -o (the finals), every other a/ā + a/ā hiatu [@hiatus — blocks/hiatus-medial-west.sc] ──
[mewari_dholpura] {a ā} {a ā} > ā / _ !#                       ; @hiatus — non-final a/ā hiatus > ā (andʰakāra > andʰaāra > ãdʰārɔ, ãdʰārũ) / (block hiatus-medial-west)
[mewari_dholpura] a {o u} > o                                  ; @hiatus — a+o/u > o (hastakūṭa > hatʰaüḍa > hathoḍo) / (block hiatus-medial-west)
[mewati_akera Goj] a {o u} > o                                  ; @hiatus — 14 lects / (changes.sc hiatus.aou)
[G] a {o u} > ɔ                               ; @hiatus — a + u (after consonant loss) contracts to open ɔ (catura > cɔrɔ, catuṣka > cɔk, *kaṭṭakūṭa > kāṭɔṛɔ)
[B M G] a > ∅ / {e ē o ō} _ C                      ; @hiatus — short a absorbed after e/o left in hiatus by consonant loss, before a consonant (as -ia- > -i-): gold oa→o 13, ea→e 5 (hill-climb round 5)
[OG] a a > a u / _ #                             ; @hiatus — -aka(ḥ) > Ap. -aü (pārakya > pārakaü)
[OG] a > ∅ / i _ v v                             ; @hiatus — -iavv- > -ivv- (-itavya: valiavva- > valivaüṁ)
[OG] a > ∅ / V _ u #                             ; @hiatus — a absorbed after a vowel before final -u (anurāgaka > aṇurāau > aṇurāu, argʰyaka > āgʰu)
[bihari-tharu Ku Garh N Dotyali eastern -OB hindi marathi-bhil -M -OM northwest -D -Khol rajasthani-gujarati] ; ── block nasals-geminate-dental: nasals-geminate-dental — NIA ṇṇ > nn, nasals theme, blocks round 2026-09-30. Stage 2 (western/north-western): the MIA geminate ṇṇ (< rṇ, ṇṇ, jñ, ṇy) becomes den [@n-dental — NIA ṇṇ > nn (blocks/nasals-geminate-dental.sc)] ──
[bihari-tharu Ku Garh N Dotyali eastern -OB hindi marathi-bhil -M -OM northwest -D -Khol rajasthani-gujarati] ṇ ṇ > n n                                    ; @n-dental — 21 lects / kárṇa > kaṇṇa > G kān, P kann, parṇá > S panu, cūrṇa > G cūn / (changes.sc n-dental.nn)
[Kannauji-Central NDu bagheli_lakshman SSr] ; a a {i ē} > n ā / _ #                       ; @morph — -ayati causatives cited as -nā (ā́nayati > ānnā)  [survey verbs: no infinitive citation]
[Kannauji-Central NDu bagheli_lakshman SSr] ; a {i ē} > n ā / _ #                         ; @morph — -ati/-atē verbs cited as -nā infinitives (likʰati > likʰnā)  [survey verbs: no infinitive citation]
[hindi -OAw -H -Aw -Brj] ; ā {i ē} > ā n ā / _ #                       ; @morph — ā-root verbs (yā́ti > jānā)  [survey verbs: no infinitive citation]
[Brj] ; ā > a / C _ #                               ; @final-vowel-loss — Ap. shortening of final -ā (jihvā́ > jībʰ); feeds the loss below   [pruned for Brj: LOO -0.0018]
[Brj] ; round 7: the three if:verb @morph -no infinitive rules were removed (neutral: ext/verb/acc unchanged, plain +0.0142 train); the verb menu supplies -no
[Aw] v > ∅ / {u ū ō o} _ V                        ; @v-loss — 4 lects / v (< -p-/-v-) lost after a rounded vowel (kū́pa > kūā, pūpa > pūā), as in H; round 6b / (changes.sc v-loss.v.uuoov)
[Aw] {ē ō ē̃ ō̃} > {e o ẽ õ}                       ; @notation — Aw (Saksena) marks ē ō only where CL lengthened a vowel (later rule), so the macron is dropped before CL
[Aw] V G $2 > $1+long $3 / # C* _ V (V) b # | # C* _ (i) a b b a #   ; @degem-lengthen — Aw add-on to length-cl-hindi-trisyllable: CL also before the gerundive -(i)abba > -ab (kaḍḍʰiabba > kāṛʰab; round 5)
[Aw Brj] ; ── block liquids-flap-to-r: Intervocalic ḍ ḍʰ > ṛ ṛʰ, then the flap merges with r. Shared by 21 lects in 6 clades (Bihari, E. Hindi, E. Pahari, Eastern, Other, W. Hindi); blocks round 2026 [Intervocalic ḍ ḍʰ > ṛ ṛʰ (blocks/liquids-flap-to-r.sc)] ──
[Aw Brj] ; ── block liquids-flap: Retroflex flapping — single post-vocalic ḍ ḍʰ > ṛ ṛʰ (NIA; liquids blocks round, 2026-09-30). The flap of the Hindi belt, Punjabi-Lahnda-Sindhi, Pahari, Bengali [@flap — Tharu flap merges with r (*dʰūḍi > dʰur)] ──
[Aw Brj] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ {ḍ ḍʰ}            ; @flap — 8 lects / kaṭāha > H kaṛāhā, kukkuṭá > H kukṛā, *dʰūḍi > H dʰūṛ; geminate ḍḍ stays a stop / (changes.sc flap.ddh.postvocalic.except)
[Aw Brj] ; ── block liquids-flap-rhotic: Flap-rhotic merger — ṛ ṛʰ > r rʰ (liquids blocks round, 2026-09-30). Stage 2 of the flap history: the retroflex flap (blocks/liquids-flap.sc, or the Dardic inte [@flap — the flap merges with r] ──
[Aw Brj] {ṛ ṛʰ} > {r rʰ}                              ; @flap — *dʰūḍi > Dang dʰur, kukkuṭá > K kɔkur, kʰaḍaka > K kʰorᵘ / (block liquids-flap-rhotic)
[H] a {i ē} > n ā / _ #                         ; @morph @hiatus — -ati/-atē verbs cited as -nā infinitives (likʰati > likʰnā)
[H Brj] V G $2 > $1+long $3 / # C* _ a n a a #       ; @degem-lengthen — Brj add-on to length-cl-hindi-hiatus: CL also before the verbal noun -ana(k)a (> -no); round 6
[west-pahari -kvar] ; ── block lar-v-b-geminate: MIA geminate vv (< rv, vy, dv) > bb. West Pahari, Rajasthani, Eastern, Pali. Laryngeals round (2026-09-30). Edit here, not in the lects. [inherited from the W. Pahari sisters (small-lect default, 2026-09-30)] ──
[west-pahari -kvar] v v > b b                                    ; @v-b — 18 lects / sárva > sabb-, parvan > pabba, dravya > dabba / (changes.sc v-b.vv)
[bhal mand cur pan] # ── end of shared prefix ──
[bhal] ā > $1+stress / # C* Short C* _             ; @accent-placement — MIA quantity stress: an ā in the second syllable takes the accent from a short first vowel (vivāhá > biā̀, pañcāśát > panʣā̀, aŋgāraka > aŋā́ri; round 3)
[bhal] Short > $1+stress / # C* _ // _ C* ā        ; @accent-placement — otherwise initial stress: a short first vowel not followed by ā
[bhal] a a {i ē} > ṇ ū / _ #                   ; @morph @hiatus — -ayati verbs cited as -ṇū infinitives
[bhal khash] # if:verb a {i ē} > ṇ ū / _ #                       ; @morph — -ati verbs cited as -ṇū infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[bhal] {ā ō ī} {i ē} > $1 ṇ ū / _ #                ; @morph @hiatus — athematic/long-vowel verbs (-āti, -ōti, -ītē) cited as -ṇū infinitives
[Sv Kalk Ush Kund] k > ∅ / V _ {a ā} #                          ; @aka @int-stop-loss — 2 lects / k is lost before a final -a/-ā (the -aka/-ikā/-uka suffixes), though stem-internal k survives (round 4) / (changes.sc aka.k.vaa)
[Ush] {k g t d c j y v h ḍ b} > ∅ / V _ V         ; @int-stop-loss @v-loss @h-loss
[Ush] a a > ā / _ C                                  ; @hiatus — two a's in hiatus (after h/stop loss) before a consonant contract to ā (mahallaka > mālo, -ī, -u, saptádaśa > satāś); word-final aa (ajá, náva) is not contracted (round 6, narrowed in the second pass)
[dardic-north -Kho -K -Kal] {a i} > ∅ / _ V // _ V #                     ; @hiatus — 3 lects / but word-final vowel sequences stay as diphthongs (-ai, -ei, -ōi endings) / (changes.sc hiatus.ai.prevocalic.except)
[Sh] {a ā} {u ū} > ō / _ C                       ; @hiatus — medial a + u (after consonant loss) contracts to ō (mātulá > mōli, *lōhila-type au) (round 6)
[Sh] ā > ∅ / _ ō                                  ; @hiatus — ā is absorbed by a following ō (kʰā́d-anaka > kʰāōno > kʰōno) (round 6)
[K] {t d} > v / {a ā} _ a #                    ; @ya-sruti — a lost t/d between a and final -a leaves a v-glide (alāta > alāv, ā́gata > āv)
[K] {k g c j t d y} > ∅ / V _ V                  ; @int-stop-loss — 105 lects / (changes.sc int-stop-loss.kgcjtdy)
[dog cur] # if:verb a a {i ē} > ṇ ā / _ #                   ; @morph — -ayati verbs cited as -ṇā infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[kgr cur] # if:verb a {i ē} > ṇ ā / _ #                       ; @morph — -ati verbs cited as -ṇā infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[west-pahari -mand -WPah -kul] ; ── block finals-syncope-anaka: Syncope before the -anaka(m) verbal noun (blocks theme "finals", 2026-09-30): the a before the -n- of -anaka(m) is lost after a closed syllable, read on the for [syncope before -naa(m) (blocks/finals-syncope-anaka.sc) — replaces `use syncope.a.vccnnaam`: vowel before the -n- of -anaka(m) lost after a closed syllable (cakṣanakam > cakkʰṇaam; round 5)] ──
[west-pahari -mand -WPah -kul] a > ∅ / V C* C _ {n ṇ} a a (m) #             ; @syncope — 13 lects / a lost before -naa(m) after a closed syllable (WPah. cʰinná+anakam > cʰinṇu, darśáyati+anakam > daśṇu; kgr. *hant+anaka > handṇā) / (changes.sc syncope.a.vccnnaam)
[kiuth khash] ; ── block finals-akam-pahari: Pahari neuter -akam (blocks theme "finals", 2026-09-30): the neuter/verbal-noun -aam (< -akam, -anakam) becomes -u, the Nepali and West Pahari infinitive/citati [-aam > -u (blocks/finals-akam-pahari.sc) — replaces `use final-nasal.aam`: neuter verbal noun -anakam > -aṇaam > -ṇu, the infinitive/citation form (round 5)] ──
[kiuth khash] a a m > u / _ #                              ; @final-nasal @hiatus — 5 lects / -aam (< -akam) > -u (N. áraṇa+akam > arnu, *aḍḍ+iyanakam > aṛinu, adyá+akam > āju) / (changes.sc final-nasal.aam)
[bundeli_atarra kaithal bagheli_lakshman Brj rajasthani-gujarati -OG -WK -Nimadi -hadothi_kelwada -marwari_bagra -OMarw -Goj kiuth] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc) — before the step that frees cluster m (rm > mm > m, Dardic rm > m), so only original single m lenites] ──
[bundeli_atarra kaithal bagheli_lakshman Brj rajasthani-gujarati -OG -WK -Nimadi -hadothi_kelwada -marwari_bagra -OMarw -Goj kiuth] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
[bundeli_atarra] ; ── block length-cl-disyllable: NIA degemination with compensatory lengthening in the stressed first syllable of an (open) disyllable: V G G > V̄ G / # C* _ V #; longer words keep a short vowe [@degem-lengthen @degem-plain — CL in the first syllable of an open disyllable (# C* _ V #), then plain degemination (blocks/length-cl-disyllable.sc; blocks round 2026-09-30)] ──
[bundeli_atarra] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[bundeli_atarra] V GL $2 > $1+long $3 / # C* _ V #            ; @degem-lengthen — CL in the first syllable of a disyllable (H hásta > hātʰ, sápta > sāt, kárṇa > kān) / (block length-cl-disyllable)
[bundeli_atarra] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[bundeli_atarra] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[bundeli_atarra] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[bagheli_lakshman] ; ── block length-cl-disyllable-closed: Bihari-Tharu CL: the first syllable of a disyllable lengthens also when the final syllable is closed (Turner's consonant-final citation stems, the -ab verbal no [@degem-lengthen @degem-plain — CL in the first syllable of a disyllable, open or closed (Bihari), then plain degemination (blocks/length-cl-disyllable-closed.sc; blocks round 2026-09-30)] ──
[bagheli_lakshman] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[bagheli_lakshman] ; ── block length-cl-disyllable: NIA degemination with compensatory lengthening in the stressed first syllable of an (open) disyllable: V G G > V̄ G / # C* _ V #; longer words keep a short vowe ──
[bagheli_lakshman] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[bagheli_lakshman] V GL $2 > $1+long $3 / # C* _ V #            ; @degem-lengthen — CL in the first syllable of a disyllable (H hásta > hātʰ, sápta > sāt, kárṇa > kān) / (block length-cl-disyllable)
[bagheli_lakshman] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[bagheli_lakshman] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[bagheli_lakshman] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[kul] a {i ē} > ṇ ā / _ #                       ; @morph @hiatus — -ati verbs cited as -ṇā infinitives
[kul] # {ā ō ī} {i ē} > $1 ṇ ā / _ #                ; @morph @hiatus — athematic/long-vowel verbs (-āti, -ōti, -ītē) cited as -ṇā infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[jaun] a a m > ō̃ / C _ #                         ; @hiatus @final-nasal — neuter verbal noun -anakam > -aṇaam > -ṇō̃, the infinitive/citation form (round 5)
[bhal] a > ∅ / V C C _ C V                          ; @syncope — 7 lects / medial a also lost after a closed syllable when a vowel follows (antarāla > antrā́l, *kʰōttarūpa > kʰōtrū; round 6 second pass) / (changes.sc syncope.a.vcccv)
[bhal bhad kgr cur khash cam] ; ── block finals-akam-pahari-long: Pahari neuter -akam with long -ū (blocks theme "finals", 2026-09-30): the verbal-noun/neuter -aam (< -akam, -anakam) becomes -ū (Bhalesi, Bhadrawahi, Churahi);  [-aam > -ū (blocks/finals-akam-pahari-long.sc) — replaces `use final-nasal.aam.2`: neuter verbal noun -anakam > -aṇaam > -ṇū, the infinitive/citation form (round 5)] ──
[bhal bhad kgr cur khash cam] a a m > ū / _ #                              ; @final-nasal @hiatus — 3 lects / -aam (< -akam) > -ū (bhal utpādayati+anakam > upāṇū, kʰā́dati+anakam > kʰāṇū, kuṇḍa+akam > kunnū) / (changes.sc final-nasal.aam.2)
[WPah] # a a {i ē} > ṇ u / _ #                   ; @morph @hiatus — -ayati verbs cited as -ṇu infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[WPah] a {i ē} > ṇ u / _ #                       ; @morph @hiatus — -ati verbs cited as -ṇu infinitives
[WPah] {i u} > ∅ / _ #                             ; @final-vowel-loss — final short i u lost (round 2)
[jaun kc mand ktg WPah cam kul] ; ── block nasals-m-v: nasals-m-v — MIA/NIA intervocalic m > v (via nasalised ṽ, nasality unwritten), nasals theme, blocks round 2026-09-30 (pass 2). Western and north-western, West P [@m-v (blocks/nasals-m-v.sc)] ──
[jaun kc mand ktg WPah cam kul] m > v / V _ V                                ; @m-v — 33 lects / grāma > gāṽ/gāv, kamala > M kavaḷ / (changes.sc m-v.m.intervocalic)
[WPah] ; ── block nasals-retroflexion: nasals-retroflexion — MIA intervocalic n > ṇ (the Prakrit "no ṇaḥ" change), nasals theme, blocks round 2026-09-30. Stage 1 of n > ṇ > n: a single dental nasal b [@n-retroflex — MIA intervocalic n > ṇ (blocks/nasals-retroflexion.sc)] ──
[WPah] n > ṇ / V _ V                                ; @n-retroflex @nur-innovation @x-nur-n-flap — 40 lects / pānī́ya > G, P, S pāṇī, vána > P vaṇ, S vaṇu, Or baṇa / (changes.sc n-retroflex.intervocalic)
[WPah] a > ∅ / V C* C _ {n ṇ} a a (m) #             ; @syncope — 13 lects / vowel before the -n- of -anaka(m) lost after a closed syllable (cakṣanakam > cakkʰṇaam; round 5) / (changes.sc syncope.a.vccnnaam)
[WPah] a a m > u / _ #                              ; @final-nasal @hiatus — 5 lects / neuter verbal noun -anakam > -aṇaam > -ṇu, the infinitive/citation form (round 5) / (changes.sc final-nasal.aam)
[cam] # a a {i ē} > ṇ ā / _ #                   ; @morph @hiatus — -ayati verbs cited as -ṇā infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[cam pan] # a {i ē} > ṇ ā / _ #                       ; @morph @hiatus — -ati verbs cited as -ṇā infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[kgr mand cur] ; ── block nasals-m-nasal: nasals-m-nasal — intervocalic m > nasalisation of the preceding vowel, nasals theme, blocks round 2026-09-30 (pass 2). Sindhi/Siraiki/Pothwari and some West Pah [@m-v — intervocalic m > nasalisation (blocks/nasals-m-nasal.sc)] ──
[kgr mand cur] V m > $1+nas / _ V                           ; @m-v — 7 lects / nā́man > nā̃, kumārī > kũāri / (changes.sc m-v.vm.prevocalic)
[pan] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc) — before the step that frees cluster m (rm > mm > m, Dardic rm > m), so only original single m lenites] ──
[pan] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
[west-pahari -bhal -dog -kiuth -khash -kvar] ; ── block nasals-m-v: nasals-m-v — MIA/NIA intervocalic m > v (via nasalised ṽ, nasality unwritten), nasals theme, blocks round 2026-09-30 (pass 2). Western and north-western, West P [inherited from the W. Pahari sisters (small-lect default, 2026-09-30)] ──
[west-pahari -bhal -dog -kiuth -khash -kvar] m > v / V _ V                                ; @m-v — 33 lects / grāma > gāṽ/gāv, kamala > M kavaḷ / (changes.sc m-v.m.intervocalic)
[dog] a a m > ā̃ / _ #                         ; @hiatus @final-nasal — neuter verbal noun -anakam > -aṇaam > -ṇā̃, the infinitive/citation form (round 5)
[cam] # {ā ō ī} {i ē} > $1 ṇ ā / _ #                ; @morph @hiatus — athematic/long-vowel verbs (-āti, -ōti, -ītē) cited as -ṇā infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[bundeli_atarra Aw] ; ── block nasals-anusvara-hindi: nasals-anusvara-hindi — anusvāra > nasal vowel, Hindi type, nasals theme, blocks round 2026-09-30. NIA: the anusvāra is absorbed as nasality; no lengthening out [@anusvara-vowel — anusvāra > nasal vowel, short before a heavy syllable (blocks/nasals-anusvara-hindi.sc)] ──
[bundeli_atarra Aw] V ṁ > $1+nas / _ C Long | _ C V C C | _ C V C V ; @anusvara-vowel @stress — 6 lects / short before a heavy syllable or ≥2 syllables: *laŋgapaṭṭa > Aw lãgoṭ / (changes.sc anusvara-vowel.vm.clongcvcccvcv)
[bundeli_atarra Aw] ; ── block nasals-anusvara-vowel: nasals-anusvara-vowel — NIA anusvāra > long nasal vowel before a consonant, nasals theme, blocks round 2026-09-30 (pass 2). The common mainland outcome (Hindi b [@anusvara-vowel — long nasal vowel elsewhere: aŋká > Aw ā̃k, jáŋgʰā > Aw jā̃gʰ, dánta > SSr dā̃t] ──
[bundeli_atarra Aw] V ṁ > $1+long+nas / _ C                      ; @anusvara-vowel — 39 lects / píṇḍa > pī̃ḍ, paŋka > pā̃k, dánta > SSr dā̃t / (changes.sc anusvara-vowel.vm.preconsonantal)
[Kannauji-Central bundeli_atarra bagheli_lakshman Brj] ; ── block length-cl-hindi-hiatus: Hindi-belt CL (length-cl-disyllable) counting the -aa/-ia/-ua hiatus left by -aka/-ikā/-uka as one final syllable: V G G > V̄ G / # C* _ V V #, so the extended  [@degem-lengthen @degem-plain — Hindi-belt CL in the first syllable of a disyllable, -VV hiatus included, then plain degemination (blocks/length-cl-hindi-hiatus.sc; blocks round 2026-09-30)] ──
[Kannauji-Central bundeli_atarra bagheli_lakshman Brj] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Kannauji-Central bundeli_atarra bagheli_lakshman Brj] V GL $2 > $1+long $3 / # C* _ V V #          ; @degem-lengthen — CL before the -VV hiatus of the extended stems (hasta-ka > *hattʰaa > hātʰā) / (block length-cl-hindi-hiatus)
[Kannauji-Central bundeli_atarra bagheli_lakshman Brj] ; ── block length-cl-disyllable: NIA degemination with compensatory lengthening in the stressed first syllable of an (open) disyllable: V G G > V̄ G / # C* _ V #; longer words keep a short vowe ──
[Kannauji-Central bundeli_atarra bagheli_lakshman Brj] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Kannauji-Central Brj] V GL $2 > $1+long $3 / # C* _ V #            ; @degem-lengthen — CL in the first syllable of a disyllable (H hásta > hātʰ, sápta > sāt, kárṇa > kān) / (block length-cl-disyllable)
[Kannauji-Central bundeli_atarra bagheli_lakshman Brj] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[Kannauji-Central bundeli_atarra bagheli_lakshman Brj] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Brj] V ṁ > $1+nas / _ C Long | _ C V C C             ; @anusvara-vowel @stress — no lengthening before a heavy syllable (NIA stress on the heavy syllable); the ≥2-syllable clause of the H rule hurts Brj; round 2
[Brj] ; ── block nasals-anusvara-vowel: nasals-anusvara-vowel — NIA anusvāra > long nasal vowel before a consonant, nasals theme, blocks round 2026-09-30 (pass 2). The common mainland outcome (Hindi b [@anusvara-vowel (blocks/nasals-anusvara-vowel.sc)] ──
[Brj] V ṁ > $1+long+nas / _ C                      ; @anusvara-vowel — 39 lects / píṇḍa > pī̃ḍ, paŋka > pā̃k, dánta > SSr dā̃t / (changes.sc anusvara-vowel.vm.preconsonantal)
[NDu SSr] ; ── block length-cl-final-hiatus: CL before the final vowel (length-cl-final), also when the final "vowel" is the -aa/-ia/-ua hiatus left by -aka/-ikā/-uka (a vowel sequence counts as the final  [@degem-lengthen @degem-plain — CL before the final vowel or -VV hiatus, then plain degemination (blocks/length-cl-final-hiatus.sc; blocks round 2026-09-30)] ──
[NDu SSr] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[NDu SSr] V GL $2 > $1+long $3 / _ V V #               ; @degem-lengthen — CL before the -aa/-ia/-ua of the extended stems (NDu hásta > hātʰ, *hattʰaa > hātʰā) / (block length-cl-final-hiatus)
[NDu SSr] ; ── block length-cl-final: NIA degemination with compensatory lengthening before the final vowel: V G G > V̄ G / _ V # (the geminate of the last MIA syllable lengthens the vowel before it ──
[NDu SSr] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[NDu SSr] V GL $2 > $1+long $3 / _ V #                 ; @degem-lengthen — CL before the final vowel (Ku hásta > hātʰ, bʰaktá > bʰāt, kárṇa > kān) / (block length-cl-final)
[NDu SSr] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[NDu SSr] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[NDu SSr] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[OAw Ko-Ul Ap] {ā ī ū} > {a i u} / _ #                      ; @final-a-short @vowel-shift — 2 lects / MIA shortening of final long vowels (rā́trī > rāti) / (changes.sc final-a-short.aiu)
[OAw] ; a a {i ē} > n ā / _ #                       ; @morph — -ayati causatives cited as -nā (ā́nayati > ānnā)  [OAw verbs are cited in 3sg -ai]
[OAw] ; a {i ē} > n ā / _ #                         ; @morph — -ati/-atē verbs cited as -nā infinitives (likʰati > likʰnā)  [OAw verbs are cited in 3sg -ai]
[OAw] ; ā {i ē} > ā n ā / _ #                       ; @morph — ā-root verbs (yā́ti > jānā)  [OAw verbs are cited in 3sg -ai]
[OAw] ḍ > r / V _ // _ ḍ                          ; @flap — OAw writes r for single ḍ (paṭōlá > paravara); ḍʰ stays (gūḍʰa > gūḍʰā); before degemination
[OAw H Aw] ; ── block length-cl-hindi-trisyllable: Hindi-belt CL (length-cl-hindi-hiatus) extended to OIA trisyllables whose second syllable is light and whose final short vowel is later lost, i.e. an NIA disyll [@degem-lengthen @stress @degem-plain — Hindi-belt CL, also before a light second syllable, then plain degemination (blocks/length-cl-hindi-trisyllable.sc; blocks round 2026-09-30)] ──
[OAw H Aw] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[OAw H Aw] V GL $2 > $1+long $3 / # C* _ Short C {a i u} # ; @degem-lengthen @stress — CL before a light second syllable (H kajjala > kājal, úttara > ūtar, akṣára > ākʰar; not uttāra > utār, aṭṭāla > aṭāl) / (block length-cl-hindi-trisyllable)
[OAw H Aw] ; ── block length-cl-hindi-hiatus: Hindi-belt CL (length-cl-disyllable) counting the -aa/-ia/-ua hiatus left by -aka/-ikā/-uka as one final syllable: V G G > V̄ G / # C* _ V V #, so the extended  ──
[OAw H Aw] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[OAw H Aw] V GL $2 > $1+long $3 / # C* _ V V #          ; @degem-lengthen — CL before the -VV hiatus of the extended stems (hasta-ka > *hattʰaa > hātʰā) / (block length-cl-hindi-hiatus)
[OAw H Aw] ; ── block length-cl-disyllable: NIA degemination with compensatory lengthening in the stressed first syllable of an (open) disyllable: V G G > V̄ G / # C* _ V #; longer words keep a short vowe ──
[OAw H Aw] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[H] V GL $2 > $1+long $3 / # C* _ V #            ; @degem-lengthen — CL in the first syllable of a disyllable (H hásta > hātʰ, sápta > sāt, kárṇa > kān) / (block length-cl-disyllable)
[OAw H Aw] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[OAw H Aw] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[H Aw] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[kaithal Rathawi mewari_bannoda Marw mewati_jhambaus kvar] ; ── block length-cl-final: NIA degemination with compensatory lengthening before the final vowel: V G G > V̄ G / _ V # (the geminate of the last MIA syllable lengthens the vowel before it [@degem-lengthen @degem-plain — CL before the final vowel (_ V #), then plain degemination (blocks/length-cl-final.sc; blocks round 2026-09-30)] ──
[kaithal Rathawi mewari_bannoda Marw mewati_jhambaus kvar] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Garh kaithal Rathawi mewari_bannoda Marw mewati_jhambaus kvar] V GL $2 > $1+long $3 / _ V #                 ; @degem-lengthen — CL before the final vowel (Ku hásta > hātʰ, bʰaktá > bʰāt, kárṇa > kān) / (block length-cl-final)
[kaithal Rathawi mewari_bannoda Marw mewati_jhambaus kvar] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[kaithal Rathawi mewari_bannoda Marw mewati_jhambaus kvar] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[mewari_bannoda Marw mewati_jhambaus] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[mewari_basad dhundari_badagaon] ; ── block length-cl-rg-disyllable: Rajasthani CL in the first syllable of a disyllable only: V G G > V̄ G / # C* _ C* V # (no lengthening before a light or closed final syllable, unlike length-cl [@degem-lengthen — Rajasthani CL in the first syllable of a disyllable (blocks/length-cl-rg-disyllable.sc; blocks round 2026-09-30)] ──
[mewari_basad dhundari_badagaon] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[mewari_basad dhundari_badagaon] V GL $2 > $1+long $3 / # C* _ C* V #         ; @degem-lengthen — first syllable of a disyllable (Nimadi hásta > hātʰ, kárṇa > kān) / (block length-cl-rg-disyllable)
[had marwari_bagra] a > ∅ / V _                                  ; @hiatus — 37 lects / a after another vowel absorbed / (changes.sc hiatus.a.postvocalic)
[OM] k > ∅ / a _ a (m) #                        ; @int-stop-loss — round 6: k lost in the final syllable of -aka / -akam (the tadbhava suffix; stems stay learned; -ika/-uka keep k: adhika, amuka)
[M OM Ko-Ul] ; ── block finals-akam-marathi: Marathi-Konkani neuter -akam (blocks theme "finals", 2026-09-30): -aam/-aaṁ (< -akam) > -ẽ, the neuter ending, before final-consonant loss. [-aam > -ẽ (blocks/finals-akam-marathi.sc) — replaces `use final-nasal.aamm`: neuter -akam > -aaṁ > -ẽ (before final-consonant loss)] ──
[M OM Ko-Ul] a a {m ṁ} > ẽ / _ #                          ; @final-nasal @hiatus — 2 lects / -aam (< -akam) > -ẽ (M. sāṁpratam > sā̃pẽ; the -anakam verbal noun > -ṇẽ) / (changes.sc final-nasal.aamm)
[Dhp] {ś s} v > s p                                ; @sv-sp — śv, sv > sp with a dental s (īśvará > ispara, víśva > vispa, śváśura > spaśura, svá > spa); round 7
[NiDoc] Stop Nasal > $1-asp $1                      ; @assim-sonorant — only Stop + nasal; Stop + r/v/l/y kept in Niya spelling (ātman > atvana)
[NiDoc Dhp] ; ── intervocalic lenition ───────────────────────────────────────────────
[Dhp] a y a > ē / _ {t d} i                        ; @aya-e — 2 lects / -ayati > -edi (causatives only; naya, āśaya keep aya) / (changes.sc aya-e.aya.tdi)
[M] a v a > ō / # _                              ; @aya-e — 9 lects / ava/apa- > o- (apasarati > osarṇẽ) / (changes.sc aya-e.ava.initial)
[DewasDoneDanuwar Bhoj Darai KochilaTharu Dotyali Bishnupriya AdivasiOriya OB MH Kannauji-Central OAw Brj SSr marathi-bhil thar khet srk rajasthani-gujarati -bagri_fatehabad -OG -Rj -mewati_akera -dhundari_badagaon -OMarw] C > ∅ / _ #                                  ; @final-cons-loss — 27 lects / (changes.sc final-cons-loss.c)
[Noiri marwari_bagra] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc) — before the step that frees cluster m (rm > mm > m, Dardic rm > m), so only original single m lenites] ──
[Noiri marwari_bagra] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
[bagri_fatehabad had hadothi_kelwada mewati_jhambaus Rj marwari_bagra dhundari_bamore dhundari_badagaon] ; ── block length-cl-rg-light: Rajasthani CL before the final vowel in any syllable and in a first syllable before a light (short open) syllable: V G G > V̄ G / _ V # | # C* _ C* Short C V (d [@degem-lengthen @x-rg-weight — Rajasthani CL before the final vowel and before a light syllable (blocks/length-cl-rg-light.sc; blocks round 2026-09-30)] ──
[bagri_fatehabad had hadothi_kelwada mewati_jhambaus Rj marwari_bagra dhundari_bamore dhundari_badagaon] V G $2 > $1+long $3 / _ V # | # C* _ C* Short C V ; @degem-lengthen @x-rg-weight — 6 lects / before the final vowel (hásta > hātʰ, -ikka > -īk) and in a first syllable before a light one / (changes.sc degem-lengthen.vg.vccshortcv)
[mewari_dholpura OMarw] ; ── block length-cl-rg: Rajasthani-Gujarati CL by weight (@x-rg-weight): V G G > V̄ G in the first syllable when the next syllable is final (open or closed) or light, i.e. when the fir [@degem-lengthen @x-rg-weight — Rajasthani CL: first syllable before a final or light syllable (blocks/length-cl-rg.sc; blocks round 2026-09-30)] ──
[mewari_dholpura OMarw] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[mewari_dholpura dhundari_bamore OMarw] V GL $2 > $1+long $3 / # C* _ C* V (C) # | # C* _ C* Short C V ; @degem-lengthen @x-rg-weight — first syllable before a final or light syllable (hásta > hātʰ, kárṇa > kān) / (block length-cl-rg)
[Garh] ∅ > ṇ u / C _ # // n _                        ; @morph — CDIAL's consonant-final (starred verbal) roots *ṭʰakk-, *cʰipp-, *ḍʰapp- are cited with the infinitive/past ending
[mewari_basad] ; ── block liquids-lateral-retroflexion: Lateral retroflexion — single intervocalic l > ḷ (late MIA / early NIA; liquids blocks round, 2026-09-30). Stage 1 of the staged liquid history: a broad allopho [@l-retroflex — Varghese's Malvi writes ḷ (69/119); round 1 had pruned it] ──
[mewari_basad] l > ḷ / V _ V                                ; @l-retroflex — 32 lects / kāla > M kāḷā, G kāḷũ, śāla > Or sāḷa / (changes.sc l-retroflex)
[Gowro] a ī > ī / C _ #                              ; @hiatus — 13 lects / -aī (from -a-ikā) contracts to -ī (pōstaka+ikā > potʰī, ārdraka+ikā > ādī), as in H, Bg, Ch, Hr; round 6 / (changes.sc hiatus.ai.c)
[Gowro Chil] ; ── block finals-apocope-postcons-e: Apocope after a consonant, with -ē (blocks theme "finals", 2026-09-30): final -a -ā -ē -i -ī -u are lost after a consonant in polysyllables (V C* C _ #); final  [apocope after a consonant, -o kept (blocks/finals-apocope-postcons-e.sc) — replaces `{a ā ē i ī u} > ∅ / V C* C _ #`: monosyllables keep their vowel; −0.001 to −0.008 in all 12 lects)] ──
[Gowro Chil] {a ā ē i ī u} > ∅ / V C* C _ #               ; @final-vowel-loss — final vowels other than -o lost after a consonant (Chil. síndʰu > sin, tailá > tel, dáśa > daś) / (block finals-apocope-postcons-e)
[Tor] {a i u} $1 > {ā ī ū}                        ; @hiatus — like vowels in hiatus contract to a long vowel (katamá > kām, saptádaśa > sattāś)
[Pas awan mewari_basad Nimadi had mewari_dholpura marwari_bagra] ; ── block h-loss-optional: Sporadic loss of intervocalic h at the end of the chain (hill-climb round 3, 2026-10-01). Where a lect keeps the h from OIA h and MIA aspirates (Pk. -h- < -kh-  [sporadic (blocks/h-loss-optional.sc; hill-climb round 3)] ──
[Pas awan mewari_basad Nimadi had mewari_dholpura marwari_bagra] optional: h > / V _ V                        ; @h-loss — sporadic intervocalic h loss (agnidʰāna > N agenu, aragʰaṭṭa > Or araṭa) / (block h-loss-optional)
[OP P kcch L srk awan] G $1 > $2 / _ {ā ī ū ē ō}                    ; @degem-plain — 2 lects / pretonic: geminate simplifies before a (stressed) long vowel (*pēṭṭāra > piṭārī) / (changes.sc degem-plain.g.aiueo.2)
[awan] a a {i ē} > u ṇ / _ #                       ; @morph @hiatus — -ayati > -uṇ
[awan] ; if:verb a {i ē} > u ṇ / _ #                           ; @morph — Awankari infinitive -uṇ   [round 7: @morph removed]
[Rj dhundari_bamore] ; ── block length-cl-rg-final: Rajasthani CL by weight (length-cl-rg: first syllable before a final or light syllable), then CL before the final vowel in any syllable and before a light sylla [@degem-lengthen @x-rg-weight — Rajasthani CL by weight + before the final vowel (length-cl-rg, length-cl-rg-light) (blocks/length-cl-rg-final.sc; blocks round 2026-09-30)] ──
[Rj dhundari_bamore] ; ── block length-cl-rg: Rajasthani-Gujarati CL by weight (@x-rg-weight): V G G > V̄ G in the first syllable when the next syllable is final (open or closed) or light, i.e. when the fir ──
[Rj dhundari_bamore] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Rj] V GL $2 > $1+long $3 / # C* _ C* V (C) # | # C* _ C* Short C V ; @degem-lengthen @x-rg-weight — first syllable before a final or light syllable (hásta > hātʰ, kárṇa > kān) / (block length-cl-rg)
[Rj dhundari_bamore] ; ── block length-cl-rg-light: Rajasthani CL before the final vowel in any syllable and in a first syllable before a light (short open) syllable: V G G > V̄ G / _ V # | # C* _ C* Short C V (d ──
[Rj dhundari_bamore] V G $2 > $1+long $3 / _ V # | # C* _ C* Short C V ; @degem-lengthen @x-rg-weight — 6 lects / before the final vowel (hásta > hātʰ, -ikka > -īk) and in a first syllable before a light one / (changes.sc degem-lengthen.vg.vccshortcv)
[bhad cur khash cam pan] ; ── block finals-syncope-anaka: Syncope before the -anaka(m) verbal noun (blocks theme "finals", 2026-09-30): the a before the -n- of -anaka(m) is lost after a closed syllable, read on the for [syncope before -naa(m) (blocks/finals-syncope-anaka.sc)] ──
[bhad cur khash cam pan] a > ∅ / V C* C _ {n ṇ} a a (m) #             ; @syncope — 13 lects / a lost before -naa(m) after a closed syllable (WPah. cʰinná+anakam > cʰinṇu, darśáyati+anakam > daśṇu; kgr. *hant+anaka > handṇā) / (changes.sc syncope.a.vccnnaam)
[ktg] a a m > õ / _ #                         ; @hiatus @final-nasal — neuter verbal noun -anakam > -aṇaam > -ṇõ, the infinitive/citation form (round 5)
[Bishnupriya AdivasiOriya Hajong OB MH Khandesi DungraBhili hal PauriBareli mewari_bannoda RathwiBareli northwest -Khol -L rajasthani-gujarati -mewari_basad -mewari_dholpura west-pahari] {s ḣ t d m n} > ∅ / _ #                      ; @final-cons-loss — 30 lects / only OIA word-final consonants (starred roots keep theirs) / (changes.sc final-cons-loss.shtdmn)
[kiuth WPah] # if:verb ∅ > ṇ u / C _ #                          ; @morph — consonant-final (starred verbal) roots cited as -ṇu infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[kgr cam kul] ∅ > ṇ ā / C _ #                          ; @morph — consonant-final (starred verbal) roots cited as -ṇā infinitives
[kc] # ∅ > ṇ o / C _ #                          ; @morph — consonant-final (starred verbal) roots cited as -ṇo infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[ktg] # if:verb ∅ > ṇ õ / C _ #                          ; @morph — consonant-final (starred verbal) roots cited as -ṇõ infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[bhal] # if:verb ∅ > ṇ ū / C _ #                          ; @morph — consonant-final (starred verbal) roots cited as -ṇū infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[cur] # if:verb ∅ > ṇ ā / C _ #                          ; @morph — consonant-final (starred verbal) roots cited as -ṇā infinitives  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[west-pahari] ; ══ NIA / West Pahari ════════════════════════════════════════════════════
[poth bagri_fatehabad dhundari_bamore mand] ; ── block hiatus-vh-loss: Hiatus theme — loss of intervocalic v and h (blocks round, 2026-09-30). The NIA loss of the weak intervocalic continuants that creates a second round of hiatus: [@v-loss @h-loss — blocks/hiatus-vh-loss.sc; lṓhita > làu] ──
[poth bagri_fatehabad dhundari_bamore mand] ; ── block hiatus-v-loss: Hiatus theme — loss of intervocalic v (blocks round, 2026-09-30). The NIA loss of intervocalic v (incl. < MIA p) that opens a second round of hiatus (dīpa > dīv ──
[poth dhundari_bamore mand] v > ∅ / V _ V                                ; @v-loss — intervocalic v (incl. < p) lost (dīpa > dīva > diyā, kūpa > kuā, kapila > kailo) / (block hiatus-v-loss)
[poth bagri_fatehabad dhundari_bamore mand] h > ∅ / V _ V                                ; @h-loss — intervocalic h (< MIA aspirates) lost (lṓhita > làu, gōdʰūma > gɔū̃) / (block hiatus-vh-loss)
[poth awan] ; if:!accented,!ext a > ā / C _ #                  ; @morph — masc. -ā citation rule removed in round 6 (bare-vs-extended artifact: the -ā comes from the -aka extension)
[srk] ; if:verb a {i ē} > a ṇ / C _ #                       ; @morph — -ati verbs cited as -aṇ infinitives   [round 7: @morph removed]
[bihari-tharu -DewasDoneDanuwar -Darai -Majhi -Bote S] i a > ī / _ #                                ; @hiatus — 14 lects / -ikā > -ia > -ī / (changes.sc hiatus.ia.final)
[OP] ∅ > i / C _ #                               ; @morph — consonant-final CDIAL roots (*gʰall) are verbal: cited as stem + -i
[OP] a a {i ē} > i / _ #                         ; @morph @hiatus — verbs cited as stem + -i (kalpayati > kappi)
[OP] ; if:starred,!ext a > ə / C _ #                    ; @morph — Turner's reconstructed (*) a-stems take the extended -ā (MIA -aka) citation form, attested a-stems -u; ə = placeholder so the -ā escapes the final -ā shortening (round 2)   [dropped: starred-headword citation shortcut; the extension menu derives these endings]
[OP] ; a > u / C _ #                               ; @morph — -a stems: dir. sg. -u (jala > jalu)   [round 7: @morph removed]
[Khandesi PalyaBareli Noiri mewati_akera] V G_std $2 > $1+long $3 / _ V #              ; @degem-lengthen — 12 lects / hattʰa > hātʰ / (changes.sc degem-lengthen.vg.v)
[VB] optional: a {a ā} > o / _ #                  ; @hiatus — 11 lects / sporadic (the source has both -ō and -ā): -aka > -aa > -o (Bhil/Gujarati-type masc.: āṇḍaka > āṇḍō, āmraka > ambō) / (changes.sc hiatus.aaa.final)
[VB] o > ō / _ #                                  ; @notation — srinivasa writes the masc. ending -ō
[marathi-bhil -Bhatri -hal -M -OM -Ko-Ul] ; ══ NIA (Bhil) ═══════════════════════════════════════════════════════════
[Bhatri] V > $1-acc                                  ; @acc-loss
[Bhatri] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[Bhatri] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[Bhatri Vasavi OP] ; ── block finals-ap-shortening: Apabhraṁśa shortening of final -ā (blocks theme "finals", 2026-09-30): late MIA final long -ā shortens to -a, so that it is later lost with the old short -a by  [Ap. final -ā > -a (blocks/finals-ap-shortening.sc) — replaces `use final-a-short.a`: final -ā written -a (GGS)] ──
[Bhatri Vasavi OP] ā > a / _ #                                  ; @final-a-short @final-shortening @final-vowel-loss @vowel-shift — 41 lects / final -ā > -a (Bi. ā́rā > ār, kʰáṭvā > kʰāṭ, nāvā́ > nāv) / (changes.sc final-a-short.a)
[eur as-ll RomWel RomBalk Zarg Sv Kalk Kho Ush Kund Gowro Bshk Chil Tor Pk NiDoc As-Ul northwest -D -thar west-pahari -bhal -bhad -ktg -cur -khash -kul -pan] ; ── block nasals-postnasal-voicing: nasals-postnasal-voicing — NT > ND, nasals theme, blocks round 2026-09-30. A voiceless stop, aspirated or not, is voiced after a nasal (or anusvāra-turned-nasal [@nasal-voicing — NT > ND incl. aspirates (blocks/nasals-postnasal-voicing.sc)] ──
[eur as-ll RomWel RomBalk Zarg Sv Kalk Kho Ush Kund Gowro Bshk Chil Tor Pk NiDoc As-Ul northwest -D -thar west-pahari -bhal -bhad -ktg -cur -khash -kul -pan] Nasal {k kʰ c cʰ ṭ ṭʰ t tʰ p pʰ} > $1 {g gʰ j jʰ ḍ ḍʰ d dʰ b bʰ}@2 ; @nasal-voicing — 10 lects / dánta > awan dand, káṇṭaka > awan kaṇḍā, śaŋkú > Kho śoŋg, páñca > Kho ponj / (changes.sc nasal-voicing.nasalkkhcc)
[awan] m {b bʰ} > m m                               ; @nasal-assim — 2 lects / mb mbʰ > mm (Hindko; rejected by the L, srk, P data) / (changes.sc nasal-assim.mbbh)
[kcch] {g d ḍ b j} > {gʰ dʰ ḍʰ bʰ jʰ} / Nasal _    ; @x-nw-postnasal-aspiration — Kutchi aŋguṣṭʰa > āŋgʰūṭʰo, undura > ūndʰar
[thar Khol kcch srk] d > ḍ / # _                                  ; @x-nw-d-retroflex — 3 lects / initial d- > ḍ-/ᶑ- (divasa > ḍẽh) / (changes.sc d-retroflex)
[S] {ḍ b j g} {ḍ b j g} > {ᶑ ɓ ʄ ɠ}@1 {ᶑ ɓ ʄ ɠ}@2   ; @implosive — plain voiced geminates only (*gōḍḍa > goᶑo, adya > ajja > aʄu); aspirated ones keep the aspirate (argʰa > agʰu, dīrgʰa > ḍrigʰo)
[S] {ḍ b} > {ᶑ ɓ} / # _                       ; @implosive — initial
[S] {g j} > {ɠ ʄ} / # _ // # _ !#* {ɠ ʄ ᶑ ɓ}   ; @implosive — initial velar/palatal, not when the word already has an implosive (dissimilation: *gōḍḍa > goᶑo, gaḍḍa > gaᶑo)
[S] {t tʰ d dʰ} r > {ṭ ṭʰ ḍ ḍʰ} r               ; @tr-retroflex — tr > ṭr (udrá > uḍru)
[D] Long > $1+stress / # !Long* _               ; @accent-placement — accent on the first long vowel (Shina-type; Weinreich kóon, póo, cíi, gíiri); words without a long vowel get none (a default placement scores worse; round 3)
[D] V > $1+stress / _ C* # // _ !#* Long | Long !#* _   ; @accent-placement — words without a long vowel are accented on the last syllable (Weinreich baríś, kirmá, caagá; round 7: ext_acc −0.011 train/test, +9 accent pairs net, segmental scores unchanged)
[D] ; if:verb a {i ē} > n a / _ #                         ; @morph — verbs cited with the -na infinitive (kʰādati > kʰāna, rōdati > roina)   [round 7: @morph removed]
[D] {ñ ṇ n m} Stop > $1                          ; @cluster-simplify @nc-simplify — 2 lects / stop lost after a nasal (síndʰu > sina, kāṇḍa > kóon) / (changes.sc nc-simplify.nnnmstop)
[D] ŋ Stop > $2                                 ; @cluster-simplify — but ŋg > g (aŋgúli > agúla), as in Shina
[marathi-bhil -Bhatri -Khandesi -hal -M -Bhilali -OM -VB] a {a ā} > o / _ #                            ; @hiatus — 11 lects / -aka > -aa > -o (Gujarati-type masc.) / (changes.sc hiatus.aaa.final)
[OM] a a > ā / _ #                                ; @hiatus — 17 lects / -aka > -aa > -ā (ghōṭaka > gʰoṛā; extension-aware score 0.3725 → 0.3485 in Dang) / (changes.sc hiatus.aa.final.2)
[OM] n > ṇ / V _ ẽ #                              ; @n-retroflex — round 6: n > ṇ before the neuter ending (-aṇē)
[Bhatri] C > ∅ / _ #                                  ; @final-cons-loss — 27 lects / (changes.sc final-cons-loss.c)
[Bhatri hal] ; ══ NIA (Halbic) ═════════════════════════════════════════════════════════
[northwest -S -L] a v a > a u / _ C                            ; @aya-e — 6 lects / ava > au before a consonant (apalāpa > valāu, avarā > aul) / (changes.sc aya-e.ava.preconsonantal)
[P] ā v a > ā u / _ C                           ; @aya-e — āva > āu before a consonant, as ava > au: causative -āpana(ka) > -āvaṇā > -āuṇā (prāpayati > pāuṇā; round 5)
[Rana] ; ── block liquids-flap-to-r: Intervocalic ḍ ḍʰ > ṛ ṛʰ, then the flap merges with r. Shared by 21 lects in 6 clades (Bihari, E. Hindi, E. Pahari, Eastern, Other, W. Hindi); blocks round 2026 [Intervocalic ḍ ḍʰ > ṛ ṛʰ (blocks/liquids-flap-to-r.sc)] ──
[Rana] ; ── block liquids-flap: Retroflex flapping — single post-vocalic ḍ ḍʰ > ṛ ṛʰ (NIA; liquids blocks round, 2026-09-30). The flap of the Hindi belt, Punjabi-Lahnda-Sindhi, Pahari, Bengali [@flap — Tharu flap merges with r (*dʰūḍi > dʰur)] ──
[Rana] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ {ḍ ḍʰ}            ; @flap — 8 lects / kaṭāha > H kaṛāhā, kukkuṭá > H kukṛā, *dʰūḍi > H dʰūṛ; geminate ḍḍ stays a stop / (changes.sc flap.ddh.postvocalic.except)
[Rana] ; ── block liquids-flap-rhotic: Flap-rhotic merger — ṛ ṛʰ > r rʰ (liquids blocks round, 2026-09-30). Stage 2 of the flap history: the retroflex flap (blocks/liquids-flap.sc, or the Dardic inte [@flap — the flap merges with r] ──
[Rana] {ṛ ṛʰ} > {r rʰ}                              ; @flap — *dʰūḍi > Dang dʰur, kukkuṭá > K kɔkur, kʰaḍaka > K kʰorᵘ / (block liquids-flap-rhotic)
[Bote] a a > o / _ #                                ; @hiatus — 20 lects / -aka > -aa > -o (Nepali-type masc. -o: dʰāgo, kā̃ṭo) / (changes.sc hiatus.aa.final)
[Darai Bote Shum Wot Dm Tor northwest -D -thar -S -Khol -srk hadothi_kelwada kc] ; ── block finals-aka-iu: The -ikā/-ika and -uka endings (blocks theme "finals", 2026-09-30): after intervocalic k is lost, -ia and -ua contract to long -ī and -ū, which survive apocope  [-ikā/-uka > -ī/-ū (blocks/finals-aka-iu.sc) — replaces `use aka.iiuuaa.final`: -ikā/-iya > -iā > -ī, -uka > -uā > -ū] ──
[Darai Bote Shum Wot Dm Tor northwest -D -thar -S -Khol -srk hadothi_kelwada kc] {i ī u ū} {a ā} > {ī ī ū ū}@1 / _ #          ; @aka @hiatus — 8 lects / -ikā/-uka > -ia/-ua > -ī/-ū (G. kárṇikā > kānī, kuñcikā > kũcī, kārpaṭika > kāpṛī) / (changes.sc aka.iiuuaa.final)
[srk awan] a {i u} > {e o}@2                            ; @ai-au — 5 lects / secondary diphthongs from hiatus monophthongise (as in OIA-dog.sc) / (changes.sc ai-au.aiu)
[awan] a > u / V _ ṇ a #                           ; @vowel-shift — the -uṇ infinitive also after a stem vowel left in hiatus by consonant loss (before the hiatus rules) …
[awan] ∅ > v / V _ u ṇ a #                         ; @hiatus — … with a glide v: *kōṣati > kṑvuṇ, piṁṣáti > pī̀vuṇ, *rahati > rā̀vuṇ (round 7)
[OP P poth srk awan] repeat: {a ā} {a ā} > ā                      ; @hiatus — 22 lects / rājana > rāaṇ > rāṇ / (changes.sc hiatus.aaaa.repeat)
[OP S P poth srk awan] a > ∅ / {i ī u ū ē ō} _                      ; @hiatus — 7 lects / sūkara > sūar > sūr / (changes.sc hiatus.a.iiuueo)
[poth] ; use ai-au.aiu                                ; @ai-au — removed round 7: Pothwari keeps the secondary diphthong from hiatus, as Punjabi does (nakulá > naul, tráyōviṁśati > taoeī; ext −0.0020 train, test flat)
[S] ā v a > ā i / _ ṇ                         ; @aya-e — āva > āi before the verbal-noun -ṇ-: the causative -āpanakam > -āvaṇu > -āiṇu (útpātayati > upāiṇu, *uccayatē > ucāiṇu; round 6 second pass)
[WK kgr mand] i ā > ī / _ #                                ; @hiatus — 3 lects / fem. -ikā > -iā > -ī (round 4) / (changes.sc hiatus.ia.final.2)
[kiuth mand] ; ── block finals-pahari-east: Inner Pahari finals with -ā (blocks theme "finals", 2026-09-30): final -a and -ā are lost except the second vowel of the -aa left by -aka, which then contracts  [apocope of -a -ā, then -aa > -ā (blocks/finals-pahari-east.sc)] ──
[kiuth mand] {a ā} > ∅ / _ # // a _                       ; @final-vowel-loss — 4 lects / final -a -ā lost, not the second a of -aka > -aa (khash gʰara > gʰar, pʰulla > pʰull; kiuth kárṇa > kān) / (changes.sc final-vowel-loss.aa.except)
[kiuth mand] ; ── block finals-aka-east: The -aka ending in the centre and east, final only (blocks theme "finals", 2026-09-30): the masculine nominative -akaḥ (MIA -aa) contracts to -ā, which is why t ──
[kiuth mand] a a > ā / _ #                                ; @aka @hiatus — 11 lects / -aka > -aa > -ā (kul. kāṇá+aka > kāṇā, ūná+aka > uṇā, uparōdʰa+aka > urā) / (changes.sc aka.aa.final)
[kiuth] ; ── block finals-syncope-pahari: West Pahari syncope (blocks theme "finals", 2026-09-30): a medial short vowel of an open non-initial syllable is lost (VCVC), then medial a also after a closed  [W. Pahari syncope (blocks/finals-syncope-pahari.sc)] ──
[kiuth] {a i u} > ∅ / V C _ C                        ; @syncope — 9 lects / open non-initial syllable (kgr. cālana+ikā > cālṇī, apakṣá+aka > aukʰā; WPah. *muṣala > muśḷ) / (changes.sc syncope.aiu.vcc)
[kiuth] a > ∅ / V C C _ C V                          ; @syncope — 7 lects / also after a closed syllable (kgr. *gilla+araka > gilrā, *paddʰara+aka > padrā, *vartmapāla > batvāl) / (changes.sc syncope.a.vcccv)
[cur] {s ś} > h / V _                              ; @s-h — 2 lects / postvocalic sibilant > h / (changes.sc s-h.ss.postvocalic)
[Kal Wot bhatr ktg cur kul] Nasal {k c ṭ t p} > $1 {g j ḍ d b}           ; @nasal-voicing — 19 lects / (changes.sc nasal-voicing.nasalkcttp)
[jaun] V Nasal > $1+nas / _ Stop                    ; @anusvara-vowel — nasal before a stop > vowel nasalisation
[mewari_basad dhundari_bamore] a a > o / _ #                                ; @hiatus — 20 lects / -aka > -ao > -o (masc. extended stems) / (changes.sc hiatus.aa.final)
[mewari_basad bagri_fatehabad Nimadi Rj mewati_akera dhundari_bamore dhundari_badagaon Goj kiuth] a > ∅ / V _                                  ; @hiatus — 37 lects / a after another vowel absorbed / (changes.sc hiatus.a.postvocalic)
[G] {a i e o ā ī ē ō} G $2 > $1+long $3 / # C* _ C* V (C) # | # C* _ C* Short C V | # C* _ C* e v v   ; @degem-lengthen @x-rg-weight — first syllable, if the next syllable is final or light (the -evv- of -itavya counts as light: its e is later lost, arpitavya > āpvũ)
[rajasthani-gujarati -mewari_basad -WK -mewari_dholpura -marwari_bagra -OMarw] ; ── block finals-syncope-rajasthani: Rajasthani syncope before the -ḍo/-lo/-ro/-ṇo/-Cī suffix syllables (blocks theme "finals", 2026-09-30): medial a is lost before a final -ḍ ḷ l ṛ r ṇ n + -o/-ī ( [syncope before -Co/-Cī suffixes (blocks/finals-syncope-rajasthani.sc) — replaces `use syncope.a.vccdllrrnnoi`: the suffix vowel of -aḍa-/-ala-/-ara-/-aṇa- is lost before -o/-ī (ṭūkḍo, kukro, hasṇo; round 7 group form)] ──
[rajasthani-gujarati -mewari_basad -WK -mewari_dholpura -marwari_bagra -OMarw] a > ∅ / V C (C) _ {ḍ ḷ l ṛ r ṇ n} {o ī} #    ; @syncope @x-rg-dla — 4 lects / medial a lost before -Co/-Cī (Marw. karpaṭa+aḍaka > kapaḍɔ, sīvana+ikā > sīvṇī, kurkurá+ikā > kukrī) / (changes.sc syncope.a.vccdllrrnnoi)
[marwari_bagra] a > ∅ / V C (C) _ {ḍ ḷ l ṛ} {o ī} #          ; @syncope @x-rg-dla — 7 lects / the suffix vowel of -aḍa-/-ala- is lost before -o/-ī (ṭūkḍo, pʰūlḍo, camṛī; *-aḍaka > -ḍo, *-allaka > -lo) / (changes.sc syncope.a.vccdllroi)
[MH] ; [round 7: dropped `use v-b.vv.2` (vv > bb, the A. -iba source): MH's toulmin cites bare verb stems, so it only acted on nouns, wrongly (gōrvara > gavar, cárvati > cavvā); ext −0.0007 train, test =, 2 better, 0 worse]
[Buksa Kathoriya Majhi Garh Dotyali eastern marathi-bhil -PauriBareli -M -mewari_bannoda -Vasavi -OM -RathwiBareli -Ko-Ul] a > ∅ / i _ C                                ; @hiatus — 7 lects / a absorbed after i before a consonant (-iavva > -ivva), as in A. and B. / (changes.sc hiatus.a.ic)
[A S hadothi_kelwada] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[A S hadothi_kelwada] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[Bhatri Rathawi hal mewari_bannoda Bhilali thar] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc) — before the step that frees cluster m (rm > mm > m, Dardic rm > m), so only original single m lenites] ──
[Bhatri Rathawi hal mewari_bannoda Bhilali thar] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
[hindi -OAw] ; ── block liquids-flap: Retroflex flapping — single post-vocalic ḍ ḍʰ > ṛ ṛʰ (NIA; liquids blocks round, 2026-09-30). The flap of the Hindi belt, Punjabi-Lahnda-Sindhi, Pahari, Bengali [@flap — second, post-nasal flap after nasal-vowel formation (kuṇḍá > kū̃ṛī, gōvr̩nda > gõṛo)] ──
[hindi -OAw] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ {ḍ ḍʰ}            ; @flap — 8 lects / kaṭāha > H kaṛāhā, kukkuṭá > H kukṛā, *dʰūḍi > H dʰūṛ; geminate ḍḍ stays a stop / (changes.sc flap.ddh.postvocalic.except)
[Kannauji-Central bundeli_atarra Brj SSr] ; ── block nasals-anusvara-hindi: nasals-anusvara-hindi — anusvāra > nasal vowel, Hindi type, nasals theme, blocks round 2026-09-30. NIA: the anusvāra is absorbed as nasality; no lengthening out [@anusvara-vowel — anusvāra > nasal vowel, short before a heavy syllable (blocks/nasals-anusvara-hindi.sc)] ──
[Kannauji-Central bundeli_atarra Brj SSr] V ṁ > $1+nas / _ C Long | _ C V C C | _ C V C V ; @anusvara-vowel @stress — 6 lects / short before a heavy syllable or ≥2 syllables: *laŋgapaṭṭa > Aw lãgoṭ / (changes.sc anusvara-vowel.vm.clongcvcccvcv)
[Kannauji-Central bundeli_atarra Brj SSr] ; ── block nasals-anusvara-vowel: nasals-anusvara-vowel — NIA anusvāra > long nasal vowel before a consonant, nasals theme, blocks round 2026-09-30 (pass 2). The common mainland outcome (Hindi b [@anusvara-vowel — long nasal vowel elsewhere: aŋká > Aw ā̃k, jáŋgʰā > Aw jā̃gʰ, dánta > SSr dā̃t] ──
[Kannauji-Central bundeli_atarra Brj SSr] V ṁ > $1+long+nas / _ C                      ; @anusvara-vowel — 39 lects / píṇḍa > pī̃ḍ, paŋka > pā̃k, dánta > SSr dā̃t / (changes.sc anusvara-vowel.vm.preconsonantal)
[kaithal] ; ── block nasals-anusvara-vowel: nasals-anusvara-vowel — NIA anusvāra > long nasal vowel before a consonant, nasals theme, blocks round 2026-09-30 (pass 2). The common mainland outcome (Hindi b [@anusvara-vowel (blocks/nasals-anusvara-vowel.sc)] ──
[kaithal] V ṁ > $1+long+nas / _ C                      ; @anusvara-vowel — 39 lects / píṇḍa > pī̃ḍ, paŋka > pā̃k, dánta > SSr dā̃t / (changes.sc anusvara-vowel.vm.preconsonantal)
[northwest -D -Khol -L] ; ── block hiatus-medial-nw: Hiatus theme — the Sindhi-Punjabi medial contraction (blocks round, 2026-09-30). The general a(C)a contraction (`hiatus-medial`) and, at the same point in the S [@hiatus — blocks/hiatus-medial-nw.sc; medial a(y)a from stop loss contracts to e (as in OIA-H.sc)] ──
[northwest -D -Khol -L] ; ── block hiatus-medial: Hiatus theme — medial a(C)a contraction (blocks round, 2026-09-30). After the MIA loss of intervocalic stops/y, a-a before a consonant contracts to e and a-ā to ──
[northwest -D -Khol -L] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here ──
[thar kcch khet] a a > e / C _ C                              ; @hiatus — medial a(C)a > e (kadala > kaala > Bhoj kerā, kadalī > kayalī > kelā) / (block hiatus-medial-core)
[thar kcch] a ā > ā                                      ; @hiatus — a-ā hiatus > ā (kumbʰakāra > kumʰār, andʰakāra > anʰār) / (block hiatus-medial-core)
[Buksa Rana Dang DewasDoneDanuwar OAw VB northwest -D -Khol] repeat: {a ā} {a ā} > ā                      ; @hiatus — a/ā hiatus > ā (andʰakāra > andʰaāra > ā̃dʰār, kumbʰakāra > kumʰār, rājana > rāaṇ > rāṇ) / (block hiatus-medial)
[northwest -D -Khol -L] a > ∅ / {i ī u ū ē ō} _                      ; @hiatus — a absorbed after i/u/ē/ō (sūkara > sūar > sūr) / (block hiatus-medial-nw)
[H] V ṁ > $1+nas / _ C Long | _ C V C C | _ C V C V // _ C a n a a # | # C* _ C Short C {a i u} #   ; @anusvara-vowel @stress — no lengthening outside the NIA stressed syllable; but the root syllable before the verbal noun -ana(k)a is stressed (bandʰati > bā̃dʰnā, váñcati > bā̃cnā; 55 better, 4 worse); round 6b Round 7: nor in an OIA trisyllable with a light second syllable and a final short vowel (later lost), whose first syllable is the NIA stressed one, as in the round-7 CL extension: ántara > ā̃tar, *daŋgara > dā̃gar; train ext −0.0003, test −0.0006, 21 better / 10 worse
[H] V ṁ > $1+long+nas / # C* _ C                 ; @anusvara-vowel — first syllable: long nasal vowel (píṇḍa > pī̃ḍ)
[H] ; ── block nasals-anusvara-short: nasals-anusvara-short — anusvāra > short nasal vowel before a consonant (no lengthening), nasals theme, blocks round 2026-09-30 (pass 2). The step after a lect' [@anusvara-vowel — non-initial syllables: short nasal vowel (round 2) (blocks/nasals-anusvara-short.sc)] ──
[H] V ṁ > $1+nas / _ C                           ; @anusvara-vowel — 6 lects / G tʰãbʰvũ (stambʰ-), Or/Halbi short Ṽ before C / (changes.sc anusvara-vowel.vm.preconsonantal.2)
[OAw] V ṁ > $1+long+nas / # C* _ C V (V) #         ; @anusvara-vowel — only in a first-syllable penult, like CL (maṇḍá > māṁḍa; elsewhere short V + ṁ); round 2
[hindi] {i ī} {a ā} > ī / _ #                        ; @hiatus — 10 lects / -ikā/-iya > -ī (gʰaṭikā > gʰaṛī) / (changes.sc hiatus.iiaa.final.2)
[Kannauji-Central] ; ── block finals-aka-i: The -ikā/-ika ending (blocks theme "finals", 2026-09-30): after intervocalic k is lost, -ia/-iā contracts to -ī, which survives apocope (finals-apocope*); -uka  [-ikā > -ī (blocks/finals-aka-i.sc) — replaces `use aka.iaa.final`: -ika/-ikā > -ia/-iā > -ī] ──
[Kannauji-Central] i {a ā} > ī / _ #                            ; @aka @hiatus — 20 lects / -ikā/-ika > -ia > -ī (mwr. kīṭá+ikā > kīḍī, kukkuṭá+ikā > kūkḍī, bʰārika > bʰārī) / (changes.sc aka.iaa.final)
[Kannauji-Central Brj] {a ā} a > o / _ #                            ; @hiatus — 2 lects / -aka > -o (W. Hindi masc. -o: Brj. gʰoṛo) / (changes.sc hiatus.aaa.final.2)
[kaithal H] {u ū} {a ā} > ū / _ #                        ; @hiatus — 2 lects / -uka/-uā > -ū (bālukā > bālū); round 4 / (changes.sc hiatus.uuaa.final)
[nuristani] {k g t d y v b} > ∅ / V _ V                  ; @int-stop-loss @v-loss — 7 lects / ā́gata > *ā, jātá > *ʣā- / (changes.sc int-stop-loss.kgtdyvb)
[Chitwan Bhoj Morang Bote hindi -OAw OSi thar kcch Kt] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here [@hiatus — blocks/hiatus-medial-core.sc; medial a(y)a from stop loss contracts to e (kadalī > kayalī > kelā)] ──
[bihari-tharu -Darai hindi Bhatri DungraBhili Bhilali VB Ko-Ul OSi northwest -D -Khol -khet Kt cam] a a > e / C _ C                              ; @hiatus — medial a(C)a > e (kadala > kaala > Bhoj kerā, kadalī > kayalī > kelā) / (block hiatus-medial-core)
[bihari-tharu -Darai -Majhi Ku Garh N Chil hindi Bhatri DungraBhili Bhilali VB Ko-Ul OSi northwest -D -L Kt PNur mewari_dholpura cam] a ā > ā                                      ; @hiatus — a-ā hiatus > ā (kumbʰakāra > kumʰār, andʰakāra > anʰār) / (block hiatus-medial-core)
[B hindi -OAw] ; ── block finals-aka-contract: -aka > -ā by general hiatus contraction (blocks theme "finals", 2026-09-30): once intervocalic k is lost, every a/ā + a/ā hiatus contracts to ā; word-finally th [a/ā hiatus contraction, -aka > -ā (blocks/finals-aka-contract.sc) — replaces `use aka.aaaa`: (round 4: restored; -aka > -ā) vowels in hiatus after stop loss contract   [pruned for bundeli_atarra: LOO -0.0004]] ──
[eastern -Or -OB hindi -OAw] {a ā} {a ā} > ā                              ; @aka @hiatus — 19 lects / a/ā hiatus contracts to ā (H. *āṇvaka > ānā, āmalaka > ā̃vlā; B. andʰakāra > ā̃dʰār) / (changes.sc aka.aaaa)
[rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewati_akera -Goj] repeat: G $1 > $2                            ; @degem-plain — 11 lects / (changes.sc degem-plain.g.repeat)
[Khandesi PalyaBareli DungraBhili Vasavi Bhilali Noiri RathwiBareli] ; ── block length-cl-first-nc: CL before a homorganic nasal + stop, the anusvāra counterpart of CL before a geminate: a > ā / _ N Stop (dánta > dāt, candrá > cānd), then first-syllable CL and [@degem-lengthen @anusvara-vowel @degem-plain — a > ā before NC, first-syllable CL, plain degemination (blocks/length-cl-first-nc.sc; blocks round 2026-09-30)] ──
[Khandesi PalyaBareli] a > ā / _ Nasal Stop                         ; @anusvara-vowel @degem-lengthen — 12 lects / a lengthened before a retained homorganic nasal + stop (Bhilali dánta > dāt, candrá > cānd) / (changes.sc anusvara-vowel.a.nasalstop)
[Khandesi PalyaBareli DungraBhili Vasavi Bhilali Noiri RathwiBareli] ; ── block length-cl-first: NIA degemination with compensatory lengthening (CL) in the first syllable: V G G > V̄ G / # (C) _, whatever follows, then plain degemination (length-degem-plain ──
[Khandesi PalyaBareli DungraBhili Vasavi Bhilali Noiri RathwiBareli] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Khandesi PalyaBareli DungraBhili Bhilali Noiri RathwiBareli] V GL $2 > $1+long $3 / # (C) _               ; @degem-lengthen — first-syllable CL (B hásta > hāt, sápta > sāt, kárṇa > kān) / (block length-cl-first)
[Khandesi PalyaBareli DungraBhili Vasavi Bhilali Noiri RathwiBareli] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[Khandesi PalyaBareli DungraBhili Vasavi Bhilali Noiri RathwiBareli] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[PalyaBareli Bhilali Noiri RathwiBareli] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[DungraBhili] {ō ṓ} > ū                                   ; @vowel-shift — inherited ō > ū (gōla > gūli, rōṣa > rūṣ), the back partner of ē > ī; it must precede ā > ō (drag chain: +0.009 if placed after) (round 2)
[hal WK] ; ── block length-cl-first: NIA degemination with compensatory lengthening (CL) in the first syllable: V G G > V̄ G / # (C) _, whatever follows, then plain degemination (length-degem-plain [@degem-lengthen @degem-plain — CL in the first syllable (# (C) _), then plain degemination (blocks/length-cl-first.sc; blocks round 2026-09-30)] ──
[hal WK] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[hal WK] V GL $2 > $1+long $3 / # (C) _               ; @degem-lengthen — first-syllable CL (B hásta > hāt, sápta > sāt, kárṇa > kān) / (block length-cl-first)
[hal WK] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[hal WK] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[hal WK] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[K] a > u / C a _ #                              ; @aka @hiatus — the masc. -aka ending, its k lost, becomes -au > -ᵘ (the unscored u-mātrā), which then umlauts the root vowel (kʰaḍaka > kʰorᵘ, kāṇaka > kɔnᵘ, káṇṭaka > konḍᵘ) (round 7)
[K] i ā > ɨ / C _ #                              ; @aka @hiatus — likewise fem. -ikā > -iā > -ᵘ̈ (ɨ, the ü-mātrā), with its own umlaut (kuñcikā > kunzᵘ̈, gʰaṭikā > gürᵘ̈, *āḍikā > ə̄rᵘ̈) (round 7)
[Kund] {k g t c j y v h} > ∅ / V _ V               ; @int-stop-loss @v-loss @h-loss
[pog] ; ── block hiatus-absorb: Hiatus theme — absorption of a by a neighbouring vowel (blocks round, 2026-09-30). After the loss of intervocalic consonants (incl. the NIA v/h loss), short a i [inherited from the Kashmiric sisters (small-lect default, 2026-09-30)] ──
[pog] a > ∅ / V _                                  ; @hiatus — a absorbed after a vowel (kāhala > kaaḷ > kāḷ; -iya > -i after final-vowel loss) / (block hiatus-absorb)
[dod] {k g t d c j y v p} > ∅ / V _ V             ; @int-stop-loss @v-loss
[pog] {k g t d c j y v h b} > ∅ / V _ V           ; @int-stop-loss @v-loss @h-loss
[dardic-north -Sh -Sv -Kho -Ush -Kal] {a i} > ∅ / _ V                              ; @hiatus — 5 lects / (changes.sc hiatus.ai.prevocalic)
[K WK mewari_dholpura] ; ── block hiatus-absorb: Hiatus theme — absorption of a by a neighbouring vowel (blocks round, 2026-09-30). After the loss of intervocalic consonants (incl. the NIA v/h loss), short a i [@hiatus — blocks/hiatus-absorb.sc; a after another vowel absorbed] ──
[K WK mewari_dholpura] a > ∅ / V _                                  ; @hiatus — a absorbed after a vowel (kāhala > kaaḷ > kāḷ; -iya > -i after final-vowel loss) / (block hiatus-absorb)
[mewari_bannoda Bhilali] ; ── block finals-ap-shortening: Apabhraṁśa shortening of final -ā (blocks theme "finals", 2026-09-30): late MIA final long -ā shortens to -a, so that it is later lost with the old short -a by  [Ap. final -ā > -a (blocks/finals-ap-shortening.sc) — replaces `use final-a-short.a`: MIA shortening of final -ā (mū́lā > mūḷ)] ──
[mewari_bannoda Bhilali] ā > a / _ #                                  ; @final-a-short @final-shortening @final-vowel-loss @vowel-shift — 41 lects / final -ā > -a (Bi. ā́rā > ār, kʰáṭvā > kʰāṭ, nāvā́ > nāv) / (changes.sc final-a-short.a)
[Vasavi Bhilali] # ── Bhil post-MIA prefix: first 2 of 4 shared changes (round-7 unification) ──
[K] ∅ > v / # _ {u ū o ō ɔ}                     ; @prothesis — ujjvala > vɔzul, ṓṣṭʰa > vuṭʰ, újjāyatē > vuzun; ɔ added in round 7 (it is the lowered u); placed before the -ᵘ umlaut in round 7, so an umlaut o does not take it (accʰa > oʦʰᵘ, agryá > ogᵘ)
[K] ā > ɔ / _ C (C) (C) u #                      ; @umlaut — before the -ᵘ mātrā ā > ɔ (kāla-ka > kɔlᵘ, jā́la-ka > zɔlᵘ, asāra-ka > asɔrᵘ) … (round 7)
[K] a > u / # C* _ C (C) (C) ɨ #                 ; @umlaut — … before the -ᵘ̈ mātrā a > ü (scored as u: paṭṭa-ikā > püṭᵘ̈, kʰara > kʰürᵘ̈) …
[K] ā > ə̄ / _ C (C) (C) ɨ #                      ; @umlaut — … and ā > ə̄ (ā́rā > ə̄rᵘ̈, kapā́la > kvə̄jᵘ̈, gaṇikā > gə̄ñᵘ̈) …
[K] ɔ > ə̄̆ / _ C (C) (C) ɨ #                      ; @umlaut — … ɔ (from u) > ə̄̆ (*kubba > kə̄̆bᵘ̈, tumba > tə̄̆mbᵘ̈, muṇḍa > mə̄̆nḍᵘ̈) …
[K] {n l} > {ñ j} / _ ɨ #                        ; @palatalisation @umlaut — n and l palatalise before the -ᵘ̈ mātrā of -ikā (tanikā > tüñᵘ̈, jáni-kā > züñᵘ̈, jā́la-ikā > zə̄jᵘ̈, kīla > kījᵘ̈) (round 7)
[K] ɛ > y o / # C* _ C (C) (C) u #               ; @umlaut — … ɛ > yo before -ᵘ (cʰinná > cʰyonᵘ, *ṭikka > ṭyokᵘ, nímba > nyombᵘ) …
[K] i > y u / C _ C (C) u #                      ; @umlaut — … and i > yu before -ᵘ, the partner of ɛ > yo (antimá-ka > andyumᵘ, aṣṭamá-ka > aiṭʰyumᵘ, dákṣiṇa-ka > dacʰyunᵘ) (round 7)
[K] a > o / # C* _ C (C) (C) u #                 ; @umlaut — … and a root (first-syllable) a > o before -ᵘ (accʰa > oʦʰᵘ, kʰaḍaka > kʰorᵘ, paṭṭa > poṭᵘ); a non-initial a becomes u instead (next rule) (round 7)
[K] a > u / V C (C) _ C (C) u #                  ; @umlaut — … while a non-initial a of the final syllable becomes u before -ᵘ (the -anaka infinitive: ʦal-anaka > ʦalun; akṣára-ka > acʰur, kukkuṭá-ka > kɔkur). Replaces the round-6 unconditioned umlaut.vccc after apocope: bare -a nouns keep a (ā́sana > āsan, ántara > andar, mārjana > māzan, sāhasa > sāhas); ext −0.0134/−0.0137, plain +0.0033 is the bare-vs-extended artifact (the -ur/-ul forms carry the -ᵘ of -aka: kɔtʰulᵘ, zɔzurᵘ, kanguvᵘ) (round 7)
[Phal] {g t d c j y v} > ∅ / V _ V                 ; @int-stop-loss @v-loss — intervocalic k survives
[Phal] V > $1+stress / _ C* V C* #                  ; @accent-placement — stress on the (old) penult, placed before hiatus contraction and final-vowel loss (round 3)
[Phal] V > $1+stress / # C* _ C* #                  ; @accent-placement — monosyllables carry the accent (tone) (round 3)
[Phal] a i > e / _ #                                ; @hiatus — final -ai (from -ati after t loss) contracts to -e: the 3sg present > Liljegren's citation -e (bʰéśe, kríne) (round 5)
[Phal] {a i ā ō} > ∅ / _ V                         ; @hiatus
[dardic-north -Kho] ; ── Clusters II ─────────────────────────────────────────────────────────
[Phal K] ; ── block mia-dardic-degem: Dardic reduction of stop clusters and geminates to a single consonant (Shina, Kashmiri; blocks theme "mia", pass 2, 2026-09-30): stop + stop > the second, then  [@cluster-simplify @degem-plain — stop + stop > the second, geminates simplify (blocks/mia-dardic-degem.sc; blocks round 2026-09-30, pass 2)] ──
[Phal K] Stop Stop > $2                               ; @cluster-simplify @degem-plain — 17 lects / stop + stop > the second (sapta > sat-, bʰakta > bʰat-, dugdʰa > dud-) / (changes.sc degem-plain.stopstop)
[Phal K] C $1 > $1                                    ; @cluster-simplify @degem-plain — 18 lects / geminates simplify (kaṭṭa > kaṭ-, pakka > pak-, *tt < rt > t) / (changes.sc degem-plain.c)
[Sh Sv Ush] C $1 > $1                                    ; @cluster-simplify @degem-plain — 18 lects / other geminates (ll, nn, śś) / (changes.sc degem-plain.c)
[Kho] ; ── block liquids-d-lateral: Lateral outcome of the intervocalic retroflex stops — ṭ ḍ > ḷ (liquids blocks round, 2026-09-30). Where the retroflex stops became a lateral rather than a flap: [@int-retroflex @flap] ──
[Kho] {ṭ ḍ} > ḷ / V _ V                            ; @flap @int-retroflex — 4 lects / kūṭa > Si kuḷa, guḍá > Md guḷa, kʰēṭa > Kho kʰéḷi / (changes.sc flap.td.intervocalic.3)
[Kho] t > r / V _ V                               ; @t-l — intervocalic t > r (śatá > śor)
[Kho] {k g} > ∅ / i _ a #                          ; @int-stop-loss — -ika > -i (*kṣudʰikā > ʦ̣ʰui), not -iɣ
[Kho] {k g} > ∅ / V C* u _ {a ā} #                 ; @int-stop-loss @aka — suffixal -uka > -u, like -ika > -i (*upēṣukā > veṣú); root k stays ɣ (yū́kā > źoɣ) (round 4)
[Kho] {k g} > ɣ / V _ V                           ; @int-spirant
[Kho] {p b d v y h c j} > ∅ / V _ V               ; @int-stop-loss
[as-ll dardic-north] Stop Stop > $2                               ; @cluster-simplify @degem-plain — 17 lects / (changes.sc degem-plain.stopstop)
[Sh] ŋ {k g gʰ} > g                              ; @cluster-simplify — but ŋg > g, after lenition (áŋgāra > agār)
[dardic-north -Kho -Kund -K -Kal] ; ── block lar-deaspiration-sibilant: Shina-type loss of aspiration in ṭʰ/tʰ after a sibilant before a vowel (finally they keep it): Shina, Phalura, Ushojo. (Romani has its own post-sibilant loss in [@deaspiration — ṭʰ tʰ after a sibilant lose aspiration before a vowel (blocks/lar-deaspiration-sibilant.sc)] ──
[dardic-north -Kho -Kund -K -Kal] {ṭʰ tʰ} > {ṭ t} / {s ṣ} _ V                  ; @deaspiration — 3 lects / aŋguṣṭʰá > Phal. agúṭo, ṓṣṭʰa > ōṭo / (changes.sc deaspiration.thth.ssv)
[dardic-north -Kalk -Kho -Phal -K -Kal] {s ṣ} > ∅ / _ Stop                           ; @cluster-simplify @s-stop-loss — 3 lects / s/ṣ + stop > stop, after lenition (hásta > hāt, aŋguṣṭʰá > agúṭo) / (changes.sc s-stop-loss)
[PauriBareli mewati_akera] G_std $1 > $2                                ; @cluster-simplify @degem-plain — 83 lects / (changes.sc degem-plain.g)
[Shum] {v y} > ∅ / V _ V                            ; @v-loss — 9 lects / intervocalic v, y lost / (changes.sc v-loss.vy.intervocalic)
[srk mewari_basad dhundari_bamore] ; ── block finals-aka-iu: The -ikā/-ika and -uka endings (blocks theme "finals", 2026-09-30): after intervocalic k is lost, -ia and -ua contract to long -ī and -ū, which survive apocope  [-ikā/-uka > -ī/-ū (blocks/finals-aka-iu.sc) — replaces `use aka.iiuuaa.final`: -ikā/-iya > -iā > -ī, -uka > -uā > -ū] ──
[srk mewari_basad dhundari_bamore] {i ī u ū} {a ā} > {ī ī ū ū}@1 / _ #          ; @aka @hiatus — 8 lects / -ikā/-uka > -ia/-ua > -ī/-ū (G. kárṇikā > kānī, kuñcikā > kũcī, kārpaṭika > kāpṛī) / (changes.sc aka.iiuuaa.final)
[Shum Wot Dm Tor Tir Pas] {i ē} > ∅ / a _ #                            ; @hiatus — 4 lects / final -ai/-aē (from -ati/-atē after t loss) loses its second vowel before apocope, so the stem-final consonant stays lenited (Pas *avalōḍayati → oroṛ); round 5 / (changes.sc hiatus.ie)
[Pas] {a ā ē ō i u} > ∅ / V C* _ #                 ; @final-vowel-loss — 2 lects / (round 2: {a ā ē ō i u} generalised; polysyllables only, also after a vowel (-ikā > -iā > -i) monosyllables keep their vowel; −0.001 to −0.008 in all 12 lects) / (changes.sc final-vowel-loss.aaeoiu.final-polysyllabic)
[Wot] a > ∅ / a _ #                               ; @hiatus — final -aa (from -aya-/-aka- after consonant loss) contracts before apocope; round 6
[Gaw] ; ── block hiatus-vh-loss: Hiatus theme — loss of intervocalic v and h (blocks round, 2026-09-30). The NIA loss of the weak intervocalic continuants that creates a second round of hiatus: [@v-loss @h-loss — blocks/hiatus-vh-loss.sc; intervocalic h] ──
[Gaw] ; ── block hiatus-v-loss: Hiatus theme — loss of intervocalic v (blocks round, 2026-09-30). The NIA loss of intervocalic v (incl. < MIA p) that opens a second round of hiatus (dīpa > dīv ──
[Gaw] h > ∅ / V _ V                                ; @h-loss — intervocalic h (< MIA aspirates) lost (lṓhita > làu, gōdʰūma > gɔū̃) / (block hiatus-vh-loss)
[Gaw Shum Wot Dm Tor] ; ── block finals-apocope-postcons: Apocope after a consonant in polysyllables (blocks theme "finals", 2026-09-30): final -a -ā -i -u are lost only after a consonant (V C* C _ #), so vowels in hia [apocope after a consonant (blocks/finals-apocope-postcons.sc) — replaces `use final-vowel-loss.aaiu.vcc`: monosyllables keep their vowel; −0.001 to −0.008 in all 12 lects)] ──
[Gaw Shum Wot Dm Tor] {a ā i u} > ∅ / V C* C _ #                   ; @final-vowel-loss — 2 lects / final vowels lost after a consonant (Shum. *ārta > āt, kālá > kāl, kukkuṭá > kukuṛ) / (changes.sc final-vowel-loss.aaiu.vcc)
[northwest -D -thar -S -Khol -khet] a ī > ī / _ #                                ; @hiatus — 11 lects / -aī > -ī after consonant loss (ālaya-ikā > ālī, āpad-ikā > āī; round 6) / (changes.sc hiatus.ai.final)
[P] ; use syncope.a.cna                            ; @syncope — the stem vowel is lost before the heavy final -ṇā: verbal noun -anaka > -aṇaa > -aṇā > -ṇā (aṭṭanaka > aṭṭṇā, aŋkanaka > aŋgṇā; round 5)   [round 7: removed, ext-LOO −0.0002]
[hindi] {ō ē o e} {a i} > {ō ē o e} / _ C            ; @hiatus — 6 lects / ō/ē + a/i in hiatus contracts (*cʰōkara > cʰorā, jyautiṣika > josī); round 6 / (changes.sc hiatus.oeoeai.preconsonantal)
[H] i ī > ī                                      ; @hiatus — 5 lects / i + ī left in hiatus by stop loss contracts, as in H and Ku (kr̩ta-ikā > kiī > kiṛī/kī, -viṁśati-ikā > -vī; pred -iī 33, gold 0; hill-climb round 5) / (changes.sc hiatus.ii)
[poth] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc) — before the step that frees cluster m (rm > mm > m, Dardic rm > m), so only original single m lenites] ──
[poth] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
[poth] a > ∅ / C _ ṇ ā #                            ; @syncope — 2 lects / stem vowel lost before the heavy final -ṇā, as in P (*hiss > hisṇā; round 6 second pass) / (changes.sc syncope.a.cna)
[OG] V G $2 > $1+long $3 / # C* _                 ; @degem-lengthen — 5 lects / first syllable, whatever the word length (OG ūjama, āgi) / (changes.sc degem-lengthen.vg.first-syllable)
[OG OMarw] V ṁ > $1+long ṁ / # C* _ C // _ C Long C     ; @anusvara-vowel @x-rg-weight — 2 lects / first syllable only, not before a heavy long-vowel syllable (as G) / (changes.sc anusvara-vowel.vm.cc.except)
[mewari_basad OMarw] ; ── block hiatus-medial-west: Hiatus theme — the Rajasthani medial contraction (blocks round, 2026-09-30). Where the final -aa has already become -o (the finals), every other a/ā + a/ā hiatu [@hiatus — blocks/hiatus-medial-west.sc; medial a/ā + a/ā contract to ā (andʰakāra > andʰaāra > ãdʰārɔ); round 7] ──
[mewari_basad] {a ā} {a ā} > ā / _ !#                       ; @hiatus — non-final a/ā hiatus > ā (andʰakāra > andʰaāra > ãdʰārɔ, ãdʰārũ) / (block hiatus-medial-west)
[mewari_basad OMarw] a {o u} > o                                  ; @hiatus — a+o/u > o (hastakūṭa > hatʰaüḍa > hathoḍo) / (block hiatus-medial-west)
[Gaw Chil Dm rajasthani-gujarati -mewari_basad -OG -Nimadi -Marw -mewari_dholpura -dhundari_bamore -OMarw -Goj jaun] ; ── block finals-apocope-long: NIA apocope with long -ā, in polysyllables (blocks theme "finals", 2026-09-30): final -a -ā -i -u are lost after a preceding vowel (V C* _ #), so OIA monosyllab [apocope incl. -ā, in polysyllables (blocks/finals-apocope-long.sc) — replaces `use final-vowel-loss.aaiu`: incl. fem. -ā] ──
[Gaw Chil Dm rajasthani-gujarati -mewari_basad -OG -Nimadi -Marw -mewari_dholpura -dhundari_bamore -OMarw -Goj jaun] {a ā i u} > ∅ / V C* _ #                     ; @final-vowel-loss — 3 lects / final vowels lost in polysyllables (Marw. adyá > āj, aṣṭā́ > āṭʰ, kárṇa > kān) / (changes.sc final-vowel-loss.aaiu.final-polysyllabic)
[mewari_basad] a > ∅ / V C _ C ī #                         ; @syncope — medial a lost before a final -Cī (músala > mūsḷī)
[OG] ; ── block finals-aka-i: The -ikā/-ika ending (blocks theme "finals", 2026-09-30): after intervocalic k is lost, -ia/-iā contracts to -ī, which survives apocope (finals-apocope*); -uka  [-ikā > -ī (blocks/finals-aka-i.sc) — replaces `use aka.iaa.final`] ──
[OG] i {a ā} > ī / _ #                            ; @aka @hiatus — 20 lects / -ikā/-ika > -ia > -ī (mwr. kīṭá+ikā > kīḍī, kukkuṭá+ikā > kūkḍī, bʰārika > bʰārī) / (changes.sc aka.iaa.final)
[mewari_basad OG OMarw] repeat: G $1 > $2                            ; @degem-plain — 11 lects / (changes.sc degem-plain.g.repeat)
[Goj] ; ── block liquids-flap: Retroflex flapping — single post-vocalic ḍ ḍʰ > ṛ ṛʰ (NIA; liquids blocks round, 2026-09-30). The flap of the Hindi belt, Punjabi-Lahnda-Sindhi, Pahari, Bengali [@flap] ──
[Goj] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ {ḍ ḍʰ}            ; @flap — 8 lects / kaṭāha > H kaṛāhā, kukkuṭá > H kukṛā, *dʰūḍi > H dʰūṛ; geminate ḍḍ stays a stop / (changes.sc flap.ddh.postvocalic.except)
[P] {a i} > ∅ / V C (C) _ C {ā ī ū} #             ; @syncope — medial a i lost before a heavy final syllable, also after a geminate; after the flap (hariṇikā > harnī, śēkʰara > sihrā, *carmaḍaka > camṛā; round 6)
[P] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[P] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[S] a {i u} > {e o}@2 / _ C                    ; @hiatus — medial ai au from consonant loss contract before a consonant (*satēra > seru, prayukta > potu, yakr̩t > jero)
[dardic-south -Gowro -Chil -bhatr -Tir D] {a i} > ∅ / V _                              ; @hiatus — 11 lects / second vowel of a hiatus dropped (kʰādati > kʰā(n)a, final vowels otherwise kept) / (changes.sc hiatus.ai.postvocalic)
[marathi-bhil -Bhatri -DungraBhili -hal -M -mewari_bannoda -OM -VB -Ko-Ul] u > a / V C* _ C C* V                        ; @vowel-shift — 8 lects / medial (mostly penultimate) u reduced to a before a consonant (kukkuṭa > kūkaḍo type); accent-independent (round 2). Round 4: the C spares -uka > -ua (ext-aware) / (changes.sc vowel-shift.u.vcccv)
[Vasavi Bhilali] # ── end of shared prefix ──
[hal] ; ── block nasals-anusvara-nasal: nasals-anusvara-nasal — NIA anusvāra > short nasal vowel (no lengthening), nasals theme, blocks round 2026-09-30 (pass 2). West Pahari (bhad, jaun, khash, ktg,  [@anusvara-vowel (blocks/nasals-anusvara-nasal.sc)] ──
[hal] V ṁ > $1+nas                                 ; @anusvara-vowel — 13 lects / áŋgāra > ãgār (chain), bʰaṇṭākī > bʰãṭāī (chain) / (changes.sc anusvara-vowel.vm)
[kc] {ḍ ḍʰ} > {ṛ ṛʰ} / V _                        ; @flap — 33 lects / (changes.sc flap.ddh.postvocalic)
[pan] ; use s-h.ss.postvocalic   — dropped round 6 third pass (ext −0.0022, test =)
[dog kgr cam pan] ; ── block finals-aka-east-apocope: -aka > -ā followed by apocope (blocks theme "finals", 2026-09-30): the masculine -aa (< -akaḥ) contracts to -ā (finals-aka-east), then final short -a -i -u are  [-aka > -ā, then apocope (blocks/finals-aka-east-apocope.sc) — replaces `use aka.aa.final`; `use final-vowel-loss.aiu`: -aka > -aa > -ā (masc. direct case) / short final vowels lost; the contracted -ā stays (round 4)] ──
[dog kgr cam pan] ; ── block finals-aka-east: The -aka ending in the centre and east, final only (blocks theme "finals", 2026-09-30): the masculine nominative -akaḥ (MIA -aa) contracts to -ā, which is why t ──
[dog kgr pan] a a > ā / _ #                                ; @aka @hiatus — 11 lects / -aka > -aa > -ā (kul. kāṇá+aka > kāṇā, ūná+aka > uṇā, uparōdʰa+aka > urā) / (changes.sc aka.aa.final)
[dog kgr cam pan] ; ── block finals-apocope: NIA apocope (blocks theme "finals", 2026-09-30): loss of the short final vowels -a -i -u that the Apabhraṁśa stage left word-final (after finals-ap-shortening w ──
[dog cam pan] {a i u} > ∅ / V C* _ #                       ; @final-vowel-loss — 3 lects / final short -a -i -u lost in polysyllables (H. akṣára > ākʰar, agní > āg, áŋga > ā̃g) / (changes.sc final-vowel-loss.aiu.final-polysyllabic)
[Majhi Dotyali Bshk Shum Wot Tir Khol Rj west-pahari -bhal -kiuth -kgr -cam] ; ── block hiatus-vh-loss: Hiatus theme — loss of intervocalic v and h (blocks round, 2026-09-30). The NIA loss of the weak intervocalic continuants that creates a second round of hiatus: [@v-loss @h-loss — blocks/hiatus-vh-loss.sc] ──
[Majhi Dotyali Bshk Shum Wot Tir Khol Rj west-pahari -bhal -kiuth -kgr -cam] ; ── block hiatus-v-loss: Hiatus theme — loss of intervocalic v (blocks round, 2026-09-30). The NIA loss of intervocalic v (incl. < MIA p) that opens a second round of hiatus (dīpa > dīv ──
[Majhi Dotyali Bshk Gaw Shum Wot Tir Khol bagri_fatehabad Rj west-pahari -bhal -kiuth -kgr -cam] v > ∅ / V _ V                                ; @v-loss — intervocalic v (incl. < p) lost (dīpa > dīva > diyā, kūpa > kuā, kapila > kailo) / (block hiatus-v-loss)
[Majhi Dotyali Bshk Shum Wot Dm Tir Khol Rj west-pahari -bhal -kiuth -kgr -cam] h > ∅ / V _ V                                ; @h-loss — intervocalic h (< MIA aspirates) lost (lṓhita > làu, gōdʰūma > gɔū̃) / (block hiatus-vh-loss)
[cam] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina [@degem-plain — NIA plain degemination (GL = G∪G2∪G3) (blocks/length-degem-plain.sc; blocks round 2026-09-30)] ──
[cam] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[cam] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[cam kul pan] {i u} {a ā} > {ī ū} / _ #                    ; @hiatus — 4 lects / -ikā/-uka > -iā/-ua > -ī/-ū (round 4) / (changes.sc hiatus.iuaa.final)
[cur] a a > ā / _ #                                ; @hiatus — 17 lects / -aka > -aa > -ā (ghōṭaka > gʰoṛā; extension-aware score 0.3725 → 0.3485 in Dang) / (changes.sc hiatus.aa.final.2)
[Bhilali cam] V G_std $2 > $1+long $3 / _ V #              ; @degem-lengthen — 12 lects / hattʰa > hātʰ / (changes.sc degem-lengthen.vg.v)
[Bhilali] ; ── block hiatus-medial: Hiatus theme — medial a(C)a contraction (blocks round, 2026-09-30). After the MIA loss of intervocalic stops/y, a-a before a consonant contracts to e and a-ā to [@hiatus — blocks/hiatus-medial.sc; round 6: interior a(C)a left by consonant loss > e] ──
[Bhilali] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here ──
[Bhilali] repeat: {a ā} {a ā} > ā                      ; @hiatus — a/ā hiatus > ā (andʰakāra > andʰaāra > ā̃dʰār, kumbʰakāra > kumʰār, rājana > rāaṇ > rāṇ) / (block hiatus-medial)
[OM] + > ∅                                       ; @notation — drop the root/suffix shield
[M OM Ko-Ul] ; ══ NIA (Marathi) ════════════════════════════════════════════════════════
[M] V a {i e ē} > ṇ ẽ / _ #                     ; @morph @hiatus — -ayati, -āyatē, -īyatē verbs cited as -ṇẽ infinitives
[M] a {i ē} > ṇ ẽ / C _ #                       ; @morph @hiatus — -ati verbs cited as -ṇẽ (kʰánati > kʰaṇṇẽ)
[M] i {a ā} > ī                                 ; @hiatus — -ikā > -iā > -ī (also word-internally)
[Rathawi hal M] u {a ā} > ū / _ #                            ; @hiatus — 3 lects / round 4: -uka > -ua > -ū / (changes.sc hiatus.uaa.final)
[DungraBhili mewari_bannoda] ; ── block nasals-anusvara-bhil: nasals-anusvara-bhil — lengthening before NC, then anusvāra > nasal vowel, Bhil/Halbic type, nasals theme, blocks round 2026-09-30. a is lengthened before a ret [@anusvara-vowel — a > ā before NC, ṁ > long nasal vowel (blocks/nasals-anusvara-bhil.sc)] ──
[DungraBhili mewari_bannoda] ; ── block nasals-nc-lengthen: nasals-nc-lengthen — a > ā before a retained homorganic nasal + stop, Bhil/Halbic and Adivasi Oriya, nasals theme, blocks round 2026-09-30 (pass 2). nasals-anus [@anusvara-vowel @degem-lengthen — dánta > DungraBhili ḍāṭ, candrá > mewari_bannoda cānd] ──
[DungraBhili mewari_bannoda] ; ── block nasals-anusvara-vowel: nasals-anusvara-vowel — NIA anusvāra > long nasal vowel before a consonant, nasals theme, blocks round 2026-09-30 (pass 2). The common mainland outcome (Hindi b [@anusvara-vowel — dánta > mewari_bannoda dā̃t] ──
[DungraBhili mewari_bannoda] V ṁ > $1+long+nas / _ C                      ; @anusvara-vowel — 39 lects / píṇḍa > pī̃ḍ, paŋka > pā̃k, dánta > SSr dā̃t / (changes.sc anusvara-vowel.vm.preconsonantal)
[eastern -Or -OB -B] ; ── block finals-akam-east: Eastern -akam vs -aka (blocks theme "finals", 2026-09-30): after intervocalic k is lost, the neuter -aã (< -akam) loses its nasal vowel to -a, while the masculi [-aã > -a, -aa > -ā (blocks/finals-akam-east.sc) — replaces `use final-nasal.aa`; `use aka.aaaa`: neuter -aaṁ (< -akam) > -a, against masc. -aa > -ā (as in A.) / andʰakāra > andʰaāra > ā̃dʰār] ──
[eastern -Or -B] a ã > a / _ #                                ; @final-nasal @hiatus — 7 lects / neuter -aã (< -akam) > -a (A. úkʰya+āpitavyakam > ukʰāiba, kaṣati+āpitavyakam > kahāiba) / (changes.sc final-nasal.aa)
[eastern -Or -OB -B] ; ── block finals-aka-contract: -aka > -ā by general hiatus contraction (blocks theme "finals", 2026-09-30): once intervocalic k is lost, every a/ā + a/ā hiatus contracts to ā; word-finally th ──
[Ko-Ul] V a {i e ē} > t ā / _ #                     ; @morph @hiatus — -ayati, -āyatē verbs cited in the present -tā
[mewari_bannoda] a {a ā} > o / _ #                            ; @hiatus — 11 lects / round 4: -aka > -aa > -o (masc.) / (changes.sc hiatus.aaa.final)
[cur] {i u} {a ā} > {ī ū} / _ #                    ; @hiatus — 4 lects / -ika/-ikā, -uka > -ī, -ū (contraction with length; round 2) / (changes.sc hiatus.iuaa.final)
[Bhoj Sunha Mth Darai Majhi Bi Bote KochilaTharu AdivasiOriya MH KS hindi -OAw marathi-bhil -M -OM -VB -Ko-Ul thar poth L khet srk WK Marw mewati_jhambaus G Goj cur kul] ; ── block finals-apocope: NIA apocope (blocks theme "finals", 2026-09-30): loss of the short final vowels -a -i -u that the Apabhraṁśa stage left word-final (after finals-ap-shortening w [NIA apocope (blocks/finals-apocope.sc) — replaces `use final-vowel-loss.aiu`] ──
[Bhoj Sunha Mth Darai Majhi Bi Bote KochilaTharu AdivasiOriya MH KS hindi -OAw marathi-bhil -M -OM -VB -Ko-Ul thar poth L khet srk mewari_basad WK Marw mewati_jhambaus G Goj kgr cur kul] {a i u} > ∅ / V C* _ #                       ; @final-vowel-loss — 3 lects / final short -a -i -u lost in polysyllables (H. akṣára > ākʰar, agní > āg, áŋga > ā̃g) / (changes.sc final-vowel-loss.aiu.final-polysyllabic)
[marathi-bhil -DungraBhili -M -OM -Ko-Ul] ; ── block finals-syncope-long: Syncope before a final long-vowel syllable (blocks theme "finals", 2026-09-30): medial a after a single consonant is lost before a final syllable with -o -ā -ī  [syncope before a final long vowel (blocks/finals-syncope-long.sc) — replaces `use syncope.a.vccoaiu`: round 4: medial a lost before a final long-vowel syllable (-aḍaka > -ḍo, -araka > -ro)] ──
[marathi-bhil -DungraBhili -M -OM -Ko-Ul] a > ∅ / V C _ C {o ā ī ū} #                  ; @syncope — 4 lects / medial a lost before a final long-vowel syllable (Bhili kukkuṭá+aka > kūkḍo, jūrṇá+allaka > jūnlo) / (changes.sc syncope.a.vccoaiu)
[N] m > ũ / ā _ a #                              ; @nasal-vocalisation — single intervocalic m after ā in a final syllable becomes the nasal glide ũ (āmá > āũ, vā́ma > bāũ, śyāmá > sāũ); geminate mm (kárman > kām) is untouched; round 7
[bihari-tharu -Chitwan -Bhoj -Morang -Darai] ; ── block hiatus-medial-east: Hiatus theme — the eastern Bihari-Tharu medial contraction (blocks round, 2026-09-30). The core contraction (`hiatus-medial-core`: a(C)a > e between consonants, [@hiatus — blocks/hiatus-medial-east.sc; round 7: the a-ā hiatus left by stop loss contracts to ā (kaṇṭaka-aka > kā̃ṭā), as in 15 sister lects] ──
[bihari-tharu -Chitwan -Bhoj -Morang -Darai] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here ──
[Bote] a a > e / C _ C                              ; @hiatus — medial a(C)a > e (kadala > kaala > Bhoj kerā, kadalī > kayalī > kelā) / (block hiatus-medial-core)
[Majhi Bote] a ā > ā                                      ; @hiatus — a-ā hiatus > ā (kumbʰakāra > kumʰār, andʰakāra > anʰār) / (block hiatus-medial-core)
[bihari-tharu -Chitwan -Bhoj -Morang -Darai] a a > e / _ C                                ; @hiatus — a-a > e before a consonant in any position (āgatalla > āaal > āel, kadala > kerā) / (block hiatus-medial-east)
[MagahiNepal] a a > e / _ C                                ; @hiatus — 15 lects / medial a-a hiatus (after stop loss) contracts to e (kadala > kaala > kerā) / (changes.sc hiatus.aa.preconsonantal)
[Or] ; [round 7: removed legacy @morph `∅ > ā / C _ #` + `if:verb a {i ē} > i b ā / C _ #` + `if:verb ā {i ē} > i b ā / _ #` — neutral removal (jointly: ext −0.0000 / −0.0002, verb =; plain +0.0370 / +0.0441, the bare headword: -ā/-ibā citations come from the extension and verb menus). Kept `V a {i ē} > i b ā / C _ #`: removing it fails ext (+0.0001 train, +0.0003 test)]
[Or] V a {i ē} > i b ā / C _ #                   ; @morph @hiatus — -Vyati/-ayati verbs cited as -ibā infinitives
[Or] ; ── block finals-aka-contract: -aka > -ā by general hiatus contraction (blocks theme "finals", 2026-09-30): once intervocalic k is lost, every a/ā + a/ā hiatus contracts to ā; word-finally th [a/ā hiatus contraction, -aka > -ā (blocks/finals-aka-contract.sc) — replaces `use aka.aaaa`: andʰakāra > andʰaāra > ā̃dʰār] ──
[Or] {a ā} {a ā} > ā                              ; @aka @hiatus — 19 lects / a/ā hiatus contracts to ā (H. *āṇvaka > ānā, āmalaka > ā̃vlā; B. andʰakāra > ā̃dʰār) / (changes.sc aka.aaaa)
[Or] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[Or] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[Or B] {i ī} > ∅ / {i ī} _                          ; @hiatus — 2 lects / ii contracted after consonant loss, as in Or. (aśītí > āsi, the -śīti numerals; dīpikā-type -iyi) / (changes.sc hiatus.ii.ii)
[bihari-tharu -Kathoriya -Majhi -Bote Garh Or OM VB thar kcch WK Rj WPah] ; ── block liquids-lateral-retroflexion: Lateral retroflexion — single intervocalic l > ḷ (late MIA / early NIA; liquids blocks round, 2026-09-30). Stage 1 of the staged liquid history: a broad allopho [@l-retroflex — stage 1: single intervocalic l > ḷ] ──
[bihari-tharu -Chitwan -Rana -KochariyaEastDanuwar -Kathoriya -Majhi -Bote Garh Or thar WK Rj WPah] l > ḷ / V _ V                                ; @l-retroflex — 32 lects / kāla > M kāḷā, G kāḷũ, śāla > Or sāḷa / (changes.sc l-retroflex)
[bihari-tharu Ku N Dotyali eastern -Or hindi Bhatri Khandesi DungraBhili hal PauriBareli Vasavi Noiri northwest -P OG hadothi_kelwada mewati_jhambaus OMarw Goj] ; ── block liquids-rhotacism: Rhotacism of the retroflex lateral — ḷ > r (liquids blocks round, 2026-09-30). Stage 2 of the staged liquid history, narrower than stage 1 (blocks/liquids-later [@l-r — single intervocalic l > r (pʰála > pʰar, kajjala > kājar)] ──
[bihari-tharu Ku N Dotyali eastern -Or hindi Bhatri Khandesi DungraBhili hal PauriBareli Vasavi Noiri northwest -P OG hadothi_kelwada mewati_jhambaus OMarw Goj] ḷ > r                                        ; @l-r — kāla > *kāḷa > Brj kāro, pʰála > Bhoj pʰar, galati > S ɠaraṇu / (block liquids-rhotacism)
[Or] V m V > $1 $3+nas                     ; @m-v — intervocalic m lost, its nasality carried by the following vowel (kumāra > kuā̃ra, āmalaka > ãũḷā)
[Sh Sv pog Kund Kal] ; ── block mia-shina-clusters: Shina/Chitrali/Kashmiri late cluster simplification (blocks theme "mia", 2026-09-30): r lost before a consonant after a vowel, dental + y > palatal, C + y/v > C [@assim-r @palatal-Cy @cluster-simplify — postvocalic rC > C, t d dʰ + y > c j j, C + y/v > C (blocks/mia-shina-clusters.sc; blocks round 2026-09-30)] ──
[dardic-north -Kalk -Kho -Phal -K -dod] r > ∅ / V _ C                                ; @assim-r @cluster-simplify — 10 lects / postvocalic r + C > C (karma > kam, sarpa > sap, cárma > cam) / (changes.sc assim-r.r.vc)
[dardic-north -Kalk -Kho -Ush -dod] {t d dʰ} y > {c j j}                         ; @palatal-Cy — 5 lects / t d dʰ + y > c j j (satya > sac, adya > aj, madʰya > maj) / (changes.sc palatal-Cy.tddhy)
[Sh pog Kal] C {y v} > $1                                 ; @cluster-simplify — 18 lects / C + y/v > C (dvāra > dar, kanyā > kan, pakva > pak) / (changes.sc cluster-simplify.cyv)
[bihari-tharu -Rana -Dang -Morang -KochariyaEastDanuwar Ku Garh N Dotyali dardic-north -Kalk -Kho -Phal -Kal dardic-south -Mai -Tir -Pas eastern hindi marathi-bhil northwest -OP -P rajasthani-gujarati -OG -G west-pahari -bhal] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc) — before the step that frees cluster m (rm > mm > m, Dardic rm > m), so only original single m lenites] ──
[bihari-tharu -Rana -Dang -Morang -KochariyaEastDanuwar Ku Garh N Dotyali dardic-north -Kalk -Kho -Phal -Kal dardic-south -Mai -Tir -Pas eastern hindi marathi-bhil northwest -OP -P rajasthani-gujarati -OG -G west-pahari -bhal] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
[OP] v > ∅ / ā _ {u i} #                       ; @v-loss — v (< p, v, m) lost between ā and the final vowel (ārāma > ārāu, tāpa > tāu; round 6)
[P L] G $1 > $2 / {ā ī ū ē ō} _                    ; @degem-plain — 5 lects / geminates simplify after a long vowel (vārttā > vāt) / (changes.sc degem-plain.g.aiueo)
[OP P kcch poth srk awan] C > ∅ / Long _ $1                            ; @degem-plain — 5 lects / a geminate after a long vowel is simplified (a heavy syllable keeps at most two morae), as in bhad (kārttiká > kāti, cʰattvará > cʰā-, pārśvá > pāś-; 4 better, 0 worse; test ext −.0027; round 7) / (changes.sc degem-plain.c.long1)
[northwest -D -thar -Khol -khet] G_std $1 > $2 / a {u i} _                    ; @degem-plain — 3 lects / geminates also simplify after a secondary diphthong au ai, which counts as long (caturdvāra > caubārā, *kapalla > kaul; round 6 second pass) / (changes.sc degem-plain.g.aui)
[srk] ; ── block nasals-m-nasal: nasals-m-nasal — intervocalic m > nasalisation of the preceding vowel, nasals theme, blocks round 2026-09-30 (pass 2). Sindhi/Siraiki/Pothwari and some West Pah [@m-v — intervocalic m > nasalisation (as in Sindhi) (blocks/nasals-m-nasal.sc)] ──
[srk] V m > $1+nas / _ V                           ; @m-v — 7 lects / nā́man > nā̃, kumārī > kũāri / (changes.sc m-v.vm.prevocalic)
[Chitwan Dang DewasDoneDanuwar Morang KochariyaEastDanuwar Majhi Bote KochilaTharu] ; ── block nasals-nd-nn-tharu: nasals-nd-nn-tharu — NIA nasal + voiced stop > geminate nasal, eastern Tharu/Danuwar type (Bengali-like), nasals theme, blocks round 2026-09-30 (pass 2). Follow [@nasal-assim — nasal + voiced stop > geminate nasal (eastern Tharu, Bengali-type: lamba > nāma, bandʰá > bān) (blocks/nasals-nd-nn-tharu.sc)] ──
[Chitwan Dang DewasDoneDanuwar Morang KochariyaEastDanuwar Majhi Bote KochilaTharu] Nasal {g j ḍ d b gʰ jʰ ḍʰ dʰ bʰ} > $1 $1     ; @nasal-assim — 5 lects / lamba > Majhi lāmo, Morang nāma, bandʰá > Morang bān / (changes.sc nasal-assim.nasalgjddb)
[bihari-tharu -DewasDoneDanuwar -Darai -Majhi -Bote] ; ── block length-cl-disyllable-closed: Bihari-Tharu CL: the first syllable of a disyllable lengthens also when the final syllable is closed (Turner's consonant-final citation stems, the -ab verbal no [@degem-lengthen @degem-plain — CL in the first syllable of a disyllable, open or closed (Bihari), then plain degemination (blocks/length-cl-disyllable-closed.sc; blocks round 2026-09-30)] ──
[bihari-tharu -DewasDoneDanuwar -Darai -Majhi -Bote] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[bihari-tharu -DewasDoneDanuwar -Darai -Majhi -Bote bagheli_lakshman] V GL $2 > $1+long $3 / # C* _ V C C* #       ; @degem-lengthen — CL before a closed final syllable (Bhoj taptall > tātal, Mth arpáyati > *appab > āpab) / (block length-cl-disyllable-closed)
[bihari-tharu -DewasDoneDanuwar -Darai -Majhi -Bote] ; ── block length-cl-disyllable: NIA degemination with compensatory lengthening in the stressed first syllable of an (open) disyllable: V G G > V̄ G / # C* _ V #; longer words keep a short vowe ──
[bihari-tharu -DewasDoneDanuwar -Darai -Majhi -Bote] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[bihari-tharu -DewasDoneDanuwar -Darai -Majhi -Bote bundeli_atarra OAw Aw bagheli_lakshman] V GL $2 > $1+long $3 / # C* _ V #            ; @degem-lengthen — CL in the first syllable of a disyllable (H hásta > hātʰ, sápta > sāt, kárṇa > kān) / (block length-cl-disyllable)
[bihari-tharu -DewasDoneDanuwar -Darai -Majhi -Bote] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[bihari-tharu -DewasDoneDanuwar -Darai -Majhi -Bote] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[DewasDoneDanuwar Bote BH Hajong MH KS Bhatri Khandesi PalyaBareli DungraBhili OM Bhili] ; ── block length-cl-first: NIA degemination with compensatory lengthening (CL) in the first syllable: V G G > V̄ G / # (C) _, whatever follows, then plain degemination (length-degem-plain [@degem-lengthen @degem-plain — CL in the first syllable (# (C) _), then plain degemination (blocks/length-cl-first.sc; blocks round 2026-09-30)] ──
[DewasDoneDanuwar Bote BH Hajong MH KS Bhatri Khandesi PalyaBareli DungraBhili OM Bhili] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[DewasDoneDanuwar Bote BH Hajong MH KS Bhatri Khandesi PalyaBareli DungraBhili Vasavi OM Bhili] V GL $2 > $1+long $3 / # (C) _               ; @degem-lengthen — first-syllable CL (B hásta > hāt, sápta > sāt, kárṇa > kān) / (block length-cl-first)
[DewasDoneDanuwar Bote BH Hajong MH KS Bhatri Khandesi PalyaBareli DungraBhili OM Bhili] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[DewasDoneDanuwar Bote BH Hajong MH KS Bhatri Khandesi PalyaBareli DungraBhili OM Bhili] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Bhatri Bhili] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[M] if:!root V G $2 > $1+long $3 / # C _ // _ Long ; @degem-lengthen — compensatory lengthening in the initial syllable (hattʰa > hātʰ, karpaṭa > kāpaḍ), not before a long vowel (hill-climb round 4: *kaṭṭōra > kaṭorā, *kappāla > kapāḷ; CL before a long 2nd vowel: short 112 vs long 71; gates ext −0.0001/−0.0006, verb +0.0003/−0.0010, 76 better / 65 worse; the same block was rejected for OM, MH, KS)
[M] if:!root V G $2 > $1+long $3 / # _ // _ Long | _ V V   ; @degem-lengthen — round 6: word-initial (onsetless) vowels lengthen only before a light syllable (akṣōṭa > akʰoḍ, aśru > asū, aṭṭāla > aṭāḷ, but hasta-type onset words always lengthen)
[cam kul] ; ── block finals-syncope-pahari-a: West Pahari syncope of a (blocks theme "finals", 2026-09-30): finals-syncope-open, then medial a also after a closed syllable when a vowel follows (VCCaCV). Cha [Pahari syncope of a (blocks/finals-syncope-pahari-a.sc) — replaces `use syncope.a.vcc`; `use syncope.a.vcccv`: medial a in an open non-initial syllable / medial a also lost after a closed syllable when a vowel follows, as in ktg/bhal/kiuth (váṇṭati > bɔṇḍṇā, saṁkaṭá > sɔŋgṛā, ríŋgati > riŋgṇā; round 6 failed on test ext +.0001; round 7: ext −.0014, test −.0003, verb −.0077; 17 better / 2 worse)] ──
[cam kul] ; ── block finals-syncope-open: Pahari syncope of a in an open non-initial syllable (blocks theme "finals", 2026-09-30): medial a between a vowel + consonant and a following consonant is lost  ──
[ktg] # use aka.ua.final                             ; @hiatus — -uka > -ua > -ū (round 4)  [removed round 7: without the contraction -uka > -ua loses its final a like other final vowels and gives -u (dugdʰá > dudʰu, *gʰaṇṭa > gʰàṇḍu, vatsá > báʦʦʰu); ext .3625→.3614, test .3459→.3440, 9 better / 0 worse]
[bhal kc bhad ktg WPah] ; ── block finals-pahari-west: West Pahari finals (blocks theme "finals", 2026-09-30): -aka > -o (finals-aka-west), then apocope of final -a and -ā (final -i -u went earlier), so the masculin [-aka > -o, then apocope of -a -ā (blocks/finals-pahari-west.sc) — replaces `use aka.aa.final.2`: -aka > -aa > -o (masc. direct case)] ──
[bhal kc bhad ktg WPah] ; ── block finals-aka-west: The -aka ending in the west (blocks theme "finals", 2026-09-30): after intervocalic k is lost, the masculine nominative -akaḥ (MIA -ao) contracts to -o, which i ──
[kc bhad ktg WPah] a a > o / _ #                                ; @aka @hiatus — 7 lects / -aka > -aa > -o (WPah. *ṭaṭṭa+aka > ṭaṭo, dákṣiṇa+aka > dakʰṇo; Nim. tāraká > tāro) / (changes.sc aka.aa.final.2)
[bhal kc bhad ktg WPah] {a ā} > ∅ / _ #                              ; @final-vowel-loss — 5 lects / final -a -ā lost after the -aka contraction (WPah. dūrá > dūr, pārá > pār, bālá > bāḷ) / (changes.sc final-vowel-loss.aa)
[kc] # {a á} > o / _ C* # // Long C _   ; @morph — masc. direct case -ɔ generalised to a-stems (round 2), not after a long vowel + C (-ār, -āḷ keep ∅; round 4)  [removed round 7 (@morph audit): the ext/verb menus derive the citation form]
[kc] {i ī} > e / _ #                             ; @x-wpah-final-i-e — final i lowered to e (Kochi)
[bhal] {ā ī ū} > {a i u} / _ C V C* V               ; @vowel-shift — 4 lects / long vowel shortened two or more syllables before the end (NIA weight/stress; round 2) / (changes.sc vowel-shift.aiu.cvcv)
[bhal kc mand] ; ── block finals-syncope-open: Pahari syncope of a in an open non-initial syllable (blocks theme "finals", 2026-09-30): medial a between a vowel + consonant and a following consonant is lost  [syncope of a in an open syllable (blocks/finals-syncope-open.sc) — replaces `use syncope.a.vcc`: medial a in an open non-initial syllable] ──
[bhal kc mand cam kul] a > ∅ / V C _ C                              ; @syncope — 6 lects / medial a lost after an open syllable (mand niranna+aka > nirnā, ríṣ+allaka > rislā) / (changes.sc syncope.a.vcc)
[ktg] # {a á} > o / _ C* # // Long C _ | V {r ḷ l} _   ; @morph — masc. direct case -ɔ generalised to a-stems (final-syllable a > ɔ; round 2), not after a long vowel + C or after a liquid (-ār, -āḷ, -ra/-la stems keep ∅; round 4)  [removed round 7: with -ɔ distinct from -o the bare a-stem is better left bare; the -aka menu variant supplies -ɔ]
[khash] ; ── block finals-pahari-east: Inner Pahari finals with -ā (blocks theme "finals", 2026-09-30): final -a and -ā are lost except the second vowel of the -aa left by -aka, which then contracts  [apocope of -a -ā, then -aa > -ā (blocks/finals-pahari-east.sc)] ──
[khash] {a ā} > ∅ / _ # // a _                       ; @final-vowel-loss — 4 lects / final -a -ā lost, not the second a of -aka > -aa (khash gʰara > gʰar, pʰulla > pʰull; kiuth kárṇa > kān) / (changes.sc final-vowel-loss.aa.except)
[khash] ; ── block finals-aka-east: The -aka ending in the centre and east, final only (blocks theme "finals", 2026-09-30): the masculine nominative -akaḥ (MIA -aa) contracts to -ā, which is why t ──
[khash] a a > ā / _ #                                ; @aka @hiatus — 11 lects / -aka > -aa > -ā (kul. kāṇá+aka > kāṇā, ūná+aka > uṇā, uparōdʰa+aka > urā) / (changes.sc aka.aa.final)
[pan] r > ∅ / _ C                                  ; @assim-r @cluster-simplify — 16 lects / (changes.sc assim-r.r.preconsonantal)
[kiuth kgr ktg cur khash pan] ; ── block finals-syncope-pahari: West Pahari syncope (blocks theme "finals", 2026-09-30): a medial short vowel of an open non-initial syllable is lost (VCVC), then medial a also after a closed  [W. Pahari syncope (blocks/finals-syncope-pahari.sc) — replaces `use final-vowel-loss.aa`; `use syncope.a.vcc`; `use syncope.a.vcccv`: medial a in an open non-initial syllable / medial a also lost after a closed syllable when a vowel follows, as in bhal/kiuth/khash (*adʰyantara > jεndrɔ, *ṭiŋkara > ṭɪŋgṛɔ, *buḍḍʰal- > bv́ḍ͡ḷʰɔ); round 6 rejected it on the bare-headword accent; round 7 with ext_acc: ext .3482→.3441, test .3261→.3220, 63 reflexes better / 0 worse] ──
[dog kiuth kgr ktg cur khash pan] {a i u} > ∅ / V C _ C                        ; @syncope — 9 lects / open non-initial syllable (kgr. cālana+ikā > cālṇī, apakṣá+aka > aukʰā; WPah. *muṣala > muśḷ) / (changes.sc syncope.aiu.vcc)
[kiuth kgr ktg cur khash cam kul pan] a > ∅ / V C C _ C V                          ; @syncope — 7 lects / also after a closed syllable (kgr. *gilla+araka > gilrā, *paddʰara+aka > padrā, *vartmapāla > batvāl) / (changes.sc syncope.a.vcccv)
[west-pahari -kiuth -kgr -kvar -cam] ṇ > n / {r ṛ ḷ ṇ} _                          ; @n-dissimilation — 5 lects / ṇ > n after another retroflex/r / (changes.sc n-dissimilation.rrln)
[Bhatri M mewari_bannoda Bhilali Noiri RathwiBareli VB Bhili KharI As-Ul D thar Khol kcch khet Goj jaun kiuth kc kgr mand cur WPah kul] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina [@degem-plain — NIA plain degemination (GL = G∪G2∪G3) (blocks/length-degem-plain.sc; blocks round 2026-09-30)] ──
[Bhatri M mewari_bannoda Bhilali Noiri RathwiBareli VB Bhili KharI As-Ul D thar Khol kcch khet Goj jaun kiuth kc kgr mand cur WPah kul] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[bihari-tharu -Darai -Majhi Garh eastern -Or -A -AdivasiOriya -B hindi -NDu -H -Aw -SSr marathi-bhil -hal -PauriBareli -Ko-Ul KharI As-Ul D thar Khol kcch khet Goj west-pahari -bhal -dog -bhad -ktg -khash -cam -pan] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[northwest -D -thar -Khol -khet -awan] Long > $1+stress / _ C* V C* #              ; @accent-placement — a long penult (before apocope) carries the accent: gōpālá > guā́llā, prahēlikā > pahḗlī, vatsatará > vacʰḗrā, *maurdʰaka > Gill móḍā; words without a long penult get none (round 3)
[northwest -D -thar -Khol -khet -awan] V > $1+stress / # C* _ // _ !#* Long C* V C* #   ; @accent-placement — no long penult: the first syllable carries the accent/tone (hastatala > tʰèḷī, lṓhita > láū; round 7; exception widened to any later long penult so a word gets one accent: bʰaginīpati > *bʰaiṇī́vṇā, not two)
[H] {ō ē} {a i} > {ō ē} / _ C                    ; @hiatus — 2 lects / ō/ē + a/i in hiatus contracts (*cʰōkara > cʰorā, jyautiṣika > josī, *gurjarakēṭṭikā > gujreṭī); round 6 / (changes.sc hiatus.oeai.preconsonantal)
[Bishnupriya] ; ── block length-cl-all: NIA degemination with compensatory lengthening in every syllable: V G G > V̄ G, then plain degemination (length-degem-plain) of what is left after a long vowel  [@degem-lengthen @degem-plain — CL in every syllable, then plain degemination (blocks/length-cl-all.sc; blocks round 2026-09-30)] ──
[Bishnupriya] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Bishnupriya] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[Bishnupriya] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[eastern -Or -AdivasiOriya -OB -B] ; ── block nasals-anusvara-length-east: nasals-anusvara-length-east — anusvāra > vowel length, nasality unwritten, north Bengal/Assam survey lects (Kamtapuri, Rajbanshi, Surjapuri, Bishnupriya, Hajong [@anusvara-vowel — toulmin writes no nasality here (blocks/nasals-anusvara-length-east.sc)] ──
[eastern -Or -AdivasiOriya -OB -B] V ṁ > $1+long / _ C                          ; @anusvara-vowel — 5 lects / kaṇṭa > BH kāṭā / (changes.sc anusvara-vowel.vm.preconsonantal.3)
[northwest -D -thar -P -Khol bhal] Long > $1+stress / # C* _                   ; @accent-placement — first syllable heavy (long vowel) takes the accent/tone (NIA weight stress, placed before apocope; round 3)
[northwest] Short > $1+stress / # C* _ C C              ; @accent-placement — first syllable heavy (closed) takes the accent (garbʰarūpa > gàbʰrū; round 3)
[northwest -P -srk] Long > $1+stress / # C* Short C _           ; @accent-placement — light first syllable: the accent goes to a long second syllable (palāśá > pʰulā̀, vivāhá > viyā̀; round 3)
[northwest -D] Short > $1+stress / # C* _ C V // _ !#* Long; @accent-placement — light first syllable and no later long vowel: first syllable (round 3)
[poth awan] Short > $1+stress / # C* _ V                ; @accent-placement — a light first syllable in hiatus with the next vowel takes the accent (madʰū́ka > màūā, divasá > dḕō̃, gupʰáti > gùnuṇ; round 7)
[awan] a > u / C _ ṇ a #                           ; @vowel-shift — a > u before final -ṇa: the Hindko infinitive -uṇ from the verbal noun -aṇa (ākʰyāti > ākʰuṇ, kárṣati > kassuṇ; round 6)
[B northwest -OP -D -thar -S -Khol] ; ── block finals-apocope-schwa: Punjabi-Lahnda apocope (blocks theme "finals", 2026-09-30): final short -a -i -u and the reduced -ə from final -ā (finals-schwa-shortening*) are lost. [apocope incl. -ə (blocks/finals-apocope-schwa.sc) — replaces `use final-vowel-loss.aiu.2`] ──
[B northwest -OP -D -thar -S -Khol] {a i u ə} > ∅ / _ #                          ; @final-vowel-loss — 5 lects / final -a -i -u -ə lost (P. akṣára > akkʰar, agní > agg, áŋga > aŋg) / (changes.sc final-vowel-loss.aiu.2)
[P] Stop r > $1 $1 a r / Short _ #               ; @anaptyxis — a final stop + r kept by cr-keep-optional takes an anaptyctic a after apocope, geminating the stop after a short vowel (cakrá > cakkar, citrá > cittar, ágra > aggar, cʰidrá > cʰiddar; hill-climb round 2)
[P G] ∅ > a / Stop _ r #                           ; @anaptyxis — elsewhere plain a (mū́tra > mūtar, kṣḗtra > kʰetar, candrá > candar; hill-climb round 2)
[P] ṇ > n / {r ṛ ṇ} _                            ; @n-dissimilation — 2 lects / -ṇā > -nā after r ṛ ṇ (sārayati > sārnā) / (changes.sc n-dissimilation.rrn)
[poth awan] G $1 > $2 / _ #                              ; @degem-plain — 2 lects / word-final geminates simplify (saptá > sat, *cʰatti > cʰat, ákṣi > akʰ; round 6) / (changes.sc degem-plain.g.final)
[poth] i > e / _ V                                 ; @vowel-shift — i lowered before a vowel in hiatus (divasa > dias > Lothers dɛ̤āṛ, teyāṛ)
[bhad] ; use syncope.aiu.vcc   — dropped round 6 third pass: Bhadrawahi keeps the medial vowel (maraṇ, -aṇ infinitives); removal: ext −0.0010, verb −0.0172 (test −0.0347)
[bhal bhad cur khash pan] ṇ > n / {r ṛ ḷ ṇ ṭ ḍ} _                      ; @n-dissimilation — 4 lects / ṇ > n after another retroflex/r (incl. ṭ ḍ) / (changes.sc n-dissimilation.rrlntd)
[kvar] ; ── block lar-whp-affrication: West Pahari affrication: the palatal stops become dental affricates (c cʰ j jʰ > ʦ ʦʰ ʣ ʣʰ), and ñ follows them to the dental nasal. Kotgarhi, Kochi, Kullui, Bh [@affrication — c cʰ j jʰ > ʦ ʦʰ ʣ ʣʰ, ñ > n before them (blocks/lar-whp-affrication.sc)] ──
[kvar] {c cʰ j jʰ} > {ʦ ʦʰ ʣ ʣʰ}                    ; @affrication — 6 lects / cʰárdati > bhad. ʦʰaḍḍ-, *kacca > bhal. kaʦʦo, pañca > pānʦ / (changes.sc affrication.cchjjh)
[kvar] {ʦʰ ʣʰ} > {ʦ ʣ}                              ; @deaspiration — Kvari: the new affricates lose aspiration (was {c cʰ j jʰ} > {ʦ ʦ ʣ ʣ})
[Ku Garh] ; ── block length-cl-final-cvcv: CL before the final vowel (length-cl-final), extended to the first syllable of a CVGGVCV trisyllable with a light second syllable: V G G > V̄ G / # C* _ V C V # [@degem-lengthen @degem-plain — CL before the final vowel and in the first syllable of CVCCVCV, then plain degemination (blocks/length-cl-final-cvcv.sc; blocks round 2026-09-30)] ──
[Ku Garh] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Ku Garh] V GL $2 > $1+long $3 / # C* _ V C V #        ; @degem-lengthen — first syllable of a trisyllable with a light second syllable (Garh, Ku kajjala > kājal) / (block length-cl-final-cvcv)
[Ku Garh] ; ── block length-cl-final: NIA degemination with compensatory lengthening before the final vowel: V G G > V̄ G / _ V # (the geminate of the last MIA syllable lengthens the vowel before it ──
[Ku Garh] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Ku] V GL $2 > $1+long $3 / _ V #                 ; @degem-lengthen — CL before the final vowel (Ku hásta > hātʰ, bʰaktá > bʰāt, kárṇa > kān) / (block length-cl-final)
[Ku Garh] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[Ku Garh] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Ku] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[Ku] ; if:starred,!ext a > o / C _ #                       ; @morph — Turner's *a-stems are back-formed from NIA masc. -o/-ā forms: cite them with -o (only after a consonant, so extended variants are not doubled)   [dropped: starred-headword citation shortcut; the extension menu derives these endings]
[Dotyali] ; if:starred,!ext a > o / _ #                       ; @morph — Turner's *a-stems are back-formed from NIA masc. -o/-ā forms: cite them with -o   [dropped: starred-headword citation shortcut; the extension menu derives these endings]
[Dotyali] ; ── block liquids-flap-to-r: Intervocalic ḍ ḍʰ > ṛ ṛʰ, then the flap merges with r. Shared by 21 lects in 6 clades (Bihari, E. Hindi, E. Pahari, Eastern, Other, W. Hindi); blocks round 2026 [Intervocalic ḍ ḍʰ > ṛ ṛʰ (blocks/liquids-flap-to-r.sc)] ──
[Dotyali] ; ── block liquids-flap: Retroflex flapping — single post-vocalic ḍ ḍʰ > ṛ ṛʰ (NIA; liquids blocks round, 2026-09-30). The flap of the Hindi belt, Punjabi-Lahnda-Sindhi, Pahari, Bengali [@flap — Tharu flap merges with r (*dʰūḍi > dʰur)] ──
[Dotyali] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ {ḍ ḍʰ}            ; @flap — 8 lects / kaṭāha > H kaṛāhā, kukkuṭá > H kukṛā, *dʰūḍi > H dʰūṛ; geminate ḍḍ stays a stop / (changes.sc flap.ddh.postvocalic.except)
[Dotyali] ; ── block liquids-flap-rhotic: Flap-rhotic merger — ṛ ṛʰ > r rʰ (liquids blocks round, 2026-09-30). Stage 2 of the flap history: the retroflex flap (blocks/liquids-flap.sc, or the Dardic inte [@flap — the flap merges with r] ──
[Garh N Dotyali] ; ── block length-cl-final: NIA degemination with compensatory lengthening before the final vowel: V G G > V̄ G / _ V # (the geminate of the last MIA syllable lengthens the vowel before it [@degem-lengthen @degem-plain — CL before the final vowel (_ V #), then plain degemination (blocks/length-cl-final.sc; blocks round 2026-09-30)] ──
[Garh N Dotyali] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Garh N Dotyali] V GL $2 > $1+long $3 / _ V #                 ; @degem-lengthen — CL before the final vowel (Ku hásta > hātʰ, bʰaktá > bʰāt, kárṇa > kān) / (block length-cl-final)
[Garh N Dotyali] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[Garh N Dotyali] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Garh N Dotyali] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[kvar] a a > ɔ / _ # // V _                        ; @hiatus @aka — -aka > -aa > -ɔ (masc. direct case; Jouanne writes -ɔ: kulā́ya > koḷɔ, gʰōṭa > goṛɔ; round 4, ɔ since round 7)
[N] a a {i ē} > n u / _ #                        ; @morph @hiatus — -ayati verbs > -nu (cōrayati > cornu)
[N] ; if:starred,!ext a > o / C _ #                       ; @morph — Turner's *a-stems are back-formed from NIA masc. -o/-ā forms: cite them with -o (only after a consonant, so extended -uka/-ikā variants are not doubled)   [dropped: starred-headword citation shortcut; the extension menu derives these endings]
[Garh kvar] a a > u / _ #                                ; @hiatus — 2 lects / after a further vowel (-V.a.a, *tapaka > tau) the old -u outcome remains (round 4 note) / (changes.sc hiatus.aa.final.3)
[kvar] ; ── block finals-pahari-west: West Pahari finals (blocks theme "finals", 2026-09-30): -aka > -o (finals-aka-west), then apocope of final -a and -ā (final -i -u went earlier), so the masculin [-aka > -o, then apocope of -a -ā (blocks/finals-pahari-west.sc) — replaces `use aka.aa.final.2`: -aka > -aa > -o (masc. direct case)] ──
[kvar] ; ── block finals-aka-west: The -aka ending in the west (blocks theme "finals", 2026-09-30): after intervocalic k is lost, the masculine nominative -akaḥ (MIA -ao) contracts to -o, which i ──
[kvar] a a > o / _ #                                ; @aka @hiatus — 7 lects / -aka > -aa > -o (WPah. *ṭaṭṭa+aka > ṭaṭo, dákṣiṇa+aka > dakʰṇo; Nim. tāraká > tāro) / (changes.sc aka.aa.final.2)
[kvar] {a ā} > ∅ / _ #                              ; @final-vowel-loss — 5 lects / final -a -ā lost after the -aka contraction (WPah. dūrá > dūr, pārá > pār, bālá > bāḷ) / (changes.sc final-vowel-loss.aa)
[kvar] {a i u} > ∅ / V C _ C                        ; @syncope — 9 lects / medial short vowel in an open non-initial syllable (round 2) / (changes.sc syncope.aiu.vcc)
[poth dog] ; ── block lar-deaspiration-initial-devoicing: Punjabi-type tonogenesis, segmental half: initial voiced aspirates become voiceless unaspirated (bʰ dʰ gʰ jʰ ḍʰ > p t k c ṭ, with low tone, which the @tone rule [@deaspiration @tone — initial voiced aspirates > voiceless unaspirated + tone (blocks/lar-deaspiration-initial-devoicing.sc)] ──
[poth dog] {bʰ dʰ gʰ jʰ ḍʰ} > {p t k c ṭ} / # _         ; @deaspiration @tone — 2 lects / gʰara > poth. kar, bʰaginīpati > L. bàṇiā / (changes.sc deaspiration.bhdhghjhdh.initial)
[Kannauji-Central bundeli_atarra bagheli_lakshman Brj] a > ∅ / V C _ n ā #                          ; @syncope — 2 lects / round 7: the vowel of the verbal noun -ana(k)a syncopates after a (C)VC root (-na infinitive, as in Bagheli: celna); train ext −0.00003, 2 better / 1 worse, test ±0 / (changes.sc syncope.a.vcna)
[hindi -OAw] ; ── block hiatus-hindi-mid: Hiatus theme — Hindi-area contractions after the a/ā contraction (blocks round pass 2, 2026-09-30). i + ī > ī and a mid vowel absorbing a following a/i before a [@hiatus — blocks/hiatus-hindi-mid.sc; ō/ē + a/i in hiatus contracts (*cʰōkara > cʰorā, jyautiṣika > josī); round 6] ──
[hindi L] i ī > ī                                      ; @hiatus — i + ī > ī (ākʰēṭika + -ikā > aheriī > aherī, duḣkʰitikā > dukʰī) / (block hiatus-hindi-mid)
[hindi -OAw] {ō ē o e} {a i} > {ō ē o e} / _ C            ; @hiatus — ō/ē + a/i > ō/ē before a consonant (*cʰōkara > cʰorā, jyautiṣika > josī) / (block hiatus-hindi-mid)
[hindi -kaithal -OAw] ; ── block length-hindi-pretonic: Hindi-belt pretonic shortening: a long vowel shortens before the NIA stressed syllable — before the stressed diminutive -i(y)ā, and in the first syllable before [@x-hindi-pretonic-shortening @stress — pretonic shortening before -i(y)ā and before a stressed final CV̄C (blocks/length-hindi-pretonic.sc; blocks round 2026-09-30)] ──
[hindi -OAw] {ā ī ū} > {a i u} / _ C i (y) ā #            ; @x-hindi-pretonic-shortening @stress — 6 lects / before the stressed -i(y)ā (H stʰālī + -iyā > tʰaliyā) / (changes.sc pretonic-shortening.aiu.ciya)
[hindi -kaithal -OAw] {ā ī ū} > {a i u} / # C* _ C Long C #        ; @x-hindi-pretonic-shortening @stress — 2 lects / first syllable before a stressed final CV̄C (H āhāra > ahār) / (changes.sc pretonic-shortening.aiu.cclongc)
[bundeli_atarra] ; a > ∅ / V C _ C V                           ; @syncope — schwa deletion (agravāṭa > agvāṛā)   [pruned for bundeli_atarra: LOO -0.0005]
[bagheli_lakshman] ; a > ∅ / V C _ C V                           ; @syncope — schwa deletion (agravāṭa > agvāṛā)   [pruned for bagheli_lakshman: LOO -0.0005]
[Brj] {a ã} > ∅ / V C _ C V                        ; @syncope — schwa deletion (agravāṭa > agvāṛā); round 7: short nasal ã (from ṁ outside the stressed syllable) syncopates too, as in H and Aw; train ext −0.0001, test −0.0014, 1 better / 0 worse
[H] {a i ã ĩ} > ∅ / V C _ C V // _ C a C V        ; @syncope — schwa and i syncope, right to left (laṛikā > laṛkā, parinī > parnī, mukʰiyā > mukʰyā); round 6b. Round 7: short nasal a/i (from ṁ or a nasal cluster outside the stressed syllable, and ãv < -m-) syncopate like oral ones (kaḍambá + -ikā > kaṛbī, palyaŋka + -aka > palkā, daśamá > dasãvā > dasvā̃; the losers keep ã in -ãg-/-ãḍ- before an extension: bahãgī); train ext −0.0011, test −0.0008, 116 better / 9 worse
[kaithal] {a i} > ∅ / V C _ C V // _ {n ṇ} {ā a} # | _ C a C V   ; @syncope — (round 6b: i too) schwa syncope as in Khari Boli (bakri, cʰokri, camḍi, musli, mogrā), right-to-left; not before the -ṇā/-nā infinitive (Hr keeps -aṇā); round 6
[kaithal] ; ── block nasals-retroflexion: nasals-retroflexion — MIA intervocalic n > ṇ (the Prakrit "no ṇaḥ" change), nasals theme, blocks round 2026-09-30. Stage 1 of n > ṇ > n: a single dental nasal b [inherited from the Sindhic sisters (small-lect default, 2026-09-30)] ──
[kaithal] n > ṇ / V _ V                                ; @n-retroflex @nur-innovation @x-nur-n-flap — 40 lects / pānī́ya > G, P, S pāṇī, vána > P vaṇ, S vaṇu, Or baṇa / (changes.sc n-retroflex.intervocalic)
[Kannauji-Central] ; a > ∅ / V C _ C V                           ; @syncope — schwa deletion (agravāṭa > agvāṛā)   [pruned for Kannauji-Central: never fires]
[NDu] ; a > ∅ / V C _ C V                           ; @syncope — schwa deletion (agravāṭa > agvāṛā)   [pruned for NDu: LOO -0.0002]
[SSr] ; a > ∅ / V C _ C V                           ; @syncope — schwa deletion (agravāṭa > agvāṛā)   [pruned for SSr: LOO -0.0010]
[Aw] i > ∅ / _ a C #                             ; @hiatus — i lost in hiatus before a final closed syllable (cariab > carab, kʰāiab > kʰāab); round 5
[Aw] {ā a} a > ā / _ C #                         ; @hiatus — the resulting -āa-/-aa- contracts (kʰāab > kʰāb); round 5
[K] {n ṇ} y > ñ                                 ; @palatal-Cy — anya > añ
[Kalk K] optional: r {t d} > {ṭ ḍ}@2 / _ V            ; @rt-retroflex — 2 lects / r + dental > retroflex (*kartati > kaṭ-, *ārta > āṭ) / (changes.sc rt-retroflex.rtd.prevocalic)
[Phal] optional: r {t d tʰ dʰ} > {ṭ ḍ ṭʰ ḍʰ}@2              ; @rt-retroflex — r + dental > retroflex before metathesis (śūrtá > ṣūṛ, *varta > bāṭ)
[Sh Kalk Phal] Stop V r C > $1 r $2 $4                      ; @metathesis — 2 lects / r moves into the initial syllable, after lenition (dīrgʰá > dríg, kárman > trām) / (changes.sc metathesis.stopvrc)
[Sv Kalk Ush Kund] ; ── block mia-shina-clusters-z: Shina cluster simplification with dy > ź (Sv, Kalk, Ush; blocks theme "mia", pass 2, 2026-09-30): the mia-shina-clusters sequence with the Shina reflex of d/dʰ  [@assim-r @palatal-Cy @cluster-simplify — postvocalic rC > C, d dʰ + y > ź, C + y/v > C (blocks/mia-shina-clusters-z.sc; blocks round 2026-09-30, pass 2)] ──
[Sv Kalk Kund] r > ∅ / V _ C                                ; @assim-r @cluster-simplify — 10 lects / postvocalic r + C > C (karma > kam-, sarpa > sap-, cárma > cam-) / (changes.sc assim-r.r.vc)
[Sv Kalk Ush Kund] {d dʰ} y > ź                                 ; @palatal-Cy — 3 lects / d dʰ + y > ź (adya > aź-, madʰya > maź-, udyāna > uź-) / (changes.sc palatal-Cy.ddhy)
[Ush] {a á ā ā́} > ō / _ n                         ; @a-rounding — a before n > ō, after r + C loss (kárṇa > kōn, dánta > dōn)
[Sh Sv Ush dardic-south -Shum -Dm -Tir -Pas] C r > $1                                     ; @cluster-simplify — 8 lects / consonant + r > consonant (pralēpayati > pal-, gr̩hiṇī > gēn), after lenition / (changes.sc cluster-simplify.cr)
[dardic-north -Kho -Phal -K -Kal] ; ── block cr-release: End of the cluster window: the onset protected by cr-keep-optional / r-anticipation-optional becomes stop + r again, and later rules (flaps, tr > ṭr, …) see an  [end of the cluster window (blocks/cr-release.sc)] ──
[dardic-north -Kho -Phal -K -Kal] {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} > {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} r ; @cr-keep — protected stop + r released after the cluster stage / (block cr-release)
[Ush] ; ── block finals-apocope-shina: Shina-type apocope in polysyllables (blocks theme "finals", 2026-09-30): final vowels are lost after one to three consonants preceded by a vowel, but not the -o [apocope in polysyllables, -o kept (blocks/finals-apocope-shina.sc) — replaces `{a i u e ā ī ū ē ō} > ∅ / V (C) C _ # | V C C C _ #`: not in monosyllables, and not the -o of -aka nouns] ──
[Ush] {a i u e ā ī ū ē ō} > ∅ / V (C) C _ # | V C C C _ # ; @final-vowel-loss — final vowel lost in polysyllables, -o kept (Sh. áŋgāra > agār, aṣṭā́ > aṭ, āmrá > ām) / (block finals-apocope-shina)
[Phal] ; ── block mia-shina-clusters: Shina/Chitrali/Kashmiri late cluster simplification (blocks theme "mia", 2026-09-30): r lost before a consonant after a vowel, dental + y > palatal, C + y/v > C [@assim-r @palatal-Cy @cluster-simplify — postvocalic rC > C, t d dʰ + y > c j j, C + y/v > C (blocks/mia-shina-clusters.sc; blocks round 2026-09-30)] ──
[Phal] r > ∅ / V _ C                                ; @assim-r @cluster-simplify — 10 lects / postvocalic r + C > C (karma > kam, sarpa > sap, cárma > cam) / (changes.sc assim-r.r.vc)
[dardic-north -Kho -pog -Kund -K -dod] ; ── Initials ────────────────────────────────────────────────────────────
[K] ; ── block r-anticipation-optional: Sporadic r-anticipation (Dardic/NW 'metathesis'): the r of the first-syllable coda or of the second syllable's onset cluster is copied into the onset of the ini [sporadic (blocks/r-anticipation-optional.sc)] ──
[K] optional: {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} > {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} / # _ V r C | # _ V C r ; @metathesis — sporadic r-anticipation into the initial onset (tarka > trakk-, kárman > krom) / (block r-anticipation-optional)
[pog] r > ∅ / V _ C                                ; @assim-r @cluster-simplify — 10 lects / r + C > C, after lenition (karpaṭa > kapur, not *kavur) / (changes.sc assim-r.r.vc)
[Kho] C {y v} > $1                                 ; @cluster-simplify — 18 lects / after lenition: skuṭyatē > iskuṭík / (changes.sc cluster-simplify.cyv)
[Kho] ; ── Edges ───────────────────────────────────────────────────────────────
[Kho Kal] ∅ > i / # _ Sib C                            ; @prothesis — 2 lects / snasā > isnos / (changes.sc prothesis.sibc)
[Dang Garh N Sh Kho Phal Kal kaithal Aw OP P poth Marw had mewari_dholpura dhundari_badagaon kiuth bhad kul] ; ── block v-b-skip-optional: Sporadic escape of initial v- from v- > b- (2026-10-01): some words keep v- where the lect otherwise has b- (P viśvāsa > vasāh beside vāla > bāl). Mechanism: an [sporadic (blocks/v-b-skip-optional.sc)] ──
[Dang Garh N Sh Kho Phal Kal kaithal Aw OP P poth Marw had mewari_dholpura dhundari_badagaon kiuth bhad kul] optional: v > ʋ / # _                        ; @v-b — sporadic: initial v- escapes v- > b- (viśvāsa > P vasāh) / (block v-b-skip-optional)
[D] {a ā} m a > o m / _ #                        ; @vowel-shift — -(ā)ma > -om (kárman > kom, nā́man > nom, stambʰa > tom; round 6)
[kvar] V Nasal > $1+long / _ Stop                   ; @anusvara-vowel — NC > V̄ + (pre)nasalised C; Jouanne writes the nasal as a tilde on the stop (kāg̃i), which the scorer drops
[D dog kvar] ; ── block lar-deaspiration-voiced: Loss of aspiration in the voiced aspirates (bʰ dʰ gʰ jʰ ḍʰ > b d g j ḍ), everywhere in the word. Shared across families: Dardic (Shinaic, Kohistani, Kunar, Kash [@deaspiration — voiced aspirates lose aspiration (blocks/lar-deaspiration-voiced.sc)] ──
[D dog kvar] {bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ}               ; @deaspiration @tone — 26 lects / dʰūma > Sh. dūm, bʰūmi > K. bum, gʰr̩tá > Gaw. gʌd / (changes.sc deaspiration.bhdhghjhdh)
[Goj] {j jʰ} > y                                 ; @x-rg-j-y — Gujari j > y (jana > yano, adya > ay, vidyut > bīyilī)
[Goj] ; ── block finals-apocope-two-step: Two-step apocope in polysyllables (blocks theme "finals", 2026-09-30): final short -a -i -u are lost (finals-apocope: Nimadi kárṇa > kān, dánta > dā̃t, hásta >  [short vowels, then -ā, lost in polysyllables (blocks/finals-apocope-two-step.sc)] ──
[Goj] ; ── block finals-apocope: NIA apocope (blocks theme "finals", 2026-09-30): loss of the short final vowels -a -i -u that the Apabhraṁśa stage left word-final (after finals-ap-shortening w ──
[Goj] {a i u} > ∅ / V C* _ #                       ; @final-vowel-loss — 3 lects / final short -a -i -u lost in polysyllables (H. akṣára > ākʰar, agní > āg, áŋga > ā̃g) / (changes.sc final-vowel-loss.aiu.final-polysyllabic)
[Goj] ā > ∅ / V C* _ # // a _                      ; @final-vowel-loss — 3 lects / final -ā lost in polysyllables, not after a (Nimadi rakṣā > rākʰ, aṣṭā́ > āṭʰ, jáṭā > jaḍ) / (changes.sc final-vowel-loss.a.final-polysyllabic.except)
[Nimadi] ; ── block length-cl-rg-disyllable: Rajasthani CL in the first syllable of a disyllable only: V G G > V̄ G / # C* _ C* V # (no lengthening before a light or closed final syllable, unlike length-cl [@degem-lengthen — Rajasthani CL in the first syllable of a disyllable (blocks/length-cl-rg-disyllable.sc; blocks round 2026-09-30)] ──
[Nimadi] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Nimadi] V GL $2 > $1+long $3 / # C* _ C* V #         ; @degem-lengthen — first syllable of a disyllable (Nimadi hásta > hātʰ, kárṇa > kān) / (block length-cl-rg-disyllable)
[Nimadi] G $1 > $2                                    ; @cluster-simplify @degem-plain — 83 lects / (changes.sc degem-plain.g)
[OG OMarw] ; ── block finals-ap-shortening: Apabhraṁśa shortening of final -ā (blocks theme "finals", 2026-09-30): late MIA final long -ā shortens to -a, so that it is later lost with the old short -a by  [Ap. final -ā > -a (blocks/finals-ap-shortening.sc) — replaces `use final-a-short.a`: Ap. shortening of final -ā] ──
[OG OMarw] ā > a / _ #                                  ; @final-a-short @final-shortening @final-vowel-loss @vowel-shift — 41 lects / final -ā > -a (Bi. ā́rā > ār, kʰáṭvā > kʰāṭ, nāvā́ > nāv) / (changes.sc final-a-short.a)
[bihari-tharu central-east-pahari-romani -as-ll -Lom dardic-north -pog -K dardic-south -Gaw -Shum -Mai -Pas eastern hindi Bhatri hal mewari_bannoda Vasavi VB OP D thar P kcch poth rajasthani-gujarati -OG -WK -G -marwari_bagra west-pahari] ; ── block lar-v-b: Initial v- > b- (NIA): Hindi, Bihari, Eastern, Pahari, Rajasthani, Punjabi, Lahnda, Sindhi, Dardic, Romani … the most widespread NIA consonant change. Geminate  [@v-b — initial v- > b- (blocks/lar-v-b.sc)] ──
[bihari-tharu central-east-pahari-romani -as-ll -Lom dardic-north -pog -K dardic-south -Gaw -Shum -Mai -Pas eastern hindi Bhatri hal mewari_bannoda Vasavi VB OP D thar P kcch poth rajasthani-gujarati -OG -WK -G -marwari_bagra west-pahari] v > b / # _                                  ; @v-b — 88 lects / vadʰū́ > H. bahū, vā́la > H. bāl / (changes.sc v-b.v.initial)
[OAw] ; {a ā} {a ā} > ā                             ; @hiatus — vowels in hiatus after stop loss contract  [OAw spelling keeps hiatus]
[OAw] {u ū} a > u / _ #                           ; @hiatus — -uka > -ua > -u, the OAw masc. -u (gālu, lāhu, bʰālu); round 6
[OAw] ∅ > y / # (C) a _ ā                        ; @ya-sruti — glide between a first-syllable a and ā in hiatus (prayāṇa > payāna, ajānant > ayāna, vātālī > bayāri); round 6
[OAw] ; ── block hiatus-medial-hindi: Hiatus theme — the Hindi-area medial contraction (blocks round, 2026-09-30). The general a(C)a contraction (`hiatus-medial`) followed by the two contractions th [@hiatus — blocks/hiatus-medial-hindi.sc; medial a(y)a from stop loss contracts to e (kadalī > kayalī > kelā)] ──
[OAw] ; ── block hiatus-medial: Hiatus theme — medial a(C)a contraction (blocks round, 2026-09-30). After the MIA loss of intervocalic stops/y, a-a before a consonant contracts to e and a-ā to ──
[OAw] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here ──
[OAw] ; {a i u} > ∅ / _ #                           ; @final-vowel-loss  [OAw spelling keeps final short vowels]
[OAw] ; a > ∅ / V C _ C V                           ; @syncope — schwa deletion (agravāṭa > agvāṛā)  [OAw spelling keeps medial a]
[Kho Phal Kal Gowro kaithal Aw bagheli_lakshman OP P poth Marw had mewari_dholpura dhundari_badagaon kiuth bhad kul] ; ── block v-b-release: Release of the v-b-skip-optional placeholder: ʋ > v (2026-10-01). Include right after the line that brings v- > b-. [releases the v-b-skip-optional placeholder] ──
[Kho Phal Kal Gowro kaithal Aw bagheli_lakshman OP P poth Marw had mewari_dholpura dhundari_badagaon kiuth bhad kul] ʋ > v                                        ; @v-b — release (blocks/v-b-skip-optional.sc) / (block v-b-release)
[L] a > ∅ / V C (C) _ C V // V C C _ C a ṇ #                         ; @syncope — medial a lost between single consonants or after a geminate (*parsula > paslī, *maṇṭʰara > maṇḍʰrā), except a geminate stem before the infinitive -aṇ (uggʰaṛaṇ; round 6)
[L] v > ∅ / ā _ #                               ; @v-loss — final -āv (< -āpa, -āva) > -ā (tāpá > tā, udvāpa > ubā; round 6)
[mewari_bannoda] u > a / V C* _ C C* V                        ; @vowel-shift — 8 lects / medial (mostly penultimate) u reduced to a before a consonant (kukkuṭa > kūkaḍo type); accent-independent (round 2). Round 4: the C spares -uka > -ua (ext-aware) / (changes.sc vowel-shift.u.vcccv)
[khet awan] ; ── block finals-syncope: NIA schwa syncope (blocks theme "finals", 2026-09-30): after apocope, medial short a in an open syllable between single consonants, with a vowel following, is l [NIA schwa syncope (blocks/finals-syncope.sc) — replaces `use syncope.a.vccv`: medial a] ──
[khet awan] a > ∅ / V C _ C V                            ; @syncope — 16 lects / medial a lost in VCaCV (thar kaccara+aka > kacro, karpaṭa+aka > kapṛo, kṣurapra+ikā > kʰurpī) / (changes.sc syncope.a.vccv)
[awan] Short > ∅ / V C (C) _ C u ṇ #                     ; @syncope — the stem vowel is lost before the heavy infinitive -uṇ, also after a geminate (*utkarati > ukruṇ, *udgalati > ugluṇ; round 6)
[Khol] a {i u} > {e o}@2                            ; @ai-au — 5 lects / secondary diphthongs from hiatus monophthongise (as in OIA-dog.sc) / (changes.sc ai-au.aiu)
[Khol] a > e / V _ {n ṇ} a #                       ; @vowel-shift — the -en infinitive also after a stem vowel left in hiatus by consonant loss (before the hiatus rules) …
[Khol] ∅ > y / V _ e {n ṇ} a #                     ; @hiatus — … with a glide y: kʰā́dati > xāyen, gā́yati > gāyen, rōdati > royen (cf. Hindko -vuṇ; round 7)
[Khol] repeat: {a ā} {a ā} > ā                      ; @hiatus — 22 lects / aa > ā / (changes.sc hiatus.aaaa.repeat)
[Khol] a > ∅ / {i ī u ū ē ō} _                      ; @hiatus — 7 lects / a lost after another vowel / (changes.sc hiatus.a.iiuueo)
[Khol] a > e / C _ {n ṇ} a #                     ; @vowel-shift — a > e before final -na: the Kholosi infinitive -en from the verbal noun -ana (karṓti > karen, *kaḍḍʰati > kadden; round 6)
[northwest -D -S] ∅ > y / {i ī} _ o #                          ; @hiatus — 3 lects / glide y between a front vowel and the final -o (dīpaka > ḍīyo, gōṣṭʰika > goṭʰyo; round 7) / (changes.sc hiatus.0.iio)
[Noiri] c > s                                        ; @deaffrication — 3 lects / deaffrication c > s (nīcá > niso) / (changes.sc deaffrication.c)
[Ko-Ul] ; ── block length-cl-final: NIA degemination with compensatory lengthening before the final vowel: V G G > V̄ G / _ V # (the geminate of the last MIA syllable lengthens the vowel before it [@degem-lengthen @degem-plain — CL before the final vowel (_ V #), then plain degemination (blocks/length-cl-final.sc; blocks round 2026-09-30)] ──
[Ko-Ul] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Ko-Ul] V GL $2 > $1+long $3 / _ V #                 ; @degem-lengthen — CL before the final vowel (Ku hásta > hātʰ, bʰaktá > bʰāt, kárṇa > kān) / (block length-cl-final)
[Ko-Ul] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[Ko-Ul] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Ko-Ul] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[Garh] k > g / ṁ _                                  ; @voicing — post-nasal voicing ŋk > ŋg before the nasal is absorbed into the vowel (páŋka > pā̃gu, vaŋkú > bā̃gu, kaŋkaṇa > kaŋgaṇ); round 7
[eastern -OB -B] ; ── block length-cl-first-short: First-syllable CL blocked before a long vowel: V G G > V̄ G / # (C) _ // _ Long, then plain degemination (length-degem-plain). Odia and Adivasi Odia. [@degem-lengthen @degem-plain — first-syllable CL, not before a long vowel (A. *kappāla > kapāl, karpāsa > kapāh, *karttāra > kaṭārī; hásta > hāt), then plain degemination (blocks/length-cl-first-short.sc; hill-climb round 1)] ──
[eastern -OB -B] V G2 $2 > $1+long $3 / # (C) _ // _ {ā ī ū ē ō} ; @degem-lengthen — 2 lects / first-syllable CL, not before a long vowel (Or hásta > hāta, kárṇa > kāna) / (changes.sc degem-lengthen.vg2.c.except)
[eastern -OB -B] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[eastern -OB -B] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[eastern -Bishnupriya -Hajong -OB -B] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[A] a > ā / # (C) _ Nasal Stop                   ; @anusvara-vowel @degem-lengthen — 2 lects / also before a kept NC cluster (aŋgāra > āŋgār) / (changes.sc anusvara-vowel.a.cnasalstop)
[B] V G2 $2 > $1+long $3 / # C* _                ; @degem-lengthen — 5 lects / first syllable only, as in A. and Or. (bʰatta > bʰāt) / (changes.sc degem-lengthen.vg2.first-syllable)
[B] G2 $1 > $2                                   ; @degem-plain — 10 lects / geminates left unlengthened (after ṁ, etc.) / (changes.sc degem-plain.g2)
[MB OB] ; ── block length-cl-all-nc: CL before a homorganic nasal + stop (a > ā / _ N Stop) and before every geminate (length-cl-all). Old and Middle Bengali. [@degem-lengthen @anusvara-vowel @degem-plain — a > ā before NC, CL in every syllable, plain degemination (blocks/length-cl-all-nc.sc; blocks round 2026-09-30)] ──
[MB OB] ; ── block length-cl-all: NIA degemination with compensatory lengthening in every syllable: V G G > V̄ G, then plain degemination (length-degem-plain) of what is left after a long vowel  ──
[MB OB] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Bishnupriya MB OB] V GL $2 > $1+long $3                         ; @degem-lengthen — CL in every syllable (OB/MB hásta > hātʰa, bʰaktá > bʰāta, kárṇa > kāṇa) / (block length-cl-all)
[MB OB] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina ──
[MB OB] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[bihari-tharu Ku Garh N Dotyali eastern -Or -OB hindi -OAw -bagheli_lakshman DungraBhili Rathawi hal M mewari_bannoda VB Ko-Ul northwest -OP bagri_fatehabad WK hadothi_kelwada mewati_jhambaus Rj marwari_bagra dhundari_bamore dhundari_badagaon] ; ── block nasals-anusvara-vowel: nasals-anusvara-vowel — NIA anusvāra > long nasal vowel before a consonant, nasals theme, blocks round 2026-09-30 (pass 2). The common mainland outcome (Hindi b [@anusvara-vowel (blocks/nasals-anusvara-vowel.sc)] ──
[bihari-tharu Ku Garh N Dotyali eastern -Or -OB hindi -OAw -bagheli_lakshman DungraBhili Rathawi hal M mewari_bannoda VB Ko-Ul northwest -OP bagri_fatehabad WK hadothi_kelwada mewati_jhambaus Rj marwari_bagra dhundari_bamore dhundari_badagaon] V ṁ > $1+long+nas / _ C                      ; @anusvara-vowel — 39 lects / píṇḍa > pī̃ḍ, paŋka > pā̃k, dánta > SSr dā̃t / (changes.sc anusvara-vowel.vm.preconsonantal)
[Garh Bhatri DungraBhili M VB Ko-Ul kc cam] ; ── block hiatus-medial: Hiatus theme — medial a(C)a contraction (blocks round, 2026-09-30). After the MIA loss of intervocalic stops/y, a-a before a consonant contracts to e and a-ā to [@hiatus — blocks/hiatus-medial.sc; a(y)a > e in the interior] ──
[Garh Bhatri DungraBhili M VB Ko-Ul kc cam] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here ──
[Garh M kc] a a > e / C _ C                              ; @hiatus — medial a(C)a > e (kadala > kaala > Bhoj kerā, kadalī > kayalī > kelā) / (block hiatus-medial-core)
[M kc] a ā > ā                                      ; @hiatus — a-ā hiatus > ā (kumbʰakāra > kumʰār, andʰakāra > anʰār) / (block hiatus-medial-core)
[Garh Bhatri DungraBhili M Ko-Ul kc cam] repeat: {a ā} {a ā} > ā                      ; @hiatus — a/ā hiatus > ā (andʰakāra > andʰaāra > ā̃dʰār, kumbʰakāra > kumʰār, rājana > rāaṇ > rāṇ) / (block hiatus-medial)
[Noiri] ; ── block lar-h-loss-postvocalic: Loss of postvocalic h (leaving tone/length in some sources): Bhil (Khandesi, Noiri, Mewari of Bannoda), Haryanvi (Kaithal), Kotgarhi, Mandeali. Laryngeals round [@h-loss — postvocalic h lost (blocks/lar-h-loss-postvocalic.sc)] ──
[Noiri] h > ∅ / V _                                  ; @h-loss @tone — 6 lects / bāhú > bā́ / (changes.sc h-loss.h.postvocalic)
[M] a {i ī} > ī / _ #                           ; @hiatus — round 6: final -aī > -ī (apavaraka+ikā > ovrī, āḍʰakikā > āḍʰī)
[Bote] ; ── block finals-syncope-bihari: Bihari/Tharu syncope before a final long-vowel syllable (blocks theme "finals", 2026-09-30): medial a between single consonants (stops, nasals, sonorants, s; no [syncope before a final long-vowel syllable (blocks/finals-syncope-bihari.sc)] ──
[Bote] a > ∅ / V {Stop Nasal Son s} _ {Stop Nasal Son s} {ā ī} # ; @syncope — 7 lects / medial a lost before a final -Cā/-Cī (Bi. ārātrika > ārtī, kaŋkaṇa+aka > kãknā, kartana+ikā > kaṭnī) / (changes.sc syncope.a.vstopnasalsons)
[KochariyaEastDanuwar KochilaTharu M Khol kcch srk] ; ── block finals-apocope: NIA apocope (blocks theme "finals", 2026-09-30): loss of the short final vowels -a -i -u that the Apabhraṁśa stage left word-final (after finals-ap-shortening w [NIA apocope (blocks/finals-apocope.sc) — replaces `use final-vowel-loss.aiu`: (moved before medial syncope: a final closed syllable keeps its vowel)] ──
[KochariyaEastDanuwar KochilaTharu M Khol kcch srk] {a i u} > ∅ / V C* _ #                       ; @final-vowel-loss — 3 lects / final short -a -i -u lost in polysyllables (H. akṣára > ākʰar, agní > āg, áŋga > ā̃g) / (changes.sc final-vowel-loss.aiu.final-polysyllabic)
[srk] a > ∅ / V C (C) _ C V                         ; @syncope — medial a lost between single consonants or after a geminate (as in L; round 6)
[northwest -D -thar -P -Khol -khet] G_std $1 > $2 / _ C                          ; @degem-plain — 4 lects / a geminate simplifies before a consonant after syncope (passlī > paslī; round 6) / (changes.sc degem-plain.g.preconsonantal)
[S] G $1 > $2                                    ; @degem-plain — Sindhi degeminates without lengthening
[S] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ r               ; @flap
[RomBalk Zarg DungraBhili mewari_bannoda Bhili northwest -OP -D -S -poth] ; ── block finals-syncope: NIA schwa syncope (blocks theme "finals", 2026-09-30): after apocope, medial short a in an open syllable between single consonants, with a vowel following, is l [NIA schwa syncope (blocks/finals-syncope.sc) — replaces `use syncope.a.vccv`: medial a] ──
[RomBalk Zarg DungraBhili mewari_bannoda Bhili northwest -OP -D -S -poth] a > ∅ / V C _ C V                            ; @syncope — 16 lects / medial a lost in VCaCV (thar kaccara+aka > kacro, karpaṭa+aka > kapṛo, kṣurapra+ikā > kʰurpī) / (changes.sc syncope.a.vccv)
[kcch] ; ∅ > o / C _ #                               ; @morph — masc. -o citation rule removed in round 6 (bare-vs-extended artifact)
[khet] ṇ > n / {r ṛ ḷ ṇ} _                          ; @n-dissimilation — 5 lects / ṇ > n after another retroflex/r / (changes.sc n-dissimilation.rrln)
[Khol] {i u} > {e o} / _ C                         ; @vowel-shift — Kholosi lowers short i u before a consonant (putra > peter, piccʰa > pʰec)
[Khol] cʰ > ś                                      ; @x-nw-ch-sh — cʰikkā > śekken
[Khol] {ṭ ṭʰ ḍ ṇ} > {t tʰ d n}                     ; @deretroflexion — Kholosi loses the retroflex stops (pēṭṭa > pet)
[Sh Khol] s > z / V _ V                                ; @int-voicing @x-nw-s-z — 2 lects / dáśa > dazerā / (changes.sc int-voicing.s)
[G] V ṁ > $1+long+nas / # C* _ C // _ C Long C   ; @anusvara-vowel @x-rg-weight — first syllable only, and not before a heavy long-vowel syllable (stress there: ãgār, ãjīr vs ā̃gaṇ, jā̃gʰ, dā̃tɔ)
[rajasthani-gujarati -mewari_basad -OG -WK -G -marwari_bagra -OMarw -Goj] ; ── block nasals-anusvara-west: nasals-anusvara-west — anusvāra > nasal vowel, Rajasthani type, nasals theme, blocks round 2026-09-30. NIA: the anusvāra (< MIA homorganic nasal before a stop)  [@anusvara-vowel — anusvāra > nasal vowel, long before a light syllable (blocks/nasals-anusvara-west.sc)] ──
[rajasthani-gujarati -OG -WK -G -OMarw -Goj] V ṁ > $1+long+nas / _ C {a i u} | _ C #      ; @anusvara-vowel — 13 lects / lengthened before a light/final syllable: dánta > Rj, had dā̃t, páñca > had, Nimadi pā̃c / (changes.sc anusvara-vowel.vm.caiuc)
[rajasthani-gujarati -mewari_basad -OG -WK -G -marwari_bagra -OMarw -Goj] ; ── block nasals-anusvara-nasal: nasals-anusvara-nasal — NIA anusvāra > short nasal vowel (no lengthening), nasals theme, blocks round 2026-09-30 (pass 2). West Pahari (bhad, jaun, khash, ktg,  [@anusvara-vowel — short nasal vowel elsewhere (before a heavy syllable): áŋgāra > ãgār (chain; Nimadi aŋgār), bʰaṇṭākī > bʰãṭāī (chain)] ──
[Or Bhatri rajasthani-gujarati -mewari_basad -OG -G -marwari_bagra -OMarw -Goj jaun dog bhad ktg khash] V ṁ > $1+nas                                 ; @anusvara-vowel — 13 lects / áŋgāra > ãgār (chain), bʰaṇṭākī > bʰãṭāī (chain) / (changes.sc anusvara-vowel.vm)
[Sh] ; ── block lar-v-b-all: v > b in every position (after intervocalic v is lost): Bengali, Odia, Halbi, Shina, Old Bengali … Laryngeals round (2026-09-30). Edit here, not in the lects. [@v-b — v > b everywhere (blocks/lar-v-b-all.sc)] ──
[Sh] v > b                                        ; @v-b — 11 lects / sarva > sabba > B. sab, vidyut > bijuli / (changes.sc v-b.v)
[M] a > ∅ / V C _ C {ā ī ū ẽ} #                 ; @syncope — round 4: medial a lost before a final long-vowel syllable (-aḍaka > -ḍā, -araka > -rā)
[M WK G] ; ── block finals-syncope-aiu: Syncope of short a i u in VC_CV (blocks theme "finals", 2026-09-30): Marathi, Gujarati, Kutchi-type lects lose any short medial vowel of an open syllable betwee [syncope of a i u in VC_CV (blocks/finals-syncope-aiu.sc) — replaces `use syncope.aiu.vccv`: medial short vowel in an open syllable (āvaraṇa > āvraṇ)] ──
[M WK G] {a i u} > ∅ / V C _ C V                      ; @syncope — 3 lects / medial short vowel lost in VCVCV (M. utkuṭaka > ukḍā, avigʰna+aka > avgʰā; G. ánaśana > aṇsaṇ) / (changes.sc syncope.aiu.vccv)
[Bishnupriya] ; ── block lar-a-rounding-east: Bengali-type rounding of short a (Bengali, Kamta/Rajbanshi, Bishnupriya): a is raised to o before a high vowel in the next syllable (vowel harmony), and the rem [@a-rounding — a > o before a high vowel, first-syllable a > ɔ (blocks/lar-a-rounding-east.sc)] ──
[Bishnupriya] a > o / # C* _ C* {i u ī ū}                  ; @a-rounding — 2 lects / before a high vowel in the next syllable: *raśuna > B. rosun, hariṇá > horin / (changes.sc a-rounding.cciuiu)
[Bishnupriya] a > ɔ / # C* _                               ; @a-rounding — 2 lects / first-syllable a > ɔ: pʰála > pʰɔl, kadala > kɔla, gʰara > gʰɔr / (changes.sc a-rounding.first-syllable)
[A] a > ∅ / _ # // i b _                       ; @final-vowel-loss — only -a (the -iba citation ending exempt); -i -u survive
[Hajong] a > o / # C* _ | _ C a # // _ ŋ                  ; @a-rounding — a > ɔ in the first syllable and before a (later lost) final -a (vānara > banor, pʰala > pʰɔl, kadala > kola); not before ŋ (gaŋ, bʰaŋa, aŋgul)
[G] ; ── block nasals-anusvara-short: nasals-anusvara-short — anusvāra > short nasal vowel before a consonant (no lengthening), nasals theme, blocks round 2026-09-30 (pass 2). The step after a lect' [@anusvara-vowel — short nasal vowel elsewhere (tʰãbʰvũ) (blocks/nasals-anusvara-short.sc)] ──
[G] V ṁ > $1+nas / _ C                           ; @anusvara-vowel — 6 lects / G tʰãbʰvũ (stambʰ-), Or/Halbi short Ṽ before C / (changes.sc anusvara-vowel.vm.preconsonantal.2)
[G] {i u} > a / V C* _ C #                      ; @vowel-shift — short i/u in a final closed syllable (after apocope) > a (aniccʰā > aṇacʰ, *avalipya > oḷap, undura > ũdar); tatsamas keep u (dāruṇ)
[G] h > ∅ / C _ | _ C                           ; @h-loss — h lost next to a consonant after syncope (durlabʰa > dulɔ, *dēhalla > delī, *lābʰana > lāṇī)
[G] ḷ l > l                                     ; @assim-sonorant @x-rg-dla — ḷ + l from syncope in -ala-/-alla- merge to l (*akṣatal- + -alla- > ākʰlɔ)
[G] e > ∅ / V C* _ v ũ #                        ; @syncope — the ε of -evvaü (< -itavyakam) is lost before -vũ (likʰevũ > likʰvũ, kʰāevũ > kʰāvũ)
[VB] {a i u} > ∅ / _ #                            ; @final-vowel-loss — 31 lects / (changes.sc final-vowel-loss.aiu)
[bhal dog] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc) — before the step that frees cluster m (rm > mm > m, Dardic rm > m), so only original single m lenites] ──
[bhal dog] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
[bhal cur] G $1 > $2 // a _                            ; @degem-plain — except after short a (agni > agg, kúṣṭʰa > kuṭṭʰ, *kacca > kaʦʦo)
[bhal] ; ── block lar-deaspiration-voiced-medial: Non-initial voiced aspirates lose aspiration (initial ones survive, with tone/breathiness). Rajasthani (Bagri, Hadothi, Mewari, Dholpuri) and West Pahari (Bhale [@deaspiration @tone — non-initial voiced aspirates (blocks/lar-deaspiration-voiced-medial.sc)] ──
[bhal] {gʰ jʰ ḍʰ dʰ bʰ} > {g j ḍ d b} / !# _        ; @deaspiration @tone — 7 lects / andʰakāra > andāro, garbʰa > gābu / (changes.sc deaspiration.ghjhdhdhbh.cond)
[M VB dog] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina [@degem-plain — NIA plain degemination (GL = G∪G2∪G3) (blocks/length-degem-plain.sc; blocks round 2026-09-30)] ──
[M VB dog] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[M VB dog] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[M] ; ── block nasals-retroflexion: nasals-retroflexion — MIA intervocalic n > ṇ (the Prakrit "no ṇaḥ" change), nasals theme, blocks round 2026-09-30. Stage 1 of n > ṇ > n: a single dental nasal b [@n-retroflex — MIA intervocalic n > ṇ (blocks/nasals-retroflexion.sc); M extends it below] ──
[M] n > ṇ / V _ V                                ; @n-retroflex @nur-innovation @x-nur-n-flap — 40 lects / pānī́ya > G, P, S pāṇī, vána > P vaṇ, S vaṇu, Or baṇa / (changes.sc n-retroflex.intervocalic)
[marathi-bhil -PalyaBareli -OM -Ko-Ul] ; ── block nasals-retroflexion-noninitial: nasals-retroflexion-noninitial — add-on to nasals-retroflexion: n > ṇ in every non-initial position (before consonants and word-finally too), Marathi and the Bh [@n-retroflex — non-initial n > ṇ (blocks/nasals-retroflexion-noninitial.sc)] ──
[marathi-bhil -PalyaBareli -OM -Ko-Ul] n > ṇ / !# _                                 ; @n-retroflex — 5 lects / non-initial n > ṇ: jānāti > M jāṇ, pānīya > M pāṇī / (changes.sc n-retroflex.cond)
[Si Md OSi] ; ── block lar-insular-deaffrication: Insular (Sinhala, Dhivehi) deaspiration and deaffrication, in order: every aspirate loses its aspiration; OIA intervocalic d (and dʰ) is lost; intervocalic c is [@deaspiration @int-stop-loss @affrication — Insular deaspiration, d-loss, c-voicing, c j > s d (blocks/lar-insular-deaffrication.sc)] ──
[Si Md OSi] ; ── block lar-deaspiration-all: Loss of aspiration in every aspirate: the voiced aspirates (the shared lar-deaspiration-voiced), then the voiceless ones (kʰ cʰ ṭʰ tʰ pʰ > k c ṭ t p). Insular ( [@deaspiration — all series (hásta > Si. at-a, bʰāga > bā, kʰaḍga > kaga)] ──
[Si Md OSi] ; ── block lar-deaspiration-voiced: Loss of aspiration in the voiced aspirates (bʰ dʰ gʰ jʰ ḍʰ > b d g j ḍ), everywhere in the word. Shared across families: Dardic (Shinaic, Kohistani, Kunar, Kash [@deaspiration — voiced aspirates (bʰāga > Si. bā, gʰr̩tá > Si. giya)] ──
[Md] {bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ}               ; @deaspiration @tone — 26 lects / dʰūma > Sh. dūm, bʰūmi > K. bum, gʰr̩tá > Gaw. gʌd / (changes.sc deaspiration.bhdhghjhdh)
[OSi] {kʰ cʰ ṭʰ tʰ pʰ} > {k c ṭ t p}               ; @deaspiration — 2 lects / voiceless aspirates (hásta > Si. at-a, kʰaḍga > Si. kaga) / (changes.sc deaspiration.khchththph)
[Si Md OSi] d > ∅ / V _ V                                ; @int-stop-loss — 3 lects / OIA d (and dʰ) lost before j > d (pā́da > pā, paya) / (changes.sc int-stop-loss.d)
[Si Md OSi] c > j / V _ V                                ; @int-voicing — 2 lects / intervocalic c voiced before j > d (kāca > kada, kávaca > kavada) / (changes.sc int-voicing.c)
[Si Md OSi] {c j} > {s d}                                ; @affrication @deaffrication — 3 lects / c > s, j > d (citrá > sita, jána > dena) / (changes.sc affrication.cj)
[pog Md] ; ── block lar-h-loss-initial: Loss of word-initial h: Dhivehi, Kashmiri (Poguli), Tirahi, Shumashti, Kvari, Vasavi. Laryngeals round (2026-09-30). Edit here, not in the lects. [@h-loss — initial h lost (blocks/lar-h-loss-initial.sc)] ──
[pog Md] h > ∅ / # _                                  ; @h-loss — 5 lects / hásta > Md. at, halá > al, hastatala > Kvari ātʰēli / (changes.sc h-loss.h.initial)
[Rathawi] if:!ext ∅ > o / {ḍ ḷ r k ṭ} _ #                     ; @morph — nouns in the extension suffixes -ḍa- -la- -ra- -ka- cited with masc. -o
[Ko-Ul] a > ∅ / V C _ Nasal Stop                     ; @syncope — round 6: medial a lost before a nasal+stop cluster (present participle car-anta > cart-, vikar-anta > vikr-; palyaŋka > pālki)
[Ko-Ul] Nasal > ∅ / C _ C                           ; @cluster-simplify — round 6: a nasal between consonants is lost (car-nt- > cart-: cartā, giḷtā, caḍtā)
[Ko-Ul] ; ── block nasals-nc-lengthen: nasals-nc-lengthen — a > ā before a retained homorganic nasal + stop, Bhil/Halbic and Adivasi Oriya, nasals theme, blocks round 2026-09-30 (pass 2). nasals-anus [@anusvara-vowel — a lengthened before the retained homorganic nasal (candrá > cānd) (blocks/nasals-nc-lengthen.sc)] ──
[Ko-Ul] a > ā / _ Nasal Stop                         ; @anusvara-vowel @degem-lengthen — 12 lects / dánta > dānt > DungraBhili ḍāṭ, candrá > mewari_bannoda cānd / (changes.sc anusvara-vowel.a.nasalstop)
[Md] y > d / # _                                 ; @y-j — Dhivehi y- > d- (yantra > dantura, yaṣṭi > dor̆i); not in CDIAL Si.
[Tir Pas] n {t tʰ} > n d                               ; @nasal-voicing — 2 lects / post-nasal voicing (as Pashai) / (changes.sc nasal-voicing.ntth)
[dardic-south] C v > $1                                     ; @cluster-simplify — 11 lects / C + v > C (jvará > zar, dvā́daśa > dāś) / (changes.sc cluster-simplify.cv)
[bhatr] {v y} > ∅ / V _ V                            ; @v-loss — 9 lects / intervocalic v, y lost / (changes.sc v-loss.vy.intervocalic)
[bhatr] a a > a                                      ; @hiatus — 2 lects / (changes.sc hiatus.aa)
[dardic-south -Gaw -Shum -Dm] r {t tʰ d dʰ ṇ n} > {ṭ ṭʰ ṛ ṛ n n}@2         ; @assim-r @cluster-simplify — 8 lects / r + dental > single retroflex/dental (ardʰá > aṛ, kárṇa > kan) / (changes.sc assim-r.rtthddhnn)
[Gaw Pk Si Dhp] p > v / V _ V                                ; @int-p-v — 119 lects / (changes.sc int-p-v)
[Pk] y > ∅ / V _ V // {a ā} _ {a ā}              ; @int-stop-loss — inherited intervocalic y lost except between a-vowels (valaya, ālaya keep y)
[Pk] {k g c j t d} > y / {a ā} _ {a ā}           ; @ya-sruti @int-stop-loss — sāgara > sāyara
[Pk Ap] {k g c j t d} > ∅ / V _ V                    ; @int-stop-loss — 2 lects / (changes.sc int-stop-loss.kgcjtd)
[Pk] ē > i / {a ē} _ #                           ; @morph — 3sg middle -atē > -ai, -ayatē > -ēi (active ending)
[Pk Ap] ; ── Dentals / initials ──────────────────────────────────────────────────
[Pk Ap] ; ── block nasals-retroflexion-initial: nasals-retroflexion-initial — add-on to nasals-retroflexion: n > ṇ everywhere, word-initial and geminate too (the Prakrit grammarians' "no ṇaḥ sarvatra"), Prakr [@n-retroflex (blocks/nasals-retroflexion-initial.sc)] ──
[Pk Ap] n > ṇ                                        ; @n-retroflex — 2 lects / nára > Pk ṇara, nayana > Pk ṇayaṇa / (changes.sc n-retroflex)
[OM Pk OP] y > j / ṁ _                                 ; @y-j — y after the anusvāra is a stop onset, j (Pischel §252: sáṁyukta > Pk saṁjutta, saṁyōga > OP saṁjog, saṁyāvá > OM sāñjavarī). The chain left ṁy (gold ṁy 0; hill-climb round 7)
[Sh] ; ── block v-b-release: Release of the v-b-skip-optional placeholder: ʋ > v (2026-10-01). Include right after the line that brings v- > b-. [releases the v-b-skip-optional placeholder] ──
[Sh] ʋ > v                                        ; @v-b — release (blocks/v-b-skip-optional.sc) / (block v-b-release)
[bihari-tharu central-east-pahari-romani Kho K Kal dod Gaw Shum Wot Chil Dm Tir eastern hindi marathi-bhil -PauriBareli Pk Ap Md OSi northwest Gmb rajasthani-gujarati -Goj west-pahari] y > j / # _                                  ; @y-j — 101 lects / (changes.sc y-j.initial)
[S] ; ∅ > u / C _ #                               ; @morph — Sindhi nouns end in a vowel: consonant-final (root / suffix) headwords take masc. -u (*kakudʰaṭṭ > kūhaṭu, *cuttaḍ > cutaṛu; round 2)   [round 7: @morph removed]
[S P] {ā ī ū ē ō} Nasal > $1+nas / _ Stop          ; @anusvara-vowel — 2 lects / nasal + stop after a LONG vowel > nasal vowel + stop (short vowels keep the nasal consonant: aŋka > aŋg); āntrá > ā̃d, pāṇḍú > pā̃ḍū, nānda > nā̃d; Gurmukhi writes bindī on long vowels, ṭippī (n) after short ones, so this may be partly orthographic (round 2) / (changes.sc anusvara-vowel.aiueonasal.stop)
[awan] %source rensch-hallberg-oleary1992 gill
[S] ã u > õ / _ #                               ; @hiatus — ordinal -ama > -ãu > -õ (aṣṭama > aṭʰõ, daśama > ᶑahõ)
[hindi -bundeli_atarra -OAw -NDu -SSr] v > u / V _ #                                ; @x-hindi-v-vocalise — 4 lects / final -v (< -m-, -p-) vocalises (nā́man > nāũ) / (changes.sc v-vocalise)
[S] a ī > ī / C _ #                              ; @hiatus — 13 lects / -aī (from -aka + -ikā) contracts (ciṭaka > ciṛī, dhaṭaka > dʰaṛī) / (changes.sc hiatus.ai.c)
[S] a > i / V C* C _ {ṇ r ṛ l} {o ī i} #       ; @raising — a > i before a final -ṇo/-ṇī/-ro/-ṛo/-lo/-rī after a non-initial syllable (vartana > vaṭiṇo, mantʰaraka > mandʰiro, kuṇḍaḍaka > kuniṛo, śyāmalaka > sā̃viro; liquids added round 6)
[S] a > i / V C* C _ {r ṛ l} a ṇ u #             ; @raising — the same before the verbal-noun ending (*uttʰalyati > utʰilaṇu, *ullaṭati > uliṛaṇu, *ubbal > uɓiraṇu; round 6)
[kcch] ∅ > y / ā _ ṇ ū #                         ; @glide — glide y between a stem-final ā and the infinitive -ṇū (kʰā́dati > kʰāyṇū, *karmāpayati > kamāyṇū; round 6 second pass)
[kcch] u > a u / # _                               ; @vowel-shift — Kutchi initial u- > au- (utkalati > aukkarṇū; probably [ɔ])
[hal M mewari_bannoda] y > j                                        ; @y-j — 3 lects / initial and post-consonantal y / (changes.sc y-j)
[PalyaBareli DungraBhili Rathawi PauriBareli mewari_bannoda Vasavi Bhili] ; ── block lar-s-h: s > h everywhere: Bhil. Laryngeals round (2026-09-30). Edit here, not in the lects. [@s-h — s > h everywhere (blocks/lar-s-h.sc)] ──
[PalyaBareli DungraBhili Rathawi PauriBareli mewari_bannoda Vasavi Bhili] s > h                                        ; @s-h — 8 lects / divasá > dahaḍo, sarpá > hāp, sūcī́ > hūī / (changes.sc s-h.s)
[ktg] Gx $1 > $2                                  ; @degem-plain — all geminates except ṭṭʰ (kept: ʣeṭṭʰ, koṭṭʰ, hoṭṭʰ)
[ktg] u C > $2 v / # _ ā                          ; @x-wpah-u-metathesis — initial u- lost with labialisation of the next consonant before ā (uttāná > tvaṇɔ; round 2)
[ktg] ṭ ṭ > ṭ                                     ; @degem-plain — unaspirated ṭṭ
[Or Bhatri jaun dog bhad ktg khash] ; ── block nasals-anusvara-nasal: nasals-anusvara-nasal — NIA anusvāra > short nasal vowel (no lengthening), nasals theme, blocks round 2026-09-30 (pass 2). West Pahari (bhad, jaun, khash, ktg,  [@anusvara-vowel (blocks/nasals-anusvara-nasal.sc)] ──
[WPah] a a > o / _ #                                ; @hiatus — 20 lects / -aka > -ao > -o (masc. extended stems) / (changes.sc hiatus.aa.final)
[DungraBhili] ; ── block lar-deaspiration-voiced: Loss of aspiration in the voiced aspirates (bʰ dʰ gʰ jʰ ḍʰ > b d g j ḍ), everywhere in the word. Shared across families: Dardic (Shinaic, Kohistani, Kunar, Kash [@deaspiration — voiced aspirates lose aspiration (blocks/lar-deaspiration-voiced.sc)] ──
[DungraBhili] {bʰ dʰ gʰ jʰ ḍʰ} > {b d g j ḍ}               ; @deaspiration @tone — 26 lects / dʰūma > Sh. dūm, bʰūmi > K. bum, gʰr̩tá > Gaw. gʌd / (changes.sc deaspiration.bhdhghjhdh)
[mewari_bannoda] c > s                                        ; @deaffrication — 3 lects / deaffrication c > s (nīcá > niso) / (changes.sc deaffrication.c)
[Or B] ; ── block liquids-flap: Retroflex flapping — single post-vocalic ḍ ḍʰ > ṛ ṛʰ (NIA; liquids blocks round, 2026-09-30). The flap of the Hindi belt, Punjabi-Lahnda-Sindhi, Pahari, Bengali [@flap] ──
[Or B] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ {ḍ ḍʰ}            ; @flap — 8 lects / kaṭāha > H kaṛāhā, kukkuṭá > H kukṛā, *dʰūḍi > H dʰūṛ; geminate ḍḍ stays a stop / (changes.sc flap.ddh.postvocalic.except)
[BH Hajong] ; ── block finals-apocope-ai: NIA apocope of -a and -i (blocks theme "finals", 2026-09-30): final short -a and -i are lost while -u survives (Bengali, Bishnupriya's neighbours BH and Hajong, [apocope of -a -i (blocks/finals-apocope-ai.sc) — replaces `use final-vowel-loss.ai.2`: -a and short -i (as in B.); MIA -ī (< -in) survives] ──
[BH Hajong] {a i} > ∅ / _ #                              ; @final-vowel-loss — 8 lects / final -a -i lost, -u kept (B. akṣára > ākʰar, agní > āg, aŋgúli > āŋul) / (changes.sc final-vowel-loss.ai)
[B] a ã > a / _ #                                ; @final-nasal @hiatus — 7 lects / neuter -aã (< -akam) > -a after the apocope, so the -a survives (kar-āpanakam > karānaã > karāna, ugāna, ucʰṛāna; hill-climb round 1) / (changes.sc final-nasal.aa)
[B] h > ∅ / V _ #                                 ; @h-loss — word-final h lost after apocope (kaṭāha > kaṛā, gā́tʰā > gā)
[Bishnupriya] a > o / _ C a #                             ; @a-rounding — non-initial a > o before a (later lost) final -a (vānara > banor, kambalá > kɔmbol). Round 7: split into harmony / first-syllable ɔ / o once ɔ stopped being folded
[Bishnupriya] ; ── block finals-apocope-a: NIA apocope of -a only (blocks theme "finals", 2026-09-30): final short -a is lost while final -i -u survive (Maithili, eastern Tharu, Bishnupriya, Desia; Balka [apocope of -a only (blocks/finals-apocope-a.sc) — replaces `use final-vowel-loss.a`: only -a] ──
[Bishnupriya] a > ∅ / _ #                                  ; @final-vowel-loss — 20 lects / final -a lost, -i -u kept (Mth. aŋkurá > ā̃kur, adyá > āj, áŋga > ā̃g) / (changes.sc final-vowel-loss.a)
[K] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc) — before the step that frees cluster m (rm > mm > m, Dardic rm > m), so only original single m lenites] ──
[K] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
[Gaw Shum Wot Dm] ; ── block mia-dardic-clusters: Dardic cluster simplification (Kohistani, Kunar, Pashai; blocks theme "mia", 2026-09-30): clusters reduce to a single consonant without the mainland MIA geminat [@cluster-simplify @assim-stop @assim-r — Dardic cluster simplification without geminates (blocks/mia-dardic-clusters.sc; blocks round 2026-09-30)] ──
[Gaw Shum Wot Dm] C v > $1                                     ; @cluster-simplify — 11 lects / C + v > C (dvāra > dar, jihvā > jib-, pakva > pak) / (changes.sc cluster-simplify.cv)
[Gaw Shum] y > ∅ / C _                                  ; @cluster-simplify — 25 lects / C + y > C (satya > sat, kanyā > kana, madʰya > mad) / (changes.sc cluster-simplify.y.postconsonantal)
[Gaw Shum Wot Dm] {k p g b} {t tʰ d dʰ} > $2                   ; @assim-stop @cluster-simplify — 12 lects / k p g b + dental > dental (bʰakta > bʰat, sapta > sat, dugdʰa > dud-) / (changes.sc assim-stop.kpgbtthddh)
[Gaw Shum Wot Dm] C $1 > $1                                    ; @cluster-simplify @degem-plain — 18 lects / geminates simplify (putta > put, kaṭṭa > kaṭ, pakka > pak) / (changes.sc degem-plain.c)
[Gaw Shum Wot Dm] r > ∅ / _ C                                  ; @assim-r @cluster-simplify — 16 lects / r before a consonant lost (karma > kam, sarpa > sap, dīrgʰa > dīg) / (changes.sc assim-r.r.preconsonantal)
[Gaw] s C > $2 / # _                               ; @initial-degem — 5 lects / initial s + C > C (skandʰá > kan) / (changes.sc initial-degem.sc)
[dardic-south] ; ══ degemination, nasal clusters, word edges ═════════════════════════════
[dardic-south -Gaw -Chil] {t d} {k kʰ g gʰ p pʰ b bʰ} > $2             ; @assim-stop @cluster-simplify — 4 lects / dental stop + labial/velar stop > the second (utpaṭati > ōpuṛ, mudga-type); round 6 / (changes.sc assim-stop.tdkkhgghpp)
[dardic-north -K dardic-south -Gowro -Shum -Wot -Pas nuristani -Ash -Wg] t s > ʦ                                      ; @affrication @cluster-simplify — 13 lects / ts > ʦ (útsa > ūʦ, mátsya > maʦ-); round 6 / (changes.sc affrication.ts)
[Gaw Wot Dm Tor Tir Pas] {k p g b} {t tʰ d dʰ} > $2                   ; @assim-stop @cluster-simplify — 12 lects / labial/velar stop + dental stop > dental without gemination (saptá > sat, taptá > tat, bʰaktá > bʰā̌tʰ, *gʰr̩pta > git); all 12 lects; round 6 / (changes.sc assim-stop.kpgbtthddh)
[dardic-south -bhatr -Pas] ; ── block nasals-mb-m: nasals-mb-m — mb(ʰ) > m, Dardic (Bashkarik, Gowro, Indus Kohistani, Tirahi), nasals theme, blocks round 2026-09-30 (pass 2). [@nasal-cluster @cluster-simplify — mb > m (jámbʰa-type, stambʰa > tʰam); round 6 (blocks/nasals-mb-m.sc)] ──
[dardic-south -bhatr -Pas] m {b bʰ} > m                                 ; @cluster-simplify @nasal-cluster — 4 lects / stambʰa > tʰam, jámbʰa-type mb > m / (changes.sc nasal-cluster.mbbh)
[Chitwan DewasDoneDanuwar KochariyaEastDanuwar Bote N bhatr Mai eastern As-Ul Ash] {ī ū ī̃ ū̃} > {i u ĩ ũ}                      ; @length-loss @nur-innovation @vowel-shift — 31 lects / i/ī, u/ū merged, also the ī ū from compensatory lengthening (kʰura > kʰur, tikta > titta > tit) / (changes.sc length-loss.iuiu)
[Tir eastern -Bishnupriya -BH -A -Hajong] ś > s                                        ; @sibilant-s — 5 lects / ś > s (toulmin writes s) / (changes.sc sibilant-s.s)
[eastern -BH -A -MB -MH -KS Bhatri hal bagri_fatehabad mewati_jhambaus] ; ── block lar-v-b-all: v > b in every position (after intervocalic v is lost): Bengali, Odia, Halbi, Shina, Old Bengali … Laryngeals round (2026-09-30). Edit here, not in the lects. [@v-b — v > b everywhere (blocks/lar-v-b-all.sc)] ──
[eastern -BH -A -MB -MH -KS Bhatri hal bagri_fatehabad mewati_jhambaus] v > b                                        ; @v-b — 11 lects / sarva > sabba > B. sab, vidyut > bijuli / (changes.sc v-b.v)
[Sh Kho Kund K Kal Bshk Shum Wot Dm Tor Tir eastern -Or -BH -MB -OB Kannauji-Central NDu Aw SSr marathi-bhil -PalyaBareli -PauriBareli -M -Vasavi -Bhilali -OM -RathwiBareli -Ko-Ul NiDoc Md As-Ul D mewari_basad hadothi_kelwada mewati_jhambaus Rj] ; ── block nasals-dental-merger: nasals-dental-merger — NIA ṇ > n everywhere, nasals theme, blocks round 2026-09-30. Stage 2 (Hindi belt, Bihari, Tharu, Nepali …): every ṇ — the MIA intervocali [@n-dental — NIA ṇ > n (blocks/nasals-dental-merger.sc)] ──
[Sh Kho Kund K Kal Bshk Shum Wot Dm Tor Tir eastern -Or -MB -OB Kannauji-Central NDu Aw SSr marathi-bhil -PalyaBareli -PauriBareli -M -Vasavi -Bhilali -OM -RathwiBareli -Ko-Ul NiDoc Md As-Ul D mewari_basad hadothi_kelwada mewati_jhambaus Rj] ṇ > n                                        ; @n-dental — 66 lects / pānī́ya > H, Bhoj pānī, guṇá > H gun, vána > Bhoj ban / (changes.sc n-dental.n)
[Kt] m > ∅ / C e _ #                             ; @final-cons-loss — the -m of the neuter -akam is lost after -aam > -em (cāra > cāre, *kʰucc > kuce, *ḍuḍḍa > ḍūḍe, pū́rva > puve, vāsá > vase; akam-best preds -em 47, gold -e 48 vs -em 2) ; gates ext −0.0046/−0.0019, verb −0.0035/−0.0081, 57 better / 7 worse (hill-climb round 4)
[nuristani -Kam] r > ∅ / V _ {p pʰ b bʰ v}                    ; @cluster-simplify @nur-innovation — 4 lects / R6 fourth pass: r is lost before p/b/v, after lenition (sárva > Kt súv / Ash sev / PNur *sav, dārva > dāv, *carpa > PNur *capa, durbala > Gmb dabili); rm metathesises instead (kūrma > kr̆um), as in the r-loss rule of Wg, which also covers m / (changes.sc nur-innovation.r.vpphbbhv)
[Wg] r > ∅ / V _ {p pʰ b bʰ m v}                 ; @cluster-simplify @nur-innovation — R6 fourth pass: r is lost before a labial, after lenition (so v stays): pū́rva > pə̄v, *śarva > ʦə̄v, sárva > sab, dārva > dāv, cárman > cām, nirmala > nimal, carpaṭa > capā́l; before velars it metathesises instead (*margā > mr̆aŋ). Exceptions gárbʰa > grop, dárvi > dur
[nuristani] ; ── Clusters ─────────────────────────────────────────────────────────────
[Ash Wg] {a ā} > å / C _ {n ñ ṁ} C                    ; @nur-innovation @vowel-shift — 2 lects / round 7 (marker, as Ash): a before a dental/palatal nasal cluster or anusvāra, not word-initially; see å > ũ below / (changes.sc nur-innovation.aa.cnnmc)
[PauriBareli] {c j} > {t d} {s z}@1 // _ Front            ; @affrication — c j > ts dz except before front vowels (written ts, dz)
[bihari-tharu -Morang central-east-pahari-romani dardic-north -Kho dardic-south eastern hindi -OAw -NDu -bagheli_lakshman -SSr marathi-bhil -OM -Bhili D S P L khet srk nuristani rajasthani-gujarati -mewari_basad -OG -OMarw west-pahari -kiuth] ; ── block nasals-anusvara-loss: nasals-anusvara-loss — leftover anusvāra lost without nasality, nasals theme, blocks round 2026-09-30 (pass 2). The anusvāra that survives to this point (before [@anusvara-vowel — anusvāra before s/h lost without trace (blocks/nasals-anusvara-loss.sc)] ──
[-Morang -Kho -OAw -bagheli_lakshman -NDu -SSr -OM -Bhili -Pk -Pa -Si -Dhp -Md -NiDoc -As-Ul -OSi -Ap -KharI -OP -awan -kcch -thar -poth -OG -OMarw -mewari_basad -kiuth] ṁ > ∅                                        ; @anusvara-loss @anusvara-vowel — 53 lects / pāṁśu > awan pā, māṁsá > Si mas, áṁsa > Si as / (changes.sc anusvara-loss)
[bhal] {ṇ n} {ḍ d} > n n                           ; @nasal-assim — voiced dental/retroflex stop assimilated to the nasal (daṇḍá > ḍann)
[K] ; ── block mia-shina-clusters: Shina/Chitrali/Kashmiri late cluster simplification (blocks theme "mia", 2026-09-30): r lost before a consonant after a vowel, dental + y > palatal, C + y/v > C [@assim-r @palatal-Cy @cluster-simplify — postvocalic rC > C, t d dʰ + y > c j j, C + y/v > C (blocks/mia-shina-clusters.sc; blocks round 2026-09-30)] ──
[K] r > ∅ / V _ C                                ; @assim-r @cluster-simplify — 10 lects / postvocalic r + C > C (karma > kam, sarpa > sap, cárma > cam) / (changes.sc assim-r.r.vc)
[K] C {y v} > $1                                 ; @cluster-simplify — 18 lects / C + y/v > C (dvāra > dar, kanyā > kan, pakva > pak) / (changes.sc cluster-simplify.cyv)
[pog K dod] ; ── Affricates, sibilants, nasals ───────────────────────────────────────
[pog K] ; ── block lar-dardic-affrication: Dardic affrication of the palatal stops: c cʰ j > ʦ ʦʰ z (Kashmiri, Kohistani, Kunar). Included by lar-dardic-laryngeals where the deaspiration follows directly [@affrication — c cʰ j > ʦ ʦʰ z (blocks/lar-dardic-affrication.sc)] ──
[pog K] {c cʰ j} > {ʦ ʦʰ z}                          ; @affrication — 10 lects / cáturdaśa > Chil. ʦondʌś, jihvā́ > zīb, cʰāyā́ > Gaw. ʦʰā́ilat / (changes.sc affrication.cchj)
[K] m r > m b / V _                              ; @mr-mb — 5 lects / MIA medial mr > mb with an epenthetic stop (āmrá > amba, tāmrá > tamba); unified at round-7 integration / (changes.sc mr-mb.mr.postvocalic)
[K dod] ; ── block ks-cch-optional: Sporadic kṣ > ccʰ beside kkʰ (MIA doublets: akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta). `optional:` = sporadic: forks the derivation where it applies (best [sporadic (blocks/ks-cch-optional.sc)] ──
[K dod] optional: k ṣ > c cʰ                         ; @ks — sporadic kṣ > ccʰ beside kkʰ (akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta) / (block ks-cch-optional)
[K dod] k ṣ > cʰ                                     ; @ks — 2 lects / kṣība > cʰivun (after the affrication, which it escapes) / (changes.sc ks.ks.4)
[K] {t d} s > ʦʰ                                ; @ts-cluster — ts > ʦʰ (vatsá > voʦʰ, mátsya > maʦʰ)
[K] {ś ṣ} > h // _ {i ī e ē y ɨ}                ; @s-h — ś, ṣ > h except before front vowels (śará > har, kuṣáti > kuhun, śīná > śīn), s kept; the -ᵘ̈ (< -ikā) counts as front (kaṣá-ikā > küśᵘ̈, round 7)
[Kund] ; ── Initials ────────────────────────────────────────────────────────────
[dardic-north -Kho] ; ── Edges ───────────────────────────────────────────────────────────────
[Sh] {ā ī ū ē ō} > {a i u e o} / _ C* V C* V              ; @vowel-shift @two-mora — long vowels shorten before the (penultimate) stressed syllable, i.e. pretonically, before apocope (stʰūla-uka > tʰulu, rā́jan-ikā > raźi) (round 6, re-gated on ext accent)
[Sh] ; ── block finals-apocope-shina: Shina-type apocope in polysyllables (blocks theme "finals", 2026-09-30): final vowels are lost after one to three consonants preceded by a vowel, but not the -o [apocope in polysyllables, -o kept (blocks/finals-apocope-shina.sc) — replaces `{a i u e ā ī ū ē ō} > ∅ / V (C) C _ # | V C C C _ #`: not in monosyllables, and not the -o of -aka nouns] ──
[Sh] {a i u e ā ī ū ē ō} > ∅ / V (C) C _ # | V C C C _ # ; @final-vowel-loss — final vowel lost in polysyllables, -o kept (Sh. áŋgāra > agār, aṣṭā́ > aṭ, āmrá > ām) / (block finals-apocope-shina)
[Sv Kal] ; ── block finals-apocope-nonhigh: Apocope of non-high vowels in polysyllables (blocks theme "finals", 2026-09-30): final -a -ā -ē -ō are lost after one to three consonants preceded by a vowel, w [non-high final vowels lost in polysyllables (blocks/finals-apocope-nonhigh.sc) — replaces `use final-vowel-loss.aaeo.vccvccc`: only non-high vowels are lost (-i and -u survive, as in Khowar), not in monosyllables] ──
[Sv Kal] {a ā ē ō} > ∅ / V (C) C _ # | V C C C _ #    ; @final-vowel-loss — 5 lects / final non-high vowels lost in polysyllables, -i -u kept (Kal. áŋga > oŋg, aṣṭā́ > aṣṭ, índra > indr) / (changes.sc final-vowel-loss.aaeo.vccvccc)
[Kal] a > ∅ / C _ i                               ; @hiatus — a + i > i (after stop loss)
[K] ∅ > v / C V _ {u ū} #                       ; @ya-sruti @hiatus — a final hiatus left by stop loss gets a glide before apocope, so the final vowel goes (kʰāduka > kʰāv, dā́tu > dāv, pādú > pāv) …
[K] ∅ > y / C V _ {a ā i ī} #                   ; @ya-sruti @hiatus — … y before a front vowel or a (cʰāyā́ > ʦʰāy, māyā́ > māy, kʰātikā > kʰay, sātí > say); onsetless āā (āyā) keeps it: gate ext −0.0020/−0.0010, net +27 (hill-climb round 3)
[nuristani -Ash] ; ── block finals-nuristani-apocope: Nuristani apocope (blocks theme "finals", 2026-09-30): after lenition, final -ā and polysyllabic -ī are lost, then final -a (marked or not) after a consonant, t [Nuristani apocope (blocks/finals-nuristani-apocope.sc) — replaces `use final-vowel-loss.a.c.2`; `use final-vowel-loss.i.vcc`; `use final-vowel-loss.aa.c`; `use final-vowel-loss.a.c.3`; `ă > a`: final -ā lost (as in PNur) / round 2: final long -ī is lost like -ā in polysyllables (nadī́, vā́śī > voś); short -i stays (Kt: dropping -i too costs +0.0100). −0.0020; accent-blind (keeping tonic -ī́ is worse in PNur, Kt, Kam, Ash). Rejected in Pr, Gmb, neutral in Wg / round 6 (coordinator decision): full apocope of final -a after a consonant, tonic -á included. Ext NED prefers it strongly (see notes R6.4); the kept final vowels of the reflexes come from the -aka extension. Accent gate waived for this bare-vs-extended artifact / posttonic final -a is lost after clusters too, tonic -á survives (gōṣṭʰá > guṣṭá) / unmark surviving marked vowels] ──
[thar kcch nuristani -Ash] ā > ∅ / C _ #                                ; @final-vowel-loss — 5 lects / final -ā lost (Kt. pántʰā > put, vḗlā > vel, mā́lā > mol) / (changes.sc final-vowel-loss.a.c.2)
[nuristani -Ash] ī > ∅ / V C* C _ #                           ; @final-vowel-loss — 4 lects / final -ī lost in polysyllables (Kt. naptī́ > nut, rā́trī > r̆otr) / (changes.sc final-vowel-loss.i.vcc)
[nuristani -Ash] {a ă} > ∅ / C _ #                            ; @final-vowel-loss — 2 lects / final -a lost after a consonant, tonic -á included (Kt. aṭṭāla > āṭol, kālá > kol, kīla > kil) / (changes.sc final-vowel-loss.aa.c)
[nuristani -Ash] ă > a                                        ; @notation — unmark surviving marked vowels / (block finals-nuristani-apocope)
[nuristani -Ash] ĭ > ∅ / C _ #                                ; @final-vowel-loss @stress — 3 lects / posttonic final -ĭ lost after a consonant, tonic -í survives (Wg. hári > zar, Kam. dárvi > dur) / (changes.sc final-vowel-loss.i.c)
[nuristani -Ash] ĭ > i                                        ; @notation — unmark surviving marked vowels / (block finals-nuristani-apocope)
[Kalk Kund K Ash] ; ── block finals-apocope-dardic: Dardic apocope of all final vowels in polysyllables (blocks theme "finals", 2026-09-30): any final vowel is lost after one to three consonants preceded by a vow [apocope in polysyllables (blocks/finals-apocope-dardic.sc) — replaces `use final-vowel-loss.v.vccvccc`: not in monosyllables (tráyaḣ > Sh ʦ̣ē, not *ʦ̣)] ──
[Kalk Kund K Ash] V > ∅ / V (C) C _ # | V C C C _ #            ; @final-vowel-loss — 3 lects / final vowel lost in polysyllables (K. áŋga > ang, ánta > and; Ash. kukkuṭá > kukuṛ) / (changes.sc final-vowel-loss.v.vccvccc)
[Kho] {y j} > {ź ź} / # _                          ; @y-j — Khowar: initial y j > ź (with v > b in lar-v-b)
[Kho] a > ∅ / V (C) C _ # | V C C C _ #          ; @final-vowel-loss — only -a is lost (-i and -u survive), and not in monosyllables
[Phal] Stress > $1+acc / _ !#* Vedic                ; @tone — flag (prosody only): a stressed vowel with the Vedic accent on a later syllable, before apocope removes that syllable (kālá, adyá, kṣurá) …
[Phal] Acc > $1-acc                                 ; @tone — … and the flagged vowel keeps its accent as a latent Vedic mark (unwritten again; the %source blocks below write or restore it) (round 7, tone-gated)
[Phal] {a i u ā ī ū ē ō} > ∅ / V (C) C _ # | V C C C _ #     ; @final-vowel-loss — not in monosyllables, not the -o of -aka nouns, not the verbal -e
[dardic-north -pog -K PalyaBareli Noiri mewati_jhambaus bhal jaun dog kc kgr khash kul] ; ── block hiatus-absorb: Hiatus theme — absorption of a by a neighbouring vowel (blocks round, 2026-09-30). After the loss of intervocalic consonants (incl. the NIA v/h loss), short a i [@hiatus — blocks/hiatus-absorb.sc; a after another vowel absorbed] ──
[dardic-north PalyaBareli Noiri WK mewari_dholpura mewati_jhambaus west-pahari -kiuth -bhad -mand -cur -cam -pan] a > ∅ / _ ā                                  ; @hiatus — a-ā > ā (kumbʰakāra > kumbʰaār > kumhār) / (block hiatus-absorb)
[dardic-north -pog -K Or PalyaBareli Noiri Marw mewati_jhambaus west-pahari -kiuth -mand -ktg -cur -kvar -WPah -cam] a > ∅ / V _                                  ; @hiatus — a absorbed after a vowel (kāhala > kaaḷ > kāḷ; -iya > -i after final-vowel loss) / (block hiatus-absorb)
[marathi-bhil -PalyaBareli -hal -RathwiBareli] cʰ > s                                       ; @deaffrication — 6 lects / cʰ > s / (changes.sc deaffrication.ch)
[PauriBareli Vasavi Noiri RathwiBareli] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[PauriBareli Vasavi Noiri RathwiBareli] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[bihari-tharu eastern marathi-bhil -M -OM] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[bihari-tharu eastern marathi-bhil -M -OM] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[mewari_bannoda] o > ā / V C* _ #                             ; @final-vowel — polysyllabic -o (< -aka) > -ā, the Ahirwati/Haryanvi masc. ending (lambā, kelā, sūkʰā, baḍā, saberā); monosyllables keep -o (do, so) (hill-climb round 1)
[cur] ; ── block finals-syncope-pahari: West Pahari syncope (blocks theme "finals", 2026-09-30): a medial short vowel of an open non-initial syllable is lost (VCVC), then medial a also after a closed  [W. Pahari syncope (blocks/finals-syncope-pahari.sc) — replaces `use syncope.aiu.vcc`; `use syncope.a.vcccv`: medial short vowel in an open non-initial syllable (round 2) / medial a also lost after a closed syllable when a vowel follows (round 6 second pass)] ──
[cur] {a i u} > ∅ / V C _ C                        ; @syncope — 9 lects / open non-initial syllable (kgr. cālana+ikā > cālṇī, apakṣá+aka > aukʰā; WPah. *muṣala > muśḷ) / (changes.sc syncope.aiu.vcc)
[cur] a > ∅ / V C C _ C V                          ; @syncope — 7 lects / also after a closed syllable (kgr. *gilla+araka > gilrā, *paddʰara+aka > padrā, *vartmapāla > batvāl) / (changes.sc syncope.a.vcccv)
[kcch bhal cur cam pan] v > ∅ / V _                                  ; @v-loss — 4 lects / (changes.sc v-loss.v.postvocalic)
[Kal] {a á ā} m i > {e é e} m / V C* _ #          ; @epenthesis — in words of three or more syllables, final -i moves into the preceding syllable after m: 1sg -āmi/-ami > -aim > -em (jānāmi > jʰonem, karṣāmi > kaṣalém)
[dardic-north -K] ; ── block lar-deaspiration-tsh-final: Word-final retroflex affricate ʦ̣ʰ (< kṣ) deaspirates: Khowar, Kalasha, Phalura, Shina. Laryngeals round, pass 2 (2026-09-30). Edit here, not in the lects. [@deaspiration — word-final ʦ̣ʰ deaspirates (blocks/lar-deaspiration-tsh-final.sc)] ──
[dardic-north -K] ʦ̣ʰ > ʦ̣ / _ #                               ; @deaspiration — 3 lects / ŕ̩kṣa > Sh. iʦ̣, drā́kṣā > Kal. drāʦ̣, rakṣá > raʦ̣ / (changes.sc deaspiration.tsh.final)
[dardic-north -K] ñ > n / _ Stop                              ; @notation — ñj written nj
[Kund] {ś ṣ s} > {y y z} / V _ #                   ; @x-dardic-north-final-sibilant-lenition — word-final sibilants lenite (ākāśá > ay, dvā́daśa > bay, divasá > dīz)
[dog kiuth mand] {ū} > {u}                                    ; @vowel-shift — 2 lects / ū shortened / (changes.sc vowel-shift.u)
[kiuth] {ā ī ū} > {a i u} / _ C V C* V               ; @vowel-shift — 4 lects / long vowel shortened two or more syllables before the end (NIA weight/stress; round 2) / (changes.sc vowel-shift.aiu.cvcv)
[Darai Majhi] V G $2 > $1+long $3 / # C* _                 ; @degem-lengthen — 5 lects / in any initial syllable (Nepali-type; the disyllable-only version is worse by 0.005/0.002) / (changes.sc degem-lengthen.vg.first-syllable)
[bihari-tharu -Buksa -Rana -Bhoj -Morang -Mth -Bi eastern -Or -AdivasiOriya -MB -OB -B] ; ── block liquids-flap-to-r: Intervocalic ḍ ḍʰ > ṛ ṛʰ, then the flap merges with r. Shared by 21 lects in 6 clades (Bihari, E. Hindi, E. Pahari, Eastern, Other, W. Hindi); blocks round 2026 [Intervocalic ḍ ḍʰ > ṛ ṛʰ (blocks/liquids-flap-to-r.sc)] ──
[bihari-tharu -Buksa -Rana -Bhoj -Morang -Mth -Bi eastern -Or -AdivasiOriya -MB -OB -B] ; ── block liquids-flap: Retroflex flapping — single post-vocalic ḍ ḍʰ > ṛ ṛʰ (NIA; liquids blocks round, 2026-09-30). The flap of the Hindi belt, Punjabi-Lahnda-Sindhi, Pahari, Bengali [@flap — Tharu flap merges with r (*dʰūḍi > dʰur)] ──
[bihari-tharu -Buksa -Rana -Bhoj -Morang -Mth -Darai -Bi eastern -Or -AdivasiOriya -MB -OB -B] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ {ḍ ḍʰ}            ; @flap — 8 lects / kaṭāha > H kaṛāhā, kukkuṭá > H kukṛā, *dʰūḍi > H dʰūṛ; geminate ḍḍ stays a stop / (changes.sc flap.ddh.postvocalic.except)
[bihari-tharu -Buksa -Rana -Bhoj -Morang -Mth -Bi eastern -Or -AdivasiOriya -MB -OB -B] ; ── block liquids-flap-rhotic: Flap-rhotic merger — ṛ ṛʰ > r rʰ (liquids blocks round, 2026-09-30). Stage 2 of the flap history: the retroflex flap (blocks/liquids-flap.sc, or the Dardic inte [@flap — the flap merges with r] ──
[bihari-tharu -Buksa -Rana -Bhoj -Morang -Mth -Bi Dotyali Kalk pog K Bshk eastern -Or -AdivasiOriya -MB -OB -B Pr] {ṛ ṛʰ} > {r rʰ}                              ; @flap — *dʰūḍi > Dang dʰur, kukkuṭá > K kɔkur, kʰaḍaka > K kʰorᵘ / (block liquids-flap-rhotic)
[Mai] ; ── block nasals-postnasal-voicing: nasals-postnasal-voicing — NT > ND, nasals theme, blocks round 2026-09-30. A voiceless stop, aspirated or not, is voiced after a nasal (or anusvāra-turned-nasal [@nasal-voicing — NT > ND incl. aspirates (blocks/nasals-postnasal-voicing.sc)] ──
[Mai] Nasal {k kʰ c cʰ ṭ ṭʰ t tʰ p pʰ} > $1 {g gʰ j jʰ ḍ ḍʰ d dʰ b bʰ}@2 ; @nasal-voicing — 10 lects / dánta > awan dand, káṇṭaka > awan kaṇḍā, śaŋkú > Kho śoŋg, páñca > Kho ponj / (changes.sc nasal-voicing.nasalkkhcc)
[Tor] {t d} {k kʰ g gʰ p pʰ b bʰ} > $2             ; @assim-stop @cluster-simplify — 4 lects / dental stop + labial/velar stop > the second (utpaṭati > ōpuṛ, mudga-type); round 6 / (changes.sc assim-stop.tdkkhgghpp)
[dardic-south -Gowro -Bshk -Chil -bhatr -Mai] t s > c                                      ; @affrication @cluster-simplify — 4 lects / ts > c (útsa > ūc, mátsya > mac-); round 6 / (changes.sc affrication.ts.2)
[Phal dardic-south nuristani -Gmb] C $1 > $1                                    ; @cluster-simplify @degem-plain — 18 lects / degemination without compensatory lengthening; ordered after lenition, which geminates escape / (changes.sc degem-plain.c)
[Gowro] ; ── block hiatus-dardic: Hiatus theme — Kohistani / Kunar / Pashai hiatus (blocks round, 2026-09-30). The old intervocalic v and y are lost (before the new v < p), and the second vowel  [@v-loss @hiatus — blocks/hiatus-dardic.sc; intervocalic v, y lost] ──
[Gowro] {v y} > ∅ / V _ V                            ; @v-loss — old intervocalic v, y lost (before p > v: the new v survives) / (block hiatus-dardic)
[Gowro] {a i} > ∅ / V _                              ; @hiatus — second vowel of a hiatus absorbed (kʰādati > kʰā(n)a, -anaka > -aṽaa > -aṽa) / (block hiatus-dardic)
[Bhoj Mth Bi Ku N Gowro] ; ── block liquids-flap: Retroflex flapping — single post-vocalic ḍ ḍʰ > ṛ ṛʰ (NIA; liquids blocks round, 2026-09-30). The flap of the Hindi belt, Punjabi-Lahnda-Sindhi, Pahari, Bengali [@flap] ──
[Bhoj Mth Bi Ku N Gowro] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ {ḍ ḍʰ}            ; @flap — 8 lects / kaṭāha > H kaṛāhā, kukkuṭá > H kukṛā, *dʰūḍi > H dʰūṛ; geminate ḍḍ stays a stop / (changes.sc flap.ddh.postvocalic.except)
[Ku N] a > ∅ / # C* V C _ C a C {u o ā i} #         ; @syncope — 2 lects / second-syllable a lost when two syllables follow, as in N (utarṇo < *utaraṇo, uglaṇo, akaṛṇo) / (changes.sc syncope.a.cvccacuoai)
[Ku] a > ∅ / V C _ ṇ o #                        ; @syncope — a before the -ṇo of the verbal noun is lost (utarṇo, ucʰalṇo, āṭṇo)
[Ku] a > ∅ / {ā ī ū ē ō} C _ C {o ā i u} #      ; @syncope — penultimate a lost after a heavy (long-vowel) syllable before a gender/extension ending (-aḍaka > -ṛo, -allaka > -lo)
[Ku kiuth] ; ── block hiatus-vh-loss: Hiatus theme — loss of intervocalic v and h (blocks round, 2026-09-30). The NIA loss of the weak intervocalic continuants that creates a second round of hiatus: [@v-loss @h-loss — blocks/hiatus-vh-loss.sc; late (after apocope and syncope)] ──
[Ku kiuth] ; ── block hiatus-v-loss: Hiatus theme — loss of intervocalic v (blocks round, 2026-09-30). The NIA loss of intervocalic v (incl. < MIA p) that opens a second round of hiatus (dīpa > dīv ──
[Ku kiuth] v > ∅ / V _ V                                ; @v-loss — intervocalic v (incl. < p) lost (dīpa > dīva > diyā, kūpa > kuā, kapila > kailo) / (block hiatus-v-loss)
[Ku kiuth] h > ∅ / V _ V                                ; @h-loss — intervocalic h (< MIA aspirates) lost (lṓhita > làu, gōdʰūma > gɔū̃) / (block hiatus-vh-loss)
[bhad cam kul] {ī ū ē ō} > {i u e o}                        ; @vowel-shift — 3 lects / length lost except in ā / (changes.sc vowel-shift.iueo)
[RathwiBareli] ə > ∅ / _ #                                 ; @stress @final-vowel-loss — the marked post-tonic final vowel is lost
[PauriBareli] a a > ā                                     ; @hiatus — round 6: a-vowels brought together by v-loss contract (āgatalla > āālo > ālo)
[marathi-bhil -Bhatri -PalyaBareli -M -Vasavi -OM -VB -Ko-Ul] {ē ō} {a ā} > $1                             ; @hiatus — 6 lects / e/o absorb a following a / (changes.sc hiatus.eoaa)
[Bhili] a a > ē / C _ C                              ; @aya-e — 8 lects / a(y)a > e in the interior / (changes.sc aya-e.aa.cc.2)
[DungraBhili] a > o / C _                                 ; @a-rounding — short a > o after consonants in Dungra Bhil (round 7: the source writes it o more often than ɔ; writing ɔ here is worse once final -ɔ is in) (not a varghese-notation effect: Noiri, same source, gets worse)
[DungraBhili] c > s                                        ; @deaffrication — 3 lects / deaffrication c > s (nīcá > niso) / (changes.sc deaffrication.c)
[marathi-bhil -M -OM -VB -Ko-Ul] {a ā} {o ō} > $2 / _ #                       ; @hiatus — 11 lects / round 6: final -āo > -o after consonant loss (kʰādaka > kʰāo > kʰo) / (changes.sc hiatus.aaoo.final)
[Ku] v > ∅ / ā _ ṇ                                ; @v-loss — causative -āv(a)ṇo > -āṇo (agʰāṇo, aṭāṇo)
[Dotyali] {i ī} {a ā} > ī / _ #                        ; @hiatus — 10 lects / -ikā/-iya > -ī (gʰaṭikā > gʰaṛī) / (changes.sc hiatus.iiaa.final.2)
[Ku Pk] i ī > ī                                      ; @hiatus — 5 lects / i + ī left in hiatus when -ikā is added to an -ika/-ikā stem contracts, as in H (andʰikā + -ikā > ā̃dʰiī > ā̃dʰī, ārātrika + -ikā > ārtī; hill-climb round 2) / (changes.sc hiatus.ii)
[N] i a > i / _ {n ṇ} u #                         ; @hiatus — passive -iyaṇaaṁ > -ianu > -inu (gʰisinu, cʰalinu, oirinu)
[Garh N Ko-Ul] a > ∅ / V C _ C {o ā i u} #                  ; @syncope — 2 lects / penultimate a lost before a gender/extension ending (*ṭukka-ḍa > ṭukro, bárkara > bākʰro; -aḍaka > -ṛo, -allaka > -lo) / (changes.sc syncope.a.vccoaiu.2)
[Mth] a > ∅ / V C* C _ ṛ ā #                    ; @syncope — -aḍaka: syncope after flapping, so the flap survives after a consonant (bacʰṛā)
[Bhoj Mth Bi] ; ── block liquids-flap-rhotic-short: Partial flap-rhotic merger — ṛ > r after a short vowel (Bihari; liquids blocks round pass 2, 2026-09-30). The Bihari counterpart of blocks/liquids-flap-rhotic.s [@flap — after a short vowel the flap merges with r (paṭōlá > paror, kaṭāha > karāh); kept after long vowels (pī̃ṛ)] ──
[Bhoj Mth Bi] ṛ > r / {a i u} _                            ; @flap — 3 lects / kaṭāha > Bhoj karāh, kʰaṭikā > Mth kʰarī; kept after a long vowel (pī̃ṛ) / (changes.sc flap.r.aiu)
[Mth Bi] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[Mth Bi] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[MagahiNepal Mth Bi] {ā ī ū ā̃ ī̃ ū̃} > {a i u ã ĩ ũ} / _ C* V C* V C* V C* # ; @vowel-shift — 3 lects / antepenultimate long vowels shorten (counted before apocope of -a); the Vedic accent does not condition it / (changes.sc vowel-shift.aiuaiu.cvcvcvc)
[Mth Bi] {ī ū ī̃ ū̃} > {i u ĩ ũ} / V C* _ C* V        ; @vowel-shift — 2 lects / non-initial ī ū shorten in open medial position (sindūra > sindur, *jambūna > jāmun) / (changes.sc vowel-shift.iuiu.vccv)
[DewasDoneDanuwar] ĭ > ∅ / _ #                                 ; @final-vowel-loss @stress — reduced posttonic -ĭ lost with the apocope
[Darai] G $1 > $2                                    ; @cluster-simplify @degem-plain — 83 lects / (changes.sc degem-plain.g)
[MagahiNepal DewasDoneDanuwar Mth Darai Majhi Bi Bote KochilaTharu] ; ── block finals-syncope-bihari: Bihari/Tharu syncope before a final long-vowel syllable (blocks theme "finals", 2026-09-30): medial a between single consonants (stops, nasals, sonorants, s; no [syncope before a final long-vowel syllable (blocks/finals-syncope-bihari.sc)] ──
[MagahiNepal DewasDoneDanuwar Mth Darai Majhi Bi Bote KochilaTharu] a > ∅ / V {Stop Nasal Son s} _ {Stop Nasal Son s} {ā ī} # ; @syncope — 7 lects / medial a lost before a final -Cā/-Cī (Bi. ārātrika > ārtī, kaŋkaṇa+aka > kãknā, kartana+ikā > kaṭnī) / (changes.sc syncope.a.vstopnasalsons)
[Mth Bi] {ā̃ ī̃ ū̃} > {ã ĩ ũ} / _ C C                 ; @two-mora — 2 lects / nasal long vowels shorten before a (syncope-made) cluster (aŋgana > ãgnā, kaŋkaṇa > kãknā) / (changes.sc two-mora.aiu.cc)
[N RathwiBareli] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[N RathwiBareli] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[Darai] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[Darai] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[RathwiBareli] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here [@hiatus — blocks/hiatus-medial-core.sc; a(y)a > e in the interior] ──
[RathwiBareli] a a > e / C _ C                              ; @hiatus — medial a(C)a > e (kadala > kaala > Bhoj kerā, kadalī > kayalī > kelā) / (block hiatus-medial-core)
[RathwiBareli] a ā > ā                                      ; @hiatus — a-ā hiatus > ā (kumbʰakāra > kumʰār, andʰakāra > anʰār) / (block hiatus-medial-core)
[PauriBareli RathwiBareli] {ō o} > u / _ C                              ; @vowel-shift — 2 lects / o raised to u in non-final syllables (pōtara > pūryā, dōla > ḍūḷū) / (changes.sc vowel-shift.oo.preconsonantal)
[marathi-bhil -Bhatri -PalyaBareli -hal -M -mewari_bannoda -OM -VB -Ko-Ul] ; ── block nasals-final-dental: nasals-final-dental — word-final ṇ > n after apocope, Bhilali/Bareli, nasals theme, blocks round 2026-09-30 (pass 2). Follows nasals-retroflexion(-noninitial):  [@n-dental — round 6: word-final ṇ > n (kárṇa > kān, trī́ṇi > tīn) (blocks/nasals-final-dental.sc)] ──
[marathi-bhil -Bhatri -PalyaBareli -hal -M -mewari_bannoda -OM -VB -Ko-Ul] ṇ > n / _ #                                  ; @n-dental — 3 lects / kárṇa > kān, trī́ṇi > tīn / (changes.sc n-dental.n.final)
[Buksa Kathoriya Darai BH AdivasiOriya kaithal marathi-bhil -M -Vasavi -Ko-Ul thar poth rajasthani-gujarati -OG -G -OMarw ktg kvar kul] ; ── block hiatus-diphthong: Hiatus theme — a + i/u before a consonant (blocks round, 2026-09-30). The secondary diphthongs left by consonant loss monophthongise before a consonant: a+i > e [@hiatus — blocks/hiatus-diphthong.sc; round 6: medial a+u left by consonant loss > o (hastakūṭa > hathoḍo)] ──
[Buksa Kathoriya Darai BH AdivasiOriya kaithal marathi-bhil -M -Vasavi -Ko-Ul thar poth awan rajasthani-gujarati -OG -G -OMarw ktg kvar kul] a {i ī} > e / _ C                            ; @hiatus — a+i > e (bʰaginī > bʰaiṇ- > bʰeṇ, samākr̩ṣṭa > saiṭṭʰ- > seṭṭʰ-) / (block hiatus-diphthong)
[Buksa Kathoriya Darai BH AdivasiOriya kaithal marathi-bhil -M -Ko-Ul thar poth rajasthani-gujarati -OG -G -OMarw ktg kvar cam kul] a {u ū} > o / _ C                            ; @hiatus — a+u > o (hastakūṭa > hatʰaüḍa > hathoḍo) / (block hiatus-diphthong)
[mand pan] {a u o} > v / C _ ā                          ; @hiatus — 2 lects / vowel before ā (after consonant loss) > glide v (nimāna > nvāṇ) / (changes.sc hiatus.auo.ca)
[mand pan] ; ── block hiatus-absorb: Hiatus theme — absorption of a by a neighbouring vowel (blocks round, 2026-09-30). After the loss of intervocalic consonants (incl. the NIA v/h loss), short a i [@hiatus — blocks/hiatus-absorb.sc; post-vocalic a absorbed (kāhala > kaaḷ > kāḷ)] ──
[mand pan] a > ∅ / _ ā                                  ; @hiatus — a-ā > ā (kumbʰakāra > kumbʰaār > kumhār) / (block hiatus-absorb)
[mand] a > ∅ / V _                                  ; @hiatus — a absorbed after a vowel (kāhala > kaaḷ > kāḷ; -iya > -i after final-vowel loss) / (block hiatus-absorb)
[Wot OM cur] {ī ū} > {i u} / _ C                          ; @length-loss @vowel-shift — 2 lects / ī ū shortened before a consonant / (changes.sc length-loss.iu.preconsonantal)
[cur khash] a > ∅ / V _                                  ; @hiatus — 37 lects / post-vocalic a absorbed (kāhala > kaaḷ > kāḷ) / (changes.sc hiatus.a.postvocalic)
[bhal kc bhad ktg khash kul] ; ── block lar-whp-affrication: West Pahari affrication: the palatal stops become dental affricates (c cʰ j jʰ > ʦ ʦʰ ʣ ʣʰ), and ñ follows them to the dental nasal. Kotgarhi, Kochi, Kullui, Bh [@affrication — c cʰ j jʰ > ʦ ʦʰ ʣ ʣʰ, ñ > n before them (blocks/lar-whp-affrication.sc)] ──
[bhal kc bhad ktg khash kul] {c cʰ j jʰ} > {ʦ ʦʰ ʣ ʣʰ}                    ; @affrication — 6 lects / cʰárdati > bhad. ʦʰaḍḍ-, *kacca > bhal. kaʦʦo, pañca > pānʦ / (changes.sc affrication.cchjjh)
[bhal kc bhad ktg khash kvar kul] ñ > n / _ {ʦ ʦʰ ʣ ʣʰ}                        ; @affrication — 6 lects / ñ follows the palatals to dental (pañca > pānʦ, *kañcikā-type -nʦ-) / (changes.sc affrication.n.tstshdzdzh)
[Gowro Bshk Chil bhatr Mai] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc) — before the step that frees cluster m (rm > mm > m, Dardic rm > m), so only original single m lenites] ──
[Gowro Bshk Chil bhatr Mai] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
[dardic-south] ; ── block mia-dardic-clusters: Dardic cluster simplification (Kohistani, Kunar, Pashai; blocks theme "mia", 2026-09-30): clusters reduce to a single consonant without the mainland MIA geminat [@cluster-simplify @assim-stop @assim-r — Dardic cluster simplification without geminates (blocks/mia-dardic-clusters.sc; blocks round 2026-09-30)] ──
[dardic-south] C v > $1                                     ; @cluster-simplify — 11 lects / C + v > C (dvāra > dar, jihvā > jib-, pakva > pak) / (changes.sc cluster-simplify.cv)
[Gowro Bshk Tor bhatr Tir Pas] y > ∅ / C _                                  ; @cluster-simplify — 25 lects / C + y > C (satya > sat, kanyā > kana, madʰya > mad) / (changes.sc cluster-simplify.y.postconsonantal)
[dardic-south] {k p g b} {t tʰ d dʰ} > $2                   ; @assim-stop @cluster-simplify — 12 lects / k p g b + dental > dental (bʰakta > bʰat, sapta > sat, dugdʰa > dud-) / (changes.sc assim-stop.kpgbtthddh)
[dardic-south] C $1 > $1                                    ; @cluster-simplify @degem-plain — 18 lects / geminates simplify (putta > put, kaṭṭa > kaṭ, pakka > pak) / (changes.sc degem-plain.c)
[eur Zarg Lom dardic-south] r > ∅ / _ C                                  ; @assim-r @cluster-simplify — 16 lects / r before a consonant lost (karma > kam, sarpa > sap, dīrgʰa > dīg) / (changes.sc assim-r.r.preconsonantal)
[Bshk] {a ā i ī u} > ∅ / V C* _ #                  ; @final-vowel-loss — includes -ī (nāḍī́ > nār, bʰaginī > bān) (round 2: polysyllables only, also after a vowel (-ikā > -iā > -i) — monosyllables keep their vowel; −0.001 to −0.008 in all 12 lects)
[Bshk Wot jaun dog bhad kgr cur khash cam pan] {i u} > {ī ū} / _ #                          ; @final-lengthening — 11 lects / final high vowels long / (changes.sc final-lengthening)
[kc ktg] {a i u} > ə / # C* _ C C* Long               ; @vowel-reduction — 2 lects / short first-syllable vowel reduced to ə before a long vowel that took the stress, as in Kotgarhi (runs before length loss; round 7) / (changes.sc vowel-reduction.aiu.ccclong)
[ktg] Long > $1+stress / # C* ə C C* _            ; @accent-placement — the long vowel after a reduced first syllable carries the stress (kaṭāha > kəṛā́, saṁbʰāvayati > səbʰáuṇõ, *viccʰādayati > bəʦʰáuṇõ); latent, before the length merger (round 7)
[kc ktg] {ā ī ū ē ō} > {a i u e o} // # (C) (C) _ C (C) # ; @vowel-shift — 2 lects / length kept only in closed monosyllables / (changes.sc vowel-shift.aiueo.except.3)
[kvar WPah] {ā ī ū ē ō} > {a i u e o} // _ C (C) #       ; @vowel-shift — 2 lects / length lost except in a final closed syllable / (changes.sc vowel-shift.aiueo.except.2)
[ktg kvar WPah] ; ── block hiatus-absorb: Hiatus theme — absorption of a by a neighbouring vowel (blocks round, 2026-09-30). After the loss of intervocalic consonants (incl. the NIA v/h loss), short a i [@hiatus — blocks/hiatus-absorb.sc; post-vocalic a absorbed (kāhala > kaaḷ > kāḷ)] ──
[ktg kvar WPah] a > ∅ / V _                                  ; @hiatus — a absorbed after a vowel (kāhala > kaaḷ > kāḷ; -iya > -i after final-vowel loss) / (block hiatus-absorb)
[jaun dog kiuth kc mand ktg WPah kul] {ṭ͡ḷ ḍ͡ḷ ḍ͡ḷʰ} > {c j jʰ}                    ; @tr-affricate — 3 lects / the lateral affricate from tr > c (after the old palatals became ʦ, so no merger where affrication applies) / (changes.sc tr-affricate.tldldlh)
[kaithal Khandesi mewari_bannoda kcch mand ktg] ; ── block lar-h-loss-postvocalic: Loss of postvocalic h (leaving tone/length in some sources): Bhil (Khandesi, Noiri, Mewari of Bannoda), Haryanvi (Kaithal), Kotgarhi, Mandeali. Laryngeals round [@h-loss — postvocalic h lost (blocks/lar-h-loss-postvocalic.sc)] ──
[kaithal Khandesi mewari_bannoda kcch mand ktg] h > ∅ / V _                                  ; @h-loss @tone — 6 lects / bāhú > bā́ / (changes.sc h-loss.h.postvocalic)
[kcch] jʰ > j                                      ; @deaspiration — Kutchi jʰ > j (dahyátē > ḍajṇū, *jʰal > jalṇū, búdʰyatē > bujṇū; round 2)
[thar] ; if:!ext ∅ > o / C _ #                               ; @morph — masc. citation forms in -o (MIA -aka)
[mewari_basad bagri_fatehabad mewari_dholpura hadothi_kelwada kgr khash] ; ── block lar-deaspiration-voiced-medial: Non-initial voiced aspirates lose aspiration (initial ones survive, with tone/breathiness). Rajasthani (Bagri, Hadothi, Mewari, Dholpuri) and West Pahari (Bhale [@deaspiration @tone — non-initial voiced aspirates (blocks/lar-deaspiration-voiced-medial.sc)] ──
[BH AdivasiOriya Ko-Ul mewari_basad bagri_fatehabad mewari_dholpura hadothi_kelwada Rj marwari_bagra kgr khash] {gʰ jʰ ḍʰ dʰ bʰ} > {g j ḍ d b} / !# _        ; @deaspiration @tone — 7 lects / andʰakāra > andāro, garbʰa > gābu / (changes.sc deaspiration.ghjhdhdhbh.cond)
[rajasthani-gujarati -mewari_basad -OG -WK -hadothi_kelwada -G -OMarw] a > ∅ / # _ C V                              ; @aphaeresis — 8 lects / unstressed initial a- lost (araghaṭṭa > rahaṭ, anyākāra > nyāro) / (changes.sc aphaeresis.a.cv)
[VB] ; ── block finals-syncope: NIA schwa syncope (blocks theme "finals", 2026-09-30): after apocope, medial short a in an open syllable between single consonants, with a vowel following, is l [inherited from the Migratory sisters (small-lect default, 2026-09-30)] ──
[VB] a > ∅ / V C _ C V                            ; @syncope — 16 lects / medial a lost in VCaCV (thar kaccara+aka > kacro, karpaṭa+aka > kapṛo, kṣurapra+ikā > kʰurpī) / (changes.sc syncope.a.vccv)
[Bishnupriya Hajong] ; ── block lar-s-h-initial: Eastern initial ś > h (after the Magadhan sibilant merger; Assamese x/h): Bishnupriya, Hajong. Laryngeals round, pass 2 (2026-09-30). Edit here, not in the lect [@s-h — initial ś > h (blocks/lar-s-h-initial.sc)] ──
[Bishnupriya Hajong] ś > h / # _                                  ; @s-h — 2 lects / saptá > Bishnupriya hat, śilā > hīl, sūkara > Hajong huvar / (changes.sc s-h.s.initial)
[Bishnupriya Hajong] {ā ā̃} > {a ã}                               ; @length-loss — 3 lects / no a/ā contrast (bʰakta > bʰat) / (changes.sc length-loss.aa)
[Bishnupriya Hajong KS Rathawi PauriBareli mewari_bannoda Vasavi Noiri VB] ; ── block lar-deaspiration-final: Word-final aspirates lose aspiration (after apocope). Bhil (Noiri, Pauri/Rathwi Bareli, Vasavi, Mewari of Bannoda) and Eastern (Bishnupriya, Hajong, Kharia Thar [@deaspiration — word-final aspirates lose aspiration (blocks/lar-deaspiration-final.sc)] ──
[Bishnupriya Hajong KS Rathawi PauriBareli mewari_bannoda Vasavi Noiri VB] {kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ} > {k g c j ṭ ḍ t d p b} / _ # ; @deaspiration — 2 lects / hasta > hāt, dugdʰa > dūd, múkʰa > muk, mēgʰa > mek / (changes.sc deaspiration.khghchjhth.final)
[marathi-bhil -M -OM -Ko-Ul] ; ── block nasals-nc-loss-bhil: nasals-nc-loss-bhil — NIA loss of a nasal before a voiceless stop (no nasal vowel), Bhil/Halbic type, nasals theme, blocks round 2026-09-30 (pass 2). [@nasal-cluster — round 6: nasal lost before a voiceless stop (dánta > dāt, káṇṭaka > kāṭo; the preceding ā already long) (blocks/nasals-nc-loss-bhil.sc)] ──
[marathi-bhil -M -OM -Ko-Ul] {n ṇ ñ ŋ m} > ∅ / V _ {k c ṭ t p}            ; @nasal-cluster — 12 lects / dánta > dāt, káṇṭaka > kāṭo (the preceding a lengthened by nasals-anusvara-bhil / a.nasalstop) / (changes.sc nasal-cluster.nnnngm.vkcttp)
[Bhili] a {u ū} > o / _ C                            ; @hiatus — 8 lects / round 6: medial a+u left by consonant loss > o (hastakūṭa > hathoḍo) / (changes.sc hiatus.auu.preconsonantal)
[Gowro] r > ∅ / _ C                                  ; @assim-r @cluster-simplify — 16 lects / preconsonantal r lost after lenition (kárdama-type C not lenited) / (changes.sc assim-r.r.preconsonantal)
[Wot Chil Bhatri] ; ── block liquids-flap: Retroflex flapping — single post-vocalic ḍ ḍʰ > ṛ ṛʰ (NIA; liquids blocks round, 2026-09-30). The flap of the Hindi belt, Punjabi-Lahnda-Sindhi, Pahari, Bengali [@flap — ḍ > ṛ (kukkuṭá > kukṛa)] ──
[Wot Chil Bhatri] {ḍ ḍʰ} > {ṛ ṛʰ} / V _ // _ {ḍ ḍʰ}            ; @flap — 8 lects / kaṭāha > H kaṛāhā, kukkuṭá > H kukṛā, *dʰūḍi > H dʰūṛ; geminate ḍḍ stays a stop / (changes.sc flap.ddh.postvocalic.except)
[VB] ∅ > ɨ / C _ # // {r l ḷ} _                   ; @x-marathi-bhil-final-vowel — echo/epenthetic final vowel after consonants (agní > āgɨ; Dravidian-contact area); round 6: not after liquids (udʰār, upar, pūr)
[Garh Bhatri] ; ── block hiatus-vh-loss: Hiatus theme — loss of intervocalic v and h (blocks round, 2026-09-30). The NIA loss of the weak intervocalic continuants that creates a second round of hiatus: [@v-loss @h-loss — blocks/hiatus-vh-loss.sc; intervocalic h lost] ──
[Garh Bhatri] ; ── block hiatus-v-loss: Hiatus theme — loss of intervocalic v (blocks round, 2026-09-30). The NIA loss of intervocalic v (incl. < MIA p) that opens a second round of hiatus (dīpa > dīv ──
[Garh Bhatri] v > ∅ / V _ V                                ; @v-loss — intervocalic v (incl. < p) lost (dīpa > dīva > diyā, kūpa > kuā, kapila > kailo) / (block hiatus-v-loss)
[Garh Bhatri] h > ∅ / V _ V                                ; @h-loss — intervocalic h (< MIA aspirates) lost (lṓhita > làu, gōdʰūma > gɔū̃) / (block hiatus-vh-loss)
[Pk Ap] ; ── Notation ────────────────────────────────────────────────────────────
[Ap] ī > i                                       ; @vowel-shift — ī shortened (dīpa > dippa, adʰīna > adʰiṇṇa, aśītí > asii; round 2)
[Ap] ē > e                                       ; @vowel-shift — ē shortened (CDIAL Ap. e 19 : ē 11; round 2)
[Pa] ē > i / t _ #                               ; @morph — middle -tē cited as active -ti (upapadyatē > upapajjati)
[Pa] {ḍ ḍʰ} > {ḷ ḷʰ} / V _ V                     ; @flap @int-retroflex — gūḍʰa > gūḷʰa, ēḍa > eḷa
[Pa] ; ── block lar-v-b-geminate: MIA geminate vv (< rv, vy, dv) > bb. West Pahari, Rajasthani, Eastern, Pali. Laryngeals round (2026-09-30). Edit here, not in the lects. [@v-b — geminate vv > bb (blocks/lar-v-b-geminate.sc)] ──
[Pa] v v > b b                                    ; @v-b — 18 lects / sárva > sabb-, parvan > pabba, dravya > dabba / (changes.sc v-b.vv)
[As-Ul] ; ── Morphology ───────────────────────────────────────────────
[Pa As-Ul] ; ── Nasals / notation ────────────────────────────────────────────────────
[Pk Ap Pa] m h > mʰ                                    ; @notation — CDIAL Pk writes the mh cluster as mʰ (gimʰa, bamʰaṇa)
[Si] optional: {k g} > ∅ / a _ a #                ; @aka @ya-sruti — sporadic: final -aka escapes the -ā contraction; its k goes with the other intervocalic stops and the glide fork gives -aya (*aḍḍ > aḍaya, kara > karaya, káśā > kasaya, kúla > kulaya; gold -aya 116 vs -ā 896; gates ext −0.0047/−0.0049 vs placebo {r l} > ∅ −0.0002/−0.0002) (hill-climb round 4)
[Si] a {k g} a > ā / _ #                         ; @aka — -aka (-aga) > -ā before the hiatus a-loss (masc. -ā: ghōṭaka-type); round 4
[Si Md OSi] {k g c t y} > ∅ / V _ V                      ; @int-stop-loss — 2 lects / voiceless stops, g and y lost; d (< d, j) and b kept / (changes.sc int-stop-loss.kgcty)
[Garh] ā v > ɔ / _ ṇ                                ; @vowel-shift — causative -āv(a)ṇu > -ɔṇu (āu > ɔ, as in the secondary au > ɔ; jaḷɔṇu, pilɔṇu)
[Ku Garh N] ; ── block finals-apocope-pahari: Central/Eastern Pahari apocope (blocks theme "finals", 2026-09-30): final short -a -i and the feminine -ā are lost while final -u is kept (Nepali, Kumaoni, Garh [final -a -i -ā lost, -u kept (blocks/finals-apocope-pahari.sc) — replaces `use final-vowel-loss.aia`: short -a -i and fem. -ā; final -u kept] ──
[Ku Garh N] {a i ā} > ∅ / _ #                            ; @final-vowel-loss — 3 lects / final -a -i -ā lost, -u kept (N. akṣára > ākʰar, áŋga > āŋ, aŋgana > āŋan) / (changes.sc final-vowel-loss.aia)
[N] n > ∅ / i _ #                                ; @final-cons-loss — the -in of OIA in-stems is lost as in the nominative -ī (gōtrin > goti, dʰanín > dʰani, pʰalín > pʰali); round 7
[Ku Garh N Dotyali] y > ∅ / C _ #                                ; @y-loss — 3 lects / final -Cy left by apocope loses the glide (kalya > kāl, kulyā́ > kūl, daívya > deb), as in N. and Garh.; round 7 / (changes.sc y-loss)
[Garh] a m > ɔ̃ / _ #                                ; @vowel-shift — final -am (after apocope) > -aṽ > -ɔ̃, the m nasalising and rounding the vowel as au > ɔ (aṣṭamá > āṭʰɔ̃, daśamá > dasɔ̃, pañcamá > pā̃cɔ̃; only the -ama ordinals end so in the data); round 7
[Bhoj] {a i} > ∅ / Short {Stop Nasal Son s} _ {n ṇ} {ā ī} #   ; @syncope — medial a/i lost before -nā/-nī (jʰaraṇa > jʰarnā, gʰariṇī > gʰarnī, hariṇa > harnā); Bhoj keeps it before r (bakarā, pokʰarā)
[Buksa Rana Dang Bhoj Sunha Kathoriya Bi Dotyali] ; ── block finals-apocope: NIA apocope (blocks theme "finals", 2026-09-30): loss of the short final vowels -a -i -u that the Apabhraṁśa stage left word-final (after finals-ap-shortening w [NIA apocope (blocks/finals-apocope.sc) — replaces `use final-vowel-loss.ai`: final -u kept (N. -u)] ──
[Buksa Rana Dang Bhoj Sunha Kathoriya Bi Dotyali] {a i u} > ∅ / V C* _ #                       ; @final-vowel-loss — 3 lects / final short -a -i -u lost in polysyllables (H. akṣára > ākʰar, agní > āg, áŋga > ā̃g) / (changes.sc final-vowel-loss.aiu.final-polysyllabic)
[Ku N] ; ── block hiatus-medial: Hiatus theme — medial a(C)a contraction (blocks round, 2026-09-30). After the MIA loss of intervocalic stops/y, a-a before a consonant contracts to e and a-ā to [@hiatus — blocks/hiatus-medial.sc] ──
[Ku N] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here ──
[Ku N] a a > e / C _ C                              ; @hiatus — medial a(C)a > e (kadala > kaala > Bhoj kerā, kadalī > kayalī > kelā) / (block hiatus-medial-core)
[Ku N] repeat: {a ā} {a ā} > ā                      ; @hiatus — a/ā hiatus > ā (andʰakāra > andʰaāra > ā̃dʰār, kumbʰakāra > kumʰār, rājana > rāaṇ > rāṇ) / (block hiatus-medial)
[N] i a > i / C _ C                              ; @hiatus — 3 lects / medial -ia- left by consonant loss contracts to i (*prasvidana > pasinā, *bʰiyajyatē > bʰijnu, *taddivasam > tādi); round 7 / (changes.sc hiatus.ia.cc)
[Ku Garh N] a > ∅ / i _ C                                ; @hiatus — 7 lects / a absorbed after i before a consonant (-iavva > -ivva), as in A. and B. / (changes.sc hiatus.a.ic)
[MH] ; [round 7: dropped `use hiatus.a.2` (late iā > i): toulmin's MH keeps -iā (pradʰí > pāhiā, haridrā > haldiā, *kartyā > kɛciā); ext −0.0021 train / −0.0028 test, 7 better, 0 worse; the medial-only form below (as in BH, KS) is score-identical]
[Ku N B Bhatri] ; ── block lar-rh-deaspiration: The retroflex flap ṛʰ (< intervocalic ḍʰ) loses its aspiration: Bengali (ঢ় = ṛ), Bhatri. Laryngeals round, pass 2 (2026-09-30). Edit here, not in the lects. [@deaspiration — ṛʰ > ṛ (blocks/lar-rh-deaspiration.sc)] ──
[Ku N B Bhatri] ṛʰ > ṛ                                       ; @deaspiration — āṣāḍʰá > B. āsāṛ, āṣāḍʰī > āsāṛī (B. ঢ় = ṛ) / (block lar-rh-deaspiration)
[M Ko-Ul] s > ś / _ Front                              ; @palatalisation — 2 lects / s > ś before front vowels (śiroḷ, śeʣār) / (changes.sc palatalisation)
[M] {ī ū ī̃ ū̃} > {i u ĩ ũ} / _ C                ; @length-loss — 2 lects / high-vowel length only kept word-finally (kīṭa > kiḍā) / (changes.sc length-loss.iuiu.preconsonantal)
[M] {i u} > {ī ū} / _ C #                        ; @vowel-shift — round 6: high vowels long in a final closed syllable (uśīra > uśīr, aŋkurá > ā̃kūr, kúla > kūḷ)
[M] ā̃ > ã / V C* _ C                            ; @length-loss — round 6: long nasal ā̃ (anusvāra-vowel) only in the initial syllable; ã elsewhere (ārambʰaṇa > ārãbʰaṇ; CDIAL M. non-initial ã:ā̃ = 41:10 before d, 33:3 before b)
[M] ā > a / V C* _ C C                           ; @length-loss — round 6: non-initial ā shortened in a closed syllable created by syncope (*avagʰālayati > ohaḷṇẽ, causative -āvaṇẽ > -avṇẽ)
[M] {c j jʰ} > {ʦ ʣ ʣʰ} // _ Front              ; @affrication — except before front vowels (jānánt > ʣāṇ)
[MB B] ∅ > y / {i u e} _ ā                          ; @ya-sruti — 2 lects / hiatus glide before -ā after i, u, e (piāsī > piyāsī, kū́pa > kuyā, kētaka > keyā) / (changes.sc ya-sruti.iuea)
[B] ; ══ per-source notation (applied after the chain, only for that source's pairs) ═══════════════
[B] %source kim-kim-sangma2012garo
[B] # (2026-10-01: redundant after the upstream digraph-aspirate fix; the source now writes Cʰ) {kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ} > {k g c j ṭ ḍ t d p b} h   ; @notation — digraph aspirates (dhula, bhat)
[BH B] ; ── block lar-a-rounding-bengali: Bengali/Kamta vowel rounding: the shared Eastern a-rounding (lar-a-rounding-east: o before a high vowel, first-syllable ɔ), then the remaining non-initial a > o [@a-rounding — o before a high vowel, first-syllable ɔ, other a > o (blocks/lar-a-rounding-bengali.sc)] ──
[BH B] ; ── block lar-a-rounding-east: Bengali-type rounding of short a (Bengali, Kamta/Rajbanshi, Bishnupriya): a is raised to o before a high vowel in the next syllable (vowel harmony), and the rem [@a-rounding — hariṇá > horin, pʰála > pʰɔl] ──
[BH B] a > o / # C* _ C* {i u ī ū}                  ; @a-rounding — 2 lects / before a high vowel in the next syllable: *raśuna > B. rosun, hariṇá > horin / (changes.sc a-rounding.cciuiu)
[BH B] a > ɔ / # C* _                               ; @a-rounding — 2 lects / first-syllable a > ɔ: pʰála > pʰɔl, kadala > kɔla, gʰara > gʰɔr / (changes.sc a-rounding.first-syllable)
[BH B] a > o                                        ; @a-rounding — non-initial a > o (vatsará > bɔchor, kambalá > kɔmbol) / (block lar-a-rounding-bengali)
[BH] {c cʰ j jʰ} > {ʦ ʦʰ ʣ ʣʰ} / # _         ; @deaffrication — word-initial palatals stay affricates (toulmin ʦ ʣ: caṇa > ʦānā, cakrá > ʦākā); elsewhere c j > s z
[A] rʰ > r                                       ; @deaspiration — flapped ḍʰ > rʰ > r (as B. ṛʰ = ṛ)
[A] ś > h / V _                                 ; @s-h — pāśa > pāh, ūṣa > uh
[A] ś > x                                       ; @s-x — śubʰa > xu
[BH A WK] {c cʰ j jʰ} > {s s z z}                      ; @deaffrication — 3 lects / as in A. (cʰāpa > sāvā, jala > zɔl); toulmin also writes ʦ ʣ / (changes.sc deaffrication.cchjjh)
[BH] l > n / # _                                   ; @x-eastern-l-n — initial l > n (aŋgula > noŋul via *laŋgula; lāŋgala > nāŋāl)
[AdivasiOriya] ; ── block hiatus-vh-loss: Hiatus theme — loss of intervocalic v and h (blocks round, 2026-09-30). The NIA loss of the weak intervocalic continuants that creates a second round of hiatus: [@v-loss @h-loss — blocks/hiatus-vh-loss.sc; intervocalic h (< MIA aspirates) lost (gōdʰūma > gɔū̃, kapʰōṇi > kɔūnī)] ──
[AdivasiOriya] ; ── block hiatus-v-loss: Hiatus theme — loss of intervocalic v (blocks round, 2026-09-30). The NIA loss of intervocalic v (incl. < MIA p) that opens a second round of hiatus (dīpa > dīv ──
[AdivasiOriya] v > ∅ / V _ V                                ; @v-loss — intervocalic v (incl. < p) lost (dīpa > dīva > diyā, kūpa > kuā, kapila > kailo) / (block hiatus-v-loss)
[AdivasiOriya] h > ∅ / V _ V                                ; @h-loss — intervocalic h (< MIA aspirates) lost (lṓhita > làu, gōdʰūma > gɔū̃) / (block hiatus-vh-loss)
[BH AdivasiOriya Ko-Ul Rj marwari_bagra] ; ── block lar-deaspiration-medial: Non-initial aspirates of both series lose aspiration: the voiced ones (the shared lar-deaspiration-voiced-medial), then the voiceless ones. Rajasthani (Jaipuri, [@deaspiration — non-initial aspirates, both series (blocks/lar-deaspiration-medial.sc)] ──
[BH AdivasiOriya Ko-Ul Rj marwari_bagra] ; ── block lar-deaspiration-voiced-medial: Non-initial voiced aspirates lose aspiration (initial ones survive, with tone/breathiness). Rajasthani (Bagri, Hadothi, Mewari, Dholpuri) and West Pahari (Bhale [@deaspiration — non-initial voiced aspirates (garbʰa > gābu, sārdʰa > sāḍe)] ──
[BH AdivasiOriya Ko-Ul Rj marwari_bagra] {kʰ cʰ ṭʰ tʰ pʰ} > {k c ṭ t p} / !# _        ; @deaspiration — non-initial voiceless aspirates (úṣṭra > uṭ, íṣṭakā > iṭā, aŋguṣṭʰa > aŋtī) / (block lar-deaspiration-medial)
[Mai Tir] ; ── block nasals-postnasal-voicing: nasals-postnasal-voicing — NT > ND, nasals theme, blocks round 2026-09-30. A voiceless stop, aspirated or not, is voiced after a nasal (or anusvāra-turned-nasal [@nasal-voicing — NT > ND incl. aspirates (blocks/nasals-postnasal-voicing.sc)] ──
[Mai Tir] Nasal {k kʰ c cʰ ṭ ṭʰ t tʰ p pʰ} > $1 {g gʰ j jʰ ḍ ḍʰ d dʰ b bʰ}@2 ; @nasal-voicing — 10 lects / dánta > awan dand, káṇṭaka > awan kaṇḍā, śaŋkú > Kho śoŋg, páñca > Kho ponj / (changes.sc nasal-voicing.nasalkkhcc)
[Gowro Chil bhatr Mai] {n ṇ} {t d dʰ} > $1                          ; @cluster-simplify — 4 lects / dánta > dân / (changes.sc cluster-simplify.nntddh)
[Mai] ; ── block finals-aka-i: The -ikā/-ika ending (blocks theme "finals", 2026-09-30): after intervocalic k is lost, -ia/-iā contracts to -ī, which survives apocope (finals-apocope*); -uka  [-ikā > -ī (blocks/finals-aka-i.sc) — replaces `use aka.iaa.final`: -ika/-ikā > -ia/-iā > -ī] ──
[Mai] i {a ā} > ī / _ #                            ; @aka @hiatus — 20 lects / -ikā/-ika > -ia > -ī (mwr. kīṭá+ikā > kīḍī, kukkuṭá+ikā > kūkḍī, bʰārika > bʰārī) / (changes.sc aka.iaa.final)
[Tir] {v y} > ∅ / V _ V                            ; @v-loss — 9 lects / intervocalic v, y lost / (changes.sc v-loss.vy.intervocalic)
[bhatr Mai Tir] ; ── block finals-apocope-long: NIA apocope with long -ā, in polysyllables (blocks theme "finals", 2026-09-30): final -a -ā -i -u are lost after a preceding vowel (V C* _ #), so OIA monosyllab [apocope incl. -ā, in polysyllables (blocks/finals-apocope-long.sc) — replaces `use final-vowel-loss.aaiu`: incl. fem. -ā (saṁdʰyā > sā̃j)] ──
[bhatr Mai Tir] {a ā i u} > ∅ / V C* _ #                     ; @final-vowel-loss — 3 lects / final vowels lost in polysyllables (Marw. adyá > āj, aṣṭā́ > āṭʰ, kárṇa > kān) / (changes.sc final-vowel-loss.aaiu.final-polysyllabic)
[WK] {gʰ jʰ ḍʰ dʰ bʰ} > {g j ḍ d b} / # _          ; @deaspiration — word-initial voiced aspirates (gʰāsa > gāh, bʰaŋga > bãŋ)
[cur WPah] a i > e                                      ; @ai-au — 2 lects / secondary ai > e / (changes.sc ai-au.ai)
[dog] a {i u} > {e o}@2                            ; @ai-au — 5 lects / secondary diphthongs from hiatus monophthongised / (changes.sc ai-au.aiu)
[dog kiuth mand WPah] pʰ > f                                       ; @ph-f — 3 lects / pʰ spirantised / (changes.sc ph-f)
[WPah] h > ∅ / V _ C                               ; @h-loss — preconsonantal h lost
[dog] ∅ > a / C _ r                               ; @anaptyxis — svarabhakti vowel in a surviving Cr cluster
[OSi] d > ∅ / V _ V                                ; @int-stop-loss — 3 lects / OIA d (and dʰ) lost before j > d (as in Si) / (changes.sc int-stop-loss.d)
[Si Md OSi] ; ── block glide-optional: Sporadic hiatus glide at the end of the chain: y after a front vowel before a non-front vowel, v after a rounded vowel before a non-rounded one (kr̩ttikā > Pk.  [sporadic, MIA stage (blocks/glide-optional.sc)] ──
[Si Md OSi] optional: ∅ > y / {i ī e ē ĩ ī̃ ẽ ē̃ ɛ ɛ̃ æ} _ {a ā ã ā̃ o ō õ ō̃ u ū ũ ū̃ ɔ ɔ̃ ə} ; @ya-sruti — sporadic y glide after a front vowel (kr̩ttikā > kattiyā, śr̩gālá > siyār) / (block glide-optional)
[Si Md OSi] optional: ∅ > v / {u ū o ō ũ ū̃ õ ō̃ ɔ ɔ̃} _ {a ā ã ā̃ e ē ẽ ē̃ i ī ĩ ī̃ ɛ ɛ̃ æ ə} ; @ya-sruti — sporadic v glide after a rounded vowel (rōpa > rovā, *ḍʰauka > ḍʰovā) / (block glide-optional)
[Si Md OSi] ; ── block glide-a-optional: Sporadic hiatus glide after a/ā at the end of the chain, y or v (kātara > Bhoj. kāyar, jātá > G. jāyī, tāpaka > Bhoj. tāvā, śrā́vaṇa > Bhoj. sāvan). `optional:` [sporadic, MIA stage (blocks/glide-a-optional.sc)] ──
[Si Md OSi] optional: ∅ > y / {a ā ã ā̃} _ V             ; @ya-sruti — sporadic y glide after a/ā (kātara > kāyar, jātá > jāyī) / (block glide-a-optional)
[Si Md OSi] optional: ∅ > v / {a ā ã ā̃} _ V             ; @ya-sruti — sporadic v glide after a/ā (tāpaka > tāvā, śrā́vaṇa > sāvan) / (block glide-a-optional)
[Si] ∅ > v / a _ ā #                             ; @ya-sruti @hiatus — v glide in final -aā (< -akā, -akam): karaṇakam > karanavā, -tavyakam > -tavavā; round 5
[Si] a > ∅ / _ V                                  ; @hiatus @nur-innovation — 5 lects / a lost before another vowel after stop loss (hastatala > *hataala > atala) / (changes.sc hiatus.a.prevocalic)
[OSi] {c j} > {s d}                                ; @affrication @deaffrication — 3 lects / c > s, j > d (citrá > sita, jána > dena) / (changes.sc affrication.cj)
[Dhp] {tʰ dʰ} > s / V _ V                         ; @int-spirant — tʰ dʰ > θ ð written s (katʰā > kasa, adʰaḣ > aso)
[Dhp] {t k ṭ} > {d g ḍ} / V _ V                   ; @int-voicing — atīta > adida, kāruṇika > karuniga
[Dhp] {c j} > y / V _ V                           ; @int-spirant @ya-sruti — vícarati > viyaradi
[NiDoc] {kʰ gʰ} > h / V _ V                         ; @int-aspirate-h — only the velar aspirates (likʰati > lihati; tʰ, dʰ, bʰ are kept)
[NiDoc] {k ṭ} > {g ḍ} / V _ V                       ; @int-voicing — k ṭ voiced, t is not (Niya -ti vs Gāndhārī -di)
[NiDoc] t > d / V _ a #                             ; @int-voicing — t voiced in the final syllable, as k ṭ everywhere (jātá > jada, gʰr̩tá > gʰrida, likʰitá > lihida; the documents also write -ta); round 7
[Si Md OSi] ; ── block liquids-d-lateral: Lateral outcome of the intervocalic retroflex stops — ṭ ḍ > ḷ (liquids blocks round, 2026-09-30). Where the retroflex stops became a lateral rather than a flap: [@int-retroflex @flap — kūṭa > kuḷa] ──
[Si Md OSi] {ṭ ḍ} > ḷ / V _ V                            ; @flap @int-retroflex — 4 lects / kūṭa > Si kuḷa, guḍá > Md guḷa, kʰēṭa > Kho kʰéḷi / (changes.sc flap.td.intervocalic.3)
[Md] a > u / C _ r V                             ; @vowel-shift — a rounded before intervocalic r (karōti > kuranī, *tarati > turenī); round 2
[Md] ṭ ṭ > r̆                                     ; @x-mia-insular-tt-r — Dhivehi geminate ṭṭ > r̆ (yaṣṭi > dor̆i, kaṇṭa > kar̆i); single ṭ > ḷ
[nuristani] ; ── block nasals-nuristani-nt: nasals-nuristani-nt — Nuristani loss of the nasal in ñc/ñj and nt clusters (the vowel is nasalised in some daughters), nasals theme, blocks round 2026-09-30. Tw [@nasal-cluster — ñc ñj > c j, nt > t (blocks/nasals-nuristani-nt.sc)] ──
[nuristani] ñ > ∅                                        ; @cluster-simplify @nasal-cluster — 7 lects / ñc/ñj lose the nasal: páñca > *pãc, āmuñcati > Kt amci / (changes.sc nasal-cluster.n)
[nuristani] n > ∅ / V _ {t tʰ ṭ ṭʰ}                      ; @nasal-cluster @nur-innovation — 7 lects / nt > t (the cluster protected t from lenition): antár > *ātar, jīvantá > *jĩta / (changes.sc nasal-cluster.n.vtthtth)
[Majhi eur as-ll RomWel Zarg Lom Kund Kal PalyaBareli Si Md OSi nuristani -Ash -Pr mewari_dholpura Rj dhundari_badagaon cur] G_std $1 > $2                                ; @cluster-simplify @degem-plain — 83 lects / hattʰa > at / (changes.sc degem-plain.g)
[Si Md OSi] ; ── block nasals-insular-prenasal: nasals-insular-prenasal — Sinhala/Dhivehi nasal + stop: voiced stops become prenasalised, the nasal is lost before voiceless stops, nasals theme, blocks round 2 [@nasal-voicing @nasal-cluster — prenasalised voiced stops, nasal lost before voiceless (blocks/nasals-insular-prenasal.sc)] ──
[Si Md OSi] Nasal {g j ḍ d b} > {ⁿg ⁿj ⁿḍ ⁿd ᵐb}@2       ; @nasal-voicing — 3 lects / prenasalised stops: áŋgāra > Si aⁿgura, bandʰá > Si baⁿda, lamba > Si laᵐba / (changes.sc nasal-voicing.nasalgjddb)
[Si Md OSi] Nasal {k c ṭ t p} > $2                       ; @nasal-cluster — 3 lects / nasal lost before a voiceless stop: kaṇṭaka > Si kaṭu / (changes.sc nasal-cluster.nasalkcttp)
[Si Md OSi] ṁ {g j ḍ d b} > {ⁿg ⁿj ⁿḍ ⁿd ᵐb}@2           ; @nasal-voicing — 2 lects / anusvāra + voiced stop prenasalised as well / (changes.sc nasal-voicing.mgjddb)
[Md] i a > i / V C* _ #                           ; @hiatus — 2 lects / final -ia > -i except in disyllables (the Si rule, kept identical for consistency; Md: nā́vika > nevi); round 7 / (changes.sc hiatus.ia.final-polysyllabic)
[Md] ∅ > y / {i e} _ {a ā u}                      ; @ya-sruti — 2 lects / hiatus glide after front vowels (arcita > asiya) / (changes.sc ya-sruti.ieaau)
[Si Md OSi] {ā ī ū ē ō} > {a i u e o} // _ #             ; @vowel-shift — 2 lects / long vowels shortened (not word-finally) / (changes.sc vowel-shift.aiueo.except)
[Md] ; ── block finals-aka-u-dardic: The -uka ending in Kashmiric/Kohistani (blocks theme "finals", 2026-09-30): the vowel after a final -u left by -uka (-ua, -uā, -uō) is lost, so -uka ends in -u  [-ua > -u (blocks/finals-aka-u-dardic.sc) — replaces `use aka.aao.u`: a final vowel after u (from -uka after k loss) is absorbed: -uka > -u (round 4)] ──
[Md] {a ā ō} > ∅ / u _ #                          ; @aka @hiatus — 3 lects / -ua/-uā/-uō > -u (dod sá+uka > su; -uka > -ua > -u) / (changes.sc aka.aao.u)
[Sv Kalk Ush Kund dod mia-insular -Pk -NiDoc -Dhp -KharI] ; ── block finals-aka-u-dardic: The -uka ending in Kashmiric/Kohistani (blocks theme "finals", 2026-09-30): the vowel after a final -u left by -uka (-ua, -uā, -uō) is lost, so -uka ends in -u  [-ua > -u (blocks/finals-aka-u-dardic.sc) — replaces `use aka.aao.u`: a final vowel after u (from -uka after k loss) is absorbed: -uka > -u (round 4)] ──
[Sv Kalk Ush Kund dod mia-insular -Pk -NiDoc -Dhp -KharI] {a ā ō} > ∅ / u _ #                          ; @aka @hiatus — 3 lects / -ua/-uā/-uō > -u (dod sá+uka > su; -uka > -ua > -u) / (changes.sc aka.aao.u)
[Md] a ā > ī / _ #                               ; @hiatus — final -aā (< -akam) contracts to -ī: karaṇakam > kuranī; round 5
[Si] i a > i / V C* _ #                           ; @hiatus — 2 lects / final -ia (< -ita, -ika, -iya) contracts to -i after k/t/y loss (añjita > æⁿdi, abʰīta > abi, āyācita > ayadi), but not in disyllables where -i would leave a monosyllable: the y glide stays (gītá > giya, gʰr̩tá > giya, bʰiyás > biya); round 7 (net +29) / (changes.sc hiatus.ia.final-polysyllabic)
[Si] {e o} {a u} > {ē ō}@1 / _ #                    ; @hiatus — final -ea/-eu/-oa/-ou left by stop loss contract to a long vowel (bʰēdá > bē, yṓga > yō, cʰēka > sē; CDIAL also lists the uncontracted beya, seya); round 7
[Md] a a > ā / _ #                                ; @hiatus — 17 lects / -aka > -aa > -ā (ghōṭaka > gʰoṛā; extension-aware score 0.3725 → 0.3485 in Dang) / (changes.sc hiatus.aa.final.2)
[Si] a a > ā / _ #                                ; @hiatus — 17 lects / -aka > -aa > -ā (ghōṭaka > gʰoṛā; extension-aware score 0.3725 → 0.3485 in Dang) / (changes.sc hiatus.aa.final.2)
[Md] p > f                                       ; @ph-f — Dhivehi p > f (putrá > fut, páttra > fat)
[Md] {a i ī} > u / V r _ #                       ; @x-mia-insular-final-u — final vowel after r > u (māra > maru, bʰērī > beru)
[Md] a > ∅ / V C _ #                             ; @final-vowel-loss — dánta > dat, cittá > hit
[Md] i > ∅ / V {t n l} _ #                       ; @final-vowel-loss — final -i lost after t, n, l as -a is (kṣānti > ket, trī́ṇi > tin, vēlli > vel; doublets keti, veli remain); round 7
[Md] ; ── block lar-s-h-prevocalic: Marwari/Gujarati-type debuccalisation of s before a vowel: intervocalic (the shared lar-s-h-intervocalic) and word-initial. Marwari of Bagra, Dholpuri, Wagdi-Ko [@s-h — s > h before a vowel: initial and intervocalic, late (after degemination: áśva > assa > aha) (blocks/lar-s-h-prevocalic.sc)] ──
[Md] ; ── block lar-s-h-intervocalic: Intervocalic s > h: Lahnda, Punjabi, Sindhi, Dhivehi (late there); the Marwari-type lects add initial s- > h- (lar-s-h-prevocalic). Laryngeals round (2026-09-30 [@s-h — intervocalic (śaśa > hāho)] ──
[Md] s > h / V _ V                                ; @s-h — 8 lects / aśani > āhiṇ, divasa > P. dĩh, áśva > assa > Md. aha / (changes.sc s-h.s.intervocalic)
[Md] s > h / # _ V                                ; @s-h — initial (saptá > hāt, sūrya > hūraj, sūkara > huvar) / (block lar-s-h-prevocalic)
[Md] i ṁ > ī / _ s                               ; @anusvara-vowel — -iṁś- > -īs (viṁśati > vīs, triṁśat > tirīs)
[Md] ā > i / _ #                                 ; @vowel-shift — final -ā (< -aka, -ā) > -i (tāraka > tari, jīraka > diri)
[As-Ul] ; ── Brāhmī spelling ──────────────────────────────────────────────────────
[dardic-south -Bshk -bhatr -Tir -Pas Dhp west-pahari -pan] y > ∅ / C _                                  ; @cluster-simplify — 25 lects / y lost after a consonant / (changes.sc cluster-simplify.y.postconsonantal)
[Gowro] {ā ī ū} > {a i u}                           ; @length-loss — inherited length lost (Zoller writes few long vowels)
[bhatr] {a i} > ∅ / V _                              ; @hiatus — 11 lects / second vowel of a hiatus absorbed / (changes.sc hiatus.ai.postvocalic)
[bhatr] {ṭʰ tʰ} > {ṭ t}                             ; @deaspiration — voiceless dental/retroflex aspirates lost (Zoller and Strand agree: gōṣṭʰá > gōṭ)
[Gowro Chil bhatr Mai] k > h / V C* a _ #                           ; @x-dardic-south-final-aspiration — 2 lects / word-final -ak (from -akka) > -ah in polysyllables (Zoller -âh: kṣīrá+akka > ʦ̣ʰirâh, kōkila > kōlâh), before the aspiration of other final stops; the same shared rule as Gowro and bhatr (round 7: replaces the local kʰ > h after aspiration) / (changes.sc south-final-aspiration.k.vca)
[Chil Mai] {k t ṭ p} > {kʰ tʰ ṭʰ pʰ} / _ #              ; @x-dardic-south-final-aspiration — 2 lects / word-final voiceless stops aspirated (aśakta > aśâtʰ) / (changes.sc south-final-aspiration.kttp)
[Shum Wot Dm Tor] {n ṇ m} {t d dʰ ḍ b bʰ} > $1                 ; @cluster-simplify — 4 lects / nasal + stop > nasal (dánta > dan, lamba > lam) / (changes.sc cluster-simplify.nnmtddhdbb)
[Tor] {a ā ē ō i u} > ∅ / V C* _ #                 ; @final-vowel-loss — 2 lects / (round 2: {a ā ē ō i u} generalised; polysyllables only, also after a vowel (-ikā > -iā > -i) monosyllables keep their vowel; −0.001 to −0.008 in all 12 lects) / (changes.sc final-vowel-loss.aaeoiu.final-polysyllabic)
[Tor] y > ∅ / V _ #                               ; @v-loss — word-final y (from -iya, -yā after apocope) lost
[Tir] ; ── block h-initial-loss-optional: Sporadic loss of initial h- before a vowel, end of chain (2026-10-01): Si haṁsá > as, hánati > annavā; Pas hánati > an; Bishnupriya hásta > at. Beat a same-shap [cross-family law (blocks/h-initial-loss-optional.sc)] ──
[Tir] optional: h > / # _ V                        ; @h-loss — sporadic initial h- loss (haṁsá > Si as) / (block h-initial-loss-optional)
[Tir] ṣ > x                                       ; @s-x — Tirahi ṣ > x (cf. NE Pashto)
[dog kiuth kc mand cur khash] ; ── block cr-release: End of the cluster window: the onset protected by cr-keep-optional / r-anticipation-optional becomes stop + r again, and later rules (flaps, tr > ṭr, …) see an  [end of the cluster window (blocks/cr-release.sc)] ──
[dog kiuth kc mand cur khash] {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} > {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} r ; @cr-keep — protected stop + r released after the cluster stage / (block cr-release)
[kiuth] ; ── block nasals-anusvara-loss: nasals-anusvara-loss — leftover anusvāra lost without nasality, nasals theme, blocks round 2026-09-30 (pass 2). The anusvāra that survives to this point (before [@anusvara-vowel — leftover anusvāra dropped (nasality not transcribed) (blocks/nasals-anusvara-loss.sc)] ──
[kiuth] ṁ > ∅                                        ; @anusvara-loss @anusvara-vowel — 53 lects / pāṁśu > awan pā, māṁsá > Si mas, áṁsa > Si as / (changes.sc anusvara-loss)
[WPah] {a i u} > ∅ / V C _ C                        ; @syncope — 9 lects / medial short vowel in an open non-initial syllable (round 2) / (changes.sc syncope.aiu.vcc)
[WPah] a > ∅ / V C C _ C V                          ; @syncope — 7 lects / medial a also lost after a closed syllable when a vowel follows (round 6 second pass) / (changes.sc syncope.a.vcccv)
[NiDoc Dhp KharI] ; ══ Kharoṣṭhī spelling ═══════════════════════════════════════════════════
[NiDoc Dhp] ś c > c                                     ; @notation — Kharoṣṭhī writes śc with the c̄ sign, as NiDoc (āścarya > ac̄aria, paścā́t > pac̄a, paścimá > pac̄ima); round 7
[NiDoc] {d dʰ} > n / ṁ _                            ; @nasal-assim — Niya nd, ndʰ > ṁn (bandʰana > baṁnana, cʰindati > cʰiṁnati)
[NiDoc Dhp] Nasal $1 > ṁ $2 // # _                      ; @notation — geminate nasals written ṁN (anna > aṁna, dʰamma > dʰaṁma), not word-initially: jñ- > ññ- is written ñ- (jñāna > ñana, jñātá > ñada), as NiDoc; round 7
[Dhp] C $1 > $2                                   ; @notation — geminates not written (tátra > tata)
[NiDoc] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina [@degem-plain — NIA plain degemination (GL = G∪G2∪G3) (blocks/length-degem-plain.sc; blocks round 2026-09-30)] ──
[NiDoc] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[NiDoc] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[S P kcch L srk awan] {ē ō} > {e o}                               ; @notation — Turner writes e o without macron
[As-Ul awan] {ē ō ē̃ ō̃} > {e o ẽ õ}                       ; @notation — Aw (Saksena) marks ē ō only where CL lengthened a vowel (later rule), so the macron is dropped before CL
[Si Md OSi] ; ── block lar-si-umlaut-liquid: Sinhala late umlaut across a liquid: a > i before r/ḷ/l + i, a > u before r + u. Sinhala, Old Sinhala. Laryngeals round (2026-09-30). [@umlaut — a > i / u across a liquid before i / u (blocks/lar-si-umlaut-liquid.sc)] ──
[Si Md OSi] a > i / # C _ {r ḷ l} i                      ; @umlaut — 2 lects / pari- > piri-, paṭi- > piḷi- / (changes.sc umlaut.crlli)
[Si Md OSi] a > u / C _ r u                              ; @umlaut — 2 lects / antara > aturu, kēsara > kesuru / (changes.sc umlaut.cru)
[Si Md OSi] ; ── block nasals-anusvara-loss: nasals-anusvara-loss — leftover anusvāra lost without nasality, nasals theme, blocks round 2026-09-30 (pass 2). The anusvāra that survives to this point (before [@anusvara-loss — anusvāra lost everywhere without lengthening, as Si (replaces the old ṁ before voiceless stops only): māṁsá > mas, pāṁśu > fas, saṁvatsará > (h)aharu; after -iṁś- > -īs; round 7 (blocks/nasals-anusvara-loss.sc)] ──
[Si Md OSi] ṁ > ∅                                        ; @anusvara-loss @anusvara-vowel — 53 lects / pāṁśu > awan pā, māṁsá > Si mas, áṁsa > Si as / (changes.sc anusvara-loss)
[OSi] {ṁ ñ} > ∅ / _ C                             ; @anusvara-loss — anusvāra not written before consonants (kāñcana > kasun)
[OSi] {ⁿg ⁿj ⁿḍ ⁿd ᵐb} > {g j ḍ d b}             ; @notation — Old Sinhala script does not write prenasalisation (áŋga > aga, ámbara > abara)
[OSi S] m r > m b / V _                              ; @mr-mb — 5 lects / MIA medial mr > mb with an epenthetic stop (āmrá > amba, tāmrá > tamba); unified at round-7 integration / (changes.sc mr-mb.mr.postvocalic)
[P] m r > m b / V _                              ; @mr-mb — 5 lects / MIA medial mr > mb with an epenthetic stop (āmrá > amba, tāmrá > tamba); unified at round-7 integration / (changes.sc mr-mb.mr.postvocalic)
[P L] %source gill
[awan] %end
[awan] %source rensch-hallberg-oleary1992
[awan] a > ā / _ G $1                              ; @notation — Rensch writes the phonetic length of a before a geminate (hásta > hātʰ, agní > āg; round 7)
[khet] %source elfenbein
[khet] {g j b d} > {ɠ ʄ ɓ ɗ} / # _                 ; @notation @implosive — only Elfenbein's phonetic transcription records Khetrani initial implosives (ɗūr, ɓolī); CDIAL/Grierson forms do not
[khet awan] %end
[srk awan] %source rensch-hallberg-oleary1992 gill
[poth] %source lothers-lothers2010pahari gill liljegren-hindukush
[khet] %source elfenbein grierson-lsi1928
[L] G $1 > $2                                   ; @notation — this wordlist rarely writes gemination (Gill 2/52 forms vs CDIAL 1137/7065)
[awan] G $1 > $2                                    ; @cluster-simplify @degem-plain — 83 lects / residual geminates (vaḍra > *barr > baro, culla > culo) / (changes.sc degem-plain.g)
[P poth khet srk] G $1 > $2                                    ; @cluster-simplify @degem-plain — 83 lects / residual geminates (vaḍra > *barr > baro, culla > culo) / (changes.sc degem-plain.g)
[P L awan] ; ── block lar-deaspiration-initial-devoicing: Punjabi-type tonogenesis, segmental half: initial voiced aspirates become voiceless unaspirated (bʰ dʰ gʰ jʰ ḍʰ > p t k c ṭ, with low tone, which the @tone rule [@deaspiration @tone — initial voiced aspirates > voiceless unaspirated + tone (blocks/lar-deaspiration-initial-devoicing.sc)] ──
[P L awan] {bʰ dʰ gʰ jʰ ḍʰ} > {p t k c ṭ} / # _         ; @deaspiration @tone — 2 lects / gʰara > poth. kar, bʰaginīpati > L. bàṇiā / (changes.sc deaspiration.bhdhghjhdh.initial)
[dod] V > ∅ / V C _ #                             ; @final-vowel-loss — after a single consonant only
[dod] a > ō / _ #                                 ; @vowel-shift — surviving final -a (after clusters) becomes the masc. -ō; retagged as a sound change in round 6
[pog] V > ∅ / C _ # // C n _                      ; @final-vowel-loss — (the -nu of consonant-stem infinitives keeps its vowel)
[pog] ; ── block finals-aka-u-dardic: The -uka ending in Kashmiric/Kohistani (blocks theme "finals", 2026-09-30): the vowel after a final -u left by -uka (-ua, -uā, -uō) is lost, so -uka ends in -u  [-ua > -u (blocks/finals-aka-u-dardic.sc) — replaces `use aka.aao.u`: a final vowel after u (from -uka after k loss) is absorbed: -uka > -u (round 4)] ──
[pog] {a ā ō} > ∅ / u _ #                          ; @aka @hiatus — 3 lects / -ua/-uā/-uō > -u (dod sá+uka > su; -uka > -ua > -u) / (changes.sc aka.aao.u)
[Sv pog Ush Kund Phal] ; ── block finals-k-loss: Loss of a final -k after apocope (blocks theme "finals", 2026-09-30): the k of -uka/-ika/-aka variants that apocope leaves word-final in a polysyllable is lost  [final -k lost after apocope (blocks/finals-k-loss.sc) — replaces `use final-cons-loss.k.vcv`: a final k left by apocope in polysyllables is lost (áṇu-uka > āṇu, taptá-uka > tātu, *akṣadāruka > aʦ̣ʰāru) (round 6; installed over a +0.0013 test-verb gate miss caused by one mislinked pair, *vunāti > sil)] ──
[Sv pog Ush Kund Phal] k > ∅ / V C* V _ #                           ; @final-cons-loss — 3 lects / final -k lost in polysyllables after apocope (Phal. *akṣadāruka > aʦ̣ʰāru, áṇu+uka > āṇu) / (changes.sc final-cons-loss.k.vcv)
[Phal] a k a m > ū m / _ #                         ; @hiatus @aka — neuter -akam contracts to -ūm (aka > o, raised before the nasal: cʰōr-akam > ʦ̣ʰorū́m, *diy-akam > dyūm) (round 6; installed over a +0.0014 test-verb gate miss caused by one mislinked pair, upapatí > vī́vay)
[Phal] Short > $1-stress / _ !#* Long | Long !#* _       ; @accent-placement — in a word with a long vowel, short vowels do not carry the stress …
[Phal] Long > $1+stress / _ !Long* #                ; @accent-placement — … the (last) long vowel does: after apocope the old penult stress shifted onto the syllable with length, which Liljegren marks with tone (ʦ̣ʰor-ū́m, bʰā̂r-type; *akṣadāruka > aʦ̣ʰā̂ru) (round 7)
[Ush Phal Kal] ; ── Per-source notation ─────────────────────────────────────────────────
[Sh] o > u / ō n _ #                              ; @vowel-shift — final -o dissimilates to -u after -ōn- (the -ōno infinitive > Gilgiti -ōnu: bonōnu, rōnu) (round 6)
[Sh] ; ── Per-source citation conventions (applied after the chain, only for that source's pairs) ──
[Sh] %source schmidt
[Sh] o i k > ō̌ n / _ #                           ; @morph — Schmidt cites the Gilgiti infinitive -ō̌nu/-ō̌n (with his rising-tone caron)
[Phal] %source liljegren
[Sh Phal] {ā ī ū ē ō} > {ā̂ ī̂ ū̂ ē̂ ō̂}                  ; @notation — Liljegren marks tone on long vowels, falling (circumflex) being the commoner
[Sh] ō̂ > ō̌ / _ n (V) #                           ; @notation @tone — but the ō of -ōn(u), contracted from -a(n)a- (bandʰ-anaka > bonōnu), has Schmidt's rising tone: kʰā́dati > kʰō̌n, *uttʰāti > utʰyō̌n, *r̩tuvanta > yō̌nu (contracted long vowels are rising in 48% vs 16–22% for inherited ones) (round 7, tone-gated)
[Sh Phal northwest -OP -thar -S -Khol -kcch] %end
[P awan] %source CDIAL
[P] ḷ > l                                        ; @notation — CDIAL writes the Punjabi retroflex lateral as l (gill's phonetic list writes ḷ in 7 of 8 cases)
[poth] %source lothers-lothers2010pahari
[G] ū̃ > ũ                                      ; @vowel-shift — nasal ū̃ > ũ (the ū merger itself is now the early rule plus no lengthening of u)
[WK] u > ū                                       ; @notation — zubair writes ū for short u
[hindi -H -Brj] v > ∅ / {u ū ō o} _ V                        ; @v-loss — 4 lects / v (< -p-/-v-) lost after a rounded vowel (kū́pa > kūā, pūpa > pūā), as in H; round 6b / (changes.sc v-loss.v.uuoov)
[H Brj] a ū > a u                                    ; @hiatus — 2 lects / round 7: a + ū left in hiatus by consonant loss becomes the diphthong au, as in H (*hastakūṭa > hatʰauṛā); train ext −0.0010, 2 better / 0 worse / (changes.sc hiatus.au)
[NDu] {a ā} {e ē i} > ɛ / _ #                    ; @hiatus @vowel-shift — round 7: final -ai/-ae (3sg -ati/-atē after t-loss) monophthongises to -ɛ, the E. Hindi 3sg (máratē > marɛ, bʰajyátē > bʰājɛ, cf. list margɛ, bʰāgatʰɛ); train ext −0.0014, 4 better / 0 worse
[hindi -OAw] {i ī} {a ā} > ī / _ C #                      ; @hiatus — 7 lects / round 7: i + a in hiatus before a final consonant contracts to ī (participle -ita + -sya > -iasa > -īs, the E. Hindi 3sg past: jarīs, sūnīs, dekʰīs; also nouns in -ias < -ika-s-); train ext −0.0008, test ext −0.0015, 6 better / 2 worse (noun -ias forms; the W. Hindi lects have no -is past) / (changes.sc hiatus.iiaa.c)
[OAw NDu H Aw bagheli_lakshman] ∅ > y / {i ī} _ {a ā}                        ; @ya-sruti — 3 lects / round 7: y glide in i/ī + a/ā hiatus, as in H (both lists write -iyā); train ext −0.0042, test −0.0031, 34 better / 17 worse / (changes.sc ya-sruti.iiaa)
[Kannauji-Central] ; round 7: removed the @morph Ø > o imperative rule (neutral: never changes a score)
[H] v > ∅ / ā _ n ā #                         ; @v-loss — causative -āvnā (< -āpanaka) > -ānā (aṭṭ > aṭānā, antar- > ãtarānā)
[Gaw kaithal NDu H Aw bagheli_lakshman thar poth awan mewari_basad WK Nimadi Marw G dhundari_bamore Goj jaun khash cam pan] a ī > ī / C _ #                              ; @hiatus — 13 lects / -aī contracts (as G §12) / (changes.sc hiatus.ai.c)
[rajasthani-gujarati -mewari_basad -G] ∅ > y / i _ o #                              ; @glide — 11 lects / y glide in final -io (evāḷiyo, māriyo; as G §12) / (changes.sc glide.io)
[Ku] ā > a / _ {u i} #                            ; @hiatus — ā shortened before a final -u/-i left in hiatus by consonant loss (gʰāta > gʰau, taḍāga > talau, *upalāya > ulau); round 7
[Ku N] ∅ > y / {ā i e} _ o #                        ; @hiatus — 2 lects / ya-śruti glide before final -o, as in N (āyo < āgata, cayo < cáya, payo < padá) / (changes.sc hiatus.0.aieo)
[Ku N] ∅ > v / {u ū o ō} _ ā #                      ; @hiatus — 2 lects / va-śruti glide between a back vowel and final -ā, as in N (śúka > suvā, *nirmukʰa > nimuvā, yugá > jūvā); round 7 / (changes.sc hiatus.0.uuooa)
[Ku Garh N] {ā ā̃} > {a ã} / # C* _ C* {ā ā̃}            ; @x-cep-pretonic-shortening — 3 lects / NIA: first-syllable ā shortened before a long ā (stress on the later long vowel; gamār, bʰãṛār, pacās, masā̃t, aŋār); the Vedic-accent version loses (+0.002…+0.011) / (changes.sc pretonic-shortening.aa.ccaa)
[Ku Garh] i > y / C _ {a ā o u}                        ; @hiatus — 2 lects / glide formation in hiatus after C (Garh. syū̃d < sīmanta, adʰyāṇ; Ku.); not N. / (changes.sc hiatus.i.caaou)
[eur RomWel Zarg Ap] y > ∅ / V _ V                                ; @int-stop-loss — 4 lects / intervocalic y lost (-aya- > -a-) / (changes.sc int-stop-loss.y)
[eur as-ll RomBalk] ; ── block finals-syncope-romani: Romani/Domari syncope (blocks theme "finals", 2026-09-30): a short a i u after one or two consonants is lost before a single consonant and a vowel, but not befo [Romani syncope (blocks/finals-syncope-romani.sc) — replaces `{a i u} > ∅ / V (C) C _ C V // _ v`: medial short vowel lost (kʰabni, terno), but not before v (calavel, not *calvel); jointly with -aya- > -ava- (alone +3 and +4 net reflexes, together +10)] ──
[eur as-ll RomBalk] {a i u} > ∅ / V (C) C _ C V // _ v           ; @syncope — medial short vowel lost in V(C)C_CV, not before v (eur. kōmalá+aka > kovlo, táruṇa+aka > terno, garbʰíṇī > kʰabni) / (block finals-syncope-romani)
[as-ll RomBalk Zarg Lom] ; ── block finals-syncope: NIA schwa syncope (blocks theme "finals", 2026-09-30): after apocope, medial short a in an open syllable between single consonants, with a vowel following, is l [NIA schwa syncope (blocks/finals-syncope.sc) — replaces `use syncope.a.vccv`: medial a syncope in open syllables (bárkara > bakri, maraṇa > marnʌ); absent in Pahari] ──
[as-ll RomBalk Zarg Lom] a > ∅ / V C _ C V                            ; @syncope — 16 lects / medial a lost in VCaCV (thar kaccara+aka > kacro, karpaṭa+aka > kapṛo, kṣurapra+ikā > kʰurpī) / (changes.sc syncope.a.vccv)
[RomWel] ∅ > v / # _ a                                ; @prothesis — Welsh Romani prothetic v- before initial a (vaŋār, vaŋlī, vavēr, vārdō)
[eur RomWel RomBalk Zarg Lom Marw dhundari_bamore] ; ── block finals-apocope-long: NIA apocope with long -ā, in polysyllables (blocks theme "finals", 2026-09-30): final -a -ā -i -u are lost after a preceding vowel (V C* _ #), so OIA monosyllab [apocope incl. -ā, in polysyllables (blocks/finals-apocope-long.sc) — replaces `use final-vowel-loss.aaiu.final-polysyllabic`: final short vowels and -ā lost in polysyllables (round 7): cʰatti > cʰat, náva > nav, trī́ṇi > tīn; monosyllables keep theirs (gāva > gā, bā); -ī/-o forms come from -ikā/-aka] ──
[eur RomWel RomBalk Zarg Lom Marw dhundari_bamore] {a ā i u} > ∅ / V C* _ #                     ; @final-vowel-loss — 3 lects / final vowels lost in polysyllables (Marw. adyá > āj, aṣṭā́ > āṭʰ, kárṇa > kān) / (changes.sc final-vowel-loss.aaiu.final-polysyllabic)
[mewari_basad Nimadi] ; ── block finals-apocope-two-step: Two-step apocope in polysyllables (blocks theme "finals", 2026-09-30): final short -a -i -u are lost (finals-apocope: Nimadi kárṇa > kān, dánta > dā̃t, hásta >  [short vowels, then -ā, lost in polysyllables (blocks/finals-apocope-two-step.sc)] ──
[mewari_basad Nimadi] ; ── block finals-apocope: NIA apocope (blocks theme "finals", 2026-09-30): loss of the short final vowels -a -i -u that the Apabhraṁśa stage left word-final (after finals-ap-shortening w ──
[Nimadi] {a i u} > ∅ / V C* _ #                       ; @final-vowel-loss — 3 lects / final short -a -i -u lost in polysyllables (H. akṣára > ākʰar, agní > āg, áŋga > ā̃g) / (changes.sc final-vowel-loss.aiu.final-polysyllabic)
[mewari_basad Nimadi] ā > ∅ / V C* _ # // a _                      ; @final-vowel-loss — 3 lects / final -ā lost in polysyllables, not after a (Nimadi rakṣā > rākʰ, aṣṭā́ > āṭʰ, jáṭā > jaḍ) / (changes.sc final-vowel-loss.a.final-polysyllabic.except)
[mewari_basad Nimadi mewari_dholpura Goj kiuth] a ā > ā / _ #                                ; @hiatus — 4 lects / the masc. plural -akāḥ > -aā contracts to -ā after apocope, which took only post-consonantal vowels (āmra-kā > ā̃bā, kadala-kā > keḷā) / (changes.sc hiatus.aa.final.4)
[rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_akera -G -marwari_bagra -OMarw] ; ── block finals-syncope-iyo: Syncope of i before -yo (blocks theme "finals", 2026-09-30): the i of -iyo (< -ita-ka, -iyaka) is lost after a consonant (Rajasthani, Gujarati, Doteli). [-Ciyo > -Cyo (blocks/finals-syncope-iyo.sc) — replaces `use syncope.i.cyo`: -iyo > -yo after a consonant (māryɔ, dekʰyɔ, pʰūṭyo)] ──
[rajasthani-gujarati -OG -WK -hadothi_kelwada -mewati_akera -G -marwari_bagra -OMarw] i > ∅ / C _ y o #                            ; @syncope — 4 lects / -Ciyo > -Cyo (had. māráyati+itaka > māryɔ, *dēkṣati+itaka > dekʰyɔ) / (changes.sc syncope.i.cyo)
[G marwari_bagra] ā o > o / _ #                                ; @hiatus — 2 lects / -āo > -o (as G §12) / (changes.sc hiatus.ao.final.2)
[Ko-Ul] a > ∅ / V _ #                               ; @hiatus — final -a absorbed after another vowel (-ia > -i: *alavaṇika > aḷṇī)
[Ko-Ul] ; ── block length-degem-plain: NIA plain degemination — every MIA geminate still standing simplifies without lengthening (H uttāra > utār, aṭṭāla > aṭāl). The "elsewhere" half of NIA degemina [@degem-plain — NIA plain degemination (GL = G∪G2∪G3) (blocks/length-degem-plain.sc; blocks round 2026-09-30)] ──
[Ko-Ul] class GL = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ ŋ ñ ṇ n m r y v l ḷ s ś h ʦ ʦʰ ʣ ʣʰ
[Ko-Ul] GL $1 > $2                                   ; @degem-plain — remaining geminates simplify (H uttāra > utār, aṭṭāla > aṭāl; Bhoj *laḍikka > laṛikā) / (block length-degem-plain)
[Bhatri] {a ã} > {ə ə̃} / _ C                        ; @a-centralisation — round 7: short a is [ə] before a consonant (final -a of verb forms stays: mərla) (Beine writes ə 97, ʌ 74; ā is a): gʰara > gər, pʰála > pəl, dánta > dət
[N hindi -OAw -H -Aw marathi-bhil -OM -VB OP D P srk nuristani -Ash rajasthani-gujarati -WK -G -Goj bhal dog kiuth mand WPah] {ē ō ē̃ ō̃} > {e o ẽ õ}                             ; @notation — no macron on e o
[DungraBhili rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewati_jhambaus -OMarw -Goj ktg] ; ── block lar-o-lowering: Word-final o (masc. -o < -aka, past ptc. -iyo) is open ɔ: Rajasthani, Gujarati, Dungra Bhili, Kotgarhi. Laryngeals round (2026-09-30). Edit here, not in the lec [@vowel-shift — final o > ɔ (blocks/lar-o-lowering.sc)] ──
[DungraBhili rajasthani-gujarati -mewari_basad -OG -WK -Nimadi -mewati_jhambaus -OMarw -Goj ktg] o > ɔ / _ #                                  ; @o-lowering @vowel-shift — 13 lects / gʰōṭaka > gʰoṛɔ, kadala > keḷɔ, karpaṭa > kapṛɔ / (changes.sc o-lowering)
[Marw] %source marwari mewari
[had] %source hadothi mewari
[dhundari_badagaon] o > ɔ / # C* _ C C* ɔ #                    ; @vowel-shift — first-syllable o lowered before a final ɔ (sonā-ka > sɔnɔ, *cʰōkara-ka > cʰɔrɔ, stōka-ka > tʰɔḍɔ)
[dhundari_badagaon] %source dhundari marwari
[bagri_fatehabad] %source bagri
[mewari_dholpura Rj] %source mewari
[mewati_jhambaus] o > ā / V C* _ #                             ; @final-vowel — polysyllabic -o (< -aka) > -ā, the Ahirwati/Haryanvi masc. ending (lambā, kelā, sūkʰā, baḍā, saberā); monosyllables keep -o (do, so) (hill-climb round 1)
[mewati_jhambaus mewati_akera] %source mewati
[dhundari_bamore] %source dhundari
[bagri_fatehabad Marw had mewari_dholpura mewati_jhambaus Rj mewati_akera dhundari_bamore dhundari_badagaon] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} Stop > $1-nas {ŋ ŋ ŋ ŋ ñ ñ ñ ñ ṇ ṇ ṇ ṇ n n n n m m m m}@2 $2   ; @notation — survey writes a nasal vowel before a stop as V + homorganic nasal (pāñc)
[marwari_bagra] %source marwari
[mewari_basad] i > y / C _ o #                             ; @glide — -io > -yo after a consonant (māryo, dekʰyo)
[mewari_basad] %source mewari varghese-john-samuel2009malvi
[marathi-bhil -Bhatri -hal -M -mewari_bannoda -OM -VB -Ko-Ul] # (2026-10-01: redundant after the upstream digraph-aspirate fix; the source now writes Cʰ) {kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ} > {k g c j ṭ ḍ t d p b} h   ; @notation — survey transcription writes digraph aspirates
[Khandesi PalyaBareli PauriBareli Bhilali RathwiBareli Bhili] %source varkey-vunnamatla2018bareli
[Khandesi PauriBareli RathwiBareli poth rajasthani-gujarati -OG -WK -Nimadi -hadothi_kelwada -G -OMarw -Goj] {i u} > {ī ū}                               ; @notation — Lothers & Lothers write high vowels with macron (kukkuṭa > kūkṛī)
[Dhp] s tʰ > ṭ ṭʰ                                 ; @s-stop-aspirate @notation — stʰ > ṭṭʰ with retroflexion, after the no-gemination spelling: Kharoṣṭhī has a separate sign for this cluster, transliterated ṭṭh (stʰā́na > ṭṭʰana, ástʰi > aṭṭʰi, gr̩hastʰa > gahaṭṭʰa; ṣṭ stays single ṭʰ: duṣṭa > duṭʰa); round 7
[NiDoc] r m > m                                     ; @notation — Kharoṣṭhī writes rm with the m̄ sign (kárman > kam̄a, dʰárma > dʰam̄a; the macron is not scored); round 7
[NiDoc] ; ── block ks-cch-optional: Sporadic kṣ > ccʰ beside kkʰ (MIA doublets: akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta). `optional:` = sporadic: forks the derivation where it applies (best [sporadic (blocks/ks-cch-optional.sc)] ──
[NiDoc] optional: k ṣ > c cʰ                         ; @ks — sporadic kṣ > ccʰ beside kkʰ (akṣi > akkʰi ~ accʰi, kṣetra > kʰetta ~ cʰetta) / (block ks-cch-optional)
[NiDoc] k ṣ > c h / V _                              ; @ks — Niya writes intervocalic kṣ with the special letter c̄h (dakṣiṇa > dac̄hina; tokenised c h)
[NiDoc] g > ǵ / V _ V                               ; @notation @int-spirant — Kharoṣṭhī ǵ for the spirant [ɣ] (bʰāga > bʰaǵa)
[NiDoc] ḍ > ḍ ' / V _ V                             ; @notation @flap — Niya ḍ' for the lenited retroflex (kuḍa > kuḍ'a)
[NiDoc Dhp KharI] V > $1-long                                 ; @notation — no vowel length written
[mia-insular -Si -NiDoc -Pa -KharI] ; ══ Per-source notation ══════════════════════════════════════════════════
[Pk Ap] %source sheth1923
[Pk Ap] ṁ > ŋ / _ {k kʰ g gʰ}                       ; @notation — Sheth 1923 writes the homorganic nasal, CDIAL the anusvāra (paṁca / pañca)
[Pk Ap] ṁ > ñ / _ {c cʰ j jʰ}                       ; @notation
[Pk Ap] ṁ > ṇ / _ {ṭ ṭʰ ḍ ḍʰ}                       ; @notation
[Pk Ap] ṁ > n / _ {t tʰ d dʰ}                       ; @notation
[Pk Ap] ṁ > m / _ {p pʰ b bʰ}                       ; @notation
[Pk Ap] {e o} > {ē ō}                               ; @notation — Sheth marks e o long also before geminates (tēlla)
[As-Ul] %source andersen1990
[As-Ul] ṁ > m                                       ; @notation — Andersen 1990 writes m for the anusvāra (mamca, amta)
[Md] %source fritz
[Md] r̆ > ṣ                                       ; @notation — Fritz writes ṣ for the retroflex sibilant/flap CDIAL writes r̆
[OSi] %source paranavitana
[OSi] {æ ǣ} > {a ā}                                ; @notation — the Paranavitana glossary writes the umlauted vowel as a (no æ anywhere in the source: vāpī́ > vavi beside CDIAL OSi. væ̃); round 7
[OSi] {ṭ ṇ ḷ} > {ț ņ ļ}                           ; @notation — Paranavitana's Sigiri glossary (as digitised) writes ț ņ ļ for ṭ ṇ ḷ
[Dhp] %source gandhari
[Dhp] ṇ > n                                       ; @notation — the Gāndhārī dictionary writes n for ṇ (karuṇā > karuna)
[Kho] a > e / C _ i                               ; @aya-e @hiatus — a + i in hiatus > ei (dʰāpayati > deik, -ayati causatives > -eik); retagged as a sound change in round 6 so it also applies to verb forms
[dardic-north -K] m b > m / _ #                                ; @cluster-simplify — 4 lects / word-final mb (left by apocope) loses its b, as in Khowar (ḍimba > ḍim) (round 7, consistency) / (changes.sc cluster-simplify.mb.final)
[Kt Kam] ; ── block r-anticipation-optional: Sporadic r-anticipation (Dardic/NW 'metathesis'): the r of the first-syllable coda or of the second syllable's onset cluster is copied into the onset of the ini [sporadic (blocks/r-anticipation-optional.sc)] ──
[Kt Kam] optional: {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} > {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} / # _ V r C | # _ V C r ; @metathesis — sporadic r-anticipation into the initial onset (tarka > trakk-, kárman > krom) / (block r-anticipation-optional)
[Pr] {a ā} > ∅ / C _ #                           ; @final-vowel-loss — final -ā is already lost in PNur (drā́kṣā > *drās, vḗlā > *vel); the modern lects also drop final -a (kīla > Kt kíl)
[dardic-north -Sh -Kho -K -Kal nuristani] C {y v} > $1                                 ; @cluster-simplify — 18 lects / C + y/v simplify after lenition: pakvá > *pak, āsyà > *āś / (changes.sc cluster-simplify.cyv)
[Gowro nuristani] {k p} {t ṭ} > $2                             ; @cluster-simplify — 7 lects / śuktá > *suta / (changes.sc cluster-simplify.kptt)
[nuristani -Ash -Wg] t > ∅ / _ {k m ʦ ʦ̣ ś s p}                   ; @cluster-simplify — 4 lects / round 6: t lost before other obstruents and m (ātmán > āmú, utkrāma > vukṛom, *ut-stʰāti > uśt); tr survives (putrá) / (changes.sc cluster-simplify.t.kmtstsssp)
[nuristani] t > ∅ / _ {k m ʦ ʦ̣ ś p} | _ s C             ; @cluster-simplify — 3 lects / round 6 (as Kt): t lost before other obstruents and m (*mr̩tka > mūk, *pʰūtka > puk, utpātaka > upá); ts is kept for the late ts > ʦ (fourth pass: moved after post-vocalic ʦ > ś, as in Ash/Pr)2 / (changes.sc cluster-simplify.t.kmtstsspsc)
[Kt Kam] ; ── block cr-release: End of the cluster window: the onset protected by cr-keep-optional / r-anticipation-optional becomes stop + r again, and later rules (flaps, tr > ṭr, …) see an  [end of the cluster window (blocks/cr-release.sc)] ──
[Kt Kam] {kʳ kʰʳ gʳ gʰʳ cʳ cʰʳ jʳ jʰʳ ṭʳ ṭʰʳ ḍʳ ḍʰʳ tʳ tʰʳ dʳ dʰʳ pʳ pʰʳ bʳ bʰʳ} > {k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ} r ; @cr-keep — protected stop + r released after the cluster stage / (block cr-release)
[Phal Kal] %source decker1992
[Sh] %source rajapurohit2012
[Sh] o i k > ō n o / _ #                         ; @morph — Rajapurohit cites the infinitive -ṓno
[Sh PauriBareli RathwiBareli mia-insular -Si -NiDoc -Pa -KharI poth rajasthani-gujarati -OG -WK -hadothi_kelwada -G -OMarw -Goj] %end
[poth] %source gill
[poth awan] Vedic > $1-stress                          ; @notation @tone — (the carried Vedic accent is never shown in this lect; cleared so only the placed accent is written)
[poth awan] Stress > $1+acc                            ; @notation @tone — write the placed accent …
[poth awan] Stress > $1-stress                         ; @notation @tone — … as a visible mark
[poth awan] {á ã́ ā́ ā̃́ é ẽ́ ḗ ē̃́ í ĩ́ ī́ ī̃́ ó ṍ ṓ ō̃́ ú ṹ ū́ ū̃́ ɛ́ ɛ̃́ ɛ̄́ ɛ̄̃́ ɔ́ ɔ̃́ ɔ̄́ ɔ̄̃́ ə́ ə̃́} > {à ã̀ ā̀ ā̃̀ è ẽ̀ ḕ ē̃̀ ì ĩ̀ ī̀ ī̃̀ ò õ̀ ṑ ō̃̀ ù ũ̀ ū̀ ū̃̀ ɛ̀ ɛ̃̀ ɛ̄̀ ɛ̄̃̀ ɔ̀ ɔ̃̀ ɔ̄̀ ɔ̄̃̀ ə̀ ə̃̀} ; @notation @tone — Turner writes the Hindko tone (from lost h and voiced aspirates) with a grave: mahārājá > mε̄̀rā, lōhá > lṑā, vāhana > vā̀ṇ, bʰaginī > bʰε̄̀ṇ (105 of 114 marked Awankari forms)
[D] %source weinreich2008
[D] {ā ī ū e o} > {a i u e o} {a i u e o}@1   ; @notation — Weinreich writes long vowels (and e o, which are long) doubled: kóon, ʦ̣éedoos, póo
[eur RomWel RomBalk Zarg] ; ── block lar-h-loss: Loss of h in every position: Romani (Balkan, European, Welsh, Zargari). Laryngeals round (2026-09-30). Edit here, not in the lects. [@h-loss — h lost everywhere (blocks/lar-h-loss.sc)] ──
[eur RomWel RomBalk Zarg] h > ∅                                        ; @h-loss — 4 lects / hasati > asal, hēmanta > ivend, bāhu > baya / (changes.sc h-loss.h)
[eur RomWel RomBalk Zarg Lom Chil OSi] {ā ī ū ē ō} > {a i u e o}                    ; @length-loss @vowel-shift — 12 lects / Romani has no phonemic length / (changes.sc length-loss.aiueo)
[Rana Dang Bhoj Sunha Mth Bi KochilaTharu] ; ── block length-antepenult-shortening: Bihari antepenultimate shortening: a long vowel (oral or nasal) two or more syllables before the end of the word shortens, counted after apocope, i.e. outside t [@vowel-shift @stress — antepenultimate long vowels shorten (after apocope) (blocks/length-antepenult-shortening.sc; blocks round 2026-09-30)] ──
[Rana Dang Bhoj Sunha Mth Bi KochilaTharu] {ā ī ū ā̃ ī̃ ū̃} > {a i u ã ĩ ũ} / _ C* V C* V C* # ; @vowel-shift @stress — antepenultimate long vowels shorten (Bhoj yantrasvāra > jãtasār, pāḍḍarūpa > paṛarū) / (block length-antepenult-shortening)
[Mth as-ll] u > ∅ / _ #                                  ; @final-vowel-loss — 2 lects / only -u is lost (vāyú > vai); final -i kept (máni, sáli, asti, vasti; barytone -i loss +0.008, oxytone -í loss +0.002: no accent effect) and -a/-ā kept as -ʌ/-a / (changes.sc final-vowel-loss.u)
[eur as-ll RomBalk Zarg Lom] ; ── block nasals-dental-romani: nasals-dental-romani — every non-labial nasal > n (ṇ ŋ ñ merge with n), Romani/Domari/Lomavren, nasals theme, blocks round 2026-09-30 (pass 2). [@n-dental — all nasals before stops written n (angar, manro) (blocks/nasals-dental-romani.sc)] ──
[eur as-ll RomBalk Zarg Lom] {ṇ ŋ ñ} > n                                  ; @n-dental — 5 lects / áŋgāra > eur angar, káṇṭaka > eur kanro / (changes.sc n-dental.nngn)
[BH MB MH KS] ā > ∅ / i _ C                                ; @hiatus — 3 lects / late medial iā > i contraction (*bʰiyantara > bʰitar); final -iā (< -iyaka) kept (round 7: the old unconditioned iā > i ate the -iyaka ending: haridrā > haldiā, pradʰí > pāhiā) / (changes.sc hiatus.a.ic.2)
[KS] ā > a / _ C (C) V C                         ; @pretonic-shortening — ā > a before a closed (stressed) syllable, as in Bihari (KS only)
[A] ∅ > y / i _ ā                               ; @ya-sruti — hiatus glide in -iā (piāsī > piyāsī, kuñcikā > kũciyā)
[A] ∅ > v / {u o} _ ā                         ; @ya-sruti — labial hiatus glide before -ā after u, o (kʰasa > kʰahuvā, ṭēraka > ṭeruvā, -uka > -uvā)
[eur RomWel Lom Hajong] {ṭ ṭʰ ḍ ḍʰ} > {t tʰ d dʰ}                    ; @deretroflexion — 4 lects / residual retroflexes (ruṣṭa > ruśto) merge with dentals / (changes.sc deretroflexion)
[RomWel] {ṇ ñ} > n                                    ; @n-dental — ŋ kept in Welsh Romani (vaŋlī)
[RomWel Shum Khol] kʰ > x                                       ; @int-spirant @kh-x — 3 lects / kʰ > x (xur, xulav) / (changes.sc int-spirant)
[eur RomWel] ; ── block lar-romani-deaspiration-late: Late Romani (European, Welsh) aspiration losses after the shared Romani chain: cʰ > c, and the voiceless aspirates after a sibilant. Laryngeals round (2026-09-3 [@deaspiration — late cʰ > c and post-sibilant deaspiration (blocks/lar-romani-deaspiration-late.sc)] ──
[eur RomWel Goj] cʰ > c                                       ; @deaspiration — 3 lects / cʰikkā > cik / (changes.sc deaspiration.ch)
[eur RomWel] {kʰ tʰ pʰ} > {k t p} / {ś s} _               ; @deaspiration — 2 lects / after a sibilant (aŋguṣṭʰa > anguśt, hásta > vast) / (changes.sc deaspiration.khthph.ss)
[Zarg] a > ā                                        ; @notation — rezaibaghbidi2003 writes the low vowel /ɒ/ as ā (kān, māng, tʰān)
[Sh] %source liljegren-hindukush
[Sh] o i k > ō n / _ #                           ; @morph — Liljegren's wordlists cite -ōn/-ōk
[Sv Kalk] ; ── Per-source citation conventions ─────────────────────────────────────
[eur Sv Dhp Marw] %source CDIAL
[Marw] ḍ > ṛ / V _ V                               ; @notation — CDIAL's Marwari writes the flap ṛ where the surveys write ḍ (kaṛo, gʰaṛī)
[Marw] ɔ > o / _ #                                 ; @notation — CDIAL's Marwari writes the final -o closed (kaṛo, gʰoṛo)
[Goj] %source hugoniot-polster-ahmad-rajan2023easterngujari varghesekumar2015noira
[Nimadi] %source varkey-vunnamatla2018bareli
[Nimadi Goj] # (2026-10-01: redundant after the upstream digraph-aspirate fix; the source now writes Cʰ) {kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ} > {k g c j ṭ ḍ t d p b} h   ; @notation — this source writes aspirates as digraphs (kh, th)
[H] ō > u / # C* _ C Long C V                   ; @stress @x-hindi-pretonic-shortening — round 7: a first-syllable ō (mostly < ava-/apa-/upa-) shortens to u before a stressed long second syllable in a trisyllable, like ā ī ū > a i u pretonically (apasāra > usārā, avasārayati > usārnā, apadʰārayati > uhārnā); ē does not (e 16 vs i 9). Train ext −0.0002, verb −0.0009, test −0.0003, 22 better / 9 worse
[H] v > u / V _ #                                ; @x-hindi-v-vocalise — 4 lects / round 7: final -v (< -p-, -v-, -m-) vocalises to -u, as in Aw, Bg (*gāva > gāu, jīvá > jīu, *cava > cau; H. -Vu# 63× vs -Vv# 29×); train ext −0.0004, test −0.0001, 24 better / 17 worse (rejected in §12 on plain NED alone) / (changes.sc v-vocalise)
[H] ; ── block length-hindi-pretonic: Hindi-belt pretonic shortening: a long vowel shortens before the NIA stressed syllable — before the stressed diminutive -i(y)ā, and in the first syllable before [@x-hindi-pretonic-shortening @stress — pretonic shortening before -i(y)ā and before a stressed final CV̄C (blocks/length-hindi-pretonic.sc; blocks round 2026-09-30)] ──
[H] {ā ī ū} > {a i u} / _ C i (y) ā #            ; @x-hindi-pretonic-shortening @stress — 6 lects / before the stressed -i(y)ā (H stʰālī + -iyā > tʰaliyā) / (changes.sc pretonic-shortening.aiu.ciya)
[H] {ā ī ū} > {a i u} / # C* _ C Long C #        ; @x-hindi-pretonic-shortening @stress — 2 lects / first syllable before a stressed final CV̄C (H āhāra > ahār) / (changes.sc pretonic-shortening.aiu.cclongc)
[kaithal] {i u} > a / V C _ C                         ; @vowel-shift @stress — round 7: an unstressed medial high vowel (after the stressed first syllable, before a consonant) centralises to a/ə in Haryanvi (*lakkuṭa > lakaḍ, lakaṛi; vidyullatā > bijali; láśuna > lasaṇ); u alone: train ext −0.0037, 12 better / 4 worse, test ±0; adding i: a further −0.0002 (2 / 1). Worse in every neighbour
[H] {ē ō ē̃ ō̃} > {e o ẽ õ}                       ; @notation — e o without macron
[bundeli_atarra] ; Ø > ∅                                       ; @morph — root verbs cited as bare stems   [pruned for bundeli_atarra: never fires]
[bagheli_lakshman] ; Ø > ∅                                       ; @morph — root verbs cited as bare stems   [pruned for bagheli_lakshman: never fires]
[bagheli_lakshman] %source koshy2022bagheli
[bundeli_atarra] %source bundeli
[bundeli_atarra] tʰ > t ʰ                                    ; @notation — bundeli list writes dental tʰ as t + detached ʰ (30 of 34)
[bundeli_atarra bagheli_lakshman] {ṛ ṛʰ} > {ḍ ḍʰ}                             ; @notation — bundeli list does not distinguish ṛ from ḍ
[bagheli_lakshman] {t tʰ d dʰ} > {ṭ ṭʰ ḍ ḍʰ}                   ; @notation — Koshy writes dentals with the retroflex dot (saptá > saṭey)
[bagheli_lakshman] a > e                                       ; @notation — Koshy writes schwa as e (samāná > seman)
[bagheli_lakshman] {ā ū i} > {a u ī}                           ; @notation — Koshy leaves a/u length unmarked and writes i as ī
[bagheli_lakshman] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} > $1-nas                   ; @notation — Koshy rarely marks nasality (páñca > pac)
[PauriBareli Vasavi Noiri RathwiBareli] %source watters2013northerndhule
[PauriBareli Vasavi Noiri RathwiBareli] a > ā / # (C) _                            ; @notation — watters2013northerndhule writes initial-syllable a as ā (open [a] vs varkey's a)
[PauriBareli Vasavi Noiri RathwiBareli] {d dʰ} > {ḍ ḍʰ}                             ; @notation — watters2013northerndhule writes dental d as ḍ (ḍor, ḍuḍ)
[PauriBareli Vasavi Noiri RathwiBareli] {ī ū ī̃ ū̃} > {i u ĩ ũ}                      ; @length-loss @nur-innovation @vowel-shift — 31 lects / i/ī, u/ū merged, also the ī ū from compensatory lengthening (kʰura > kʰur, tikta > titta > tit) / (changes.sc length-loss.iuiu)
[mewari_bannoda] %source mewari
[PalyaBareli mewari_bannoda Bhilali Bhili] {i u} > {ī ū}                               ; @notation — varkey-vunnamatla2018bareli writes ī ū for short i u
[Sv] i l > ∅ / _ #                               ; @morph — CDIAL cites Sauji verbs as stems
[Kalk Ush] %source rensch-decker-hallberg1992
[eur Sh Sv Kalk bagheli_lakshman marathi-bhil -Bhatri -DungraBhili -Rathawi -hal -M -OM -VB -Ko-Ul P poth awan Marw Goj] %end
[P] %source arora
[srk awan] %source rensch-hallberg-oleary1992
[srk awan] {cʰ j} > {c y}                              ; @notation — Rensch et al. write no j and no cʰ: the palatal affricates appear as y and c (awan: j→y 11/11, cʰ→c 14/15 on train; jána > yaṇā, majján > mī̃y, *cʰatti > cat, púccʰa > pūc); round 7
[jaun] {ā ī ū} > {a i u} / _ C V C* V               ; @vowel-shift — 4 lects / long vowel shortened two or more syllables before the end (NIA weight/stress; round 2) / (changes.sc vowel-shift.aiu.cvcv)
[west-pahari -bhal -mand -ktg -kvar -kul] ; ══ Round 6 audit: hiatus leftovers after consonant loss ══
[mand ktg] ; ══ Accent (latent tier; round 3) ════════════════════════════════════════
[ktg] V > $1+stress / # C* _                      ; @accent-placement — modern initial stress on the first syllable (88% of Hendriksen's marked accents are word-initial; train accent LOO +0.0505; Vedic, weight/penult, ā-quantity and tone-after-aspirate variants all do worse or fail LOO, see notes §9)
[ktg] ə > $1-stress / # C* _                      ; @accent-placement — a reduced first-syllable ə is never stressed: the initial stress passes to the long vowel marked above (round 7; ext_acc train .3802→.3792, test .3324→.3309)
[ktg] ; ══ Round 6 audit: hiatus leftovers ══
[Si Md OSi west-pahari -mand -kvar -kul] V $1 > $1                                    ; @hiatus — 12 lects / identical vowels merge (round 6) / (changes.sc hiatus.v.2)
[Tor] r > ∅ / V _ #                               ; @x-dardic-south-final-r-loss — kṣīrá > ʦ̣ʰī, dūrá > dū, áŋgāra > aŋā
[Tor] ; ── block liquids-flap-rhotic: Flap-rhotic merger — ṛ ṛʰ > r rʰ (liquids blocks round, 2026-09-30). Stage 2 of the flap history: the retroflex flap (blocks/liquids-flap.sc, or the Dardic inte [@flap — *kʰōṭi > kʰōr (after final-r loss: not bled)] ──
[Tor] {ṛ ṛʰ} > {r rʰ}                              ; @flap — *dʰūḍi > Dang dʰur, kukkuṭá > K kɔkur, kʰaḍaka > K kʰorᵘ / (block liquids-flap-rhotic)
[Gaw Shum Dm] ∅ > y / # _ {e ē}                            ; @prothesis — 2 lects / glide before initial e (ḗka > yēk; as Shumashti) / (changes.sc prothesis.ee)
[Shum] ś > s / V _                                 ; @sibilant-s — non-initial ś > s
[Shum] ; ── block h-initial-loss-optional: Sporadic loss of initial h- before a vowel, end of chain (2026-10-01): Si haṁsá > as, hánati > annavā; Pas hánati > an; Bishnupriya hásta > at. Beat a same-shap [cross-family law (blocks/h-initial-loss-optional.sc)] ──
[Shum] optional: h > / # _ V                        ; @h-loss — sporadic initial h- loss (haṁsá > Si as) / (block h-initial-loss-optional)
[mand kvar] ; ══ Round 6 audit: hiatus leftovers after consonant loss ══
[Sunha Kathoriya Dotyali Bishnupriya KS Khandesi Rathawi hal PauriBareli mewari_bannoda Vasavi Noiri bagri_fatehabad Nimadi Marw had mewari_dholpura hadothi_kelwada mewati_jhambaus mewati_akera west-pahari -bhad -cur] ; ── block hiatus-medial: Hiatus theme — medial a(C)a contraction (blocks round, 2026-09-30). After the MIA loss of intervocalic stops/y, a-a before a consonant contracts to e and a-ā to [@hiatus — blocks/hiatus-medial.sc; short a absorbed before ā (kumbʰakāra > kumbʰaār > kumhār; round 4)] ──
[Sunha Kathoriya Dotyali Bishnupriya KS Khandesi Rathawi hal PauriBareli mewari_bannoda Vasavi Noiri bagri_fatehabad Nimadi Marw had mewari_dholpura hadothi_kelwada mewati_jhambaus mewati_akera west-pahari -bhad -cur] ; ── block hiatus-medial-core: Hiatus theme — the shared core of the medial a(C)a contraction (blocks round, 2026-09-30). a-a before a consonant > e, a-ā > ā. The Bihari-Tharu lects stop here ──
[Sunha Kathoriya Dotyali Bishnupriya KS Khandesi Rathawi hal PauriBareli mewari_bannoda Vasavi Noiri bagri_fatehabad Nimadi Marw had mewari_dholpura hadothi_kelwada mewati_jhambaus mewati_akera west-pahari -bhad -cur] a a > e / C _ C                              ; @hiatus — medial a(C)a > e (kadala > kaala > Bhoj kerā, kadalī > kayalī > kelā) / (block hiatus-medial-core)
[Sunha Kathoriya Dotyali Bishnupriya KS Khandesi Rathawi hal PauriBareli mewari_bannoda Vasavi Noiri bagri_fatehabad Nimadi Marw had mewari_dholpura hadothi_kelwada mewati_jhambaus mewati_akera west-pahari -bhad -cur] a ā > ā                                      ; @hiatus — a-ā hiatus > ā (kumbʰakāra > kumʰār, andʰakāra > anʰār) / (block hiatus-medial-core)
[Sunha Kathoriya Dotyali Bishnupriya KS Khandesi Rathawi hal PauriBareli mewari_bannoda Vasavi Noiri bagri_fatehabad Nimadi Marw had mewari_dholpura hadothi_kelwada mewati_jhambaus mewati_akera west-pahari -bhad -cur] repeat: {a ā} {a ā} > ā                      ; @hiatus — a/ā hiatus > ā (andʰakāra > andʰaāra > ā̃dʰār, kumbʰakāra > kumʰār, rājana > rāaṇ > rāṇ) / (block hiatus-medial)
[west-pahari -ktg -WPah] i u > u / _ #                                ; @hiatus — 5 lects / -iu > -u (round 6) / (changes.sc hiatus.iu.final)
[kiuth] a i > ī / _ #                               ; @hiatus — -ai > -ī (-akikā, -ati leftovers; round 6)
[mand] o i > o / _ C                                ; @hiatus — 2 lects / medial oi > o (round 6) / (changes.sc hiatus.oi.preconsonantal)
[jaun dog kiuth kc mand ktg WPah kul] V $1 > $1 / _ C                              ; @hiatus — 2 lects / identical medial vowels merge (dā́yādya > dāāʣ > dāj; round 6) / (changes.sc hiatus.v.preconsonantal)
[kul] ɔ o > o / _ C                               ; @hiatus — the rounded ɔ contracts with a following o as identical vowels do (*adʰaōṣṭʰa > ɔoṭʰ > oṭʰ, kapʰōṇi > kɔoṇ > koṇ; round 7)
[kul] {o a} ā > ā / _ #                            ; @hiatus — 2 lects / -oā > -ā (round 6) / (changes.sc hiatus.oaa.final)
[west-pahari -mand -kvar] a e > e / _ C                                ; @hiatus — 3 lects / medial ae > e (round 6) / (changes.sc hiatus.ae.preconsonantal)
[kiuth kgr cam] {a ā} > ∅ / # _ h V                          ; @aphaeresis — 3 lects / initial a-/ā- lost before h (< OIA voiced aspirate): ākʰēṭa > āheṛā > heṛā, *adʰaōṣṭʰa > hoṭʰ (round 6 second pass) / (changes.sc aphaeresis.aa.hv)
[bhal ktg khash] a i > e / _ C                                ; @hiatus — 3 lects / medial ai > e (bʰaginī > bʰaiṇ- > bʰε̃ṇ; round 6) / (changes.sc hiatus.ai.preconsonantal)
[RomBalk] kʰ > x / # _                                 ; @kh-x — initial kʰ > x (xar, xai̦o)
[RomBalk] {tʰ kʰ pʰ} > {t k p}                         ; @notation — the Rom. Balk. source marks aspiration only on cʰ (tan < stʰāna, pir < *pʰir-)
[RomBalk] V > $1+stress / V C* _ C* #                  ; @accent-placement — Romani oxytone stress: last syllable of a polysyllabic output (after apocope); monosyllabic stems are left unstressed because the attested stress falls on the unpredicted suffix (kʰer vs koró, pinró); Vedic worse than none (§9)
[as-ll] {kʰ cʰ ṭʰ tʰ pʰ} > {k c ṭ t p}               ; @deaspiration — 2 lects / voiceless aspirates too (Macalister writes no aspiration: kʰara > kʌri) / (changes.sc deaspiration.khchththph)
[as-ll] {ṭ ḍ} > {t d}                                ; @deretroflexion — Domari has no retroflex series
[as-ll] a > ʌ / V C* _                                ; @vowel-reduction — short a outside the first syllable is reduced to ʌ [ə] (kánʌr, xálʌ, śtirʌr, kúrtʌ); long ā (merged with a only later, by the notation rule) stays a
[as-ll] {ā ī ū} > {a i u}                        ; @notation — CDIAL (Macalister) Domari rarely marks length of ā ī ū; ē ō written long
[as-ll] V > $1+stress / _ C* V C* #                  ; @accent-placement — Domari (Macalister) stresses the penult of the output word (máni, tʌ́rnʌ, jaúʌr); monosyllables unmarked; Vedic accent carried is worse than none (§9)
[Ku] {ī ū ī̃ ū̃} > {i u ĩ ũ} // _ #                ; @length-loss — non-final ī ū shortened; final -ī -ū kept (dʰartī)
[Garh] {a ā} u > ɔ                                  ; @vowel-shift — secondary au > ɔ (after h/v loss: bʰɔr, nɔ̃, cɔk)
[Garh] {a ā} i > ɛ                                  ; @vowel-shift — secondary ai > ɛ (bɛṇ < bʰaginī, kaṛɛ)
[Garh] ṇ > n / {r ṛ ḷ ṇ} _                          ; @n-dissimilation — 5 lects / ṇ > n after another retroflex/r / (changes.sc n-dissimilation.rrln)
[MagahiNepal Chitwan DewasDoneDanuwar Mth Darai AdivasiOriya MH mewari_dholpura mewati_jhambaus Goj] ; ── block finals-apocope-a: NIA apocope of -a only (blocks theme "finals", 2026-09-30): final short -a is lost while final -i -u survive (Maithili, eastern Tharu, Bishnupriya, Desia; Balka [apocope of -a only (blocks/finals-apocope-a.sc) — replaces `use final-vowel-loss.a`: only -a; final -i -u kept (as Mth)] ──
[MagahiNepal Chitwan DewasDoneDanuwar Mth Darai AdivasiOriya MH mewari_dholpura mewati_jhambaus Goj] a > ∅ / _ #                                  ; @final-vowel-loss — 20 lects / final -a lost, -i -u kept (Mth. aŋkurá > ā̃kur, adyá > āj, áŋga > ā̃g) / (changes.sc final-vowel-loss.a)
[Morang] ; ── block finals-apocope-ai: NIA apocope of -a and -i (blocks theme "finals", 2026-09-30): final short -a and -i are lost while -u survives (Bengali, Bishnupriya's neighbours BH and Hajong, [apocope of -a -i (blocks/finals-apocope-ai.sc) — replaces `use final-vowel-loss.ai`: -a and -i lost; -u kept] ──
[Morang] {a i} > ∅ / _ #                              ; @final-vowel-loss — 8 lects / final -a -i lost, -u kept (B. akṣára > ākʰar, agní > āg, aŋgúli > āŋul) / (changes.sc final-vowel-loss.ai)
[bihari-tharu -Mth] ∅ > y / i _ {ā ā̃} #                         ; @ya-sruti — 11 lects / glide in the -iā hiatus (-iyaka > -iyā) / (changes.sc ya-sruti.iaa)
[Darai] ; ── block finals-syncope-bihari: Bihari/Tharu syncope before a final long-vowel syllable (blocks theme "finals", 2026-09-30): medial a between single consonants (stops, nasals, sonorants, s; no [syncope before a final long-vowel syllable (blocks/finals-syncope-bihari.sc) — replaces `use syncope.a.vstopnasalsons`: medial a lost before a final long-vowel syllable (-aka/-ikā contractions); not next to h (round 6)] ──
[Darai] a > ∅ / V {Stop Nasal Son s} _ {Stop Nasal Son s} {ā ī} # ; @syncope — 7 lects / medial a lost before a final -Cā/-Cī (Bi. ārātrika > ārtī, kaŋkaṇa+aka > kãknā, kartana+ikā > kaṭnī) / (changes.sc syncope.a.vstopnasalsons)
[MagahiNepal Mth] ; ── block length-antepenult-shortening: Bihari antepenultimate shortening: a long vowel (oral or nasal) two or more syllables before the end of the word shortens, counted after apocope, i.e. outside t [@vowel-shift @stress — antepenultimate long vowels shorten (after apocope) (blocks/length-antepenult-shortening.sc; blocks round 2026-09-30)] ──
[MagahiNepal Mth] {ā ī ū ā̃ ī̃ ū̃} > {a i u ã ĩ ũ} / _ C* V C* V C* # ; @vowel-shift @stress — antepenultimate long vowels shorten (Bhoj yantrasvāra > jãtasār, pāḍḍarūpa > paṛarū) / (block length-antepenult-shortening)
[Bhoj] a > i / ā _ l #                              ; @hiatus — āa > āi before the -l of the -al verbal noun (kʰā-alla > kʰāil, osāil, onāil; cf. Mth āab > āeb)
[bihari-tharu] {i ī} i > i                                  ; @hiatus — 7 lects / i-i hiatus (after stop loss) contracts (dīpa-iyaka > diiyā > diyā) / (changes.sc hiatus.iii)
[Shum Dm] {ū ē} > {u e}                                ; @length-loss — 2 lects / partial: ū ē shortened (as Shumashti) / (changes.sc length-loss.ue)
[Mai] ; ── block v-b-skip-optional: Sporadic escape of initial v- from v- > b- (2026-10-01): some words keep v- where the lect otherwise has b- (P viśvāsa > vasāh beside vāla > bāl). Mechanism: an [sporadic (blocks/v-b-skip-optional.sc)] ──
[Mai] optional: v > ʋ / # _                        ; @v-b — sporadic: initial v- escapes v- > b- (viśvāsa > P vasāh) / (block v-b-skip-optional)
[DewasDoneDanuwar] a {i ē} > l a / C _ #                         ; @morph @hiatus — verbs cited in the -la (l-past) form (máratē > marla, *dēkṣati > dekʰla)
[MagahiNepal] C i > i $1 / ā _ #                          ; @epenthesis — round 7: final -i anticipated into a preceding ā syllable (Magahi epenthesis: agní > āgi > āig, catvā́ri > cāri > cāir; Thakur also writes duvāir, kāilh); cʰāti keeps -i
[bihari-tharu N Dotyali eastern -Or -OB hindi -kaithal -Aw -bagheli_lakshman] ; ── block late-mergers: Late NIA mergers, order-free: initial v- > b-, ṇ > n, initial y- > j-. Shared by 20 lects in 5 clades (Bihari, E. Hindi, Early NIA, Eastern, Other); blocks roun [package (blocks/late-mergers.sc)] ──
[bihari-tharu N Dotyali eastern -Or -OB hindi -kaithal -Aw -bagheli_lakshman] ; ── block lar-v-b: Initial v- > b- (NIA): Hindi, Bihari, Eastern, Pahari, Rajasthani, Punjabi, Lahnda, Sindhi, Dardic, Romani … the most widespread NIA consonant change. Geminate  [@v-b — initial v- > b- (blocks/lar-v-b.sc)] ──
[bihari-tharu N Dotyali eastern -Or -OB hindi -kaithal -Aw -bagheli_lakshman] v > b / # _                                  ; @v-b — 88 lects / vadʰū́ > H. bahū, vā́la > H. bāl / (changes.sc v-b.v.initial)
[bihari-tharu N Dotyali eastern -Or -OB hindi -kaithal -Aw -bagheli_lakshman] ; ── block nasals-dental-merger: nasals-dental-merger — NIA ṇ > n everywhere, nasals theme, blocks round 2026-09-30. Stage 2 (Hindi belt, Bihari, Tharu, Nepali …): every ṇ — the MIA intervocali [@n-dental — NIA ṇ > n (blocks/nasals-dental-merger.sc)] ──
[bihari-tharu N Dotyali eastern -Or -OB hindi -kaithal -Aw -bagheli_lakshman] ṇ > n                                        ; @n-dental — 66 lects / pānī́ya > H, Bhoj pānī, guṇá > H gun, vána > Bhoj ban / (changes.sc n-dental.n)
[bihari-tharu N Dotyali eastern -Or -OB hindi -kaithal -Aw -bagheli_lakshman] y > j / # _                                  ; @y-j — 101 lects / (changes.sc y-j.initial)
[Dm Mai] ; ── block lar-v-b: Initial v- > b- (NIA): Hindi, Bihari, Eastern, Pahari, Rajasthani, Punjabi, Lahnda, Sindhi, Dardic, Romani … the most widespread NIA consonant change. Geminate  [@v-b — initial v- > b- (blocks/lar-v-b.sc)] ──
[Dm Mai] v > b / # _                                  ; @v-b — 88 lects / vadʰū́ > H. bahū, vā́la > H. bāl / (changes.sc v-b.v.initial)
[Tir] a a > a                                      ; @hiatus — 2 lects / (changes.sc hiatus.aa)
[Bshk Chil Dm Tir Pas] {ā ī ū ē ō} > {a i u e o} / _ C C* V         ; @x-dardic-south-length-loss — 3 lects / long vowels shortened outside the final syllable of the modern word (as Pashai); round 6 / (changes.sc south-length-loss)
[Gowro Bshk Wot Chil Mai] ; ── block finals-syncope-dardic: Dardic syncope in the penultimate syllable (blocks theme "finals", 2026-09-30): a short a i u between single consonants is lost when a closed or open final syll [penultimate syncope (blocks/finals-syncope-dardic.sc) — replaces `use syncope.aiu.vccvc`: short vowel of an open penult syncopated in words of 3+ syllables (late, after apocope; as Indus Kohistani); round 6] ──
[Gowro Bshk Wot Chil Mai] {a i u} > ∅ / V C _ C V C* #                 ; @syncope — 3 lects / penultimate short vowel lost (kambalá+akka > kamlǎh, *gaḍa-baḍa > gǎḍbaḍ) / (changes.sc syncope.aiu.vccvc)
[Bshk] {k ṭ t p} > {g ḍ d b} / {ŋ ṇ n m} _          ; @nasal-voicing — 7 lects / voiceless stops voiced after a nasal, as in Indus Kohistani, Gowro and Bhateri (káŋkata > kēŋg, *taŋka > taŋgṓr, *riŋk > riŋg; 6 words, none worse); round 7 / (changes.sc nasal-voicing.kttp.ngnnm)
[Dang Garh N Mai] ; ── block v-b-release: Release of the v-b-skip-optional placeholder: ʋ > v (2026-10-01). Include right after the line that brings v- > b-. [releases the v-b-skip-optional placeholder] ──
[Dang Garh N Mai] ʋ > v                                        ; @v-b — release (blocks/v-b-skip-optional.sc) / (block v-b-release)
[Dotyali] ∅ > y / V _ V                                ; @ya-sruti — hiatus glide written by the field source (āgata > āyo, dita > diyo); absent from Turner's N.
[Dotyali] ; ── block finals-syncope-iyo: Syncope of i before -yo (blocks theme "finals", 2026-09-30): the i of -iyo (< -ita-ka, -iyaka) is lost after a consonant (Rajasthani, Gujarati, Doteli). [-Ciyo > -Cyo (blocks/finals-syncope-iyo.sc) — replaces `i > ∅ / C _ y o #`: i before the glide of final -yo is lost after a consonant (-itaka > -iyo > -yo: māráyati > māryo, máratē > maryo, suptá > sutyo, *ut-stʰita > uṭʰyo); round 7] ──
[Dotyali] i > ∅ / C _ y o #                            ; @syncope — 4 lects / -Ciyo > -Cyo (had. māráyati+itaka > māryɔ, *dēkṣati+itaka > dekʰyɔ) / (changes.sc syncope.i.cyo)
[G] {ē ō} > {e o}                               ; @notation — no length mark on e o
[OG OMarw] ∅ > y / {ā e} _ a #                          ; @glide — 2 lects / y glide in final -āa/-ea (pāya, rāya; as OG) / (changes.sc glide.aea)
[Nimadi] %source varkey-vunnamatla2018bareli vunnamatla-john-samuvel2012nimadi
[Nimadi] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} Stop > $1-nas {ŋ ŋ ŋ ŋ ñ ñ ñ ñ ṇ ṇ ṇ ṇ n n n n m m m m}@2 $2   ; @notation — survey writes a nasal vowel before a stop as V + homorganic nasal (pāñc)
[DungraBhili Noiri] %source varghesekumar2015noira
[DungraBhili Noiri srk awan Nimadi] u > ū                                       ; @notation — and u mostly as ū (awan u→ū 23 vs u 15: *kuḍa > kūṛī, dugdʰá > dūd), like Lothers for Pothwari; round 7
[OP] %source shackle-auto
[OP] {ṛ ṇ} > {r n}                               ; @notation — the automatically extracted Shackle glossary mostly writes the flap as r and often ṇ as n (train, where the chain predicts ṛ: r 55 vs ṛ 25; ṇ: n 94 vs ṇ 88; the hand-checked Shackle list has r 12 vs ṛ 43, n 36 vs ṇ 77): kūṭaka > kūrā, pāṭáyati > pāri, añjana > añjanu; round 7
[Sh] %source degener-shina2008 buddruss-shina1996 backstrom1992 grierson-lsi1928
[Sh] o i k > ∅ / _ #                             ; @morph — these sources cite bare stems or finite forms
[Sv] %source knobloch2020sauji decker1992 liljegren-hindukush
[Sv] i l > i l o / _ #                           ; @morph — these sources cite the participle/infinitive -ilo
[Kalk] %source liljegren-hindukush
[Kalk] ū n > i l / _ #                             ; @morph — Liljegren's wordlist cites the -il form
[bundeli_atarra] {u ũ} > {ū ū̃}                              ; @notation — bundeli list writes u long by default (ū:372 vs u:75)
[bundeli_atarra] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} {k kʰ g gʰ} > $1-nas ŋ $2  ; @notation — nasal vowel + stop written as V + homorganic nasal (aŋgūṭī)
[bundeli_atarra] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} {c cʰ j jʰ} > $1-nas ñ $2  ; @notation
[bundeli_atarra] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} {ṭ ṭʰ ḍ ḍʰ} > $1-nas ṇ $2  ; @notation
[bundeli_atarra] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} {t tʰ d dʰ} > $1-nas n $2  ; @notation
[bundeli_atarra] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} {p pʰ b bʰ} > $1-nas m $2  ; @notation
[Sh Sv Kalk bundeli_atarra] %end
[Sh D] %source backstrom1992
[Mai] i > ī / # C* _ C #                          ; @vowel-shift @x-dardic-south-monosyllable-lengthening — i lengthened in a closed monosyllable of the modern word (after apocope and high-vowel shortening): íṣṭakā > ī̂ṭʰ, kiri > kī̂r, kīla > kī̂l, ŕ̩kṣa > ī̂ʦ̣ʰ; Zoller 31 long vs 12 short; u (41/41) and a (48/82) are not lengthened (with u: plain +0.0079); 53 better / 24 worse; round 7
[Mai] {a ā i ī u ū e ē o ō ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃} > {ǎ ā̌ ǐ ī̌ ǔ ū̌ ě ē̌ ǒ ō̌ ã̌ ā̃̌ ĩ̌ ī̃̌ ũ̌ ū̃̌ ẽ̌ ē̃̌ õ̌ ō̃̌} / _ {ṽ m} #   ; @tone — Zoller's rising contour on the accented final syllable before a final nasal (the -aṽ < -ana(ka) of infinitives and nouns, final -m): kālayati > kālǎṽ, gáḍati > gaṛǎṽ, ḍōmba > ḍū̌m, *dʰamm > dʰǎm; rising 79 % of 157 train -ṽ reflexes; round 7 (tone-gated)
[Mai] {a ā i ī u ū e ē o ō ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃} > {ǎ ā̌ ǐ ī̌ ǔ ū̌ ě ē̌ ǒ ō̌ ã̌ ā̃̌ ĩ̌ ī̃̌ ũ̌ ū̃̌ ẽ̌ ē̃̌ õ̌ ō̃̌} / {bʰ dʰ gʰ ḍʰ h b g} _ C* #   ; @tone @depressor — rising contour after a breathy or voiced labial/velar onset, which lowers the start of the tone: gʰāsá > gʰā̌, gʰr̩tá > gʰī̌, anāhāra > nihā̌r, káŋkata > kaŋgǎh, *gāva > gā̌, *ṭabbara > ṭabǎr (b 60 %, g 63 % rising in train); d ḍ z j and v stay falling-majority (all voiced obstruents: worse; +v z: test worse); an OIA-traced breathy onset (latent tier) instead of the modern one scores the same; round 7 (tone-gated)
[Mai] {a ā i ī u ū e ē o ō ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃} > {â ā̂ î ī̂ û ū̂ ê ē̂ ô ō̂ ã̂ ā̃̂ ĩ̂ ī̃̂ ũ̂ ū̃̂ ẽ̂ ē̃̂ õ̂ ō̃̂} / _ C* #   ; @tone — otherwise the accented final syllable carries Zoller's falling contour (57 % of noun tones): kaṇṭa > kâṇḍ, dʰā́sas > dʰā̂s, kṣīrá+akka > ʦ̣ʰirâh; the Vedic accent does not choose the contour (Vedic-accented final vowel → falling, alone or for long vowels, adds nothing: §Round 7); round 7 (tone-gated)
[Gowro bhatr] {a ā i ī u ū e ē o ō ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃} > {ǎ ā̌ ǐ ī̌ ǔ ū̌ ě ē̌ ǒ ō̌ ã̌ ā̃̌ ĩ̌ ī̃̌ ũ̌ ū̃̌ ẽ̌ ē̃̌ õ̌ ō̃̌} / _ C* # ; @tone — 2 lects / Zoller's Gowro realises the accented final syllable with a rising contour (74 % of noun tones, 94 % of verbs): adyá > ā̌z, ágra > agǎr, kárṇa+akka > kanǎh; no subgroup (Vedic accent, OIA length, aspirate, weight) has a falling majority large enough to pass (Vedic long vowel → falling: train +0.0003); round 7 (tone-gated) / (changes.sc tone.aaiiuueeoo.last-syllable)
[Tor] {i u} > {ī ū} / _ #                          ; @final-lengthening — 11 lects / word-final high vowels long / (changes.sc final-lengthening)
[Tir Pas] Stress > $1+acc                              ; @tone — 2 lects / the breathy-onset mark made visible (acute as a placeholder) before the latent tier is cleared for accent placement; round 7 / (changes.sc tone.stress)
[Tir Pas] V > $1-stress                               ; @tone — (as above); round 7
[Wot] ś > ∅ / V _ C                               ; @cluster-simplify — preconsonantal ś (from ṣ in ṣṭ ṣk, after the ṣ > ś merger) lost after a vowel: gōṣṭʰá > goṭ, dr̩ṣṭá > diṭ, śúṣka > śuk, piṣṭá > pīṭ; 6 better / 0 worse; ext −0.0041/−0.0066; round 7
[nuristani] ; ── Vowels ───────────────────────────────────────────────────────────────
[nuristani -Ash] {ī ū ī̃ ū̃} > {i u ĩ ũ}                      ; @length-loss @nur-innovation @vowel-shift — 31 lects / high vowels lose length: mū́la > *mul, nī́la > *nila / (changes.sc length-loss.iuiu)
[Ash Kt] {a ā} i > e / _ #                            ; @ai-au @hiatus — 2 lects / round 5: final *-ai (from -ati, -ayati, -ādi … after intervocalic loss) contracts to -e instead of losing the i: dāmayati > dāmé, *cakṣati > ṣaʦ̣é, *cʰidati > ʦie. Derives Strand's Kt verb citation -e from the 3sg present by sound change / (changes.sc ai-au.aai.final)
[Kt Kam] a ā > ā                                      ; @hiatus — 2 lects / round 7 (as PNur): a + ā after a lost consonant contracts to long ā, which then takes the last-syllable ā > o (*hayanavāra > ʣā̃vór, kṣapā́ > ṣóã); ext −0.0007/0, ext_acc −0.0009/0, 5 reflexes better, 0 worse / (changes.sc hiatus.aa.3)
[Kt Kam Pr] ; ── block hiatus-nuristani: Hiatus theme — Nuristani hiatus resolution (blocks round, 2026-09-30). After the loss of intervocalic stops, Nuristani resolves hiatus in favour of the second v [@hiatus — blocks/hiatus-nuristani.sc; contraction after intervocalic loss (the second vowel goes): ā́gata > *ā, *svakuliya > Kt suli] ──
[nuristani -Ash -Wg -Gmb] {a ā i} > ∅ / V _                            ; @hiatus — second vowel lost after a vowel (ā́gata > *āa > ā, *svakuliya > Kt suli) / (block hiatus-nuristani)
[Kt Kam Pr] a > ∅ / _ V                                  ; @hiatus @nur-innovation — a lost before a vowel (kakubʰá > Kt kuv) / (block hiatus-nuristani)
[Kt Kam Pr] ā > ∅ / V C* _ V                             ; @hiatus — non-initial ā yields to a following vowel (nāsāpuṭa > nāsúr̆, jā́mātr̩ > zami) / (block hiatus-nuristani)
[Wg] {a ā i u} > ∅ / V _                         ; @hiatus — contraction after intervocalic loss (the second vowel goes): ā́gata > *ā, *svakuliya > Kt suli
[nuristani] {a ā} > ∅ / V _                              ; @hiatus — 2 lects / contraction after intervocalic loss: ā́gata > *ā / (changes.sc hiatus.aa.postvocalic)
[nuristani] å > ∅ / V _                                  ; @hiatus — 2 lects / round 7: the marked vowel contracts like a / (changes.sc hiatus.a.postvocalic.2)
[PNur] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[PNur] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[nuristani -Wg] ; ── block r-vocalic-multi: r̩ with several outcomes (2026-10-01): beside the lect's own outcome, a, i or u (kr̩ta > kaa ~ kia ~ kua; MIA Ashokan āvaḍikā ~ ediśa, Dhp kidiśa ~ kicʰa). Repl [r̩ > (own) | a | i | u (blocks/r-vocalic-multi.sc)] ──
[nuristani -Wg] optional: {r̩ ŕ̩} > {a á} | {i í} | {u ú}    ; @r-vocalic — r̩ > (own) | a | i | u / (block r-vocalic-multi)
[Gmb] {a i} > ∅ / _ V                              ; @hiatus — 5 lects / (changes.sc hiatus.ai.prevocalic)
[nuristani] r̩ > i                                       ; @r-vocalic — 118 lects / śŕ̩ŋgāra > *ṣiŋara, hŕ̩daya > *ʣird- (PNur often keeps *r̩) / (changes.sc r-vocalic)
[nuristani] C V r > $1 r $2 / # _ C                      ; @metathesis @nur-innovation — 6 lects / r of a closed first syllable moves into the onset: kūrma > Kt kr̆um, *margā > Kt mr̆oŋ, dīrgʰá > Kt dragar̆ / (changes.sc metathesis.cvr.c)
[nuristani -Wg -Gmb] a > ā / # _ C                                ; @nur-innovation @vowel-shift — 3 lects / initial a- = PNur *ā-: antár > *ātar, áśru > *āʦ̣u / (changes.sc nur-innovation.a.c)
[Kt] ; ══ Kt-specific (Kamkata-vari) ═══════════════════════════════════════════
[Wg] ; ══ Wg-specific (Waigali / Kalasha-ala) ══════════════════════════════════
[Wg] ∅ > i / # _ ś t {a ā r}                     ; @prothesis — R6 fourth pass: prothetic i- before initial št + a/ā/r (stara > iśtor, staraṇa > iśtaræ̃, stāva > iśtāo, strī́ > iśtri); not before u (stū́pa > śtu; *stumbʰa > uśtum has u-)
[Pr] ; ══ Pr-specific (Prasun / Vasi-vari) ═════════════════════════════════════
[nuristani -Gmb -Kam -PNur] ʣ > z                                        ; @nur-innovation @x-nur-dz-z — 4 lects / PNur *ʣ > Kt z: hariṇá > zar̆-, jarā́ > zar̆ / (changes.sc nur-innovation.dz)
[Kam] y > i / # _                                 ; @notation — Strand writes *i- for initial y-: yávasa > *ios
[Kam] ; ══ Kam-specific (Kamviri) ═══════════════════════════════════════════════
[Kt Kam] ṛ > r̆                                       ; @flap @nur-innovation — 2 lects / PNur *ṛ > Kt ř: gala > gar̆ák / (changes.sc flap.r.2)
[Ash Kt Kam] r > r̆ / {k g p b m v} _                     ; @flap @nur-innovation — 3 lects / round 6: r after a labial or velar is the flap ř (kūrma > kr̆um, grāhá > gr̆óv, bʰrā́tr̩ > br̆ó, prádadāti > pr̆e, mriyátē > mr̆é); after dentals it stays r (putrá, dravá > dru): Strand and CDIAL alike / (changes.sc flap.r.kgpbmv)
[Kt Kam] r > r̆ / # _                                 ; @nur-innovation @x-nur-initial-r — 2 lects / initial r- > ř-: rā́trī > r̆ōtr̩ / (changes.sc nur-innovation.r.initial.2)
[Pr] ʦ > z / V _                                 ; @nur-innovation @int-voicing — *ʦ voiced after vowels: kā́śatē > koz, vrīhí > vuzi, -daśa > -līz
[nuristani -Gmb -PNur] r > ź / # _                                  ; @nur-innovation @x-nur-initial-r — 3 lects / initial r- > ź-: rṓka- > źɔ, rambʰati > źama / (changes.sc nur-innovation.r.initial)
[Ash] ; ══ Ash-specific (Ashkun) ════════════════════════════════════════════════
[nuristani -Gmb -Pr] V > $1+nas / V C* _ {n ṇ} #                  ; @nur-innovation — 2 lects / R6 fourth pass (as Ash): final nasal of a polysyllable (after the CDIAL/Buddruss apocope) > nasal vowel (káraṇa > karã, niṣádana > niṣã́, aśáni > aśē̃́) / (changes.sc nur-innovation.v.vcnn)
[nuristani -Gmb -Pr] {n ṇ} > ∅ / V C* V _ #                       ; @nur-innovation — 2 lects / R6 fourth pass: see the rule above / (changes.sc nur-innovation.nn.vcv)
[Gmb] ; ══ Gmb-specific (Tregami) ═══════════════════════════════════════════════
[Gmb] r > ʣ̣ / # _                                 ; @nur-innovation @x-nur-initial-r — initial r- > ʣ̣-: rakta > ʣ̣atǻ, rṓka- > ʣ̣ō, rā́trī > ʣ̣ātr
[nuristani -Ash -PNur] o > u                                        ; @nur-innovation @vowel-shift — 5 lects / PNur *o > u: paura > púr, rṓka > r̆ū- / (changes.sc nur-innovation.o)
[Kam] {u i} > ∅ / V C _ C V                        ; @syncope — round 6: Kam loses a short high vowel between single consonants after the first syllable (āmuñcati > āmcá, ārōgya > ārgú, mā́nuṣa > mānśa, *bahira > bārá); a-syncope stays rejected (§8.3)
[Ash Gmb Kam] ; ── block nasals-nuristani-nd: nasals-nuristani-nd — Nuristani loss of a voiced stop after a nasal (the counterpart of nasals-nuristani-nt, later in the chain), nasals theme, blocks round 202 [@nasal-cluster @cluster-simplify — voiced stop lost after nasal: daṇḍá > ḍún (blocks/nasals-nuristani-nd.sc)] ──
[Ash Gmb Kam] {d ḍ b g} > ∅ / Nasal2 _                     ; @cluster-simplify @nasal-cluster — 4 lects / voiced stop lost after a nasal (nd > n, mb > m, ŋg > ŋ) / (changes.sc nasal-cluster.ddbg.nasal2)
[Kt Kam] ā > a / V C* _ #                             ; @nur-innovation @vowel-shift — 2 lects / round 7 (as Kam): a word-final non-initial ā (left by apocope of -ka) is reduced before the last-syllable ā > o, so it stays a (utpātaka > upá; Kam *ut-stʰāti > uśtá). In Kt it rests on one etymon: ext −0.0003/0, ext_acc −0.0007/0 (utpātaka better, *āṇḍajāta > āzo worse) / (changes.sc nur-innovation.a.final-polysyllabic)
[Kt Kam] ā > o / C _ C* #                             ; @nur-innovation @vowel-shift — 2 lects / Kt/Kam: ā in the last syllable of the modern word > o (after *o > u): mā́sa > mós/mús, bʰāga > bó, śā́lā > pśól, aṭṭāla > āṭól; ā stays when a final vowel survives (kāṇá > kār̆á, dvā́ra > dārá). Accent-blind: a tonic-penult ā́ > o adds nothing (+0.0003) / (changes.sc nur-innovation.a.cc)
[Kt] i > a / # C* _ C V #                        ; @vowel-shift @nur-innovation — round 6: first-syllable i of a (post-apocope) disyllable that ends in a vowel is reduced to ə (written a): *nijyati > najé, *nīḍāti > nar̆e, lihati > lazé, *niccʰāya > naʦé, r̩ṣabʰá > aṣə. The weak partner of monosyllable a > u; the final syllable carries the stress (§9)
[Kt Kam] a > u / # C C* _ C* #                        ; @nur-innovation @vowel-shift — 2 lects / Kt/Kam: a of a (post-apocope) monosyllable > u, the short partner of ā > o (náva > nú, padá > pú, dánta > dút, páñca > púc, sáras > súr, yakṣá > iuṣ; −0.0077). Accent-blind: first-syllable a of OIA disyllables gives u in 44% of tonic vs 43% of pretonic syllables (Kt; Kam 42% vs 36%) / (changes.sc nur-innovation.a.ccc)
[Kt Kam] a > u / V C* C _ {s ʦ ʦ̣ ś ṣ k t ṭ z ʣ} #    ; @nur-innovation @vowel-shift — 2 lects / round 6: a of a final closed syllable of polysyllables > u before obstruents (*ārasa > ārús, vasantá > vasút, cáturdaśa > śtrúʦ, *kuḍakk- > kur̆uk); not before r/r̆/m (kakkʰaṭa > kakar̆, maraṇa > mará) / (changes.sc nur-innovation.a.vccststssskttz)
[Kt] ; ── block nasals-nuristani-nd: nasals-nuristani-nd — Nuristani loss of a voiced stop after a nasal (the counterpart of nasals-nuristani-nt, later in the chain), nasals theme, blocks round 202 [@nasal-cluster @cluster-simplify — voiced stop lost after nasal: daṇḍá > ḍun, skandʰá > kan (blocks/nasals-nuristani-nd.sc)] ──
[Kt] {d ḍ b g} > ∅ / Nasal2 _                     ; @cluster-simplify @nasal-cluster — 4 lects / voiced stop lost after a nasal (nd > n, mb > m, ŋg > ŋ) / (changes.sc nasal-cluster.ddbg.nasal2)
[Kt] u > ∅ / V C _ C V                           ; @syncope — R6 fourth pass: medial u syncopated between single consonants (āmuñcati > āmcé, aŋgúli > āŋǐur̆ via āŋl-, *utpusati > aps, mā́nuṣa > mānśa); as Kam
[Wg Gmb Pr] ʦ > ś / V _                                  ; @nur-innovation @x-nur-ts-sh — 2 lects / Wg/Gmb: post-vocalic *ʦ (< *ć) > ś: páñcadaśa > Wg pacīś, dáśa > Gmb daś / (changes.sc nur-innovation.ts.postvocalic)
[Pr] a > u // _ #                                ; @nur-innovation @vowel-shift — Pr a > u/ɨ (rounding/raising of *a). R6 fourth pass: not word-finally; a final a (monosyllables gatá > ga, nakʰá > na, yábʰati > iza; the -a of extended stems ā́sa > āsa, áp > ā́və) stays a/ə. −0.0147/−0.0155 ext, 121 better vs 27 worse
[Pr] t r > c                                     ; @nur-innovation @tr-affricate — Pr *tr > č: tráyaḣ > cī, triká > cigi, *tr̩pu > cüū
[Pr] r > ∅ / C _ | V _ C                         ; @cluster-simplify @nur-innovation — Pr loses r in clusters on either side (no metathesis): prāṇín > pīē̃, prámucyatē > pɨ̄ci, kárṣati > kṣə
[Pr] ā > a / C _ // _ #                          ; @nur-innovation @vowel-shift — Pr: non-initial ā shortened (Morgenstierne's Pr rarely marks length). R6 fourth pass: not word-finally; a final ā of monosyllables stays long (śā́kʰā > ʦāv, dʰrā́yati > lār, pāyáyati > pār, śrāva > ʦ̣ā)
[Pr] ʦ̣ > ṭ                                       ; @nur-innovation @x-nur-pr-ts-t — Pr deaffricates the retroflex affricate: mākṣiká > māṭek, śrṓṇi > ṭō̃
[Pr] e > i                                       ; @nur-innovation @vowel-shift — Pr raises *e to i: mēha > mī́zə, dēha > diz
[Ash] å > o / # C* _ C* #                         ; @vowel-shift @nur-innovation — round 6: Ash a before a lost nasal cluster > o (nasalised in the sources) in a post-apocope monosyllable: dánta > dó̃t, páñca > pó̃ʦ, gandʰá > gōn, māṁsá > mós, *kʰandati > kōn; in longer words it stays a (páñcadaśa > paʦūs, saṁdaṁśá > sandús). Cf. Kt/Kam monosyllable a > u
[Ash] å > a                                       ; @notation — round 6: unmark
[nuristani] ∅ > i / # _ s t                              ; @prothesis — 2 lects / R6 fourth pass: prothetic i- before initial st (stŕ̩ > istā́, strī́ > istrī́, stara > istārā́; cf. stʰū́ṇā > istũ) / (changes.sc prothesis.st)
[Ash] ʦ > s                                       ; @nur-innovation @x-nur-ts-s — Ash deaffricates *ʦ (< PIIr *ć): kā́śatē > kas, ūnaviṁśati > osã́
[nuristani -Gmb -Pr] t s > ʦ                                      ; @affrication @cluster-simplify — 13 lects / round 6: ts > ʦ; fourth pass: placed after post-vocalic *ʦ > ś, since the ts-affricate stays ʦ (útsa > ɨʦ, *pītsa > pɨʦ, vatsalá > vuʦalá) / (changes.sc affrication.ts)
[Pr] {u i} > ∅ / V C _ C V C* #                  ; @syncope — R6 fourth pass: Pr syncopates a short high vowel (incl. u < *a) in the penult between single consonants (āmuñcati > āmju, dʰamáni > lümī́, kr̩muká > kumlugi, prasupti > pust); passes the extension-aware accent gate
[Ash] c > ʦ                                       ; @nur-innovation @affrication — then *č > ʦ (chain shift; the reverse order costs 0.009 NED)
[nuristani -Gmb -PNur] ; ── Accent (round 3; latent tier only, plain NED unchanged) ──
[eur Zarg Sv Kalk Kho Kund dardic-south -Shum -Chil nuristani -Wg -Kam mand] V > $1+stress / _ C* #                      ; @accent-placement — accent (Gawri's tone) on the last syllable of the modern form (dʰaráṇī > dʌrín, *kʰōjja > kʰòjā́nt, kṣárati > cʰō̌r); round 7: under the extension-aware accent score this beats round 3's 'long final vowel, else first syllable' (ext_acc −0.0034/−0.0076; that choice was made on bare headwords, which lack the accented suffixes)
[Bshk] {a ā i ī u ū e ē o ō ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃} > {ǎ ā̌ ǐ ī̌ ǔ ū̌ ě ē̌ ǒ ō̌ ã̌ ā̃̌ ĩ̌ ī̃̌ ũ̌ ū̃̌ ẽ̌ ē̃̌ õ̌ ō̃̌} / {kʰ tʰ ṭʰ pʰ ʦʰ ʦ̣ʰ cʰ} _ C* #   ; @tone @depressor — Gawri's accented syllable is rising (not high) after a voiceless aspirate onset (Bashkarik keeps kʰ tʰ ṭʰ pʰ ʦʰ cʰ): pʰā́la > pʰā̌l, kʰā́dati > kʰā̌nt, *cʰindati > cʰǐn; after aspirates Gawri writes 38 % rising, 46 % low, 8 % high (n = 26); low instead of rising fails the test veto, and low tone after a lost voiced aspirate (bʰ > b) or after h is worse on train; round 7 (tone-gated)
[dardic-south -Gaw -Shum -Wot -Tir -Pas] ; ══ per-source notation blocks (applied after the chain, only to that source; not chain rules) ══
[Kho] {a i u} > ∅ / V C _ C V C                     ; @syncope — late medial syncope of a short vowel between single consonants before a closed (stressed) final syllable (*adʰikatara > aihro type) (round 6)
[Kho] ; ── Per-source notation ─────────────────────────────────────────────────
[Sv Kho] %source decker1992
[Kalk Gowro Bshk Chil Tor bhatr Mai] %source rensch-decker-hallberg1992
[Gowro Bshk bhatr Mai] {â ā̂ î ī̂ û ū̂ ê ē̂ ô ō̂ ã̂ ā̃̂ ĩ̂ ī̃̂ ũ̂ ū̃̂ ẽ̂ ē̃̂ õ̂ ō̃̂} > {a ā i ī u ū e ē o ō ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}  ; @notation — (as above)
[Pas] {á ā́ í ī́ ú ū́ é ḗ ó ṓ ã́ ā̃́ ĩ́ ī̃́ ṹ ū̃́ ẽ́ ē̃́ ṍ ō̃́} > {ǎ ā̌ ǐ ī̌ ǔ ū̌ ě ē̌ ǒ ō̌ ã̌ ā̃̌ ĩ̌ ī̃̌ ũ̌ ū̃̌ ẽ̌ ē̃̌ õ̌ ō̃̌} / _ C* #   ; @tone @depressor — Lehr's rising tone on the accented last syllable where an OIA aspirate or h stood before its vowel (Pashai loses them): gʰāsá > gā̌ns, hásta > ā̌st, hā́rdi > ā̌ṛ, dʰūmá > dū̌m, kʰáṭvā > kā̌ṭ; voiced aspirates and h only −0.0004 vs −0.0006 train; round 7 (tone-gated)
[Tir Pas] {á ā́ í ī́ ú ū́ é ḗ ó ṓ ã́ ā̃́ ĩ́ ī̃́ ṹ ū̃́ ẽ́ ē̃́ ṍ ō̃́} > {a ā i ī u ū e ē o ō ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃} ; @tone — 2 lects / the placeholder removed elsewhere; round 7 / (changes.sc tone.aaiiuueeoo)
[Wg] å > ũ / # C* _ C* #                         ; @vowel-shift @nur-innovation — round 7: Wg a before a (lost) nasal cluster > nasal ũ in a post-apocope monosyllable, the Wg counterpart of Ash å > o and Kt/Kam monosyllable a > u (dánta > dũt/dū̃t, páñca > pũc, pántʰā > pũt, kantʰā > kū̃t, *vañja > mũʦ); polysyllables keep a (antár > ater, páñcadaśa > pacīś, kāntā > kā̃ta). With the CDIAL long-nasal notation: ext −0.0012/−0.0009 with full apocope, −0.0014/0 without
[Wg] å > a                                       ; @notation — round 7: unmark
[nuristani -Ash -Gmb -Pr] V > $1+stress / # C* _ C* #                 ; @accent-placement — monosyllables
[nuristani -Ash -Gmb -Pr] V > $1+stress / # C* V C* _ C* #            ; @accent-placement — disyllables (after apocope): last syllable, including a surviving final -a/ə (kār̆á); the Vedic accent instead of this for vowel-final accented disyllables costs +0.0071
[Wg Kam] V > $1+stress / V !#* _ C* V C* #           ; @accent-placement — three or more syllables: penult (Strand Kam trisyllables: penult 66%, antepenult 31%, final 4%)
[nuristani -Gmb -PNur] ; ══ Per-source notation (not chain rules) ════════════════════════════════
[Dm Pas Wg Kt Pr] %source CDIAL
[Gowro Bshk bhatr Mai Pas] {ǎ ā̌ ǐ ī̌ ǔ ū̌ ě ē̌ ǒ ō̌ ã̌ ā̃̌ ĩ̌ ī̃̌ ũ̌ ū̃̌ ẽ̌ ē̃̌ õ̌ ō̃̌} > {a ā i ī u ū e ē o ō ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}                                   ; @notation — (as above)
[Sh Sv Kalk Phal Gowro Bshk Chil Tor bhatr Mai D] {a i u} > {ā ī ū}                           ; @notation — Backstrom's survey transcription marks (phonetic) length on most vowels (as in OIA-Sh.sc)
[Nimadi] ī > i                                       ; @notation — source writes i for ī (no i-length contrast)
[Kannauji-Central bundeli_atarra] %source kannauji
[Brj] ; Ø > n o                                     ; @morph — root verbs cited as -no infinitives   [pruned for Brj: never fires]
[DungraBhili Noiri] {ī ī̃} > {i ĩ}                               ; @notation — varghesekumar2015noira writes no long i
[DungraBhili Noiri] {t d tʰ dʰ} > {ṭ ḍ ṭʰ ḍʰ}                   ; @notation — varghesekumar2015noira writes dental stops with the underdot (dánta > ḍāṭ)
[Bhili] %source pattanaik-koul2003varli
[Bhili] {a ã} > {ɔ ɔ̃} / _ C                         ; @notation — round 7: this source records Varli, where short a is ɔ before a consonant (gʰara > gɔr, náva > nɔv, dáśa > dɔs); Varli ā stays a
[bagheli_lakshman Brj M] %source bundeli
[bagheli_lakshman Brj] tʰ > t ʰ                                    ; @notation — bundeli list writes dental tʰ as t + detached ʰ (30 of 34)
[kaithal] %source webster2024haryanvi
[Kannauji-Central bundeli_atarra kaithal bagheli_lakshman Brj] {ṛ ṛʰ} > {ḍ ḍʰ}                             ; @notation — kannauji list does not distinguish ṛ from ḍ
[Kannauji-Central bundeli_atarra] # (2026-10-01: redundant after the upstream digraph-aspirate fix; the source now writes Cʰ) {kʰ cʰ ṭʰ tʰ pʰ} > $1-asp h                  ; @notation — kannauji list writes voiceless aspirates as digraphs (kh, ch, ph)
[Kannauji-Central bundeli_atarra bagheli_lakshman Brj] {u ũ} > {ū ū̃}                              ; @notation — kannauji list writes u long by default
[NDu] ; Ø > ∅                                       ; @morph — root verbs cited as bare stems   [pruned for NDu: never fires]
[SSr] ; Ø > ∅                                       ; @morph — root verbs cited as bare stems   [pruned for SSr: never fires]
[NDu SSr] %source chattisgarhi
[NDu SSr] {u i ũ ĩ} > {ū ī ū̃ ī̃}                     ; @notation — chattisgarhi list writes high vowels long throughout (i:12 vs ī:256)
[hindi -kaithal -OAw -H -Aw] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} {k kʰ g gʰ} > $1-nas ŋ $2  ; @notation — nasal vowel + stop written as V + homorganic nasal (aŋgūṭī)
[hindi -kaithal -OAw -H -Aw] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} {c cʰ j jʰ} > $1-nas ñ $2  ; @notation
[hindi -kaithal -OAw -H -Aw] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} {ṭ ṭʰ ḍ ḍʰ} > $1-nas ṇ $2  ; @notation
[hindi -kaithal -OAw -H -Aw] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} {t tʰ d dʰ} > $1-nas n $2  ; @notation
[hindi -kaithal -OAw -H -Aw] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ õ} {p pʰ b bʰ} > $1-nas m $2  ; @notation
[SSr] {e ẽ} > {ɛ ɛ̃} / _ C                        ; @notation — round 7: the Surgujia part of the list writes /e/ as ɛ before a consonant (kɛrā, cɛrī, dɛh, pɛṭ, lɛṭgā, bɛrʰ; e only in ek, dekʰ-, yeh-); train ext −0.0103, test ±0. Not in NDu (same source key, the Chhattisgarhi part writes e ~ ɛ evenly: worse)
[kaithal] {ū ī ū̃ ī̃} > {u i ũ ĩ}                     ; @notation — Webster writes high vowels short (u:86 vs ū:10)
[M] {ʦ ʣ ʣʰ ś} > {c j jʰ s}                    ; @notation — the bundeli wordlist does not write ʦ ʣ ś
[M] ∅ > a / C _ #                               ; @notation — the bundeli wordlist writes a final -a after consonants
[Ko-Ul] %source IndoLanguages
[Ko-Ul] ẽ > ɛ̃                                       ; @notation — round 7: the IndoLanguages list writes nasal e as ɛ̃ (neuter kaḷśɛ̃, āḷśɛ̃; śɛ̃ḷɛ̃)
[Bhatri hal] %source beine2017bhatri
[hal] {a ã} > {ə ə̃}                               ; @notation — round 7: Beine writes short a as ə (Woods as a)
[Bhatri hal] {ā ī ū ā̃ ī̃ ū̃} > {a i u ã ĩ ũ}              ; @notation — Beine writes long ā as a (short a is ə/ʌ: round 7 rule above) and no i/u length
[OM] ; (no ʦ/ʣ: Devanagari च ज do not write the Marathi affrication)
[OM] ẽ > ē / ṇ _ #                               ; @notation — final -ẽ after ṇ written -ē (the -aṇē infinitive: ṇē 560 vs ṇēṁ 108 in Tulpule; round 6: formerly a morph-tagged, ṇ-only rule, so it was off for verbs) …
[OM] ẽ > ē ṁ / _ #                               ; @notation — … elsewhere Tulpule writes the nasal vowel with anusvara, -ēṁ (एं: the neuter -akam > āgaḷēṁ, āŋgaḍēṁ; Cēṁ 195 vs Cē 42): gate ext −0.0013/−0.0012, net +89 (hill-climb round 3)
[OM] i > ī / _ #                                 ; @notation — final i written long (Marathi orthography: -ī)
[OM] ∅ > a / C _ # // ṁ _                        ; @notation — inherent -a written word-finally (not after a final anusvara: -ēṁ, not -ēṁa; hill-climb round 5)
[OM] ∅ > a / C _ ṇ ē #                           ; @notation — inherent a before the infinitive -ṇē (lātaṇē)
[OM] %source tulpule1999
[OM] # (2026-10-01: redundant after the upstream digraph-aspirate fix; the source now writes Cʰ) {kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ ʦʰ ʣʰ} > {k g c j ṭ ḍ t d p b ʦ ʣ} h   ; @notation — Tulpule writes digraph aspirates (CDIAL's OM. forms keep ʰ)
[bhal] ; ══ Round 6 audit ══
[west-pahari -ktg -kvar -WPah] a o > o / _ #                                ; @hiatus — 4 lects / -ao > -o (nikaṭam+aka > niṛao > neḍo; round 6) / (changes.sc hiatus.ao.final)
[west-pahari -bhad -khash -kvar] a ū > ū / _ #                                ; @hiatus — 4 lects / -aū > -ū (round 6) / (changes.sc hiatus.au.final)
[jaun dog kiuth mand ktg WPah kul] i o > o / _ #                                ; @hiatus — 2 lects / -io > -o (-iyaka: kāla+iyaka > kalio > kalo; round 6) / (changes.sc hiatus.io.final)
[bhal bhad kgr cur khash cam pan] C > ∅ / _ $1 $1                              ; @cluster-simplify — 3 lects / a triple consonant (C + geminate, e.g. k + kkʰ from kṣ) loses its first member (tatkṣaṇa > takkʰaṇ; round 6) / (changes.sc cluster-simplify.c.11)
[jaun kiuth bhad WPah pan] ; ══ Round 7 ══
[S P poth awan west-pahari -dog -khash -kvar] C > ∅ / Long _ $1                            ; @degem-plain — 5 lects / a geminate after a long vowel is simplified (a heavy syllable keeps at most two morae), as in bhad (kārttiká > kāti, cʰattvará > cʰā-, pārśvá > pāś-; 4 better, 0 worse; test ext −.0027; round 7) / (changes.sc degem-plain.c.long1)
[kul] ; ══ Round 6 audit ══
[kvar] ; ── block hiatus-diphthong: Hiatus theme — a + i/u before a consonant (blocks round, 2026-09-30). The secondary diphthongs left by consonant loss monophthongise before a consonant: a+i > e [@hiatus — blocks/hiatus-diphthong.sc; medial au > o (round 6)] ──
[kvar] a {i ī} > e / _ C                            ; @hiatus — a+i > e (bʰaginī > bʰaiṇ- > bʰeṇ, samākr̩ṣṭa > saiṭṭʰ- > seṭṭʰ-) / (block hiatus-diphthong)
[kvar] a {u ū} > o / _ C                            ; @hiatus — a+u > o (hastakūṭa > hatʰaüḍa > hathoḍo) / (block hiatus-diphthong)
[jaun dog kul] ; ── per-source notation (applied after the chain only to that source's pairs) ──
[kul] %source blair2021kullu
[kul] a > ə                                       ; @notation — [blair] short a written ə, its Kullui value (kadala > ˈkeḷə, adyá > əj; round 7)
[kul] {ʦ ʦʰ ʣ ʣʰ} > {c cʰ j jʰ}                   ; @notation — [blair] palatal letters for the affricates
[jaun] %source john2008jaunsari
[jaun kul] {ā ī ū ē ō} > {a i u e o}                   ; @notation — [blair] length rarely marked
[kul] ɔ > ə                                       ; @notation — [blair] Kullui ɔ written ə (adyá > əj, aṣṭā́ > ətʰ; kullui-org writes ɔ; round 7: ə scores better than a on train)
[kul] ∅ > ə / C _ #                               ; @notation — [blair] final reduced vowel transcribed, as ə (āmrá > āmbə, kadala > ˈkeḷə; round 7: was a)
[jaun] # (2026-10-01: redundant after the upstream digraph-aspirate fix; the source now writes Cʰ) {kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ} > {k g c j ṭ ḍ t d p b} h   ; @notation — [john] digraph aspirates
[jaun] {ā̃ ī̃ ū̃ ē̃ ō̃ ã ĩ ũ ẽ õ} > {a i u e o a i u e o} ; @notation — [john] nasalisation mostly not written
[jaun] ś > ç                                       ; @notation — [john] ç for ś
[jaun] ṛ > ḍ                                       ; @notation — [john] ḍ for the flap ṛ (lakaḍi, kukaḍi, kapaḍa; round 6 second pass)
[dog] %source brightbill-turner2007dogri
[Ash] %source CDIAL buddruss-wama2006
[Ash] i e > ∅ / _ #                               ; @morph @hiatus @notation — CDIAL/Buddruss cite Ashkun verbs as bare stems, not in Strand's -ie
[Ash Kt dog] ā > a                                       ; @notation — [brightbill-turner] ā written without macron (panī, sona, kela)
[bihari-tharu -Bhoj -Mth -Darai -Bi] {ī ū ī̃ ū̃} > {i u ĩ ũ}                      ; @length-loss @nur-innovation @vowel-shift — 31 lects / no i/ī, u/ū contrast (boehm writes only short high vowels) / (changes.sc length-loss.iuiu)
[Bishnupriya] {ṭ ṭʰ} > {t tʰ}                               ; @deretroflexion — retroflex stops lost (*pēṭṭa > pɛt; Tibeto-Burman / Meitei contact)
[bihari-tharu Ku Garh N Dotyali dardic-north -Sh -pog -Ush -Kund -dod Gowro Bshk Chil Tor bhatr Mai eastern hindi -Aw marathi-bhil -OM -VB northwest nuristani rajasthani-gujarati west-pahari -kc -ktg -cur -kvar -pan] {ē ō ē̃ ō̃} > {e o ẽ õ}                     ; @notation — the survey sources write e o without macron
[Darai] {a ā} > {ə a}                                ; @vowel-shift — length > quality, as in Nepali: short a is [ə], long ā is [a] (catvā́ri > ʦar, ásti > əsi, kr̩tá > kəri-)
[bihari-tharu -Bhoj -Morang -Mth -Bi -Bote -KochilaTharu] ; ══ per-source transcription (not chain rules) ═══════════════════════════════
[Dang Kathoriya] %source mitchell-eichentopf2013tharu
[Dang] {t tʰ d dʰ} > {ṭ ṭʰ ḍ ḍʰ}                   ; @notation — mitchell's Dang lists write dental stops with retroflex letters (18 dental vs 103 retroflex)
[KS] a > ∅ / V C _ C ā #                          ; @syncope — 2 lects / medial a lost before a heavy final -ā (< -aka): markaṭa(ka) > mākrā, *gōṭṭa > goṭlā; late, Bengali-type / (changes.sc syncope.a.vcca)
[KS] %source toulmin
[Bishnupriya] i > ī                                       ; @notation — the kim-kim survey writes /i/ as ī
[Bishnupriya] %source kim-kim2008bishnupriya
[Bishnupriya B KS] {e ẽ} > {ɛ ɛ̃}                                ; @notation — toulmin's KS writes the mid front vowel ɛ (tailá > tɛl, kṣḗtra > kʰɛt, vḗlā > bɛlā)
[Dang Kathoriya B] {ā ā̃} > {a ã}                               ; @notation — mitchell marks a/ā length only sporadically (gāũ ~ gaũ for the same word)
[Kho Kal] {a i u e o} > {ā ī ū ē ō}                   ; @notation — Decker's survey transcription writes (phonetic) length on nearly every vowel
[Wg Kt Pr] {i u} > {ī ū} / _ #                         ; @notation — Morgenstierne/Turner mark final high vowels long
[Kt] a > ə / V C* _                              ; @notation — round 7: Morgenstierne's Kati (via Turner) writes the a of every non-first syllable ə, the -aka extension's final vowel included (andʰakāra > andərə, ā́sa > asə, kʰarvá > kavə, hári > zər̆ə); first-syllable a stays a (60 a vs 13 ə). ext −0.0064/−0.0017, verb-menu −0.0039/−0.0153
[Wg] {e o} > {ē ō}                               ; @notation — Morgenstierne/Turner write the long mid vowels ē ō
[Wg] {ũ ẽ õ ĩ} > {ū̃ ē̃ ō̃ ī̃}                       ; @notation — round 7: Morgenstierne/Turner write Waigali nasal vowels long (ū̃ 32 vs ũ 8, ē̃ 19 vs ẽ 5), Strand short (ũ 28, ẽ 24): dánta > CDIAL dū̃t, Strand dũt
[Dang Kathoriya Sh Sv Kalk Kho Phal hindi -OAw -H -Aw Bhatri DungraBhili hal M OM Noiri Bhili Ko-Ul OP D P srk awan nuristani -Gmb -Kam -PNur Nimadi jaun dog kul] %end
[OAw] {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ẽ õ ō̃} > $1-nas ṁ              ; @notation — OAw writes nasal vowels as V + anusvāra (kāṁvarū)
[mand] ŋ > n / # _                                 ; @cluster-simplify — word-initial ŋ (left by aphaeresis: áŋgāra > ŋgār) > n (ngār; round 6 fourth pass)
[Dotyali] {ī ū ī̃ ū̃} > {i u ĩ ũ}                      ; @length-loss @nur-innovation @vowel-shift — 31 lects / i/ī, u/ū merged, also the ī ū from compensatory lengthening (kʰura > kʰur, tikta > titta > tit) / (changes.sc length-loss.iuiu)
[Dang] %source boehm
[Dang] e > y ā / C _                               ; @notation — boehm writes Dangaura /e/ after a consonant as yā ([ʲæ]: dyākʰ, pyāṭ, byā̃t); webster/mitchell write e (æ, ʸā rarely)
[Gowro Bshk Chil Tor bhatr Mai] ʦ̣ʰ > ʦ̣                                     ; @notation — RDH 1992 never writes aspiration on the retroflex affricate
[Kho Phal nuristani -Gmb -PNur -Pr] %source strand
[Sv] %source knobloch2020sauji
[Sv] i l o > ā n / _ #                           ; @morph — Knobloch cites most verbs with the -ān- stem
[Kalk] %source kalkoti hultman2023kalkoti
[Kalk] a > æ / C _ C                               ; @notation — the Kalkoti dictionary (Hultman et al.) writes æ for a between consonants, which the survey lists write a
[Ush] {i u} > {ī ū}                               ; @notation — the Rensch/Decker survey writes high vowels long
[Dang Sv Kalk Ush dardic-south -Gaw -Shum -Wot -Dm -Tir] %end
[Buksa Chitwan Rana Dang Sunha Kathoriya] %source webster
[Buksa Rana Dang] a {i u} > {æ ɔ}@2 / _ C                     ; @notation — webster writes the preconsonantal diphthongs ai au as æ ɔ where boehm writes ai au (bʰæs ~ bʰãis < mahiṣá, bæṭʰ < úpaviśati, bʰæyā < bʰrā́tr̩)
[Buksa Rana Dang] {ã a} {ĩ i} > æ̃ / _ C                       ; @notation — likewise nasal (bʰæ̃s-)
[Buksa Chitwan Rana Dang Sunha Kathoriya] {i u ĩ ũ} > {ī ū ī̃ ū̃}                     ; @notation — webster writes high vowels long throughout
[Rana] ā > o / V C* !{y} _ #                         ; @morph — webster's Rana has the masc. -o ending where boehm has -ā/-u (seto, kāro, lambo); a source-dialect ending, not notation
[DewasDoneDanuwar KochariyaEastDanuwar] %source regmi-thakur2016danuwar
[DewasDoneDanuwar KochariyaEastDanuwar] {kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ} > {k g c j ṭ ḍ t d p b} h   ; @notation — regmi-thakur write aspirates as digraphs (ghar, āt̺h)
[B] s > ś                                          ; @notation — writes the Bengali sibilant ś (śona, maś)
[DewasDoneDanuwar KochariyaEastDanuwar B] {ṭ ḍ ṛ} > {t d r}                           ; @notation — no retroflex letters (the apical diacritic t̺ d̺ marks dentals and retroflexes alike; diacritic dropped in scoring)
[MagahiNepal] %source thakur-thakur2016magahi
[Majhi] %source chalise2014majhi
[Majhi] c > t s                                     ; @notation — chalise writes the palatal affricates as ts dz (tsār, pā̃ts, dziu)
[Majhi] j > d z                                     ; @notation
[MagahiNepal Majhi] {ṭ ṭʰ ḍ ḍʰ} > {t tʰ d dʰ}                   ; @notation — thakur-thakur2016magahi writes no retroflex stops (betā, ātʰ)
[Darai] %source dhakal2011darai
[Darai] {ī ū ā̃ ī̃ ū̃} > {i u ã ĩ ũ}                 ; @notation — dhakal2011darai writes no length for i u or nasal vowels (a vs ə is quality: see the @vowel-shift rule)
[Darai] {c cʰ j jʰ} > {ʦ ʦʰ ʣ ʣʰ}                   ; @notation — dhakal2011darai writes palatals as ʦ ʣ (ʦar < catvā́ri); probably also phonetic (Nepali-type dental affricates)
[Bshk] %source gawri
[Bshk] a > æ                                       ; @notation — the Gawri dictionary writes (Kalam) Gawri /a/ as æ in most words; CDIAL and RDH write a
[Mai] %source CDIAL rensch-decker-hallberg1992 liljegren-hindukush
[Sh Sv Kalk Ush] %source liljegren-hindukush
[Sv] a > ā                                       ; @notation — Liljegren's wordlist writes a long
[Sh Sv Kalk Ush] {c j} > {ʦ̣ ʣ̣}                               ; @notation — Liljegren's wordlist writes the palatal affricates as retroflex ʦ̣ ʣ̣ (in every Shinaic lect)
[Kho] i k > ∅ / _ #                               ; @morph — Strand cites verbs as bare stems/imperatives
[Wg] u > ɨ                                       ; @notation — Strand writes Waigali's centralised high vowel ɨ where Morgenstierne/Turner write u (kúla > kɨl, duḣkʰá > dɨk)
[nuristani -Ash -Kam -Pr] y > i / # _                                 ; @notation — Strand writes initial y- as i/ǐ (Kt iuṣ, ǐuṣ < yakṣá)
[Kam] ṣ > s / _ ṭ                                 ; @notation — Strand's Kam. forms write the retroflex cluster sṭ, never ṣṭ (aṣṭā́ > usṭ, aṣṭā́daśa > āsṭiʦ, ṓṣṭʰa > isṭ; sṭ 23 vs ṣṭ 0; hill-climb round 5)
[Ash Kam] a > ā / # (C) (C) _                         ; @notation @x-nur-a-quality — Strand writes the full vowel [a] of the first syllable â (house ā) and the reduced later a as a (ə): kakkʰaṭa > kākár̆, amítra > āmátra
[Ash] y > i / # _                                 ; @notation — Strand writes initial y- as i
[dardic-north -pog -Kund -K -dod nuristani -Gmb -PNur -Pr] %end
[Wg] %source CDIAL buddruss-waigali1992
[Wg] a > ∅ / C _ #                               ; @notation — Morgenstierne/Turner and Buddruss cite Waigali without the final ə that Strand writes (−0.0027 vs. dropping -a in the chain)
[Sh Kal Ash] %source CDIAL
[Sh] o i k > o i k i̯ / _ #                       ; @morph — CDIAL's Kohistani infinitive is written -oiki̯
[Kho] %source strand decker1992
[Phal] %source CDIAL strand decker1992 liljegren-hindukush kalkoti zoller2023
[Phal] e > ∅ / C _ #                               ; @morph — CDIAL, Strand and Decker cite Palula verbs as bare stems
[Sh Kho Phal] %end
[Sh] %source degener-shina2008 buddruss-shina1996
[Phal] %source liljegren
[Phal] Vedic > $1+acc                               ; @tone — the flagged vowel (stress before the Vedic accent) …
[Phal] {ā́ ī́ ū́ ḗ ṓ á í ú é ó} > {ā̌ ī̌ ū̌ ē̌ ō̌ ǎ ǐ ǔ ě ǒ}   ; @notation @tone — … has Liljegren's rising tone: kālá > kā̌l, adyá > ā̌j, kṣurá > ʦ̣ʰū̌r (rising on 23% of such vowels vs 7% on the Vedic syllable itself; ext_tone −0.0002/−0.0002) (round 7, tone-gated)
[Sh Phal] Stress > $1+acc                              ; @notation @tone — the stressed vowel written (after every segmental rule, so NED is untouched) …
[Sh Phal] Vedic > $1-acc                               ; @notation @tone — … while an unstressed Vedic vowel stays unwritten
[Sh] {ā́ ī́ ū́ ḗ ṓ} > {ā̀ ī̀ ū̀ ḕ ṑ}                  ; @notation @tone — Degener and Buddruss write the falling tone of a stressed long vowel (Schmidt's circumflex) with a grave: áŋgāra > agā̀r, *akṣēma > aʦ̣ʰḕmo, tārā > tā̀ro (grave 65% of stressed long σ1 vowels; 85% when the Vedic accent stood on that syllable, 44% when it stood on the next) (round 7, tone-gated)
[Kal] {i e} k > {i e} m / _ #                     ; @morph — CDIAL cites Kalasha verbs in the 1sg -im/-em (Morgenstierne), Trail & Cooper in the infinitive -ik. Off for verbs in the menu scores: there -im/-em comes from 1sg -ami by epenthesis
[Kho Pas] %source strand
[Mai Pas] {ǎ ā̌ ǐ ī̌ ǔ ū̌ ě ē̌ ǒ ō̌ ã̌ ā̃̌ ĩ̌ ī̃̌ ũ̌ ū̃̌ ẽ̌ ē̃̌ õ̌ ō̃̌} > {a ā i ī u ū e ē o ō ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}                                   ; @notation — (as above)
[Mai] a ṽ > ∅ / _ #                               ; @morph — CDIAL/RDH cite verb stems, not Zoller's -ǎṽ infinitive
[Dm] ā > ɔ                                       ; @notation — Morgenstierne (CDIAL) writes Dameli /ā/ as ɔ (źɔ < adyá, aṣṭɔś < aṣṭā́daśa, ɔru < *āḍu); Perder and Decker write ā; round 7: ɔ, no longer folded to o
[Phal] {ā́ ī́ ū́ ḗ ṓ} > {ā̂ ī̂ ū̂ ē̂ ō̂}                  ; @notation @tone — falling is Liljegren's default tone …
[Phal] {á í ú é ó} > {â î û ê ô}                  ; @notation @tone — … also where the chain leaves the tonic vowel short (aṣṭā́ > ā̂ṣṭ, udaká > vī̂, *ut-stʰāti > utʰī̂: stressed short → F 39, H 22, R 17 pairs). The Vedic accent shifts the odds (R 7% on the Vedic syllable, 23% one syllable before it, 60% two before) but never makes rising the majority (round 7, tone-gated)
[bihari-tharu -Bhoj -Morang -Mth -Bi -Bote -KochilaTharu Sh Kho Phal Kal Bshk Dm Mai Pas Bishnupriya B KS] %end
[Hajong] ś > s                                        ; @sibilant-s — 5 lects / ś > s (ṣ kept apart) / (changes.sc sibilant-s.s)
[B] %source kim-kim-sangma-ahmad2011tripura
[B] # (2026-10-01: redundant after the upstream digraph-aspirate fix; the source now writes Cʰ) {kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ bʰ} > {k g c j ṭ ḍ t d b} h      ; @notation — digraph aspirates (pathoɾ, tʃh)
[B] pʰ > ɸ                                         ; @notation — ɸ for pʰ (ɸɔl)
[B] ; ── block lar-a-rounding-bengali: Bengali/Kamta vowel rounding: the shared Eastern a-rounding (lar-a-rounding-east: o before a high vowel, first-syllable ɔ), then the remaining non-initial a > o [@a-rounding — o before a high vowel, first-syllable ɔ, other a > o (blocks/lar-a-rounding-bengali.sc)] ──
[B] ; ── block lar-a-rounding-east: Bengali-type rounding of short a (Bengali, Kamta/Rajbanshi, Bishnupriya): a is raised to o before a high vowel in the next syllable (vowel harmony), and the rem [@a-rounding — hariṇá > horin, pʰála > pʰɔl] ──
[B] a > o / # C* _ C* {i u ī ū}                  ; @a-rounding — 2 lects / before a high vowel in the next syllable: *raśuna > B. rosun, hariṇá > horin / (changes.sc a-rounding.cciuiu)
[B] a > ɔ / # C* _                               ; @a-rounding — 2 lects / first-syllable a > ɔ: pʰála > pʰɔl, kadala > kɔla, gʰara > gʰɔr / (changes.sc a-rounding.first-syllable)
[B] a > o                                        ; @a-rounding — non-initial a > o (vatsará > bɔchor, kambalá > kɔmbol) / (block lar-a-rounding-bengali)
[B] {ā ā̃} > {a ã}                                  ; @notation — no vowel length written
[Dhp] ṁ > ∅ / _ C                                 ; @notation — CDIAL (Brough's Dhp.) does not write the anusvāra (páñca > paja, píṇḍa > piṇa)
[Dhp] ; ── block nasals-retroflexion: nasals-retroflexion — MIA intervocalic n > ṇ (the Prakrit "no ṇaḥ" change), nasals theme, blocks round 2026-09-30. Stage 1 of n > ṇ > n: a single dental nasal b [@n-retroflex — MIA intervocalic n > ṇ (blocks/nasals-retroflexion.sc)] ──
[Dhp] n > ṇ / V _ V                                ; @n-retroflex @nur-innovation @x-nur-n-flap — 40 lects / pānī́ya > G, P, S pāṇī, vána > P vaṇ, S vaṇu, Or baṇa / (changes.sc n-retroflex.intervocalic)
[bhatr Mai] %source zoller2005
[Mai] ; ∅ > ᵃ / {ṇ n} {ḍ d} _ #                     ; @notation — Zoller writes a superscript release vowel after final nasal + voiced stop (kaṇṭa > kâṇḍᵃ)   [hill-climb round 2: removed — sca.gold_tokens strips superscript ᵃ from every attested form, so the predicted ᵃ always cost one edit (22 train reflexes: kaṇḍᵃ vs kaṇḍ)]
[bhatr] ǎ > æ̌                                       ; @notation — (the same, tone-marked)
[Dm] %source perder2013dameli decker1992 liljegren-hindukush
[Dm] r̆ > r                                       ; @notation — Morgenstierne's flap r̆ is written r by the later sources
[Sh] %source backstrom1992
[Kalk] %source hultman2023kalkoti
[Kund] %source kund
[Sh Kalk Kund] Stress > $1+acc                              ; @notation @tone — the stressed vowel written (after every segmental rule, so NED is untouched) …
[Sh Kalk Kund] Vedic > $1-acc                               ; @notation @tone — … while an unstressed Vedic vowel stays unwritten
[Sh Kalk Kund] {ā́ ī́ ū́ ḗ ṓ} > {ā̌ ī̌ ū̌ ē̌ ō̌}                  ; @notation @tone — Backstrom writes the accent of every long vowel with a caron (*uddʰūḍi > udū̌, *karttāra > kaṭā̌r, āryikā > ā̌jī: 93% of his marks) (round 7, tone-gated)
[Kal] %source trail-cooper1999 zoller2023
[Kal] l > ḷ                                       ; @notation — Trail & Cooper (and Zoller) write the Kalasha velarised/retroflex ḷ that CDIAL and Decker write l
[Kho] %source bashir2023 strand decker1992
[Kho] ṣ > ś / _ C                                 ; @notation — the modern sources write ś before consonants where CDIAL (Morgenstierne) writes ṣ (aṣṭā́ > ośṭ vs oṣṭ)
[Kund] {á í ú é ó} > {ǎ ǐ ǔ ě ǒ}                  ; @notation @tone — … also where the chain leaves that vowel short (it is long in the source: 6 caron vs 2 acute) (round 7, tone-gated)
[Ash] {e o} > {ē ō}                               ; @notation — Morgenstierne/Turner write the long mid vowels ē ō
[Ash] u > ū / _ #                                 ; @notation — final -u written long
[Phal] %source CDIAL strand decker1992 liljegren-hindukush kalkoti zoller2023
[Phal] Vedic > $1+stress                            ; @tone — the other sources: the flagged vowel is simply the stressed one (tone-gated, round 7)
[dardic-north -Sv -pog -Ush -K -dod Dm bhatr Mai B Dhp Ash Wg] %end
[BH Hajong B MH] ; ── block finals-syncope-bengali: Bengali-type schwa syncope (blocks theme "finals", 2026-09-30): medial a in VCaCV is lost, but not between identical consonants (ṛ_ṛ, r_r, l_l, n_n, t_t, k_k, ṭ [Bengali-type syncope (blocks/finals-syncope-bengali.sc) — replaces `use syncope.a.vccv.except`: medial a lost between single consonants (late), but not between identical consonants (no new geminates); same rule as B. (markaṭa > mākrā, *gōṭṭa > goṭlā)] ──
[BH Hajong B MH] a > ∅ / V C _ C V // ṛ _ ṛ | r _ r | l _ l | n _ n | t _ t | k _ k | ṭ _ ṭ ; @syncope — 3 lects / medial a lost in VCaCV, not between identical consonants (B. āmalaka > āmlā, añjalí+aka > ā̃jlā, *ubbal+aka > ublā) / (changes.sc syncope.a.vccv.except)
[BH MH KS] ; ── block lar-e-lowering: Bengali-type first-syllable e > ɛ before a low vowel or word end: Kamta (BH), Maithili-Hajong group (MH). Laryngeals round, pass 2 (2026-09-30). Edit here, not  [@e-lowering — first-syllable e > ɛ (blocks/lar-e-lowering.sc)] ──
[BH MH KS] e > ɛ / # C* _ C* {a ā #}                    ; @e-lowering — 2 lects / kṣḗtra > kʰɛt, vḗlā > MH. bɛlā, *pēṭṭa > pɛṭ / (changes.sc e-lowering)
[eastern -Or -AdivasiOriya -OB -B] g > ∅ / ŋ _ #                                ; @nasal-assim — 2 lects / word-final ŋg > ŋ (gáŋgā > gāŋ, śŕ̩ŋga > hiŋ); ŋg stays medially / (changes.sc nasal-assim.g.ng)
[AdivasiOriya] a > ɔ                                        ; @a-rounding — Odia/Desia a is /ɔ/ (behera writes ɔ): dánta > dɔt, *nakkʰa > nɔk, lavaṇá > nɔn, rakta > rɔktɔ; ā stays a (kavāṭa > kabat). Round 7: was a > o, scored equal only while ɔ was folded
[AdivasiOriya] {ā ā̃} > {a ã}                               ; @length-loss — 3 lects / no a/ā contrast / (changes.sc length-loss.aa)
[AdivasiOriya] {i u} > {ī ū}                               ; @notation — behera2021desia writes plain i u as ī ū
[MB] ; [round 7: removed legacy @morph `a i > e / _ #` (3sg -ai > -e, gárjati > gāje) — neutral removal after the 10:08 verb-flag fix (gárjati-type headwords now take the verb menu): ext/verb unchanged on train and test; plain +0.0154 / +0.0093, bare headword only. Before the fix it failed ext train (+0.0018)]
[B] {ē ō ē̃ ō̃} > {e o ẽ õ}                   ; @notation — e o (and nasal ẽ õ) written without macron
[Or] o > u / _ ā                                 ; @hiatus — o raised before hiatus -ā (kṣōda-ka > kʰuā, *ḍōva-ka > ḍuā)
[Or] a > u / _ {ā ā̃}                             ; @hiatus — a rounded to u before hiatus ā left by the loss of intervocalic v/p/m (labial glide): *kuṭava-ka > kuṛuā, yavākāra > juāra, maṇḍapa-ka > maṇḍuā, bʰrama > bʰuā̃, namáyati > nuā̃ibā
[AdivasiOriya] {o õ} > {ɔ ɔ̃}                               ; @o-lowering — Desia has one mid back vowel: o (< ō, ava, au) merges with ɔ (< a); behera writes o twice in 480 forms: *gōḍḍa > gɔḍ, *cʰōṭṭa > cʰɔt, gōrūpá > gɔrū, dʰavalá > dʰɔla
[Shum] a > æ / _ V                                 ; @vowel-shift — a fronted to æ before a hiatus vowel (mahiṣá > mǽiṣa, apūpá > æu, cáturdaśa > ʦæudas)
[Shum] Long > $1+stress / # !Long* _                ; @accent-placement — accent on the first long vowel (§9; 75 accent-marked train pairs)
[Shum] Short > $1+stress / # C* _ // _ !#* Long     ; @accent-placement — else on the first syllable (§9)
[K] ñ > n / _ C                                 ; @notation — CDIAL K. writes n before consonants (añcala > anzol)
[bihari-tharu Ku Garh N Dotyali dardic-north -Kalk dardic-south -bhatr eastern hindi -Kannauji-Central marathi-bhil northwest -OP nuristani rajasthani-gujarati -OG west-pahari] ; ── block spont-nasal-optional: Spontaneous nasalisation of a first-syllable vowel before a voiceless consonant or h, at the end of the chain (sarpa > sā̃p, akṣi > ā̃kʰ, mukʰa > mū̃h, púccʰa > [sporadic (blocks/spont-nasal-optional.sc)] ──
[bihari-tharu Ku Garh N Dotyali dardic-north -Kalk dardic-south -bhatr eastern hindi -Kannauji-Central marathi-bhil northwest -OP nuristani rajasthani-gujarati -OG west-pahari] optional: {ā ī ū e o ē ō} > {ā̃ ī̃ ū̃ ẽ õ ē̃ ō̃} / # C* _ {p pʰ k kʰ c cʰ ṭ ṭʰ t tʰ s ś ṣ h} ; @nasalisation — spontaneous nasalisation (sarpa > sā̃p, akṣi > ā̃kʰ, mukʰa > mū̃h) / (block spont-nasal-optional)
[-Pas -Bshk -Dm -Kt] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[-Pas -Bshk -Dm -Kt] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[Khol] ŋ > n                                       ; @notation — the Kholosi source writes [ŋ] as n before velars (aŋgula > āngūl)
[bihari-tharu -KochariyaEastDanuwar Garh N Dotyali eastern -Or -A -MB -OB -B hindi -Brj marathi-bhil -M rajasthani-gujarati -OG -OMarw] ; ── block deasp-optional: Sporadic loss of initial aspiration at the end of the chain (gʰaná > Marw. gaṇɔ, *cʰōṭṭa > Bund. coṭ, skandʰá > S. kandʰu). `optional:` = sporadic: forks the de [sporadic (blocks/deasp-optional.sc)] ──
[bihari-tharu -KochariyaEastDanuwar Garh N Dotyali eastern -Or -A -MB -OB -B hindi -Brj marathi-bhil -M rajasthani-gujarati -OG -OMarw] optional: {kʰ cʰ ṭʰ tʰ pʰ gʰ jʰ ḍʰ dʰ bʰ} > {k c ṭ t p g j ḍ d b} / # _ ; @deaspiration — sporadic initial deaspiration (gʰaná > gaṇɔ, cʰōṭṭa > coṭ) / (block deasp-optional)
[kc] {a i} e > e / _ #                           ; @hiatus — fem. -ie/-ae > -e (stʰitá+ikā > tʰie > tʰe; round 6)
[ktg khash] o u > o / _ C                                ; @hiatus — 2 lects / medial ou > o (round 6) / (changes.sc hiatus.ou.preconsonantal)
[N kc kgr ktg khash] v > u / V _ C                                ; @v-vocalisation — 3 lects / postvocalic v before a consonant > u (causative -āpayati > -āvṇ- > -auṇ-: nāmáyati > nauṇõ, vāmaná > bauṇ; round 6 second pass) / (changes.sc v-vocalisation)
[kc] ; ══ Round 7: lax vowels ══
[ktg] ; ══ Round 7: lax vowels (Hendriksen writes ɔ ɪ ə; scored as distinct vowels since 2026-09-29) ══
[ktg] i > ɪ / _ #                                 ; @vowel-shift — final -i laxed to ɪ: pippalī́ > pɪpḷɪ, kaŋkaṇa > kaŋgṇɪ, ākʰēṭika > hèṛɪ (298 attested -ɪ vs 62 -i; round 7)
[kc ktg] a > ə / V C C* _ C C | V C C* _ C #          ; @vowel-reduction — 2 lects / unstressed a of a closed non-initial syllable > ə, as in Kotgarhi (balivárda > bɔḷəd, śaṭati > śɔṛək, *avaśalati > oʦʰəḷno; round 7) / (changes.sc vowel-reduction.a.vccccvccc)
[-eur -as-ll -Lom -RomWel -RomBalk -Zarg -bhatr -Si -P -L] ; ── block glide-optional: Sporadic hiatus glide at the end of the chain: y after a front vowel before a non-front vowel, v after a rounded vowel before a non-rounded one (kr̩ttikā > Pk.  [sporadic (blocks/glide-optional.sc)] ──
[-eur -as-ll -Lom -RomWel -RomBalk -Zarg -bhatr -Si -P -L] optional: ∅ > y / {i ī e ē ĩ ī̃ ẽ ē̃ ɛ ɛ̃ æ} _ {a ā ã ā̃ o ō õ ō̃ u ū ũ ū̃ ɔ ɔ̃ ə} ; @ya-sruti — sporadic y glide after a front vowel (kr̩ttikā > kattiyā, śr̩gālá > siyār) / (block glide-optional)
[-eur -as-ll -Lom -RomWel -RomBalk -Zarg -bhatr -Si -P -L] optional: ∅ > v / {u ū o ō ũ ū̃ õ ō̃ ɔ ɔ̃} _ {a ā ã ā̃ e ē ẽ ē̃ i ī ĩ ī̃ ɛ ɛ̃ æ ə} ; @ya-sruti — sporadic v glide after a rounded vowel (rōpa > rovā, *ḍʰauka > ḍʰovā) / (block glide-optional)
[MB] a > ∅ / i _ C                                ; @hiatus — 7 lects / a absorbed after i before a consonant (-iavva > -ivva), as in A. and B. / (changes.sc hiatus.a.ic)
[OB] a a > ā / _ #                                ; @hiatus — 17 lects / -aka > -aa > -ā (ghōṭaka > gʰoṛā; extension-aware score 0.3725 → 0.3485 in Dang) / (changes.sc hiatus.aa.final.2)
[bihari-tharu -Buksa -MagahiNepal -Dang Ku Garh Dotyali dardic-north dardic-south -Mai eastern hindi -OAw -NDu marathi-bhil -hal northwest nuristani rajasthani-gujarati west-pahari -jaun -bhad -mand -cur] ; ── block glide-a-optional: Sporadic hiatus glide after a/ā at the end of the chain, y or v (kātara > Bhoj. kāyar, jātá > G. jāyī, tāpaka > Bhoj. tāvā, śrā́vaṇa > Bhoj. sāvan). `optional:` [sporadic (blocks/glide-a-optional.sc)] ──
[bihari-tharu -Buksa -MagahiNepal -Dang Ku Garh Dotyali dardic-north dardic-south -Mai eastern hindi -OAw -NDu marathi-bhil -hal northwest nuristani rajasthani-gujarati west-pahari -jaun -bhad -mand -cur] optional: ∅ > y / {a ā ã ā̃} _ V             ; @ya-sruti — sporadic y glide after a/ā (kātara > kāyar, jātá > jāyī) / (block glide-a-optional)
[bihari-tharu -Buksa -MagahiNepal -Dang Ku Garh Dotyali dardic-north dardic-south -Mai eastern hindi -OAw -NDu marathi-bhil -hal northwest nuristani rajasthani-gujarati west-pahari -jaun -bhad -mand -cur] optional: ∅ > v / {a ā ã ā̃} _ V             ; @ya-sruti — sporadic v glide after a/ā (tāpaka > tāvā, śrā́vaṇa > sāvan) / (block glide-a-optional)
[P L] ; ── block pretonic-i-a-optional: Sporadic i > a in the first syllable of words of three or more syllables, end of chain (P cʰikkāra > cakārā, citrakara > catrērā, virāmayati > varāuṇā; L). A re [sporadic (blocks/pretonic-i-a-optional.sc)] ──
[P L] optional: i > a / # C* _ C C* V C* V         ; @i-a — sporadic pretonic i > a (cʰikkāra > P cakārā) / (block pretonic-i-a-optional)
[-eur -as-ll -Lom -RomWel -RomBalk -Zarg -Sh -Kal -Phal -Bshk -Dm -Gowro -bhatr -B -A -PalyaBareli -Si -P -S -L -awan -Kam -Kt -Ash -Pr -bhad] ; ── block apheresis-optional: Sporadic loss of an initial vowel before a consonant, end of chain (āṣāḍʰá > hāṛʰū, apidʰāna > H. pidhān; 2026-10-01). Kept in the groups where it beats a same- [sporadic (blocks/apheresis-optional.sc)] ──
[-eur -as-ll -Lom -RomWel -RomBalk -Zarg -Sh -Kal -Phal -Bshk -Dm -Gowro -bhatr -B -A -PalyaBareli -Si -P -S -L -awan -Kam -Kt -Ash -Pr -bhad] optional: V > / # _ C                        ; @apheresis — sporadic initial vowel loss / (block apheresis-optional)
[-Bhoj -eur -as-ll -Lom -RomWel -RomBalk -Zarg -OB -Kannauji-Central -Ap] ; ── block h-loss-optional: Sporadic loss of intervocalic h at the end of the chain (hill-climb round 3, 2026-10-01). Where a lect keeps the h from OIA h and MIA aspirates (Pk. -h- < -kh-  [sporadic (blocks/h-loss-optional.sc; hill-climb round 3)] ──
[-Bhoj -eur -as-ll -Lom -RomWel -RomBalk -Zarg -OB -Kannauji-Central -Ap] optional: h > / V _ V                        ; @h-loss — sporadic intervocalic h loss (agnidʰāna > N agenu, aragʰaṭṭa > Or araṭa) / (block h-loss-optional)
[P] optional: ∅ > i / a h _ C                  ; @notation — CDIAL P. writes a + h before a consonant three ways: ahiC (Gurmukhi sihari: kahiṇā, rahiṇā, gahiṇ; 52), aihC (ɛ: vaihṇā, gaihṇ, paihr; 26) and ahC (vahṇ, bahṇā; 25), often for one etymon (hill-climb round 5; notation fork)
[P] optional: a h > a i h / _ C                ; @notation — … the aih spelling
[P] optional: a h > a i h / _ #                ; @notation — … also word-final (prabʰā́ > paih)
[L srk] optional: {g j ḍ b} > {ɠ ʄ ᶑ ɓ} // {ŋ ñ ṇ n m ṁ} _ ; @implosive — CDIAL pools southern (Multani/Saraiki) forms with implosives (ɠaṛā, ɠabʰ, uᶑaṇ, kaʄaṇ, kuɓɓā, ~28% of initial g j ḍ b) with plain ones; a whole-word fork, not after a nasal (as in S): endprobe −0.0151/−0.0115 vs placebo devoicing −0.0001/−0.0004 (hill-climb round 3)
[L] optional: {e o ẽ õ} > {ē ō ē̃ ō̃}           ; @notation — CDIAL L. writes e o with a macron in 162 forms and without in 581 (6 mixed), consistently per form (hōr, ōhar, vēṭʰ, kōṇ vs heṭʰ, agge); all CDIAL, so a whole-word fork, not a %source tail. endprobe ext −0.0054/−0.0049, ext_acc −0.055 vs placebo {ɛ ɔ} −0.0017/−0.0003 (hill-climb round 6; notation fork)
[Or] optional: a > u / V C* _ {r ṛ ḷ ṇ} V          ; @vowel-shift — sporadic non-initial a > u before a liquid or ṇ (añjalí > añjuḷi, *argaḍa > āguḷa, kuṭṭana > kuṭuṇī, *ujjaṭati > ujuṛibā; 9–18% of the residual cases): endprobe −0.0044/−0.0034 vs placebo a > i −0.0006/−0.0005 (hill-climb round 3)
[Bhoj Mth Bi Ku Garh OAw OM Pk Pa OP OG OMarw bhal] ; ── block final-i-short-optional: Sporadic shortening of final -ī to -i at the end of the chain (hill-climb round 2, 2026-10-01). Ku. and the Bihari lects write a short fem./diminutive -i beside [sporadic (blocks/final-i-short-optional.sc; hill-climb round 3: Early NIA and MIA, ext Δ group train/test −0.0052/−0.0052 and −0.0014/−0.0019 vs placebo −0.0005/−0.0009 and −0.0001/0)] ──
[Bhoj Mth Bi Ku Garh OAw OM Pk Pa OP OG OMarw bhal] optional: ī > i / _ #                        ; @final-shortening — sporadic final -ī > -i (Ku. ā̃dʰi, Mth. gābʰini) / (block final-i-short-optional)
[Ku] optional: l > v / V _ #                     ; @l-v — sporadic final -l > -v (añcala > ā̃cav beside ā̃cal, kajjala > kāʣav, ujjvala > uʣav, *utkāla > ukāv; CDIAL Ku. -al/-āl 72 vs -av/-āv 41). endprobe ext −0.0033/−0.0015 vs placebo l > y 0/0; C/E Pahari aggregate −0.00125/−0.00057 vs 0/0 (only Ku is affected) (hill-climb round 4)
[Pk] optional: ḍ > l / V _ V                     ; @l-retroflex — sporadic Pk. intervocalic ḍ (< ṭ ḍ) > l (Pischel §240: akkʰōḍa ~ akkʰula, aggala < *argaḍa, ukkala < utkaṭa; 155 of 811 residual ḍ): endprobe −0.0022/−0.0024 vs placebo ḍ > r −0.0001/0 (hill-climb round 3)
[Mth] ; ── block h-loss-optional: Sporadic loss of intervocalic h at the end of the chain (hill-climb round 3, 2026-10-01). Where a lect keeps the h from OIA h and MIA aspirates (Pk. -h- < -kh-  [sporadic (blocks/h-loss-optional.sc; hill-climb round 3)] ──
[Mth] optional: h > / V _ V                        ; @h-loss — sporadic intervocalic h loss (agnidʰāna > N agenu, aragʰaṭṭa > Or araṭa) / (block h-loss-optional)
[Chitwan Rana Dang Morang Kathoriya Mth Ku eur Garh N RomWel dardic-south bundeli_atarra OAw bagheli_lakshman Brj PalyaBareli PauriBareli M OM RathwiBareli Bhili NiDoc Md D thar L srk awan nuristani rajasthani-gujarati -WK -G WPah] ; ── hill-climb round 7: end-of-chain gap fixes (OIA segments the chain left unchanged, never attested here) ──
[-OP] ; ── block visarga-loss: visarga-loss — end of chain: a visarga still left in the output is lost (hill-climb round 7). These chains had no rule deleting ḣ in every position (no `final-c [end of chain: leftover visarga lost (blocks/visarga-loss.sc; hill-climb round 7)] ──
[-OP] {ḣ ḥ} > ∅                                    ; @final-cons-loss — leftover visarga lost (tráyaḣ > tīn/tre, catvā́raḣ > cār, duḣkʰá > duk) / (block visarga-loss)
[S] optional: {ɠ ʄ ᶑ ɓ} > {g j ḍ b}               ; @notation @implosive — CDIAL S. writes the implosives in some forms and plain voiced stops in others (ɠ 196 vs g 218, ʄ 118 vs j 283, ɓ 199 vs b 109 in the golds; *gacca > gacu, *ujjaṭa > ujaṛu, *udbūra > ubūraṇu beside ɠaṛu, uʄaṛu): a whole-word notation fork, not separable by source (all CDIAL). endprobe ext −0.0076/−0.0081 vs placebo {k c ṭ p} −0.0000/+0.0004 (hill-climb round 4)
[Dang Ku Garh Kho dod dardic-south OAw bagheli_lakshman Brj Bhili Md thar Goj WPah] ; ── block nasals-palatal-n: nasals-palatal-n — end of chain: ñ > n, for lects whose attested forms never write ñ (hill-climb round 7). The palatal nasal before c/j (páñca, kuñcikā) and fro [end of chain: ñ > n, this lect's sources never write ñ (blocks/nasals-palatal-n.sc; hill-climb round 7)] ──
[Dang Ku Garh Kho dod dardic-south OAw bagheli_lakshman Brj Bhili Md thar Goj WPah] ñ > n                                        ; @n-dental @notation — palatal nasal written n (páñca > Pas pānja, Gaw pānʦ; ajñānin > Ku anāṛi) / (block nasals-palatal-n)
[RomWel Gaw PalyaBareli Bhili srk mewari_basad Goj] ; ── block nasals-anusvara-leftover: nasals-anusvara-leftover — end of chain: an anusvāra no rule consumed (before a sibilant or h) becomes vowel length, as anusvara-vowel.vm.2 in the Hindi belt (v [end of chain: leftover anusvāra > length (blocks/nasals-anusvara-leftover.sc; hill-climb round 7)] ──
[RomWel Gaw PalyaBareli Bhili srk mewari_basad Goj] V ṁ > $1+long                                ; @anusvara-vowel — leftover anusvāra > length (māṁsá > Bhili māx, Goj mās; viṁśatí > vīh) / (block nasals-anusvara-leftover)
[bihari-tharu -Bhoj -Mth Ku Garh N Dotyali dardic-north dardic-south eastern hindi marathi-bhil northwest -OP nuristani rajasthani-gujarati west-pahari -kc -cam] ; ── block nasal-aspirate-plain: End of chain: aspirated nasals written plain, for the groups whose mh/nh outcomes the sources write as m/n (asmad > amha > ham, B āmi; cihna > cin). Included wi [aspirated nasals written plain (blocks/nasal-aspirate-plain.sc)] ──
[bihari-tharu -Bhoj -Mth Ku Garh N Dotyali dardic-north dardic-south eastern hindi marathi-bhil northwest -OP nuristani rajasthani-gujarati west-pahari -kc -cam] {mʰ nʰ} > {m n}                              ; @notation @h-metathesis — aspirated nasals written plain / (block nasal-aspirate-plain)
[-Pa] ; ── block nc-voiced-stop: Nasal + voiced stop, two outcomes, end of chain (2026-10-01): the cluster stays (aŋga > aŋg, kambala > kambal) or loses its stop (aŋga > aŋ, lamba > lam, índra  [NC > NC | N (blocks/nc-voiced-stop.sc)] ──
[-Pa] {ŋ m n ṇ} {g gʰ b bʰ d dʰ ḍ ḍʰ} > $1 $2 | $1 ; @nasal-cluster — NC > NC | N (aŋga > aŋg | aŋ) / (block nc-voiced-stop)
[Bhoj Bi N Kalk Kund K Bshk Dm Pas Bishnupriya Or AdivasiOriya Hajong Bhatri DungraBhili PauriBareli mewari_bannoda Vasavi Bhili Ko-Ul Si Md kvar] ; ── block h-initial-loss-optional: Sporadic loss of initial h- before a vowel, end of chain (2026-10-01): Si haṁsá > as, hánati > annavā; Pas hánati > an; Bishnupriya hásta > at. Beat a same-shap [cross-family law (blocks/h-initial-loss-optional.sc)] ──
[Bhoj Bi N Kalk Kund K Bshk Dm Pas Bishnupriya Or AdivasiOriya Hajong Bhatri DungraBhili PauriBareli mewari_bannoda Vasavi Bhili Ko-Ul Si Md kvar] optional: h > / # _ V                        ; @h-loss — sporadic initial h- loss (haṁsá > Si as) / (block h-initial-loss-optional)
[Sh] optional: a > ʌ                             ; @notation — CDIAL's Sh. forms come in two transcriptions: 324 write every short a as ʌ (vʌtu, ʌśātu, ʌgúru), 393 write a, 24 mix; a whole-word fork (endprobe −0.0132/−0.0094 vs placebo a > ə −0.0012/−0.0002; hill-climb round 3)
[-Or -A -Ko-Ul -kc] ; ── block b-lenition-multi: Intervocalic b, three outcomes, end of chain (2026-10-01): b stays, becomes v (abalā > G avḷũ), or is lost (lābu > B, Or lāu; píbati > Bshk pū). Beat a same-sha [cross-family law (blocks/b-lenition-multi.sc)] ──
[-Or -A -Ko-Ul -kc] optional: b > v | / V _ V                    ; @b-lenition — VbV > VbV | VvV | VV / (block b-lenition-multi)
[D] {ṭʰ tʰ} > {ṭ t} / {ṣ s ś} _                ; @deaspiration — end of chain: ṣṭʰ > ṣṭ (gold ṣṭʰ 0: aṣṭʰi, kāṣṭʰa); hill-climb round 7
[Kal] ; ── hill-climb round 7: end-of-chain gap fixes (OIA segments the chain left unchanged, never attested here) ──
[pog] a > u / V C (C) _ C #                        ; @umlaut — 2 lects / a of a (now) final non-initial syllable > u, as in Kashmiri (paśy-anaka > paśun, *japy- > zapun) (round 6 second pass) / (changes.sc umlaut.vccc)
[ktg] ; ── block v-loss-optional: Sporadic loss of intervocalic v at the end of the chain, western IA (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā, praviśati > paisnā, kāvikā > G. kāī). `optional:` = s [sporadic (blocks/v-loss-optional.sc)] ──
[ktg] optional: v > / V _ V                        ; @v-loss — sporadic intervocalic v loss (aṣṭāviṁśati > aṭʰāīs, nāvika > naiyā) / (block v-loss-optional)
[pog ktg] ; ── block apheresis-optional: Sporadic loss of an initial vowel before a consonant, end of chain (āṣāḍʰá > hāṛʰū, apidʰāna > H. pidhān; 2026-10-01). Kept in the groups where it beats a same- [sporadic (blocks/apheresis-optional.sc)] ──
[pog ktg] optional: V > / # _ C                        ; @apheresis — sporadic initial vowel loss / (block apheresis-optional)
[pog ktg] ; ── block nc-voiced-stop: Nasal + voiced stop, two outcomes, end of chain (2026-10-01): the cluster stays (aŋga > aŋg, kambala > kambal) or loses its stop (aŋga > aŋ, lamba > lam, índra  [NC > NC | N (blocks/nc-voiced-stop.sc)] ──
[pog ktg] {ŋ m n ṇ} {g gʰ b bʰ d dʰ ḍ ḍʰ} > $1 $2 | $1 ; @nasal-cluster — NC > NC | N (aŋga > aŋg | aŋ) / (block nc-voiced-stop)
[Kho] ; ── block nc-final-stop-loss-optional: Sporadic loss of a word-final voiced stop after a nasal at the end of the chain (hill-climb round 4, 2026-10-01). Dardic apocope leaves final -nd -ṇḍ -ŋg -mb (d [sporadic (blocks/nc-final-stop-loss-optional.sc; hill-climb round 4)] ──
[Kho] optional: {d ḍ g b} > / {n ṇ ŋ m} _ #        ; @cluster-simplify — sporadic final NC > N (dánta > Phal dān beside dānd, áŋga > Kal aŋ beside oŋg) / (block nc-final-stop-loss-optional)
[Kho dod] ; ── hill-climb round 7: end-of-chain gap fixes (OIA segments the chain left unchanged, never attested here) ──
[Dotyali Kho dod bhad khash] ; ── block m-nasal-multi: NIA loss of intervocalic (or, after apocope, final) m with nasalisation, three outcomes (2026-10-01): the m leaves a nasal vowel (gōdʰū́ma > Bund. gehū̃, dʰūmá  [cross-family law (blocks/m-nasal-multi.sc)] ──
[Dotyali Kho dod bhad khash] {a ā i ī u ū e ē o ō} m > {ã ā̃ ĩ ī̃ ũ ū̃ ẽ ē̃ õ ō̃}@1 | $1 ṽ | $1 m / _ V | _ # ; @m-nasal — Vm > Ṽ | Vṽ | Vm (gōdʰū́ma > gehū̃) / (block m-nasal-multi)
