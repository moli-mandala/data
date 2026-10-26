fixes = [
    'Skt.', 'Pkt.',
    'Panj.', 'OMar.', 'Pali', 'H.', 'Apabhraṃśa', 'Guj.',
    'O.', 'Or.', 'Nahali', 'Mar.', 'Beng.', 'Halbi', 'Nep.', 'Ta.', 'Sgh.'
]

# A small set of source entries has an unclosed initial ``<b>`` tag, making
# the first language's English definition look like more comma-separated
# headwords.  These are exact definition boundaries in the published text.
shared_gloss_boundaries = {
    '23': 'wife of an elder brother',
    '360': 'id.',
    '1075': 'desire',
    '1315': 'unhusk with the teeth',
    '1347': 'bonduc nut',
    '1699': 'munch',
    '2367': 'in the dry bed of a river or a dried-up tank',
    '3252': 'essential part',
    '3594': 'moist',
    '4251': 'hands',
}

# Exact replacements in an entry section's HTML, applied after ``repair_botanical_markup`` and
# before the language labels are read (so a repair can restore a lost label). parse.py fails if
# a repair no longer matches. "SQL" is the 2013 DEDR dump dedr_new_entry_oct2013_edited.sql.
source_markup_repairs = {
    # Words of an English gloss that DSAL set in bold, each parsed as a form of the entry.
    '50': [('a <b>Malayāḷa</b> woman', 'a Malayāḷa woman')],
    '1037': [('that <b>I</b> may look nice', 'that I may look nice')],
    '1278': [('rock at <b>Me\u0387na\u0387ṛ</b> village', 'rock at Me\u0387na\u0387ṛ village')],
    # Names of the country, not reflexes of *koṅk-: "inhabitant of the Koṅku country", "of the
    # Chēra or Kēraḷa country", "the Koṅgu country (Cēra or Kēraḷa, …)", "the Kon̄gu country".
    '2037': [(f'<b>{name}</b>', name) for name in (
        'Koṅku', 'Chēra', 'Kēraḷa', 'Kēraḷa,', 'Koṅgu', 'Cēra', 'Kon̄gu')],
    '3067': [('father of <b>Ar̤āṉ; āntai</b> father of <b>Ātaṉ; pur̤āntai</b> father of '
              '<b>Pur̤āṉ. <i>Ma.</i> tanta</b>',
              'father of Ar̤āṉ; <b>āntai</b> father of Ātaṉ; <b>pur̤āntai</b> father of '
              'Pur̤āṉ. <b><i>Ma.</i> tanta</b>')],
    # Toda women's names that attest the form glossed: "murx, in woman's name Piḷymurx".
    '3268': [("in woman's name <b>Sici\u0387z</b>, (also known as <b>Tici\u0387ζ</b>,",
              "in woman's name Sici\u0387z, (also known as Tici\u0387ζ,")],
    '4979': [("in woman's name <b>Piḷymurx (piḷy</b> silver,",
              "in woman's name Piḷymurx (piḷy silver,")],
    '4198': [('title of <b>Veḷḷāḷa</b> caste', 'title of Veḷḷāḷa caste')],
    '5545': [('a person of <b>Vēḷāḷa</b> caste', 'a person of Vēḷāḷa caste')],
    'a18': [('a class of <b>Mārān</b> Sūdras', 'a class of Mārān Sūdras')],
    # Cross-references to other entries set like forms.
    '1225': [('Or with 1221 <b>Ta. kappu</b> (Pfeiffer)', 'Or with 1221 Ta. kappu (Pfeiffer)')],
    '2661': [('For <b>eli</b>, etc., see 833 <b>Ta. eli</b>.', 'For eli, etc., see 833 Ta. eli.')],
    # The Brahui paradigm is cut by three stray bold tags, and the cross-reference "(for ēt-, see
    # 872 Ma. ēkuka)" is bold together with the next lemma.
    '3098': [('<b><i>Br.</i> tining (tir-, <i>neg.</i> ti-, <i>present</i> ēt-</b> [dial. '
              '<b>tē-], <i>imper.</i> ēte, <i>past</i> tiss-</b>) to give (for <b>ēt-</b>, see 872 '
              '<b>Ma. ēkuka); tiss</b>',
              '<b><i>Br.</i> tining (tir-, <i>neg.</i> ti-, <i>present</i> ēt- [dial. tē-], '
              '<i>imper.</i> ēte, <i>past</i> tiss-)</b> to give (for ēt-, see 872 Ma. ēkuka); '
              '<b>tiss</b>')],
    # Likewise in the Kolami run: "kak- (kakt-; also ka- instead of kak- in the present) to do,
    # make; …"; akk- and ā belong to the comparison with Nk. and 333 Ta.
    '1957': [('<b><i>Kol.</i> kak- (kakt-</b>; also <b>ka-</b> instead of <b>kak-</b> in the '
              'present)', '<b><i>Kol.</i> kak- (kakt-; also ka- instead of kak- in the present)</b>'),
             ('a form like Nk. <b>akk-</b>, s.v. 333 Ta. <b>ā</b>).',
              'a form like Nk. akk-, s.v. 333 Ta. ā).')],
    # Plain "(b) Kurub. (LSB 1.12)" after a sentence: pōke is Betta Kurumba, not Kuwi.
    '1224': [('frog.(<i>b</i>) Kurub. (<i>LSB</i> 1.12) <b>',
              'frog. <i>(b) Kurub.</i> (<i>LSB</i> 1.12) <b>')],
    # A dialect parenthesis that lost its opening bracket.
    '2209': [('<i><b>Ga.</b></i> P.) <b>', '<i><b>Ga.</b></i> (P.) <b>')],  # kōne
    '4508': [('<i><b>Go.</b></i> Tr.) <b>', '<i><b>Go.</b></i> (Tr.) <b>')],  # pōṛāl
    # "M. malabaricum, Ma." for "M. malabaricum. Ma.", hiding the Malayalam label.
    '1464': [('<b><i>M. malabaricum, Ma.</i> kāyāvu</b>',
              '<i>M. malabaricum</i>. <b><i>Ma.</i> kāyāvu</b>')],
    # Malayalam iṭala "a wood, burning very rapidly, (ML) Olea dioica"; (DCV) eṭala id.
    '436': [('<i>(ML) <b>Olea dioica</b>; (DCV)</i> <b>eṭala</b>',
             '(ML) <i>Olea dioica</i>; (DCV) <b>eṭala</b>')],
    # "modern names Errayya, Erramma (MBE 1978, p. 355)": the gloss precedes the two names.
    '865': [('modern names <b>Errayya, Erramma</b> (MBE 1978, p. 355)',
             '<b>Errayya, Erramma</b> modern names (MBE 1978, p. 355)')],
    # Toda t/θ (DSAL <at>Q</at>) alternation inside one word, "teṣ t/θwï·ṟ" (cf. twï·ṟ, 2897).
    '3245': [('<b>teṣ t/<at>Q</at>wï\u0387ṟ</b>', '<b>teṣ twï\u0387ṟ/teṣ <at>Q</at>wï\u0387ṟ</b>')],
    # Capital I for l: Pengo lēkiŋ (pl.), cf. Kui vlēka.
    '5557': [('<b><i>Pe.</i> Iēkiŋ', '<b><i>Pe.</i> lēkiŋ')],
    # DSAL text drops, restored from SQL (as for 2399 below). 421: "muṭ-caṅku mistletoe berry
    # thorn,"; the binomial is DSAL's "tetracantha" and Kannada's "A. tetracantha".
    '421': [('uvar-ccaṅkam, tetracantha. Ma. iyaṅku, iśaṅku</b> id.',
             'uvar-ccaṅkam, muṭ-caṅku</b> mistletoe berry thorn, <i>Azima tetracantha</i>. '
             '<b><i>Ma.</i> iyaṅku, iśaṅku</b> id.')],
    # 1737: SQL has "Ma. kuppa-mēni id. Ka. kuppe, kuppe-giḍa id."
    '1737': [('kuppai-mēṉi <i>Acalypha indica</i> kuppe-giḍa</b> id.',
              'kuppai-mēṉi</b> Indian acalypha, <i>Acalypha indica</i>. <b><i>Ma.</i> kuppa-mēni</b> '
              'id. <b><i>Ka.</i> kuppe, kuppe-giḍa</b> id.')],
    # 1077: SQL has "kakadg, peṇ gakadg Pogostemon speciosum; am gakadg Plectranthus macraei";
    # am gakadg is "our k.", ka·ṛ gakadg "jungle k.".
    '1077': [("<b><i>Ko.</i> kakadg, peṇ gakadg <i>Pogostemon <i>Plectranthus macraei</i></i></b>"
              "<i> ('our k.'); <b>ka\u0387ṛ gakadg</b> <i><b>Pl. nepetaefolius</b></i> ('jungle k.'). "
              "<b><i>To.</i> kaka<at>Q</at>x <i>Pogostemon speciosum.</i></b>",
              "<b><i>Ko.</i> kakadg, peṇ gakadg</b> <i>Pogostemon speciosum</i>; <b>am gakadg</b> "
              "<i>Plectranthus macraei</i> ('our k.'); <b>ka\u0387ṛ gakadg</b> <i>Pl. nepetaefolius</i> "
              "('jungle k.'). <b><i>To.</i> kaka<at>Q</at>x</b> <i>Pogostemon speciosum.</i>")],
    # 2665: SQL has "cuṇṭai Solanum torvum; S. verbascifolium" and "Ma. cuṇṭa".
    '2665': [('<b><i>Ta.</i> cuṇṭai <i>Solanum torvum; S. <i>S. pubescens. Ko.</i> <b>ciḍ muḷ</b>',
              '<b><i>Ta.</i> cuṇṭai</b> <i>Solanum torvum; S. verbascifolium</i>. '
              '<b><i>Ma.</i> cuṇṭa</b> <i>S. pubescens</i>. <b><i>Ko.</i> ciḍ muḷ</b>')],
    # 3112: SQL has "taviṭṭu-kkoyyā, taviṭṭu-cceṭi, taviṭṭu-ppar̤am hill guava,"; the binomial is
    # implied by Kannada's abbreviated R. tomentosa.
    '3112': [('taviṭṭu-kkoyyā, taviṭṭu-cceṭi, Ko. <b>tavṭ</b> id.',
              'taviṭṭu-kkoyyā, taviṭṭu-cceṭi, taviṭṭu-ppar̤am</b> hill guava, '
              '<i>Rhodomyrtus tomentosa</i>. <b><i>Ko.</i> tavṭ</b> id.')],
    # The DSAL text drops "savunte a kind of cucumber, Cucumis utilissimus", leaving Kittel's
    # botanical authority "Roxb." as a fifth Kannada "form". The 2013 DEDR SQL dump keeps
    # savunte and "a kind of cucumber,"; the binomial is Kittel's (p. 1607, s.v. savute), whose
    # Kannada definitions DEDR quotes.
    '2399': [(' savute, saute, savate, savati, Roxb. ',
              ' <b>savute, saute, savate, savati, savunte</b> a kind of cucumber, '
              '<i>Cucumis utilissimus</i> Roxb. ')],
    # Both Latin names are set bold, so the authority of the bracketed correction became a Gondi
    # form "Roxb" glossed "]".
    '2628': [('<i>Buchanania angustifolia</i><b> Roxb.</b> [better, <i>B. latifolia</i><b> Roxb</b>.]',
              '<i>Buchanania angustifolia</i> Roxb. [better, <i>B. latifolia</i> Roxb.]')],
    '4428': [('bē salt, piquancy, spirit, flavour.',
              '<b>bē</b> salt, piquancy, spirit, flavour.')],
}

# Genus names shorter than eight letters that the source sets alone in bold italics
# (``<b><i>Barbus. Ma.</i> keṇṭa</b>``, ``<i><b>Cassia</b></i>``); ``repair_botanical_markup``
# accepts longer single words without a list. This is every such name in the DSAL text.
latin_genera = frozenset({
    'Aegle', 'Alpinia', 'Arum', 'Barbus', 'Belone', 'Caranx', 'Cassia', 'Clupea', 'Cuculus',
    'Cuon', 'Cyperus', 'Panicum', 'Premna', 'Shorea', 'Solanum', 'Ulex',
})

# Definitions stated once for a whole compact subsection whose remaining
# forms are bare equivalents.  The section indices come from the source's
# slash-separated layout.
shared_section_glosses = {
    ('1159', 2): 'tears',
    ('5496', 6): 'butter',
}

# Entries whose digital text opens with ``id.`` because the first definition (usually a
# Latin plant name) was lost in digitisation, or whose first language run is not parsed.
# The antecedent is supplied from the printed DEDR so ``resolve_idem`` can expand it.
entry_initial_glosses = {
    '360': 'castor-oil plant, <i>Ricinus communis</i>',
    '575': 'sand',
    '1092': 'cart',
    '1667': 'cooing of pigeons',
    '1699': 'munch',
    '3796': 'fast, religious observance, vow',
    '5072': 'fenugreek, <i>Trigonella foenum-graecum</i>',
    '5183': 'eaves',
    '5391': 'shrub, <i>Dichrostachys cinerea</i>',
    '5509': 'wood-apple, <i>Feronia elephantum</i>',
}

abbrevs = {
    'Ta': 'Tam',
    'Ma': 'Mal',
    'Ir': 'Irula',
    'PālKu': 'PaluKurumba',
    'ĀlKu': 'AluKurumba',
    'Ālku': 'AluKurumba',  # 656, 1977
    'Kurub': 'BettaKurumba',
    'Ko': 'Kota',
    'To': 'Toda',
    'Ka': 'Kannada',
    'Koḍ': 'Kodagu',
    # A handful of entries use the non-diacritic spelling.
    'Kod': 'Kodagu',
    # Single-entry slips in the DSAL text, identified by the entry's language order:
    # 2690 Koḏ. (Ka. … Koḏ. … Tu.), 4035 Koḷ. (SR.) (Te. … Koḷ. … Nk.), 1942 Ṭe. (Tu. … Ṭe. …).
    'Koḏ': 'Kodagu',
    'Koḷ': 'Kolami',
    'Ṭe': 'Telugu',
    'Tu': 'Tulu',
    'Bel': 'Belari',
    'Kor': 'Koraga',
    'Te': 'Telugu',
    'Kol': 'Kolami',
    'Nk. (Ch.)': 'Naiki',
    'Nk': 'Naikri',
    'Pa': 'Parji',
    'Ga': 'Gadaba',
    'Go': 'Gondi',
    'Konḍa': 'Konda',
    'Koṇḍa': 'Konda',  # 443
    'Konḏa': 'Konda',  # 5183
    'Pe': 'Pengo',
    'Manḍ': 'Manda',
    'Mand': 'Manda',   # 34
    'Manḍa': 'Manda',  # 2341, 3808
    'Kui': 'Kui',
    'Kuwi': 'Kuwi',
    'Oraon': 'Kurux',
    'Kuruḵẖ': 'Kurux',
    'Kur': 'Kurux',
    'Malt': 'Malto',
    'Br': 'Brah',
    'Skt': 'OIA',
    'Turner': 'OIA',
    'CDIAL': 'OIA',
    'Pali': 'Pa',
    'Mar': 'M',
    'H': 'H',
    'Beng': 'B',
    'Pkt': 'Pk',
    'Halbi': 'hal',
    'O': 'Or',
    'Or': 'Or',
    'Nahali': 'Ni',
    'Apabhraṃśa': 'Ap',
    'Guj': 'G',
    'OMar': 'OM',
    'Nep': 'N',
    'Sgh': 'Si'
}

replacements = {
    'Koya Su.': 'KoyaSu.',
    'Koya T.': 'KoyaT.',
}

dialects = {
    # Betta Kurumba
    ('LSB', 'BettaKurumba'):                ['LSB-BettaKurumba', None],
    # Gadaba
    ('P.', 'Gadaba'):                       [None, 'pottangi'],
    ('Oll.', 'Gadaba'):                     ['Oll-Gadaba', None],
    ('S.<super>3</super>', 'Gadaba'):       ['S3-Gadaba', 'salur'],
    ('S.', 'Gadaba'):                       ['S-Gadaba', 'salur'],
    ('S', 'Gadaba'):                        ['S-Gadaba', 'salur'],
    ('S.<super>2</super>', 'Gadaba'):       ['S2-Gadaba', 'salur'],
    # Gondi
    ('ASu.', 'Gondi'):                      [None, 'adil'],
    ('S.', 'Gondi'):                        [None, 'seoni'],
    ('D.', 'Gondi'):                        [None, 'durg'],
    ('G.', 'Gondi'):                        [None, 'gommu'],
    ('Y.', 'Gondi'):                        [None, 'yavatmal'],
    ('Ch.', 'Gondi'):                       [None, 'chanda'],
    ('Ko.', 'Gondi'):                       [None, 'koya'],
    ('Ma.', 'Gondi'):                       [None, 'maria'],
    ('Mu.', 'Gondi'):                       [None, 'muria'],
    ('Ph.', 'Gondi'):                       ['Ph-Gondi', 'mandla'],
    ('KoyaSu.', 'Gondi'):                   [None, 'koya'],
    ('Pat.', 'Gondi'):                      ['Pat-Gondi', 'chanda'],
    ('LuS.', 'Gondi'):                      [None, 'chanda'],
    ('Tr.', 'Gondi'):                       ['Tr-Gondi', 'betul'],
    ('W.', 'Gondi'):                        ['W-Gondi', 'mandla'],
    ('mand.', 'Gondi'):                     [None, 'mandla'],
    ('M.', 'Gondi'):                        ['M-Gondi', 'maria'],
    ('L.', 'Gondi'):                        ['L-Gondi', 'maria'],
    ('SR.', 'Gondi'):                       ['SR-Gondi', 'adil'],
    ('A.', 'Gondi'):                        ['A-Gondi', 'adil'],
    ('KoyaT.', 'Gondi'):                    ['KoyaT-Gondi', 'gommu'],
    ('Hislop', 'Gondi'):                    ['Hislop-Drav', None],
    ('LuS.', 'Gondi'):                      ['LuS-Gondi', None],
    ('<i>DGG</i>', 'Gondi'):                ['DGG-Gondi', None],
    # Irula
    ('Z.', 'Irula'):                        ['Z-Irula;Z2-Irula', None],
    ('Zvelebil', 'Irula'):                  ['Zvelebil1980-Irula', None],
    # Kannada
    ('Hal.', 'Kannada'):                    ['Hal-Kannada', 'halakki'],
    ('Hav.', 'Kannada'):                    ['Hav-Kannada', 'havyaka'],
    ('HavS.', 'Kannada'):                   ['HavS-Kannada', 'havyaka'],
    ('PBh.', 'Kannada'):                    ['PBh-Kannada', 'pampa'],
    ('Gowda', 'Kannada'):                   ['Gowda-Kannada', 'gowda'],
    ('Gul.', 'Kannada'):                    ['Gul-Kannada', 'gulbarga'],
    ('Nanj.', 'Kannada'):                   ['Nanj-Kannada', 'nanjangud'],
    ('Bark.', 'Kannada'):                   ['Bark-Kannada', 'barkur'],
    ('Tipt.', 'Kannada'):                   ['Tipt-Kannada', 'tiptur'],
    ('Coorg', 'Kannada'):                   ['Coorg-Kannada', 'coorg'],
    ('Kuruba', 'Kannada'):                  [None, 'coorg'],
    ('Rabakavi', 'Kannada'):                ['Rabakavi-Kannada', 'rabakavi'],
    ('Sholiga', 'Kannada'):                 ['Sholiga-Kannada', 'Sholaga'],
    ('Badaga', 'Kannada'):                  ['Badaga-Kannada', 'Badaga'],
    ('Hock.', 'Kannada'):                   ['Hock-Kannada', 'Badaga'],
    ('Kitt.', 'Kannada'):                   ['Kitt-Kannada', None],
    ('K.<super>2</super>', 'Kannada'):      ['K2-Kannada', None],
    ('UNR', 'Kannada'):                     ['UNR-Kannada', None],
    ('U.P.U.', 'Kannada'):                  ['UPU-Kannada', None],
    # Kodagu
    ('Shanmugam', 'Kodagu'):                ['Shanmugam-Kodagu', None],
    # Kolami
    ('Kin.', 'Kolami'):                     [None, 'kinwat'],
    ('P.', 'Kolami'):                       [None, 'pandh'],
    ('Wagh.', 'Kolami'):                    [None, 'wagh'],
    ('SR.', 'Kolami'):                      ['SR-Kolami', None],
    ('SR', 'Kolami'):                       ['SR-Kolami', None],
    ('Pat.', 'Kolami'):                     ['Pat-Kolami', None],
    ('Pat.,', 'Kolami'):                    ['Pat-Kolami', None],
    ('Hislop', 'Kolami'):                    ['Hislop-Drav', None],
    # Konda
    ('BB', 'Konda'):                        ['BB-Konda', None],
    ('Sova', 'Konda'):                      [None, 'sova'],
    # Koraga
    ('O.', 'Koraga'):                       [None, 'onti'],
    ('T.', 'Koraga'):                       [None, 'tappu'],
    ('M.', 'Koraga'):                       [None, 'mudu'],
    # Kui
    ('K.', 'Kui'):                          ['K-Kui', None],
    # Kurux
    ('Hahn', 'Kurux'):                      ['Hahn-Kurux', None],
    ('Grignard', 'Kurux'):                  ['Grignard-Kurux', None],
    ('Bleses', 'Kurux'):                    ['Bleses-Kurux', None],
    ('Tiga', 'Kurux'):                      ['Tiga-Kurux', None],
    ('BB', 'Kurux'):                        ['Mirdha-Kurux', 'mirdha'],
    # Kuwi
    ('Su.', 'Kuwi'):                        [None, 'sunkar'],
    ('P.', 'Kuwi'):                         [None, 'bisam'],
    ('S.', 'Kuwi'):                         ['S-Kuwi', None],
    ('F.', 'Kuwi'):                         ['F-Kuwi', None],
    ('Isr.', 'Kuwi'):                       ['Isr-Kuwi', None],
    ('Mah.', 'Kuwi'):                       ['Mah-Kuwi', None],
    ('Ṭ.', 'Kuwi'):                         ['Ṭ-Kuwi', None],
    ('Ḍ.', 'Kuwi'):                         ['Ḍ-Kuwi', None],
    # Malayalam
    ('Tiyya', 'Mal'):                       ['Tiyya-Mal', 'tiyya'],
    ('Kauṭ.', 'Mal'):                       ['Kauṭ-Mal', 'OMal'],
    ('Menon', 'Mal'):                       ['Menon-Mal', 'OMal'],
    # Tamil
    ('Tinn.', 'Tam'):                       ['Tinn-Tam', 'tinn'],
    ('Koll.', 'Tam'):                       ['Koll-Tam', 'kollimalai'],
    ('NTD', 'Tam'):                         ['Koll-Tam', 'nancil'],
    ('SATD', 'Tam'):                        ['Koll-Tam', 'southarcot'],
    ('TATD', 'Tam'):                        ['Koll-Tam', 'thanjavur'],
    ('STD', 'Tam'):                         ['Koll-Tam', 'salem'],
    ('CTD', 'Tam'):                         ['Koll-Tam', 'coimbatore'],
    ('Coimbatore', 'Tam'):                  [None, 'coimbatore'],
    ('RS', 'Tam'):                          ['Koll-Tam', 'lowerperak'],
    ('PR', 'Tam'):                          ['PR-Tam', None],
    ('Devanesan', 'Tam'):                   ['Devanesan-Tam', None],
    ('PPTI', 'Tam'):                        ['PPTI-Tam', None],
    ('PN', 'Tam'):                          ['PN-Tam', None],
    ('Ag.', 'Tam'):                         ['Ag-Tam', 'nanv'],
    ('Andronov', 'Tam'):                    ['Andronov-Tam', None],
    ('Annamalai', 'Tam'):                   ['Annamalai-Tam', None],
    ('Ramnad', 'Tam'):                      [None, 'ramnad'],
    ('Asher-Radhakrishnan', 'Tam'):         ['Asher-Radhakrishnan-Tam', None],
    # Telugu
    ('K.', 'Telugu'):                       ['K-Telugu', None],
    ('also', 'Telugu'):                     ['K-also-Telugu', None],
    ('B.', 'Telugu'):                       ['B-Telugu', None],
    ('inscr.', 'Telugu'):                   ['Inscr1-Telugu', 'OTelugu'],
    ('Inscr.', 'Telugu'):                   ['Inscr-Telugu', 'OTelugu'],
    ('Inscr.<super>2</super>', 'Telugu'):   ['Inscr2-Telugu', 'OTelugu'],
    ('<i>VPK</i>', 'Telugu'):               ['VPK-Telugu', None],
    ('Merolu', 'Telugu'):                   ['Merolu-Telugu', 'merolu'],
    # Toda
    ('Sak.', 'Toda'):                       ['Sak-Toda', None],
    # Tulu
    ('B-K.', 'Tulu'):                       ['B-K-Tulu', None],
}

refs = {
    # Gondi
}
