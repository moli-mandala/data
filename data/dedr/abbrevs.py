abbrevs = {
    'Ta': 'Tam',
    'Ma': 'Mal',
    'Ir': 'Irula',
    'PālKu': 'PaluKurumba',
    'ĀlKu': 'AluKurumba',
    'Kurub': 'BettaKurumba',
    'Ko': 'Kota',
    'To': 'Toda',
    'Ka': 'Kannada',
    'Koḍ': 'Kodagu',
    'Tu': 'Tulu',
    'Bel': 'Belari',
    'Kor': 'Koraga',
    'Te': 'Telugu',
    'Kol': 'Kolami',
    'Nk': 'Naikri',
    'Pa': 'Parji',
    'Ga': 'Gadaba',
    'Go': 'Gondi',
    'Konḍa': 'Konda',
    'Pe': 'Pengo',
    'Manḍ': 'Manda',
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
    'H\.': 'H',
    'Beng': 'B',
    'Pkt': 'Pk',
    'Halbi': 'hal',
    'O\.': 'Or',
    'Or\.': 'Or',
    'Nahali': 'Ni',
    'Apabhraṃśa': 'Ap',
    'Guj\.': 'G'
}

replacements = {
    'Koya Su.': 'KoyaSu.',
    'Koya T.': 'KoyaT.',
}

dialects = {
    # Gadaba
    ('P.', 'Gadaba'):                       [None, 'pottangi'],
    ('Oll.', 'Gadaba'):                     ['Oll-Gadaba', None],
    ('S.<super>3</super>', 'Gadaba'):       ['S3-Gadaba', None],
    ('S.', 'Gadaba'):                       ['S-Gadaba', None],
    ('S', 'Gadaba'):                        ['S-Gadaba', None],
    ('S.<super>2</super>', 'Gadaba'):       ['S2-Gadaba', None],
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
    ('M.', 'Gondi'):                        ['M-Gondi', 'maria'],
    ('L.', 'Gondi'):                        ['L-Gondi', 'maria'],
    ('SR.', 'Gondi'):                       ['SR-Gondi', 'adil'],
    ('A.', 'Gondi'):                        ['A-Gondi', 'adil'],
    ('KoyaT.', 'Gondi'):                    ['KoyaT-Gondi', 'gommu'],

    ('LuS.', 'Gondi'):                      ['LuS-Gondi', None],
    ('<i>DGG</i>', 'Gondi'):                ['DGG-Gondi', None],
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
    ('SR.', 'Kolami'):                      ['SR-Kolami', None],
    ('SR', 'Kolami'):                       ['SR-Kolami', None],
    ('Pat.', 'Kolami'):                     ['Pat-Kolami', None],
    ('Pat.,', 'Kolami'):                    ['Pat-Kolami', None],
    # Konda
    ('BB', 'Konda'):                        ['BB-Konda', None],
    # Koraga
    ('O.', 'Koraga'):                       [None, 'onti'],
    ('T.', 'Koraga'):                       [None, 'tappu'],
    ('M.', 'Koraga'):                       [None, 'mudu'],
    # Kui
    ('K.', 'Kui'):                          ['K-Kui', None],
    # Kurux
    ('Hahn', 'Kurux'):                      ['Hahn-Kurux', None],
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
    # Tamil
    ('Tinn.', 'Tam'):                       ['Tinn-Tam', 'tinn'],
    # Telugu
    ('K.', 'Telugu'):                       ['K-Telugu', None],
    ('also', 'Telugu'):                     ['K-also-Telugu', None],
    ('B.', 'Telugu'):                       ['B-Telugu', None],
    ('Inscr.', 'Telugu'):                   ['Inscr-Telugu', None],
    ('<i>VPK</i>', 'Telugu'):               ['VPK-Telugu', None],
    # Tulu
    ('B-K.', 'Tulu'):                       ['B-K-Tulu', None],
}

refs = {
    # Gondi
}
