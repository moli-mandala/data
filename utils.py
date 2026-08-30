# data files -> transcription conversion filess
mapping = {
    # These older imports were manually normalized when they were ingested.  Route them through
    # an explicit preservation profile so that their transcription contract is checked rather
    # than silently bypassing the sound-profile layer.
    'dhivehi': 'house', 'gawri': 'house', 'khetrani': 'house',
    'kholosi': 'house', 'konkani': 'house', 'kundalshahi': 'house', 'kvari': 'house',
    # Both Kalkoti sources are read in the broad transcription customary among
    # Shina scholars, including its tone marking, rather than typed straight into
    # house transcription as the 2022 snapshot was.
    'kalkoti': 'kalkoti', 'hultman': 'kalkoti',
    'zadjali': 'house', 'arora': 'house', 'punjabi': 'house', 'sindhic': 'house',
    'pashai': 'house', 'paranavitana': 'house', 'tulpule': 'house', 'wolf': 'house',
    'southworth': 'southworth-marathi',
    'emeneau': 'emeneau-brahui',
    'burrow': 'dedr',
    'ghatage': 'ghatage',
    'patyal': 'cdial', 'thari': 'cdial',
    'vaagri': 'vaagri', 'cdial': 'cdial', 'palula': 'liljegren',
    'strand': 'strand', 'strand2': 'strand', 'strand3': 'strand', 'wadiyara': 'wadiyara',
    'northern': 'northern', 'toulmin': 'toulmin', 'chattisgarhi': 'chattisgarhi',
    'rajasthani': 'rajasthani', 'old_punjabi': 'cdial', 'bundeli': 'chattisgarhi',
    'tharu': 'cdial', 'kannauji': 'rajasthani', 'tharu2': 'sil-western-tharu', 'shina': 'liljegren', 'berger': 'berger',
    # Buddruss explicitly contrasts dental c, palatal č, and retroflex c̣; preserve that
    # three-way distinction instead of routing this source through Berger's orthography.
    'buddruss': 'buddruss-grangali',
    'markodi': 'markodi',  # Toulmin-style Dravidian consonants + Markodi vowels/length
    'kalasha': 'kalasha',  # Trail & Cooper's Kalasha orthography
    'zoller': 'zoller',  # Zoller's tonal Indus Kohistani pseudo-IPA -> transcription + IPA pronunciation
    'bashir': 'khowar',  # Bashir 2023 Khowar romanisation -> house transcription + phonemic IPA
    # SSNP's extractor decodes the legacy font to IPA; this profile then maps
    # that IPA into Jambu's house transcription while retaining IPA separately.
    'ssnp': 'ssnp',
    # Andersen's Ashokan forms use conventional digraph aspiration and
    # underdotted anusvara; normalize both to Jambu's house transcription.
    'andersen': 'andersen',
    'schmidt': 'schmidt-kashmiri',
    'drasi': 'drasi',
    # Degener 2008 writes Gilgit Shina in Berger's orthography with doubled
    # vowels for length and mora-positioned acutes for the pitch contrast.
    'degener': 'degener-shina',
    # Buddruss 1996 uses the same quantity/tone system for Gilgit Shina, with
    # a few source-specific optional-form and nasal-vowel spellings.
    'buddruss-shina': 'buddruss-shina',
    'yoshioka': 'yoshioka',
    'gandhari': 'gandhari',
    'kullui': 'kullui',
    'bhaskararao': 'toda',
    'hockings': 'badaga-hockings',
    'rabha': 'rabha',
    'eastern': 'eastern-magar',
    'western': 'western-tamang',
    'humla': 'humla',
    'gurung': 'gurung',
    'dotyali': 'dotyali',
    'kudiya': 'kudiya',
    'majhi': 'majhi-bote',
    'majhi-bote': 'majhi-bote',
    'kochila': 'kochila-tharu',
    'pyangaun': 'pyangaun-newar',
    'maikoti': 'maikoti-kham',
    'thakali': 'thakali',
    'mustang': 'mustang-loke',
    'kurux': 'kurux-nepal',
    'north': 'north-gorkha',
    'weinreich': 'weinreich-domaaki',
    # Boretzky & Igla write Romani in a Balkanist transcription with two affricate
    # series (č/dž beside ć/dź), the Vlax rhotic ř and schwa.
    'boretzky': 'boretzky-romani',
    'ali': 'brahui',
    'dewas': 'dewas-rai',
    'hajong': 'hajong-survey',
    'santali': 'santali-cluster',
    'sampang': 'sampang',
    'mewahang': 'mewahang',
    'chhulung': 'chhulung',
    'magahi': 'magahi-survey',
    'magar': 'magar-2024',
    # Merriam et al.'s reconstruction database deliberately combines the source
    # notation of three scholars. Preserve that notation through an explicit
    # identity profile rather than imposing a false common phonological analysis.
    'merriam': 'merriam-reconstruction',
    # NurED Form templates use the source's own carefully diacritized
    # Nuristani transcription. Preserve it losslessly in a dedicated profile.
    'nured': 'nured',
    # Lexibank LSI supplies canonical CLTS segments separately from Grierson's
    # historical spelling. Convert those segments to house transcription.
    'grierson': 'lsi',
    # Mundlay's scan and the Nagaraja-derived Wiktionary list use closely
    # related Nihali transcription; the shared profile preserves source
    # diacritics while normalizing colon length and w/v.
    'mundlay': 'nihali', 'nagaraja': 'nihali',
    # Zubair Torwali's student dictionary prints source IPA with ASCII colon
    # length; retain it in Phonemic and normalize only the display Form.
    'torwali': 'torwali-student',
    # Knobloch's Sauji grammar sketch mixes broad IPA (phonology tables) with a
    # simplified Indo-Aryanist transcription (everything else); one profile reads
    # both. make_cldf.py also routes it by citation key.
    'knobloch': 'knobloch-sauji',
    # Beine's 46 Gondi survey word lists, digitized by Rama et al., are Unicode IPA;
    # the profile maps them onto Jambu's Dravidianist house transcription while the
    # source IPA is retained in Original and Phonemic.
    'gondi': 'gondi-beine',
    # Beine's twelve Bhatri/Halbi/Oriya survey word lists are printed in his own
    # "modified IPA": a raised wedge for retroflexion, a superscript n for dentality,
    # and look-alike letters with an under-bar for the central vowels. make_cldf.py
    # also routes this source by citation key.
    'beine': 'beine-bhatri',
    # SEAlang's structured Pinnow index supplies Unicode source IPA. Preserve it
    # exactly while retaining a dedicated, exhaustively tested source profile.
    'pinnow': 'pinnow-munda',
    # Munda's structured thesis index likewise supplies source Unicode for
    # Proto-Kherwarian, pre-Mundari, and Santali comparison records.
    'munda': 'munda-proto-kherwarian',
    'zide': 'zide-sora-juray',
    'bhattacharya': 'bhattacharya-bonda',
    # Bahl's keyed Korwa vocabulary is Unicode source transcription.  The full
    # dictionary supersedes the secure BAHL excerpts previously carried in the
    # Proto-Munda seed while preserving its own stable source-record identity.
    'bahl': 'bahl-korwa',
}

# superscript forms of letters
superscript = {
    'a': 'ᵃ', 'e': 'ᵉ', 'i': 'ᶦ',
    'o': 'ᵒ', 'u': 'ᵘ', 'ü': 'ᵘ̈',
    'y': 'ʸ', 'ə': 'ᵊ', 'ŭ': 'ᵘ̆',
    'z': 'ᶻ', 'gy': 'ᵍʸ', 'h': 'ʰ',
    'ŕ': 'ʳ́', 'ĕ': 'ᵉ̆', 'n': 'ⁿ'
}

# changing language ids
change = {
    'khaś': 'khash',
    'Māl': 'Malw',
    'Brah': 'Brahui',
    'Drav': 'PDr',
    'Ga': 'Gadaba',
    'Kan': 'Kannada',
    'Kol': 'Kolami',
    'Kur': 'Kurux',
    'Mal': 'Malayalam',
    'Nk': 'Naikri',
    'Prj': 'Parji',
    'Tam': 'Tamil',
    'Tel': 'Telugu',
    'Tu': 'Tulu',
    'Go': 'Gondi',
    'OIA': 'Sk',
    'J': 'kiũth',
    'mald': 'Md',
    'kua': 'kvar',
    'Sant': 'sa',
    'Arb': 'Ar',
    'Prs': 'Pers',
    'Arb-Prs': 'Ar',
    'gamb': 'Gmb',
    'Kmd': 'Kamd',
    'Kan': 'Kannada',
    'Tam': 'Tamil',
    'Tel': 'Telugu',
    'Mal': 'Malayalam',
    'Ga': 'Gadaba',
    'Kol': 'Kolami',
    'Kur': 'Kurux',
    'Nk': 'Naikri',
    'Prj': 'Parji',
    'Sant': 'sa',
    'Tu': 'Tulu',
    'Brah': 'Brahui',
    'Domaaki': 'D',
    'Dras': 'dr',
    'Gilgit': 'gil',
    'Punial': 'punl',
    'Palas': 'pales',
    'pach': 'pch',
    'Urdu': 'H',
    'Pashto': 'Psht',
}
