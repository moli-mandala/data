# data files -> transcription conversion filess
mapping = {
    # These older imports were manually normalized when they were ingested.  Route them through
    # an explicit preservation profile so that their transcription contract is checked rather
    # than silently bypassing the sound-profile layer.
    'dhivehi': 'house', 'gawri': 'house', 'kalkoti': 'house', 'khetrani': 'house',
    'kholosi': 'house', 'konkani': 'house', 'kundalshahi': 'house', 'kvari': 'house',
    'zadjali': 'house', 'arora': 'house', 'punjabi': 'house', 'sindhic': 'house',
    'pashai': 'house', 'paranavitana': 'house', 'tulpule': 'house', 'wolf': 'house',
    'southworth': 'southworth-marathi',
    'ghatage': 'ghatage',
    'patyal': 'cdial', 'thari': 'cdial',
    'vaagri': 'vaagri', 'cdial': 'cdial', 'palula': 'liljegren',
    'strand': 'strand', 'strand2': 'strand', 'strand3': 'strand', 'wadiyara': 'wadiyara',
    'northern': 'northern', 'toulmin': 'toulmin', 'chattisgarhi': 'chattisgarhi',
    'rajasthani': 'rajasthani', 'old_punjabi': 'cdial', 'bundeli': 'chattisgarhi',
    'tharu': 'cdial', 'kannauji': 'rajasthani', 'tharu2': 'chattisgarhi', 'shina': 'liljegren', 'berger': 'berger',
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
    'ali': 'brahui',
    'dewas': 'dewas-rai',
    'hajong': 'hajong-survey',
    'santali': 'santali-cluster',
    'sampang': 'sampang',
    'mewahang': 'mewahang',
    'chhulung': 'chhulung',
    'magahi': 'magahi-survey',
    'magar': 'magar-2024',
    # Lexibank LSI supplies canonical CLTS segments separately from Grierson's
    # historical spelling. Convert those segments to house transcription.
    'grierson': 'lsi',
    # Mundlay's scan and the Nagaraja-derived Wiktionary list use closely
    # related Nihali transcription; the shared profile preserves source
    # diacritics while normalizing colon length and w/v.
    'mundlay': 'nihali', 'nagaraja': 'nihali',
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
