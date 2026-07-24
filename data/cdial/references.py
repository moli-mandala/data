"""Bibliographic abbreviations used in CDIAL's main volume and addenda."""

import re
import unicodedata


# Main preface, pp. xvi-xx, followed by the addenda preface, p. xxii. Cross-reference-only
# aliases ("See ...", "or ...") are included because they occur literally in dictionary entries.
REFERENCE_ABBREVS = {
    "ABORI", "AO", "Apte", "ArchLing", "ArchOr", "Bailey ShGr", "Bartholomae Airb",
    "BB", "BDCRI", "B.D. Jain Phon. Panj", "Beames", "BEFEO", "BelvalkarVol",
    "Bender Nal", "Bhal-dial", "BHS", "BKPD", "Bloch As", "Bloch IA", "Bloch LM",
    "Bloch Tsig", "BR", "Bray Brah", "Brough Dhp", "BSBU", "BSL", "BSOAS", "BSOS",
    "Buddruss", "G.B", "Buddruss Kan", "Buddruss Woṭ", "Burrow KharDoc", "Burrow SkLg",
    "Childers DPL", "CII", "COJ", "CPD", "CR", "Dave", "Dave GujLg", "DED", "DGW",
    "DSL", "EA", "EGS", "EI", "El", "EOL", "ES", "EVP", "EWA", "EZ",
    "FestschrHerzfeld", "FestskrBroch", "Finck AZ", "FOK", "GBVoc", "Ge",
    "Geiger Pali Gr", "Geiger PLS", "GHA", "G.M", "Grahame Bailey", "Grām",
    "Grierson BPL", "Grierson KD", "Grierson MthLg", "Grierson PiśLg", "Grierson Tor",
    "GS", "Gupta Grām", "GWZS", "Hendriksen", "Hettiaratchi Indeclinables",
    "Hettiaratchi Vowels", "HJ", "Horn NPE", "HŚS", "Hübschmann ArmGr", "Hultzsch As",
    "IEW", "IF", "IIFL", "IIJ", "IL", "JA", "JAOS", "JBORS", "JBRAS", "JGIS",
    "JGLS", "Joshi Tîka", "JRASB", "Ju", "JVEDSt", "Katre FKo", "Kern Toev", "KharI",
    "Kuiper Myth", "Kuiper PMWS", "KZ", "Lakṣmīdhar Pad", "Laufer", "LM", "Lor",
    "Lorimer BurLg", "Lorimer ḌumLg", "LSI", "Lüders BSBU", "Lüders PhilInd",
    "Maya Singh PD", "Mayrhofer HPa", "Miklosich Mund", "Miscellany", "MO",
    "Molesworth MD", "Morgenstierne", "MSL", "MW", "ND", "Neisser WGRV", "NIA",
    "NOGaw", "NOPhal", "NTS", "ODBL", "Panse Jñān", "PhonPj", "Pischel GrPk",
    "Platts UD", "PMWS", "POBh", "Pokorny", "Pre-Aryan", "PSM", "PTSD", "PW",
    "Raghu Vira", "Rajwade MDh", "Rep1", "Rep2", "RM", "RoczOrj", "Sã̄ḍesrā Phāg",
    "Saksena", "Sampson", "SBAW", "Schmidt Nachtr", "SED", "Shirt SD", "SigGr",
    "Siiger", "S. K. Chatterji", "SSS", "Stack SD", "Stein RājatTrans", "StudII",
    "Suman Braj", "S. Varma", "ThomasEIS", "TPS", "Tulpule OMR", "Uhlenbeck",
    "Varma BhalDial", "Wackernagel AiGr", "Whitney SkGr", "Wijeratne", "Wolf GWZS",
    "Woolner Gloss", "WP", "WR", "WZKSOA", "ZDMG", "ZII",
    # Addenda bibliography.
    "AFD", "AKŚ", "BKhoT", "Burrow Shwa", "ColPa", "C.Shackle", "DEDS",
    "Emeneau Sk. bhōgin-", "EVSh", "Him.I", "IB", "KhubSD", "LFG", "LKK", "LNH",
    "LOL", "LStH", "Master GrOM", "Morgenstierne ID", "Risley", "RTMV1", "RTMV2",
    "ShahidullahPresVol", "S.M.Katre", "SN", "ŚSB", "SternbachVol", "SZII", "Tau",
    "Vīsaḷa", "W",
}

# These labels are used as Sanskrit dictionary/lexicographer attestations. They belong in Tags,
# rather than in the row's bibliography relation.
ATTESTATION_ONLY = {"Apte", "Gal", "MW", "W"}

# The addenda entries do not always reproduce the typography of their own bibliography.
REFERENCE_ALIASES = {
    "C. Shackle": "C.Shackle",
    "Emeneau Sk. bhōgin-": "Emeneau Sk. bhōgin-",
    "RTMV¹": "RTMV1",
    "RTMV²": "RTMV2",
    "S. M. Katre": "S.M.Katre",
}


def _pattern(abbrev):
    bits = []
    for char in abbrev:
        if char == ".":
            bits.append(r"\.?")
        elif char.isspace():
            bits.append(r"\s+")
        else:
            bits.append(re.escape(char))
    return r"(?<![\w])" + "".join(bits) + r"(?![\w])"


_SPELLINGS = {
    abbrev: abbrev for abbrev in REFERENCE_ABBREVS - ATTESTATION_ONLY
} | REFERENCE_ALIASES
_ORDERED = sorted(_SPELLINGS, key=len, reverse=True)
_REFERENCE_RE = re.compile(
    "|".join(f"(?P<r{i}>{_pattern(abbrev)})" for i, abbrev in enumerate(_ORDERED))
)
_GROUP_TO_ABBREV = {f"r{i}": _SPELLINGS[abbrev] for i, abbrev in enumerate(_ORDERED)}


def extract_reference_ids(text):
    """Return distinct CDIAL bibliography IDs found in *text*, in citation order."""
    text = re.sub(r"<[^>]+>", "", unicodedata.normalize("NFC", text or ""))
    found = []
    for match in _REFERENCE_RE.finditer(text):
        abbrev = _GROUP_TO_ABBREV[match.lastgroup]
        if abbrev not in found:
            found.append(abbrev)
    return found


def source_field(notes):
    """Build the raw-CDIAL Source cell, retaining CDIAL itself as primary provenance."""
    return ";".join(["CDIAL", *extract_reference_ids(notes)])
