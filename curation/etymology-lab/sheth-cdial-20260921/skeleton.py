"""Crude Middle-Indo-Aryan consonant-skeleton similarity between a Sanskrit and a Prakrit form.

Used only to hold Prakrit rows whose Sheth Sanskrit 'equivalent' is not shaped like an etymon
(Sheth also prints semantic equivalents, e.g. bhukta for samāṇia). Calibrated so that fewer than
2% of the Prakrit forms CDIAL itself cites under the matched entry fall below the threshold.
"""
import difflib, re, unicodedata

VOWELS = 'aeiou'
CLASS = str.maketrans({'g': 'k', 'c': 'k', 'j': 'k', 'd': 't', 'b': 'p', 'v': 'p',
                       'y': '', 'r': '', 'h': '', 'ñ': 'n', 'ŋ': 'n'})

def _base(s):
    s = unicodedata.normalize('NFD', s.lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return unicodedata.normalize('NFC', s).replace('ʰ', '').replace('°', '').replace('*', '')

def sk_skeletons(form):
    s = _base(form).replace('r̩', 'i').replace('ṁ', 'n').replace('ḣ', '')
    s = re.sub(r'ks', 'k', s); s = re.sub(r'ts', 'c', s); s = re.sub(r'dv', 'b', s)
    s = re.sub(r'^y', 'j', s)                     # initial y- > Pk j-
    alts = [s, re.sub(r'jn', 'j', s)]             # jñ > ṇṇ or jj
    s = re.sub(r'jn', 'n', s)
    s = re.sub(r's(?=[tkpcmn])', '', s)          # sth/st/ṣṭ/sm/sn/śc/sk: sibilant lost
    s = re.sub(r'(?<=[kctpgdjb])[vy]', '', s)    # tv, dy, ty, ky: glide lost in the cluster
    out = []
    for a in (s, alts[1]):
        out.append(a.translate(CLASS))
        out.append(re.sub(r'(?<=[aeiou])[kgcjtdpb](?=[aeiou])', '', a).translate(CLASS))
    return [re.sub(r'(.)\1+', r'\1', re.sub(f'[{VOWELS}]', '', x)) for x in out]

def pk_skeleton(form):
    s = _base(form).replace('ṁ', 'n').replace('m̐', 'n')
    s = s.translate(CLASS)
    return re.sub(r'(.)\1+', r'\1', re.sub(f'[{VOWELS}]', '', s))

def similarity(sk_form, pk_form):
    pk = pk_skeleton(pk_form)
    return max(difflib.SequenceMatcher(None, sk, pk).ratio() for sk in sk_skeletons(sk_form))
