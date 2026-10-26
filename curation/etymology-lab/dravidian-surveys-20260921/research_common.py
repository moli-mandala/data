"""Candidate matching conventions only; never rewrite source transcriptions."""
import re
import unicodedata

def fold(s):
    s=s.lower().translate(str.maketrans({'ɨ':'u','ɯ':'u','ɛ':'e','ɔ':'o','ə':'a','ʌ':'a','ɪ':'i','ʊ':'u','ŋ':'ng','ⁱ':'i','ɡ':'g','ɱ':'m'}))
    s=''.join(c for c in unicodedata.normalize('NFD',s) if not unicodedata.combining(c))
    return re.sub(r'[\sːˑ·ʔ\'’]','',s)

def gloss(s):
    return s.lower().replace('ɡ','g').strip()
