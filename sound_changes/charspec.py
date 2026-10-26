"""Diagnostic innovation characters for innovations.py, read on aligned OIA → reflex pairs.

Each definition takes (w, k, o, O): the deaccented etymon tokens, a position, the aligned reflex
segment there and the full alignment map; it returns None (context not applicable) or whether the
reflex shows the innovative outcome.

"""
import collections
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sca, evaluate as ev
from align import Seg, align

MIN = 6
V = lambda s: s is not None and sca.kind(s) == 'V'
C = lambda s: s is not None and sca.kind(s) == 'C'
d = sca._deacc
STOPS = set('k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ'.split())
VASP = {'bʰ', 'dʰ', 'gʰ', 'jʰ', 'ḍʰ'}
NAS = set('n ṇ ñ ŋ m ṁ'.split())
LONG = lambda s: s is not None and (sca.feat(s, 'Long') or 'ː' in s)


def ctx(w, k):
    L = w[k - 1] if k > 0 else '#'
    R = w[k + 1] if k + 1 < len(w) else '#'
    return L, R


# each character: (name, test(w, k, out, outs) -> None (not applicable) | bool (innovative?))
# w: deaccented etymon tokens; outs: dict idx -> aligned reflex token or '∅'
def char_defs():
    D = []

    def add(name, fn):
        D.append((name, fn))

    def intervoc(seg_set):
        return lambda w, k: d(w[k]) in seg_set and V(ctx(w, k)[0]) and V(ctx(w, k)[1])
    add('ś > s', lambda w, k, o, O: (o == 's') if w[k] == 'ś' and not C(ctx(w, k)[1]) else None)
    add('s > ś (Magadhan)', lambda w, k, o, O: (o == 'ś') if w[k] == 's' and not C(ctx(w, k)[1]) else None)
    add('s > h', lambda w, k, o, O: (o in ('h', '∅')) if w[k] == 's' and V(ctx(w, k)[1]) else None)
    add('ṣ kept distinct', lambda w, k, o, O: (o in ('ṣ', 'ś', 'x')) if w[k] == 'ṣ' and not C(ctx(w, k)[1]) else None)
    def ks(target):
        def f(w, k, o, O):
            if w[k] == 'k' and k + 1 < len(w) and w[k + 1] == 'ṣ':
                o2 = O.get(k + 1, '∅')
                return o2 in target or o in target
            return None
        return f
    add('kṣ > (k)kʰ', ks({'kʰ', 'x'}))
    add('kṣ > (c)cʰ', ks({'cʰ', 'c'}))
    add('kṣ > ʦ̣(ʰ)', ks({'ʦ̣ʰ', 'ʦ̣', 'ʦ̣h'}))
    add('int. t lost', lambda w, k, o, O: (o in ('∅', 'y', 'v', 'h')) if intervoc({'t'})(w, k) else None)
    add('int. t voiced (d/ð)', lambda w, k, o, O: (o in ('d', 'ð', 'ḍ')) if intervoc({'t'})(w, k) else None)
    add('int. t > l/r', lambda w, k, o, O: (o in ('l', 'r', 'ṛ', 'ḷ')) if intervoc({'t'})(w, k) else None)
    add('int. k > ∅/y', lambda w, k, o, O: (o in ('∅', 'y')) if intervoc({'k'})(w, k) else None)
    add('int. k > g/ɣ', lambda w, k, o, O: (o in ('g', 'ɣ')) if intervoc({'k'})(w, k) else None)
    add('int. p > v/b', lambda w, k, o, O: (o in ('v', 'b', 'w')) if intervoc({'p'})(w, k) else None)
    add('int. aspirate > h', lambda w, k, o, O: (o == 'h') if intervoc({'kʰ', 'gʰ', 'tʰ', 'dʰ', 'pʰ', 'bʰ'})(w, k) else None)
    add('int. ṭ/ḍ > ṛ', lambda w, k, o, O: (o in ('ṛ', 'ṛʰ')) if intervoc({'ṭ', 'ḍ'})(w, k) else None)
    add('int. ṭ/ḍ > r', lambda w, k, o, O: (o == 'r') if intervoc({'ṭ', 'ḍ'})(w, k) else None)
    add('int. ṭ/ḍ > ḷ/l', lambda w, k, o, O: (o in ('ḷ', 'l')) if intervoc({'ṭ', 'ḍ'})(w, k) else None)
    add('int. l > r', lambda w, k, o, O: (o == 'r') if intervoc({'l'})(w, k) else None)
    add('int. l > ḷ', lambda w, k, o, O: (o == 'ḷ') if intervoc({'l'})(w, k) else None)
    add('int. n > ṇ', lambda w, k, o, O: (o == 'ṇ') if intervoc({'n'})(w, k) else None)
    add('y- > j-', lambda w, k, o, O: (o in ('j', 'z', 'ʣ', 'ź')) if w[k] == 'y' and k == 0 else None)
    add('v- > b-', lambda w, k, o, O: (o == 'b') if w[k] == 'v' and k == 0 else None)
    add('c > ʦ', lambda w, k, o, O: (o in ('ʦ', 'ʦʰ')) if w[k] == 'c' and not C(ctx(w, k)[0]) else None)
    add('c > s (deaffrication)', lambda w, k, o, O: (o in ('s', 'ś')) if w[k] == 'c' and not C(ctx(w, k)[0]) else None)
    add('initial voiced asp. > deaspirated', lambda w, k, o, O: (o in ('b', 'd', 'g', 'j', 'ḍ')) if w[k] in VASP and k == 0 else None)
    add('initial voiced asp. > voiceless', lambda w, k, o, O: (o in ('p', 't', 'k', 'c', 'ṭ', 'pʰ', 'tʰ', 'kʰ', 'cʰ', 'ṭʰ')) if w[k] in VASP and k == 0 else None)

    def cluster_first(o_first, o_second):
        pass
    # stop + stop (kt, pt, ...): first stop assimilated (geminate or dropped) vs kept
    def stopstop(w, k, o, O):
        if w[k] in STOPS and k + 1 < len(w) and w[k + 1] in STOPS and w[k] != w[k + 1] and k > 0:
            return o not in ('k', 'p', 'kʰ', 'pʰ', 'g', 'b') if w[k] in ('k', 'p', 'g', 'b') else None
        return None
    add('stop+stop assimilated', stopstop)
    def tr(target):
        def f(w, k, o, O):
            if w[k] in ('t', 'd', 'tʰ', 'dʰ') and k + 1 < len(w) and w[k + 1] == 'r':
                o2 = O.get(k + 1, '∅')
                return target(o, o2)
            return None
        return f
    add('tr/dr kept', tr(lambda o, o2: o2 == 'r' and o in ('t', 'd', 'tʰ', 'dʰ')))
    add('tr/dr > affricate', tr(lambda o, o2: o in ('ʦ̣', 'ʣ̣', 'c', 'j', 'ʦ', 'ʣ', 'ʦ̣ʰ') or o2 in ('ʦ̣', 'ʣ̣', 'c', 'j')))
    add('tr/dr > lateral', tr(lambda o, o2: o in ('ɬ', 'l', 'ṭ͡ḷ', 'ḍ͡ḷ') or o2 in ('ɬ', 'l', 'ḷ')))
    def st(w, k, o, O):
        if w[k] in ('s', 'ṣ', 'ś') and k + 1 < len(w) and w[k + 1] in ('t', 'ṭ', 'k', 'p') and k > 0:
            o2 = O.get(k + 1, '∅')
            return 'ʰ' in (o2 or '') or 'ʰ' in (o or '')
        return None
    add('s+stop > aspirate', st)
    def st_kept(w, k, o, O):
        if w[k] in ('s', 'ṣ', 'ś') and k + 1 < len(w) and w[k + 1] in ('t', 'ṭ', 'k', 'p') and k > 0:
            return o in ('s', 'ṣ', 'ś', 'h', 'x')
        return None
    add('s+stop: sibilant kept', st_kept)
    def nt(w, k, o, O):
        if w[k] in ('n', 'ṇ', 'ṁ', 'm', 'ŋ') and k + 1 < len(w) and w[k + 1] in ('t', 'ṭ', 'k', 'p'):
            o2 = O.get(k + 1, '∅')
            return o2 in ('d', 'ḍ', 'g', 'b')
        return None
    add('post-nasal voicing', nt)
    def nasv(w, k, o, O):
        if w[k] in NAS and k + 1 < len(w) and w[k + 1] in STOPS and k > 0 and V(w[k - 1]):
            ov = O.get(k - 1, '∅')
            return '̃' in __import__('unicodedata').normalize('NFD', ov or '') and o in ('∅', None)
        return None
    add('NC > nasal vowel', nasv)
    def cl(w, k, o, O):
        # short a of the first syllable before an assimilating cluster (stop+stop, r+C, C+r)
        if k == 0 or not C(w[k - 1]) and k != 0:
            pass
        if d(w[k]) == 'a' and all(not V(x) for x in w[:k]) and k + 2 < len(w) and C(w[k + 1]) and C(w[k + 2]) \
                and w[k + 1] not in NAS and w[k + 1] != w[k + 2]:
            return O.get(k) in ('ā', 'ā̃')
        return None
    add('compensatory lengthening', cl)
    add('final -a lost', lambda w, k, o, O: (o == '∅') if d(w[k]) == 'a' and k == len(w) - 1 and len(w) > 2 else None)
    def rv(outs):
        return lambda w, k, o, O: (o in outs) if d(w[k]) == 'r̩' else None
    add('r̩ > i', rv({'i', 'ī'}))
    add('r̩ > a', rv({'a', 'ā'}))
    add('r̩ > u', rv({'u', 'ū'}))
    add('ai > e', lambda w, k, o, O: (o in ('e', 'ē', 'ɛ')) if d(w[k]) == 'ai' else None)
    add('long vowels shortened (length loss)', lambda w, k, o, O: (o in ('a', 'i', 'u')) if d(w[k]) in ('ā', 'ī', 'ū') and k > 0 and V(w[k]) and not (k + 2 < len(w) and C(w[k + 1]) and C(w[k + 2])) else None)
    return D


def aligned(w, target):
    es = [Seg(x) for x in w]
    for k, s in enumerate(es):
        s.idx = k
    out = {}
    for a, b in align(es, [Seg(x) for x in target]):
        if a is not None:
            out[a.idx] = b.raw if b is not None else '∅'
    return out


