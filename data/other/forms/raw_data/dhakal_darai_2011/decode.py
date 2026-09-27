"""Decode chapter-native font glyphs; not a house transcription profile."""
import json
from pathlib import Path
import unicodedata

HERE = Path(__file__).resolve().parent
IPA93 = {'\uf0ab': 'ə', '\uf048': 'ʰ', '\uf067': 'ɡ', '\uf029': '\u0303', '\uf04e': 'ŋ'}


def decode_char(char):
    text, font = char['text'], char['fontname']
    if 'SILDoulosIPA93' in font:
        text = ''.join(IPA93.get(c, c) for c in text)
    elif text == '\uf020' and 'TimesNewRoman' in font:
        text = ' '
    if any(0xe000 <= ord(c) <= 0xf8ff for c in text):
        raise ValueError(f'Unmapped private-use glyph {text!r} in {font}')
    return text


def content_order_text(chars):
    # PDF content order preserves zero-width nasal marks before the next glyph;
    # geometric text sorting can erroneously attach them to a following consonant.
    return unicodedata.normalize('NFC', ''.join(map(decode_char, chars)))


if __name__ == '__main__':
    output = HERE / 'decoded'
    output.mkdir(exist_ok=True)
    for page in sorted((HERE / 'evidence').glob('p*-glyphs.json')):
        chars = json.loads(page.read_text())
        (output / page.name.replace('-glyphs.json', '.txt')).write_text(content_order_text(chars) + '\n')
    print('Decoded all 35 chapter pages; no lexical rows installed')
