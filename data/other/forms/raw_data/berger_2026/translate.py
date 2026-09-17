"""Translate changed definitions with the pinned Argos model, one CPU thread.

Only ctranslate2 and sentencepiece are required, in an isolated environment.
The source scan's published SHA and the model archive SHA are in README.md.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    import ctranslate2
    import sentencepiece
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('berger_translation_base', here.parent / 'berger_cleanup.py')
    b = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = b
    spec.loader.exec_module(b)
    requests = json.loads(args.requests.read_text())
    output = here / 'translations.json'
    results = json.loads(output.read_text()) if output.exists() else {}
    pending = [(key, value) for key, value in requests.items() if key not in results]
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.model / 'sentencepiece.model'))
    translator = ctranslate2.Translator(str(args.model / 'model'), device='cpu',
                                       inter_threads=1, intra_threads=1)
    for start in range(0, len(pending), 8):
        batch = pending[start:start + 8]
        source = [b.translation_source(type('Item', (), {'gloss_de': de})()) for _, de in batch]
        encoded = [tokenizer.encode(s, out_type=str) for s in source]
        if any(len(s) > 1024 for s in encoded):
            raise ValueError('Definition exceeds translation input bound; review without truncation')
        translated = translator.translate_batch(encoded, replace_unknowns=True,
                                                 max_input_length=1024, max_decoding_length=1024)
        for (key, de), src, result in zip(batch, source, translated):
            assert key == hashlib.sha256(de.encode()).hexdigest()
            en = b.clean_translation(src, tokenizer.decode(result.hypotheses[0]))
            results[key] = {'german': de, 'english': en, 'method': 'Argos de_en 1.3; machine-translated-unreviewed'}
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output)
        if start % 80 == 0:
            print(f'{min(start + 8, len(pending))}/{len(pending)} definitions translated', flush=True)


if __name__ == '__main__':
    main()
