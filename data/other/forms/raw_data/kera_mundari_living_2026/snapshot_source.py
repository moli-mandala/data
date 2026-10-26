"""Acquire the complete public snapshot; never build or install Jambu data."""
import datetime
import gzip
import hashlib
import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path

P = Path(__file__).resolve().parent
URLS = {
    'dictionary.html': 'https://livingdictionaries.app/kera-mundari',
    'about.html': 'https://livingdictionaries.app/kera-mundari/about',
    'catalogue.html': 'https://livingdictionaries.app/dictionaries',
    'terms.html': 'https://livingdictionaries.app/terms',
    'kera-mundari.db.gz': 'https://snapshots.livingdictionaries.app/dictionaries/kera-mundari.db.gz',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Jambu-source-review/1.0'}), timeout=40) as response:
                data = response.read(20_000_001)
                assert len(data) <= 20_000_000, 'Unexpected large source; manual scope review required'
                return data, dict(response.headers), response.status
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    cache = P / 'snapshot'
    cache.mkdir(exist_ok=True)
    receipt = {'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'responses': [],
               'authorization': 'User explicitly confirmed Living Dictionaries is permitted per Sarva, reiterated 2026-09-27.',
               'license': 'No blanket Creative Commons licence inferred. Source terms and dictionary metadata preserved; reuse rests on the explicit user-confirmed permission.',
               'scope': 'Complete published dictionary snapshot and documentary metadata; media binaries not downloaded.'}
    for name, url in URLS.items():
        destination = cache / name
        if destination.exists():
            data = destination.read_bytes()
            headers, status = {}, 'retained prior exact response body'
        else:
            data, headers, status = fetch(url)
            temporary = destination.with_suffix(destination.suffix + '.part')
            temporary.write_bytes(data)
            temporary.replace(destination)
        receipt['responses'].append({'url': url, 'file': str(destination.relative_to(P)), 'status': status,
                                     'bytes': len(data), 'sha256': sha(data), 'headers': headers,
                                     'body_saved_utc': datetime.datetime.fromtimestamp(destination.stat().st_mtime, datetime.timezone.utc).isoformat()})
    compressed = (cache / 'kera-mundari.db.gz').read_bytes()
    is_gzip = compressed.startswith(b'\x1f\x8b')
    raw = gzip.decompress(compressed) if is_gzip else compressed
    assert raw.startswith(b'SQLite format 3\x00')
    path = cache / 'kera-mundari.db'
    if not path.exists():
        if is_gzip:
            path.write_bytes(raw)
        else:
            os.link(cache / 'kera-mundari.db.gz', path)
    else:
        assert path.read_bytes() == raw
    db = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    tables = [x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    receipt['sqlite'] = {'file': str(path.relative_to(P)), 'bytes': len(raw), 'sha256': sha(raw),
                         'response_encoding': 'gzip validated and decompressed' if is_gzip else 'endpoint response body already SQLite despite .gz URL; retained exact received bytes and hardlinked reader path',
                         'integrity_check': 'ok', 'tables': {}}
    for table in tables:
        q = '"' + table.replace('"', '""') + '"'
        cols = [x[1] for x in db.execute('PRAGMA table_info(' + q + ')')]
        receipt['sqlite']['tables'][table] = {'count': db.execute('SELECT count(*) FROM ' + q).fetchone()[0], 'columns': cols}
    receipt['pagination'] = {'method': 'complete public SQLite snapshot, not a paginated result endpoint',
                             'proof': 'Single complete snapshot response; SQLite integrity checked; every physical table enumerated and counted. Lexical export must reconcile to all entries/senses/sentences before source-stage completion.',
                             'page_truncation_possible': False}
    (P / 'snapshot-acquisition.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt['sqlite'], ensure_ascii=False))


if __name__ == '__main__':
    main()
