"""Snapshot bounded public CFEL Koda searches; propose, never install, lexical rows."""

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
ENDPOINT = "https://cfelvb.in/api/dictionary/getDictionaryData.php"


def query_path(cache: Path, query: str) -> Path:
    return cache / (hashlib.sha256(query.encode("utf-8")).hexdigest() + ".json")


def retrieve(request: Request) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise ValueError("unexpected response size")
            return body
        except HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (TimeoutError, URLError):
            if attempt == 2:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch(cache: Path, limit: int) -> int:
    queries = json.loads((ROOT / "queries.json").read_text())["queries"]
    assert len(queries) == len(set(queries))
    cache.mkdir(parents=True, exist_ok=True)
    count = 0
    for query in queries:
        path = query_path(cache, query)
        if path.exists():
            continue
        if count == limit:
            break
        url = ENDPOINT + "?" + urlencode(
            {"limit": 100, "offset": 0, "search": query, "source": 1, "target": 104}
        )
        body = retrieve(Request(url, data=b"", headers={"User-Agent": "Jambu source review"}))
        payload = json.loads(body)
        if payload == 0:
            payload = {"count": 0, "data": []}
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError(f"unexpected publisher response for {query!r}")
        if payload.get("count", len(payload["data"])) > 100:
            raise ValueError(f"pagination needed for {query!r}")
        record = {
            "query": query,
            "url": url,
            "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "response_sha256": hashlib.sha256(body).hexdigest(),
            "response": payload,
        }
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False) + "\n")
        temp.replace(path)
        count += 1
        print(f"cached {count}: {query} ({len(payload['data'])} results)", flush=True)
        time.sleep(0.3)
    return count


def project(cache: Path) -> list[dict]:
    queries = json.loads((ROOT / "queries.json").read_text())["queries"]
    rows = []
    for query in queries:
        path = query_path(cache, query)
        if not path.exists():
            continue
        cached = json.loads(path.read_text())
        assert cached["query"] == query
        response = cached["response"]
        candidates = []
        for item in response["data"]:
            if item.get("language") != 104:
                continue
            candidates.append(
                {key: item.get(key) for key in (
                    "id", "abs_ref", "word", "word2", "word3", "ipa",
                    "grammatical_category", "domain_name"
                )}
            )
        rows.append({
            "query": query,
            "cache_file": path.name,
            "retrieved_utc": cached["retrieved_utc"],
            "response_sha256": cached["response_sha256"],
            "reported_count": response.get("count"),
            "candidates": candidates,
            "status": "publisher correspondence proposal; not an accepted Koda attestation",
        })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--fetch", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert 0 <= args.fetch <= 50, "bounded batches only"
    if args.fetch:
        fetch(args.cache, args.fetch)
    rows = project(args.cache)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"{len(rows)} query proposals; none accepted automatically")
