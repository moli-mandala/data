"""Refresh minimal publisher evidence for 17 visually selected Koda cells.

The complete HTTP responses are used in memory to compute hashes and are never
written to the source package. This command never installs data or builds Jambu.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://cfelvb.in/api/dictionary/getDictionaryData.php"
SELECTED = [
    ("Anklet", 156), ("Armlet", 129), ("Bindi", 6225),
    ("Cat", 384), ("Chick", 327), ("Crab", 468), ("Zero", 1197),
    ("Book", 2237), ("Bench", 2213), ("Decide", 2951),
    ("Eyelid", 3885), ("Face", 3804), ("Finger", 3859),
    ("Monday", 6957), ("Moment", 6958), ("Month", 7717),
    ("Morning", 6950),
]


def fetch_one(term: str, source_id: int) -> dict:
    url = ENDPOINT + "?" + urlencode({"limit": 20, "offset": 0, "search": term,
                                        "source": 1, "target": 104})
    request = Request(url, data=b"", headers={"User-Agent": "Jambu source review"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise ValueError("Publisher response exceeded bounded limit")
            break
        except (HTTPError, URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"Unexpected API response for {term}")
    items = [item for item in payload["data"]
             if item.get("id") == source_id and item.get("language") == 104]
    if len(items) != 1:
        raise ValueError(f"Expected exact Koda ID {source_id} for {term}")
    item = items[0]
    return {
        "query": term, "api_id": source_id,
        "api_word": item.get("word"), "api_ipa": item.get("ipa"),
        "api_domain": item.get("domain_name"),
        "api_updated_at": item.get("updated_at"),
        "response_sha256": hashlib.sha256(body).hexdigest(),
        "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def main():
    rows = []
    for term, source_id in SELECTED:
        rows.append(fetch_one(term, source_id))
        print(f"{len(rows)}/{len(SELECTED)} {term} -> {source_id}", flush=True)
        time.sleep(0.25)
    temp = HERE / "api-evidence.jsonl.tmp"
    temp.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows))
    temp.replace(HERE / "api-evidence.jsonl")


if __name__ == "__main__":
    main()
