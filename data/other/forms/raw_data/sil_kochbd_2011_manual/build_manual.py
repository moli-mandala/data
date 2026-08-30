#!/usr/bin/env python3
"""Expand independently hand-keyed Koch response lines into cumulative site cells."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "source_manifest.json"
SOURCE_PDF = "tmp/pdfs/kochbd_manual/silesr2011_023.pdf"
SOURCE_PDF_SHA256 = "d1b2d597c16fd0338ad47d2bf031566192c5ff4e26a6651de14a228df681fc10"

CHUNKS = [
    {
        "lines": ROOT / "manual_chunks/p043-items001-013-lines.tsv",
        "cells": ROOT / "manual_chunks/p043-items001-013-cells.tsv",
        "physical_page": 43,
        "printed_page": 42,
        "items": (1, 13),
        "expected_lines": 55,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/p043.png",
        "source_image_sha256": "88d00344a48875188a993df51cecd9eb2731c3af331eee9ec54a08486ee8c3f4",
        "source_image_dimensions": [4961, 7017],
        "review_resolution": "surviving high-resolution render",
    },
    {
        "lines": ROOT / "manual_chunks/p044-items014-018-lines.tsv",
        "cells": ROOT / "manual_chunks/p044-items014-018-cells.tsv",
        "physical_page": 44,
        "printed_page": 43,
        "items": (14, 18),
        "expected_lines": 24,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-44.png",
        "source_image_sha256": "9130020f377bfb4fe457bb9f566ce8a06dfa3bd2ab69c5e71a8da3cd85ef8d4a",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with targeted 600-dpi rechecks",
    },
    {
        "lines": ROOT / "manual_chunks/p044-items019-023-lines.tsv",
        "cells": ROOT / "manual_chunks/p044-items019-023-cells.tsv",
        "physical_page": 44,
        "printed_page": 43,
        "items": (19, 23),
        "expected_lines": 18,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-44.png",
        "source_image_sha256": "9130020f377bfb4fe457bb9f566ce8a06dfa3bd2ab69c5e71a8da3cd85ef8d4a",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with targeted 600-dpi rechecks",
    },
    {
        "lines": ROOT / "manual_chunks/p044-items024-028-lines.tsv",
        "cells": ROOT / "manual_chunks/p044-items024-028-cells.tsv",
        "physical_page": 44,
        "printed_page": 43,
        "items": (24, 28),
        "expected_lines": 14,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-44.png",
        "source_image_sha256": "9130020f377bfb4fe457bb9f566ce8a06dfa3bd2ab69c5e71a8da3cd85ef8d4a",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with targeted 600-dpi rechecks",
    },
    {
        "lines": ROOT / "manual_chunks/p045-items029-033-lines.tsv",
        "cells": ROOT / "manual_chunks/p045-items029-033-cells.tsv",
        "physical_page": 45,
        "printed_page": 44,
        "items": (29, 33),
        "expected_lines": 18,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-45.png",
        "source_image_sha256": "d126add19dfff95349d4ef0609a61863171bc1e5df4ab1e75b243a9ca3e897af",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p045-600-45.png",
        "review_image_600dpi_sha256": "dfa665d91d1085ea2d45dbd7ebd326a0a25cf997af54c8080948483ca28fbfec",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p045-items034-037-lines.tsv",
        "cells": ROOT / "manual_chunks/p045-items034-037-cells.tsv",
        "physical_page": 45,
        "printed_page": 44,
        "items": (34, 37),
        "expected_lines": 10,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-45.png",
        "source_image_sha256": "d126add19dfff95349d4ef0609a61863171bc1e5df4ab1e75b243a9ca3e897af",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p045-600-45.png",
        "review_image_600dpi_sha256": "dfa665d91d1085ea2d45dbd7ebd326a0a25cf997af54c8080948483ca28fbfec",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p045-items038-042-lines.tsv",
        "cells": ROOT / "manual_chunks/p045-items038-042-cells.tsv",
        "physical_page": 45,
        "printed_page": 44,
        "items": (38, 42),
        "expected_lines": 10,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-45.png",
        "source_image_sha256": "d126add19dfff95349d4ef0609a61863171bc1e5df4ab1e75b243a9ca3e897af",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p045-600-45.png",
        "review_image_600dpi_sha256": "dfa665d91d1085ea2d45dbd7ebd326a0a25cf997af54c8080948483ca28fbfec",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p045-items043-046-lines.tsv",
        "cells": ROOT / "manual_chunks/p045-items043-046-cells.tsv",
        "physical_page": 45,
        "printed_page": 44,
        "items": (43, 46),
        "expected_lines": 13,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-45.png",
        "source_image_sha256": "d126add19dfff95349d4ef0609a61863171bc1e5df4ab1e75b243a9ca3e897af",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p045-600-45.png",
        "review_image_600dpi_sha256": "dfa665d91d1085ea2d45dbd7ebd326a0a25cf997af54c8080948483ca28fbfec",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p046-items047-051-lines.tsv",
        "cells": ROOT / "manual_chunks/p046-items047-051-cells.tsv",
        "physical_page": 46,
        "printed_page": 45,
        "items": (47, 51),
        "expected_lines": 20,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-46.png",
        "source_image_sha256": "5b9f3054612b851ade94be5a2f9d6ff0c1f6a355b3af0f5bc9d1260868d0b735",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p046-600-46.png",
        "review_image_600dpi_sha256": "243c1eed48484eb18b8c5c9e2ba9018a5cd1f17ba6a3607e2720347d5312565a",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p046-items052-056-lines.tsv",
        "cells": ROOT / "manual_chunks/p046-items052-056-cells.tsv",
        "physical_page": 46,
        "printed_page": 45,
        "items": (52, 56),
        "expected_lines": 21,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-46.png",
        "source_image_sha256": "5b9f3054612b851ade94be5a2f9d6ff0c1f6a355b3af0f5bc9d1260868d0b735",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p046-600-46.png",
        "review_image_600dpi_sha256": "243c1eed48484eb18b8c5c9e2ba9018a5cd1f17ba6a3607e2720347d5312565a",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p046-items057-061-lines.tsv",
        "cells": ROOT / "manual_chunks/p046-items057-061-cells.tsv",
        "physical_page": 46,
        "printed_page": 45,
        "items": (57, 61),
        "expected_lines": 13,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-46.png",
        "source_image_sha256": "5b9f3054612b851ade94be5a2f9d6ff0c1f6a355b3af0f5bc9d1260868d0b735",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p046-600-46.png",
        "review_image_600dpi_sha256": "243c1eed48484eb18b8c5c9e2ba9018a5cd1f17ba6a3607e2720347d5312565a",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p047-items062-066-lines.tsv",
        "cells": ROOT / "manual_chunks/p047-items062-066-cells.tsv",
        "physical_page": 47,
        "printed_page": 46,
        "items": (62, 66),
        "expected_lines": 18,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-47.png",
        "source_image_sha256": "be22b6d5b54205f36057f2bc84dfbc762de81082a875caa38070711b9feb1c90",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p047-600-47.png",
        "review_image_600dpi_sha256": "434a3ee8729ed938f2c71ecba6d2979b9d94365776675e318c4cad1d7ab63386",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p047-items067-071-lines.tsv",
        "cells": ROOT / "manual_chunks/p047-items067-071-cells.tsv",
        "physical_page": 47,
        "printed_page": 46,
        "items": (67, 71),
        "expected_lines": 18,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-47.png",
        "source_image_sha256": "be22b6d5b54205f36057f2bc84dfbc762de81082a875caa38070711b9feb1c90",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p047-600-47.png",
        "review_image_600dpi_sha256": "434a3ee8729ed938f2c71ecba6d2979b9d94365776675e318c4cad1d7ab63386",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p047-items072-076-lines.tsv",
        "cells": ROOT / "manual_chunks/p047-items072-076-cells.tsv",
        "physical_page": 47,
        "printed_page": 46,
        "items": (72, 76),
        "expected_lines": 20,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-47.png",
        "source_image_sha256": "be22b6d5b54205f36057f2bc84dfbc762de81082a875caa38070711b9feb1c90",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p047-600-47.png",
        "review_image_600dpi_sha256": "434a3ee8729ed938f2c71ecba6d2979b9d94365776675e318c4cad1d7ab63386",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p048-items077-081-lines.tsv",
        "cells": ROOT / "manual_chunks/p048-items077-081-cells.tsv",
        "physical_page": 48,
        "printed_page": 47,
        "items": (77, 81),
        "expected_lines": 17,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-48.png",
        "source_image_sha256": "3ccaccce8cecc40c2f08a24ea4d6f061759496738e680af5f0ff25d9c032236c",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted cell rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p048-600-48.png",
        "review_image_600dpi_sha256": "082789d895e57bad5b47c6dd5197217061822b7f42c9c07d5bfd82c827f7fe0b",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p048-items082-085-lines.tsv",
        "cells": ROOT / "manual_chunks/p048-items082-085-cells.tsv",
        "physical_page": 48,
        "printed_page": 47,
        "items": (82, 85),
        "expected_lines": 10,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-48.png",
        "source_image_sha256": "3ccaccce8cecc40c2f08a24ea4d6f061759496738e680af5f0ff25d9c032236c",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted cell rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p048-600-48.png",
        "review_image_600dpi_sha256": "082789d895e57bad5b47c6dd5197217061822b7f42c9c07d5bfd82c827f7fe0b",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p048-items086-089-lines.tsv",
        "cells": ROOT / "manual_chunks/p048-items086-089-cells.tsv",
        "physical_page": 48,
        "printed_page": 47,
        "items": (86, 89),
        "expected_lines": 14,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-48.png",
        "source_image_sha256": "3ccaccce8cecc40c2f08a24ea4d6f061759496738e680af5f0ff25d9c032236c",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted right-column rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p048-600-48.png",
        "review_image_600dpi_sha256": "082789d895e57bad5b47c6dd5197217061822b7f42c9c07d5bfd82c827f7fe0b",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p048-items090-093-lines.tsv",
        "cells": ROOT / "manual_chunks/p048-items090-093-cells.tsv",
        "physical_page": 48,
        "printed_page": 47,
        "items": (90, 93),
        "expected_lines": 13,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-48.png",
        "source_image_sha256": "3ccaccce8cecc40c2f08a24ea4d6f061759496738e680af5f0ff25d9c032236c",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted right-column rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p048-600-48.png",
        "review_image_600dpi_sha256": "082789d895e57bad5b47c6dd5197217061822b7f42c9c07d5bfd82c827f7fe0b",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p049-items094-097-lines.tsv",
        "cells": ROOT / "manual_chunks/p049-items094-097-cells.tsv",
        "physical_page": 49,
        "printed_page": 48,
        "items": (94, 97),
        "expected_lines": 19,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-49.png",
        "source_image_sha256": "5aa95ced7ddd3b11e60438b72659eb2f786b81e345f8f3e74e94d0573cf75be0",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted left-column rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p049-600-49.png",
        "review_image_600dpi_sha256": "d0ccba1fad09722080ab221d66145585348c1cb4b3a2b49979a5f6f38accd755",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p049-items098-100-lines.tsv",
        "cells": ROOT / "manual_chunks/p049-items098-100-cells.tsv",
        "physical_page": 49,
        "printed_page": 48,
        "items": (98, 100),
        "expected_lines": 12,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-49.png",
        "source_image_sha256": "5aa95ced7ddd3b11e60438b72659eb2f786b81e345f8f3e74e94d0573cf75be0",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted left-column rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p049-600-49.png",
        "review_image_600dpi_sha256": "d0ccba1fad09722080ab221d66145585348c1cb4b3a2b49979a5f6f38accd755",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p049-items101-104-lines.tsv",
        "cells": ROOT / "manual_chunks/p049-items101-104-cells.tsv",
        "physical_page": 49,
        "printed_page": 48,
        "items": (101, 104),
        "expected_lines": 16,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-49.png",
        "source_image_sha256": "5aa95ced7ddd3b11e60438b72659eb2f786b81e345f8f3e74e94d0573cf75be0",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted right-column rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p049-600-49.png",
        "review_image_600dpi_sha256": "d0ccba1fad09722080ab221d66145585348c1cb4b3a2b49979a5f6f38accd755",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p049-items105-107-lines.tsv",
        "cells": ROOT / "manual_chunks/p049-items105-107-cells.tsv",
        "physical_page": 49,
        "printed_page": 48,
        "items": (105, 107),
        "expected_lines": 11,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-49.png",
        "source_image_sha256": "5aa95ced7ddd3b11e60438b72659eb2f786b81e345f8f3e74e94d0573cf75be0",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted right-column rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p049-600-49.png",
        "review_image_600dpi_sha256": "d0ccba1fad09722080ab221d66145585348c1cb4b3a2b49979a5f6f38accd755",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p050-items108-111-lines.tsv",
        "cells": ROOT / "manual_chunks/p050-items108-111-cells.tsv",
        "physical_page": 50,
        "printed_page": 49,
        "items": (108, 111),
        "expected_lines": 18,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-50.png",
        "source_image_sha256": "ebc2715bc94dc8921ad61aa1d062fb79fd50df2fe6a102961d98817924dd7ba1",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted left-column rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p050-600-50.png",
        "review_image_600dpi_sha256": "2ee918a7486ff5bf680d4c29ffd189a1b30683b44db7e81e2bdab584bda04811",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p050-items112-114-lines.tsv",
        "cells": ROOT / "manual_chunks/p050-items112-114-cells.tsv",
        "physical_page": 50,
        "printed_page": 49,
        "items": (112, 114),
        "expected_lines": 11,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-50.png",
        "source_image_sha256": "ebc2715bc94dc8921ad61aa1d062fb79fd50df2fe6a102961d98817924dd7ba1",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted left-column rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p050-600-50.png",
        "review_image_600dpi_sha256": "2ee918a7486ff5bf680d4c29ffd189a1b30683b44db7e81e2bdab584bda04811",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p050-items115-118-lines.tsv",
        "cells": ROOT / "manual_chunks/p050-items115-118-cells.tsv",
        "physical_page": 50,
        "printed_page": 49,
        "items": (115, 118),
        "expected_lines": 18,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-50.png",
        "source_image_sha256": "ebc2715bc94dc8921ad61aa1d062fb79fd50df2fe6a102961d98817924dd7ba1",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and targeted right-column rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p050-600-50.png",
        "review_image_600dpi_sha256": "2ee918a7486ff5bf680d4c29ffd189a1b30683b44db7e81e2bdab584bda04811",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p050-items119-121-lines.tsv",
        "cells": ROOT / "manual_chunks/p050-items119-121-cells.tsv",
        "physical_page": 50,
        "printed_page": 49,
        "items": (119, 121),
        "expected_lines": 11,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-50.png",
        "source_image_sha256": "ebc2715bc94dc8921ad61aa1d062fb79fd50df2fe6a102961d98817924dd7ba1",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and lower-right crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p050-600-50.png",
        "review_image_600dpi_sha256": "2ee918a7486ff5bf680d4c29ffd189a1b30683b44db7e81e2bdab584bda04811",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p051-items122-128-lines.tsv",
        "cells": ROOT / "manual_chunks/p051-items122-128-cells.tsv",
        "physical_page": 51,
        "printed_page": 50,
        "items": (122, 128),
        "expected_lines": 30,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-51.png",
        "source_image_sha256": "0b546e625b0be38dd67c7f3130c3fde62ab1b7e5d269fac8af6f83c3e7faa5a1",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p051-600-51.png",
        "review_image_600dpi_sha256": "7c67cbbe0abf92f216c83447afa1cc43774ce9b14a2816f1bcf6da0e7dfbccbb",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p051-items129-135-lines.tsv",
        "cells": ROOT / "manual_chunks/p051-items129-135-cells.tsv",
        "physical_page": 51,
        "printed_page": 50,
        "items": (129, 135),
        "expected_lines": 28,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-51.png",
        "source_image_sha256": "0b546e625b0be38dd67c7f3130c3fde62ab1b7e5d269fac8af6f83c3e7faa5a1",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and right-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p051-600-51.png",
        "review_image_600dpi_sha256": "7c67cbbe0abf92f216c83447afa1cc43774ce9b14a2816f1bcf6da0e7dfbccbb",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p052-items136-142-lines.tsv",
        "cells": ROOT / "manual_chunks/p052-items136-142-cells.tsv",
        "physical_page": 52,
        "printed_page": 51,
        "items": (136, 142),
        "expected_lines": 30,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-52.png",
        "source_image_sha256": "552435c083c1dfb9d1ce06a15f8b1d3fe95f00b3a69779335f1a92d48460a30f",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p052-600-52.png",
        "review_image_600dpi_sha256": "ff213805cffd75efca08653da6793255045418f60f9172cd7ab9915e6d8daa49",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p052-items143-149-lines.tsv",
        "cells": ROOT / "manual_chunks/p052-items143-149-cells.tsv",
        "physical_page": 52,
        "printed_page": 51,
        "items": (143, 149),
        "expected_lines": 29,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-52.png",
        "source_image_sha256": "552435c083c1dfb9d1ce06a15f8b1d3fe95f00b3a69779335f1a92d48460a30f",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and right-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p052-600-52.png",
        "review_image_600dpi_sha256": "ff213805cffd75efca08653da6793255045418f60f9172cd7ab9915e6d8daa49",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p053-items150-157-lines.tsv",
        "cells": ROOT / "manual_chunks/p053-items150-157-cells.tsv",
        "physical_page": 53,
        "printed_page": 52,
        "items": (150, 157),
        "expected_lines": 28,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-53.png",
        "source_image_sha256": "25a89f34b1370fa0719c3c9424fe027c411feaff99a71be96519d31479ac2f72",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p053-600-53.png",
        "review_image_600dpi_sha256": "abbd43b1dd2ecdab7dc7102866c2e595d25accbbfbbd7de3719fbae0a93a2a87",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p053-items158-167-lines.tsv",
        "cells": ROOT / "manual_chunks/p053-items158-167-cells.tsv",
        "physical_page": 53,
        "printed_page": 52,
        "items": (158, 167),
        "expected_lines": 26,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-53.png",
        "source_image_sha256": "25a89f34b1370fa0719c3c9424fe027c411feaff99a71be96519d31479ac2f72",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and right-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p053-600-53.png",
        "review_image_600dpi_sha256": "abbd43b1dd2ecdab7dc7102866c2e595d25accbbfbbd7de3719fbae0a93a2a87",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p054-items168-176-lines.tsv",
        "cells": ROOT / "manual_chunks/p054-items168-176-cells.tsv",
        "physical_page": 54,
        "printed_page": 53,
        "items": (168, 176),
        "expected_lines": 22,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-54.png",
        "source_image_sha256": "5e58e7f5bc68907c50104c862dfbd3ed7c1b31b367b37af1558ee7296d06c30a",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p054-600-54.png",
        "review_image_600dpi_sha256": "b4ec5c05abe723c97a1426e04c5168e70f744f1763a01ee6a17b7ebb2a8cf6b0",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p054-items177-183-lines.tsv",
        "cells": ROOT / "manual_chunks/p054-items177-183-cells.tsv",
        "physical_page": 54,
        "printed_page": 53,
        "items": (177, 183),
        "expected_lines": 29,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-54.png",
        "source_image_sha256": "5e58e7f5bc68907c50104c862dfbd3ed7c1b31b367b37af1558ee7296d06c30a",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and right-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p054-600-54.png",
        "review_image_600dpi_sha256": "b4ec5c05abe723c97a1426e04c5168e70f744f1763a01ee6a17b7ebb2a8cf6b0",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p055-items184-191-lines.tsv",
        "cells": ROOT / "manual_chunks/p055-items184-191-cells.tsv",
        "physical_page": 55,
        "printed_page": 54,
        "items": (184, 191),
        "expected_lines": 28,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-55.png",
        "source_image_sha256": "6936ba1b43e5cc60dddb528eb0ecfec2b2736c271b95367e9eefcd4e4732a02a",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p055-600-55.png",
        "review_image_600dpi_sha256": "aae68ffafa01e7e974624c47d367a5734d0196da476eed33134d6182251ccd2e",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p055-items192-199-lines.tsv",
        "cells": ROOT / "manual_chunks/p055-items192-199-cells.tsv",
        "physical_page": 55,
        "printed_page": 54,
        "items": (192, 199),
        "expected_lines": 27,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-55.png",
        "source_image_sha256": "6936ba1b43e5cc60dddb528eb0ecfec2b2736c271b95367e9eefcd4e4732a02a",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and right-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p055-600-55.png",
        "review_image_600dpi_sha256": "aae68ffafa01e7e974624c47d367a5734d0196da476eed33134d6182251ccd2e",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p056-items200-206-lines.tsv",
        "cells": ROOT / "manual_chunks/p056-items200-206-cells.tsv",
        "physical_page": 56,
        "printed_page": 55,
        "items": (200, 206),
        "expected_lines": 28,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-56.png",
        "source_image_sha256": "a9fda0dbd18e668f3e98c6d49655ff001742070753fe5e97fcc9236e024df03a",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p056-600.png",
        "review_image_600dpi_sha256": "e23464a3a684f9edd33a781204df34a42f0bdc8943e500c67c31062b691c4399",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p056-items207-213-lines.tsv",
        "cells": ROOT / "manual_chunks/p056-items207-213-cells.tsv",
        "physical_page": 56,
        "printed_page": 55,
        "items": (207, 213),
        "expected_lines": 29,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-56.png",
        "source_image_sha256": "a9fda0dbd18e668f3e98c6d49655ff001742070753fe5e97fcc9236e024df03a",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and right-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p056-600.png",
        "review_image_600dpi_sha256": "e23464a3a684f9edd33a781204df34a42f0bdc8943e500c67c31062b691c4399",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p057-items214-221-lines.tsv",
        "cells": ROOT / "manual_chunks/p057-items214-221-cells.tsv",
        "physical_page": 57,
        "printed_page": 56,
        "items": (214, 221),
        "expected_lines": 28,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-57.png",
        "source_image_sha256": "3d6d46e7392c8b0c2a87871aa3b2aa0e6f548813aa4cad0ff7831d1baa3071e0",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p057-600.png",
        "review_image_600dpi_sha256": "7d9dc0c732835722bd2b04b98c53ccf891cb866dc08d2e857b1d92dd7628a11b",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p057-items222-228-lines.tsv",
        "cells": ROOT / "manual_chunks/p057-items222-228-cells.tsv",
        "physical_page": 57,
        "printed_page": 56,
        "items": (222, 228),
        "expected_lines": 27,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-57.png",
        "source_image_sha256": "3d6d46e7392c8b0c2a87871aa3b2aa0e6f548813aa4cad0ff7831d1baa3071e0",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and right-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p057-600.png",
        "review_image_600dpi_sha256": "7d9dc0c732835722bd2b04b98c53ccf891cb866dc08d2e857b1d92dd7628a11b",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p057-item229-lines.tsv",
        "cells": ROOT / "manual_chunks/p057-item229-cells.tsv",
        "physical_page": 57,
        "printed_page": 56,
        "items": (229, 229),
        "expected_lines": 3,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-57.png",
        "source_image_sha256": "3d6d46e7392c8b0c2a87871aa3b2aa0e6f548813aa4cad0ff7831d1baa3071e0",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and lower-right crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p057-600.png",
        "review_image_600dpi_sha256": "7d9dc0c732835722bd2b04b98c53ccf891cb866dc08d2e857b1d92dd7628a11b",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p058-items230-237-lines.tsv",
        "cells": ROOT / "manual_chunks/p058-items230-237-cells.tsv",
        "physical_page": 58,
        "printed_page": 57,
        "items": (230, 237),
        "expected_lines": 24,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-58.png",
        "source_image_sha256": "e9991927b31cb59023ddf198bd4e7ba0742f8e098296bf753e100b799844f2fc",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p058-600.png",
        "review_image_600dpi_sha256": "fbad556c88949512b059cf67c8147feb00726f7cd316c6f4d853cf32576f9205",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p058-items238-246-lines.tsv",
        "cells": ROOT / "manual_chunks/p058-items238-246-cells.tsv",
        "physical_page": 58,
        "printed_page": 57,
        "items": (238, 246),
        "expected_lines": 29,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-58.png",
        "source_image_sha256": "e9991927b31cb59023ddf198bd4e7ba0742f8e098296bf753e100b799844f2fc",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page, column, and lower-left crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p058-600.png",
        "review_image_600dpi_sha256": "fbad556c88949512b059cf67c8147feb00726f7cd316c6f4d853cf32576f9205",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p059-items247-253-lines.tsv",
        "cells": ROOT / "manual_chunks/p059-items247-253-cells.tsv",
        "physical_page": 59,
        "printed_page": 58,
        "items": (247, 253),
        "expected_lines": 25,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-59.png",
        "source_image_sha256": "df1cf41fd91981070c9ee6d21135a0093c4f5ff8337c4e45e4ff3ee72ea39c82",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p059-600.png",
        "review_image_600dpi_sha256": "9733ee0fd08b0f60d8f3f6620e0843c6b2e9e27833bbf52061e269257823d14a",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p059-item254-lines.tsv",
        "cells": ROOT / "manual_chunks/p059-item254-cells.tsv",
        "physical_page": 59,
        "printed_page": 58,
        "items": (254, 254),
        "expected_lines": 6,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-59.png",
        "source_image_sha256": "df1cf41fd91981070c9ee6d21135a0093c4f5ff8337c4e45e4ff3ee72ea39c82",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page, left-column, and lower-left crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p059-600.png",
        "review_image_600dpi_sha256": "9733ee0fd08b0f60d8f3f6620e0843c6b2e9e27833bbf52061e269257823d14a",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p059-items255-261-lines.tsv",
        "cells": ROOT / "manual_chunks/p059-items255-261-cells.tsv",
        "physical_page": 59,
        "printed_page": 58,
        "items": (255, 261),
        "expected_lines": 27,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-59.png",
        "source_image_sha256": "df1cf41fd91981070c9ee6d21135a0093c4f5ff8337c4e45e4ff3ee72ea39c82",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page, right-column, and lower-right crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p059-600.png",
        "review_image_600dpi_sha256": "9733ee0fd08b0f60d8f3f6620e0843c6b2e9e27833bbf52061e269257823d14a",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p060-items262-268-lines.tsv",
        "cells": ROOT / "manual_chunks/p060-items262-268-cells.tsv",
        "physical_page": 60,
        "printed_page": 59,
        "items": (262, 268),
        "expected_lines": 31,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-60.png",
        "source_image_sha256": "c30a4df62df011cdda57c2fcdfa3d9b72a5fbe480b2dc69788a22e1f608aa2ab",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page, left-column, and middle-left crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p060-600.png",
        "review_image_600dpi_sha256": "71d230b2642014c5348fa7548f7d5a0132c5ae2e8617f8325e760edac8fdd663",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p060-items269-275-lines.tsv",
        "cells": ROOT / "manual_chunks/p060-items269-275-cells.tsv",
        "physical_page": 60,
        "printed_page": 59,
        "items": (269, 275),
        "expected_lines": 31,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-60.png",
        "source_image_sha256": "c30a4df62df011cdda57c2fcdfa3d9b72a5fbe480b2dc69788a22e1f608aa2ab",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page, right-column, middle-right, and lower-right crop rechecks",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p060-600.png",
        "review_image_600dpi_sha256": "71d230b2642014c5348fa7548f7d5a0132c5ae2e8617f8325e760edac8fdd663",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p061-items276-282-lines.tsv",
        "cells": ROOT / "manual_chunks/p061-items276-282-cells.tsv",
        "physical_page": 61,
        "printed_page": 60,
        "items": (276, 282),
        "expected_lines": 30,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-61.png",
        "source_image_sha256": "46a82766781ef33afab16cfaed915f48f9ff98296caa7c4d5bd074c44a967cb4",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p061-600.png",
        "review_image_600dpi_sha256": "afb79334da7a3bdaf369ad25ce38aff5a3f6a87e190711d6b90f1d9e465c26d8",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p061-items283-291-lines.tsv",
        "cells": ROOT / "manual_chunks/p061-items283-291-cells.tsv",
        "physical_page": 61,
        "printed_page": 60,
        "items": (283, 291),
        "expected_lines": 28,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-61.png",
        "source_image_sha256": "46a82766781ef33afab16cfaed915f48f9ff98296caa7c4d5bd074c44a967cb4",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and right-column crop recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p061-600.png",
        "review_image_600dpi_sha256": "afb79334da7a3bdaf369ad25ce38aff5a3f6a87e190711d6b90f1d9e465c26d8",
        "review_image_600dpi_dimensions": [4959, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p062-items292-300-lines.tsv",
        "cells": ROOT / "manual_chunks/p062-items292-300-cells.tsv",
        "physical_page": 62,
        "printed_page": 61,
        "items": (292, 300),
        "expected_lines": 27,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-62.png",
        "source_image_sha256": "4de4b7293d14e7c8746cd035f2080c74ed84c18f9161b14adb1c20651e18e261",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and left-column crop recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p062-600.png",
        "review_image_600dpi_sha256": "a28765272c1a3c46a1a68455a36c35dc0340452b7251c0a31b9dbef162bfc03e",
        "review_image_600dpi_dimensions": [4961, 7017],
    },
    {
        "lines": ROOT / "manual_chunks/p062-items301-307-lines.tsv",
        "cells": ROOT / "manual_chunks/p062-items301-307-cells.tsv",
        "physical_page": 62,
        "printed_page": 61,
        "items": (301, 307),
        "expected_lines": 21,
        "source_image": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/page-62.png",
        "source_image_sha256": "4de4b7293d14e7c8746cd035f2080c74ed84c18f9161b14adb1c20651e18e261",
        "source_image_dimensions": [2480, 3509],
        "review_resolution": "300-dpi full page with fresh 600-dpi full-page and right-column crop recheck",
        "review_image_600dpi": "/Users/aryamanarora/Documents/Code/jambu-all/tmp/pdfs/kochbd_manual/koch-p062-600.png",
        "review_image_600dpi_sha256": "a28765272c1a3c46a1a68455a36c35dc0340452b7251c0a31b9dbef162bfc03e",
        "review_image_600dpi_dimensions": [4961, 7017],
    },
]

SITES = {
    "0": ("Bangla", "control"),
    "b": ("Nokshi", "target"),
    "c": ("Kholchanda", "target"),
    "l": ("Bharatpur", "control"),
    "m": ("Nalchapra", "control"),
    "q": ("Uttor Nokshi", "target"),
    "r": ("Chandabhoi", "target"),
}
SITE_ORDER = {code: index for index, code in enumerate("bcqrlm0")}
FIELDS = [
    "physical_page", "printed_page", "item", "gloss", "group", "site_code",
    "site", "role", "status", "form", "visible_base", "note", "evidence_sha256",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(chunk: dict) -> list[dict[str, str]]:
    with chunk["lines"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == chunk["expected_lines"]
    assert {row["status"] for row in rows} <= {
        "attested", "blank", "ambiguous", "not_used",
    }
    assert set(map(int, (row["item"] for row in rows))) == set(
        range(chunk["items"][0], chunk["items"][1] + 1)
    )
    for row in rows:
        assert row["site_codes"] and set(row["site_codes"]) <= set(SITES)
        assert row["form"] == unicodedata.normalize("NFC", row["form"])
        assert row["visible_base"] == unicodedata.normalize("NFC", row["visible_base"])
        assert not any(0xE000 <= ord(char) <= 0xF8FF for char in row["form"])
        if row["status"] == "attested":
            assert row["form"] and not row["visible_base"]
        elif row["status"] in {"blank", "not_used"}:
            assert not row["form"] and not row["visible_base"]
        else:
            assert not row["form"] and row["visible_base"] and row["note"]
    return rows


def _append_distinct(value: str, addition: str, separator: str = " | ") -> str:
    parts = [part for part in value.split(separator) if part] if value else []
    if addition and addition not in parts:
        parts.append(addition)
    return separator.join(parts)


def expand(rows: list[dict[str, str]], chunk: dict) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for code in row["site_codes"]:
            site, role = SITES[code]
            key = (row["item"], code)
            match = next((cell for cell in by_key[key]
                          if (cell["status"], cell["form"], cell["visible_base"]) ==
                          (row["status"], row["form"], row["visible_base"])), None)
            if match is not None:
                match["group"] = _append_distinct(match["group"], row["group"])
                match["note"] = _append_distinct(match["note"], row["note"], separator=" ")
                continue
            if by_key[key]:
                repeated_statuses = {cell["status"] for cell in by_key[key]} | {row["status"]}
                assert not repeated_statuses & {"blank", "not_used"}, (
                    f"blank/not-used response overlaps another printed response at {key}"
                )
            by_key[key].append({
                "physical_page": str(chunk["physical_page"]),
                "printed_page": str(chunk["printed_page"]),
                "item": row["item"],
                "gloss": row["gloss"],
                "group": row["group"],
                "site_code": code,
                "site": site,
                "role": role,
                "status": row["status"],
                "form": row["form"],
                "visible_base": row["visible_base"],
                "note": row["note"],
                "evidence_sha256": chunk["source_image_sha256"],
            })

    cells = [cell for key in sorted(by_key, key=lambda key: (int(key[0]), SITE_ORDER[key[1]]))
             for cell in by_key[key]]
    item_sites = defaultdict(set)
    for row in cells:
        item_sites[int(row["item"])].add(row["site_code"])
    assert set(item_sites) == set(range(chunk["items"][0], chunk["items"][1] + 1))
    assert all(codes == set(SITES) for codes in item_sites.values())
    assert len(by_key) == (chunk["items"][1] - chunk["items"][0] + 1) * len(SITES)
    return cells


def write() -> None:
    all_lines: list[dict[str, str]] = []
    all_cells: list[dict[str, str]] = []
    artifacts = []
    for chunk in CHUNKS:
        rows = read_lines(chunk)
        cells = expand(rows, chunk)
        with chunk["cells"].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(cells)
        all_lines.extend(rows)
        all_cells.extend(cells)
        conceptual_count = len({(row["item"], row["site_code"]) for row in cells})
        artifacts.append({
            "physical_page": chunk["physical_page"],
            "printed_page": chunk["printed_page"],
            "items": list(chunk["items"]),
            "manual_lines": str(chunk["lines"].relative_to(ROOT)),
            "manual_lines_sha256": sha256(chunk["lines"]),
            "response_lines": len(rows),
            "expanded_cells": str(chunk["cells"].relative_to(ROOT)),
            "expanded_cells_sha256": sha256(chunk["cells"]),
            "conceptual_cells": conceptual_count,
            "expanded_rows": len(cells),
            "source_image": chunk["source_image"],
            "source_image_sha256": chunk["source_image_sha256"],
            "source_image_dimensions": chunk["source_image_dimensions"],
            "review_resolution": chunk["review_resolution"],
        })

    conceptual = {}
    disposition_priority = {"attested": 0, "ambiguous": 1, "blank": 2, "not_used": 3}
    for row in all_cells:
        key = (row["item"], row["site_code"])
        if key not in conceptual or disposition_priority[row["status"]] < disposition_priority[conceptual[key]["status"]]:
            conceptual[key] = row
    status = Counter(row["status"] for row in conceptual.values())
    roles = Counter((row["role"], row["status"]) for row in conceptual.values())
    manifest = {
        "report": "ESR 2011-023 The Koch of Bangladesh",
        "state": "manual_review_complete" if max(chunk["items"][1] for chunk in CHUNKS) == 307 else "partial_manual_review",
        "source_pdf": SOURCE_PDF,
        "source_pdf_sha256": SOURCE_PDF_SHA256,
        "source_pdf_bytes": 1116174,
        "source_pdf_pages": 91,
        "wayback_timestamp": "20170809124914",
        "wayback_original_url": "http://www-01.sil.org/silesr/2011/silesr2011-023.pdf",
        "wordlist_render": {
            "dpi": 300,
            "physical_pages": [43, 62],
            "rendered_page_count": 20,
            "workspace_path": "tmp/pdfs/kochbd_manual",
            "first_page_sha256": "bf8824a8c297c3376581e254b7ce4cee6e6f5447c5bfce506a2168856b3c9334",
            "last_page_sha256": "4de4b7293d14e7c8746cd035f2080c74ed84c18f9161b14adb1c20651e18e261",
        },
        "physical_pages_reviewed": sorted({chunk["physical_page"] for chunk in CHUNKS}),
        "printed_pages_reviewed": sorted({chunk["printed_page"] for chunk in CHUNKS}),
        "items_reviewed": [min(chunk["items"][0] for chunk in CHUNKS), max(chunk["items"][1] for chunk in CHUNKS)],
        "response_lines": len(all_lines),
        "conceptual_cells": len(conceptual),
        "expanded_rows": len(all_cells),
        "status_counts": dict(sorted(status.items())),
        "target_counts": {
            key: roles[("target", key)] for key in ("attested", "blank", "ambiguous")
        },
        "control_counts": {
            key: roles[("control", key)] for key in ("attested", "blank", "ambiguous")
        },
        "pending_items": ([] if max(chunk["items"][1] for chunk in CHUNKS) == 307
                          else [max(chunk["items"][1] for chunk in CHUNKS) + 1, 307]),
        "manual_chunks": artifacts,
        "policy": "Rendered pages supplied every reading; legacy glyphs, OCR, PDF text, and installed forms supplied or verified none.",
    }
    with MANIFEST.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        f"wrote {len(all_cells)} expanded rows for {len(conceptual)} conceptual cells "
        f"from {len(all_lines)} manual response lines"
    )


def main() -> None:
    write()


if __name__ == "__main__":
    main()
