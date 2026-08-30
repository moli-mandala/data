#!/bin/sh
set -eu

# Render Appendix B.5 for cell-by-cell visual review.  The PDF is retained in
# the workspace tmp tree and is not a checked-in redistribution.
repo_root=$(CDPATH= cd -- "$(dirname "$0")/../../../../../.." && pwd)
pdf="$repo_root/tmp/pdfs/korwa-kodaku-2022/source.pdf"
out="$repo_root/tmp/pdfs/korwa-kodaku-2022/wordlist-200dpi"
mkdir -p "$out"
pdftoppm -f 66 -l 90 -r 200 -png "$pdf" "$out/page" >/dev/null 2>&1
