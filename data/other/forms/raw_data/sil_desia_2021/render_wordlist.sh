#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace_root=$(CDPATH= cd -- "$script_dir/../../../../../.." && pwd)
pdf=${1:-"$workspace_root/tmp/pdfs/desia-2021-056/source.pdf"}
out=${2:-"$workspace_root/tmp/pdfs/desia-2021-056/rendered"}
mkdir -p "$out"

pdftoppm -f 80 -l 127 -png -r 180 "$pdf" "$out/page"
