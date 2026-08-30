#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace_root=$(CDPATH= cd -- "$script_dir/../../../../../.." && pwd)
pdf=${1:-"$workspace_root/tmp/pdfs/kurumba_2012/silesr2012_015.pdf"}
first=${2:-217}
last=${3:-436}
out=${4:-"$workspace_root/tmp/pdfs/kurumba_2012/rendered_300"}
mkdir -p "$out"
pdftoppm -f "$first" -l "$last" -png -r 300 "$pdf" "$out/page"
