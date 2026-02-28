#!/usr/bin/bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 INPUT_DIR OUTPUT_DIR" >&2
  exit 1
fi

SRC="$1"
OUT="$2"

rm -rf "$OUT"
mkdir -p "$OUT"

for f in "$SRC"/*.png; do
    base="$(basename "$f")"
    echo "Processing $f -> $OUT/$base"
    ffmpeg -y -i "$f" \
    -vf "scale='if(gte(iw,ih),iw,-1)':'if(gte(ih,iw),ih,-1)',pad='max(iw,ih)':'max(iw,ih)':'(ow-iw)/2':'(oh-ih)/2':color=white,scale=1024:1024" \
    -frames:v 1 "$OUT/$base"
done
