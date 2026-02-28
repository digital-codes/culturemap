#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 INPUT_DIR OUTPUT_FILE" >&2
  exit 1
fi

SRC="$1"
OUT="$2"

if [ ! -d "$SRC" ]; then
  echo "Error: input directory '$SRC' does not exist or is not a directory" >&2
  exit 1
fi


DUR=2.0
FADE=0.4
FPS=30

mapfile -t files < <(ls -1 "$SRC"/*.png | sort)
n=${#files[@]}
if (( n < 1 )); then
  echo "No PNG files found in $SRC"
  exit 1
fi

args=()
for f in "${files[@]}"; do
  args+=(-loop 1 -t "$DUR" -i "$f")
done

fg=""
for ((i=0; i<n; i++)); do
  fg+="[$i:v]fps=${FPS},format=yuva420p,setsar=1[v$i];"
done

if (( n == 1 )); then
  fg+="[v0]format=yuv420p[vout]"
else
  off=$(python3 - <<PY
i=1
DUR=$DUR
FADE=$FADE
print(i*DUR - i*FADE)
PY
)
  fg+="[v0][v1]xfade=transition=fade:duration=${FADE}:offset=${off}[x1];"

  for ((i=2; i<n; i++)); do
    off=$(python3 - <<PY
i=$i
DUR=$DUR
FADE=$FADE
print(i*DUR - i*FADE)
PY
)
    fg+="[x$((i-1))][v$i]xfade=transition=fade:duration=${FADE}:offset=${off}[x$i];"
  done

  fg+="[x$((n-1))]format=yuv420p[vout]"
fi

ffmpeg -y "${args[@]}" \
  -filter_complex "$fg" \
  -map "[vout]" \
  -c:v libvpx -crf 12 -b:v 2M -deadline good -cpu-used 4 \
  -pix_fmt yuv420p \
  "$OUT"

