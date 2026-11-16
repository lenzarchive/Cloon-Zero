#!/bin/sh
OUT="${1:-corpus.txt}"
MAX_BYTES="${2:-3000000}"
shift 2 2>/dev/null
DIRS="${@:-/usr/include /usr/lib/python3 /usr/share/doc}"
: > "$OUT.list"
for d in $DIRS; do
  find "$d" -type f \( -name '*.c' -o -name '*.h' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.cc' \
    -o -name '*.py' -o -name '*.rs' -o -name '*.go' -o -name '*.js' -o -name '*.java' \
    -o -name '*.md' -o -name '*.rst' -o -name '*.txt' \) -size -200k 2>/dev/null >> "$OUT.list"
done
shuf "$OUT.list" > "$OUT.shuf"
: > "$OUT"
while read -r f; do
  cat "$f" >> "$OUT" 2>/dev/null
  printf '\n' >> "$OUT"
  [ "$(wc -c < "$OUT")" -ge "$MAX_BYTES" ] && break
done < "$OUT.shuf"
rm -f "$OUT.list" "$OUT.shuf"
head -c "$MAX_BYTES" "$OUT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
wc -c "$OUT"
