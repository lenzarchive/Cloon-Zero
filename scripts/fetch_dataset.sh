#!/bin/bash
set -u

WORK="${WORK:-./dataset_work}"
OUT="${OUT:-./dataset}"
CAP_MB="${CAP_MB:-1}"
MIN_BYTES=300
MAX_BYTES=2000000
SRC="$WORK/src"
DOCS="$WORK/docs"
mkdir -p "$SRC" "$DOCS" "$OUT"

SOURCES=(
  "python|python/cpython|Doc|rst,txt"
  "c-curl|curl/curl|docs|md"
  "cpp-guidelines|isocpp/CppCoreGuidelines||md"
  "cpp-msdocs|MicrosoftDocs/cpp-docs|docs/cpp|md"
  "rust-book|rust-lang/book|src|md"
  "rust-reference|rust-lang/reference|src|md"
  "go|golang/go|doc|md,html"
  "javascript-node|nodejs/node|doc/api|md"
  "typescript|microsoft/TypeScript-Website|packages/documentation/copy/en|md"
  "ruby|ruby/ruby|doc|md,rdoc"
  "kotlin|JetBrains/kotlin-web-site|docs/topics|md"
  "java-spring|spring-projects/spring-framework|framework-docs|adoc"
  "swift|swiftlang/swift-book|TSPL.docc|md"
  "csharp|dotnet/docs|docs/csharp|md"
  "bash|dylanaraps/pure-bash-bible||md"
)

fetch_one() {
  local name="$1" repo="$2" paths="$3"
  local dir="$SRC/$name"
  if [ ! -d "$dir/.git" ]; then
    timeout 300 git clone --quiet --depth 1 --filter=blob:none --sparse \
      "https://github.com/$repo.git" "$dir" || { echo "  clone failed: $repo"; return 1; }
  fi
  if [ -n "$paths" ]; then
    timeout 300 git -C "$dir" sparse-checkout set $paths || { echo "  checkout failed: $repo"; return 1; }
  else
    timeout 300 git -C "$dir" sparse-checkout set || { echo "  checkout failed: $repo"; return 1; }
  fi
  return 0
}

strip_html() {
  perl -0pe 's/<script.*?<\/script>//gis; s/<style.*?<\/style>//gis; s/<!--.*?-->//gs; s/<[^>]+>//g; s/&lt;/</g; s/&gt;/>/g; s/&quot;/"/g; s/&nbsp;/ /g; s/&amp;/&/g'
}

clean_stream() {
  perl -CSD -pe 's/[\x{2E80}-\x{9FFF}\x{AC00}-\x{D7AF}\x{3040}-\x{30FF}\x{0400}-\x{04FF}\x{0590}-\x{06FF}\x{0900}-\x{0DFF}\x{0E00}-\x{0EFF}\x{2600}-\x{27BF}\x{10000}-\x{10FFFF}]//g' 2>/dev/null \
    | LC_ALL=C.UTF-8 iconv -f UTF-8 -t ASCII//TRANSLIT -c 2>/dev/null \
    | tr -d '\r' \
    | tr -d '\000-\010\013\014\016-\037\177' \
    | cat -s
}

build_one() {
  local name="$1" paths="$2" exts="$3"
  local dir="$SRC/$name"
  local dest="$DOCS/$name"
  rm -rf "$dest"
  mkdir -p "$dest"
  local roots=()
  if [ -n "$paths" ]; then
    for p in $paths; do roots+=("$dir/$p"); done
  else
    roots+=("$dir")
  fi
  local maxdepth=()
  [ -z "$paths" ] && maxdepth=(-maxdepth 1)
  local name_args=()
  IFS=',' read -ra ext_list <<< "$exts"
  for e in "${ext_list[@]}"; do
    [ ${#name_args[@]} -gt 0 ] && name_args+=(-o)
    name_args+=(-name "*.$e")
  done
  local cap=$((CAP_MB * 1000000))
  local total=0 count=0
  while IFS= read -r f; do
    [ "$total" -ge "$cap" ] && break
    local size
    size=$(wc -c < "$f")
    [ "$size" -lt "$MIN_BYTES" ] && continue
    [ "$size" -gt "$MAX_BYTES" ] && continue
    local target="$dest/$count.txt"
    if [[ "$f" == *.html ]]; then
      strip_html < "$f" | clean_stream > "$target"
    else
      clean_stream < "$f" > "$target"
    fi
    printf '\n\n' >> "$target"
    local got
    got=$(wc -c < "$target")
    if [ "$got" -lt "$MIN_BYTES" ]; then
      rm -f "$target"
      continue
    fi
    total=$((total + got))
    count=$((count + 1))
  done < <(find "${roots[@]}" "${maxdepth[@]}" -type f \( "${name_args[@]}" \) \
             -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null \
           | LC_ALL=C sort | shuf --random-source=<(yes 42))
  echo "$count $total"
}

MODE="${1:-all}"
SELECT="${2:-}"
MANIFEST="$OUT/MANIFEST.md"

if [ "$MODE" = "fetch" ] || [ "$MODE" = "all" ]; then
  for entry in "${SOURCES[@]}"; do
    IFS='|' read -r name repo paths exts <<< "$entry"
    if [ -n "$SELECT" ] && [ "$name" != "$SELECT" ]; then continue; fi
    echo "[fetch] $name ($repo)"
    fetch_one "$name" "$repo" "$paths"
  done
fi

if [ "$MODE" = "build" ] || [ "$MODE" = "all" ]; then
  {
    echo "# Dataset manifest"
    echo
    echo "Built: $(date -u +%Y-%m-%d)"
    echo
    echo "| Source | Repository | Commit | License file (first line) | Files | Bytes |"
    echo "|---|---|---|---|---|---|"
  } > "$MANIFEST"
  for entry in "${SOURCES[@]}"; do
    IFS='|' read -r name repo paths exts <<< "$entry"
    if [ ! -d "$SRC/$name/.git" ]; then
      echo "[skip] $name (not fetched)"
      continue
    fi
    echo "[build] $name"
    read -r count total < <(build_one "$name" "$paths" "$exts")
    commit=$(git -C "$SRC/$name" rev-parse --short HEAD 2>/dev/null)
    lic_file=$(ls "$SRC/$name" 2>/dev/null | grep -i -E '^(LICENSE|LICENCE|COPYING)' | head -1)
    lic="unknown"
    [ -n "$lic_file" ] && lic=$(tr -d '\r|' < "$SRC/$name/$lic_file" | sed 's/^\xEF\xBB\xBF//' | grep -m1 '[[:alnum:]]' | sed 's/^[[:space:]]*//' | cut -c1-60)
    echo "| $name | https://github.com/$repo | $commit | ${lic_file:-none}: $lic | $count | $total |" >> "$MANIFEST"
    echo "  files=$count bytes=$total"
  done
  find "$DOCS" -type f -name '*.txt' | shuf --random-source=<(yes 7) | xargs cat > "$OUT/corpus.txt"
  echo
  echo "Corpus: $(wc -c < "$OUT/corpus.txt") bytes -> $OUT/corpus.txt"
  echo "Manifest: $MANIFEST"
fi
