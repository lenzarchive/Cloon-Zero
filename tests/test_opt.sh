#!/bin/sh
# Opt-in features on the CPU emulator: every variant must learn, and checkpoints of different variants must be rejected.
# Usage: tests/test_opt.sh <base-binary> <qknorm-binary> <muon+qknorm-binary> <corpus.txt>
set -u
BASE=$1; QK=$2; MUON=$3; CORPUS=$4
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0
check() { if [ "$2" = ok ]; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }

head -c 400000 "$CORPUS" > "$TMP/c.txt"
for v in base qk muon; do
    case $v in base) BIN=$BASE ;; qk) BIN=$QK ;; muon) BIN=$MUON ;; esac
    "$BIN" train "$TMP/c.txt" 60 "$TMP/$v.bin" 300 --seed 5 > "$TMP/$v.log" 2>&1
    first=$(grep '^step' "$TMP/$v.log" | head -1 | awk '{print $5}')
    last=$(grep '^step' "$TMP/$v.log" | tail -1 | awk '{print $5}')
    ok=$(awk -v a="$first" -v b="$last" 'BEGIN { print (b < a - 0.02) ? "ok" : "bad" }')
    check "$v variant learns (train loss $first -> $last)" "$ok"
done

"$BASE" train "$TMP/c.txt" 70 "$TMP/muon.bin" 300 > "$TMP/x.log" 2>&1
check "default build rejects a Muon checkpoint" "$(grep -q 'different hyperparameters' "$TMP/x.log" && echo ok || echo bad)"
"$MUON" train "$TMP/c.txt" 70 "$TMP/base.bin" 300 > "$TMP/x.log" 2>&1
check "Muon build rejects an AdamW checkpoint" "$(grep -q 'different hyperparameters' "$TMP/x.log" && echo ok || echo bad)"
"$BASE" generate "$TMP/qk.bin" "the " 5 > "$TMP/x.log" 2>&1
check "default build rejects a QK-norm model" "$(grep -q 'different hyperparameters' "$TMP/x.log" && echo ok || echo bad)"
"$MUON" generate "$TMP/qk.bin" "the " 5 > "$TMP/x.log" 2>&1
check "optimizer choice does not affect inference models" "$(grep -q 'different hyperparameters' "$TMP/x.log" && echo bad || echo ok)"

[ $fail -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit $fail
