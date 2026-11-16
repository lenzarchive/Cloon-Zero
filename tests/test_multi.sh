#!/bin/sh
# Multi-GPU logic on the CPU emulator: data-parallel training must match single-GPU training.
# Usage: tests/test_multi.sh <emulator-binary> <corpus.txt>
set -u
BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
SRC=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
head -c 150000 "$SRC" > "$TMP/c.txt"
fail=0

run() { # name, env, flags...
    name=$1; envs=$2; shift 2
    ( cd "$TMP" && env $envs "$BIN" train c.txt 10 "m_$name.bin" 300 --seed 7 "$@" ) > "$TMP/$name.log" 2>&1
    echo $? > "$TMP/$name.rc"
}
losses() { grep -o 'train [0-9.]*' "$TMP/$1.log" | awk '{print $2}' | tr '\n' ' '; }
same() { # a b : compare loss sequences with tolerance
    paste -d' ' "$TMP/$1.l" "$TMP/$2.l" >/dev/null
    awk -v a="$(cat "$TMP/$1.l")" -v b="$(cat "$TMP/$2.l")" 'BEGIN{
        n=split(a,x," "); m=split(b,y," "); if(n!=m||n==0){exit 1}
        for(i=1;i<=n;i++){d=x[i]-y[i]; if(d<0)d=-d; if(d>2e-3)exit 1}}'
}
check() { if [ "$2" = ok ]; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }

run one      "EMU_DEVICES=2" --gpus 1
run two      "EMU_DEVICES=2" --gpus 2
run auto     "EMU_DEVICES=2"
run autoflag "EMU_DEVICES=2" --gpus auto
run dev10    "EMU_DEVICES=2" --devices 1,0
run cap      "EMU_DEVICES=4" --gpus auto
for n in one two auto autoflag dev10 cap; do losses $n > "$TMP/$n.l"; done

[ "$(cat $TMP/one.rc)" = 0 ] && [ -s "$TMP/one.l" ] && check "single GPU trains" ok || check "single GPU trains" bad
grep -q '^GPU: ' "$TMP/one.log" && check "--gpus 1 uses the single-GPU path" ok || check "--gpus 1 uses the single-GPU path" bad
grep -q '^GPUs: 2' "$TMP/two.log" && check "--gpus 2 uses two GPUs" ok || check "--gpus 2 uses two GPUs" bad
grep -q '^GPUs: 2' "$TMP/auto.log" && check "no flag defaults to auto (2 GPUs)" ok || check "no flag defaults to auto (2 GPUs)" bad
grep -q '^GPUs: 2' "$TMP/autoflag.log" && check "--gpus auto uses two GPUs" ok || check "--gpus auto uses two GPUs" bad
grep -q '\[1\] CPU emulation 1 (master)' "$TMP/dev10.log" && check "--devices 1,0 makes device 1 the master" ok || check "--devices 1,0 makes device 1 the master" bad
grep -q 'only 2 of 4' "$TMP/cap.log" && grep -q '^GPUs: 2' "$TMP/cap.log" && check "auto caps the GPU count at ACCUM" ok || check "auto caps the GPU count at ACCUM" bad
for n in two auto autoflag dev10 cap; do
    same one $n && check "loss curve of '$n' matches single GPU" ok || { check "loss curve of '$n' matches single GPU" bad; echo "   one: $(cat $TMP/one.l)"; echo "   $n: $(cat $TMP/$n.l)"; }
done

( cd "$TMP" && EMU_DEVICES=2 "$BIN" train c.txt 14 m_two.bin 300 --seed 8 --gpus 1 ) > "$TMP/resume.log" 2>&1
grep -q 'Resumed from checkpoint' "$TMP/resume.log" && grep -q 'step     14' "$TMP/resume.log" && check "checkpoint from 2 GPUs resumes on 1 GPU" ok || check "checkpoint from 2 GPUs resumes on 1 GPU" bad

run e1 "EMU_DEVICES=2" --gpus 3
run e2 "EMU_DEVICES=2" --gpus 1 --devices 0
run e3 "EMU_DEVICES=2" --devices 0,0
run e4 "EMU_DEVICES=2" --gpus abc
for n in e1 e2 e3 e4; do
    [ "$(cat $TMP/$n.rc)" != 0 ] && check "bad flag '$n' is rejected" ok || check "bad flag '$n' is rejected" bad
done

run lowmem "EMU_DEVICES=2 EMU_FREE_MB=1" --gpus auto
[ "$(cat $TMP/lowmem.rc)" != 0 ] && grep -q 'skipped' "$TMP/lowmem.log" && check "auto skips GPUs without enough memory" ok || check "auto skips GPUs without enough memory" bad

run solo "EMU_DEVICES=1"
grep -q '^GPU: ' "$TMP/solo.log" && [ "$(cat $TMP/solo.rc)" = 0 ] && check "one visible GPU: default stays single-GPU" ok || check "one visible GPU: default stays single-GPU" bad

[ $fail = 0 ] && echo "RESULT: PASS" || { echo "RESULT: FAIL"; exit 1; }
