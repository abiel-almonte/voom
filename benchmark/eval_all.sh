#!/usr/bin/env bash
# Usage: ./eval_all.sh checkpoints/rev1
set -u

DIR="${1:?usage: $0 <ckpt_dir> [extra args...]}"
shift || true

WEIGHTS_DIR="${DIR}/weights"
LOG_DIR="${DIR}/eval/logs"
SUMMARY="${DIR}/eval/summary.tsv"
mkdir -p "$LOG_DIR"

# header: step  ckpt  IoU  P  R  F1  mIoU
printf "step\tckpt\tIoU\tP\tR\tF1\tmIoU\n" > "$SUMMARY"

shopt -s nullglob
ckpts=( "$WEIGHTS_DIR"/*.pt )
echo "found ${#ckpts[@]} ckpts in $WEIGHTS_DIR"

for ckpt in "${ckpts[@]}"; do
    name="$(basename "$ckpt" .pt)"
    log="$LOG_DIR/${name}.log"

    if [[ -s "$log" ]] && grep -q "^Occ:" "$log"; then
        echo "[skip] $name  (log exists)"
    else
        echo "[eval] $name"
        python3 benchmark/eval.py --ckpt "$ckpt" "$@" > "$log" 2>&1 || {
            echo "  FAILED — see $log"
            continue
        }
    fi

    # Parse: "Occ:  IoU=0.xxxx  P=0.xxxx  R=0.xxxx  F1=0.xxxx"
    line="$(grep '^Occ:' "$log" | tail -1)"
    iou=$(echo "$line" | sed -n 's/.*IoU=\([0-9.]*\).*/\1/p')
    p=$(echo "$line"   | sed -n 's/.*P=\([0-9.]*\).*/\1/p')
    r=$(echo "$line"   | sed -n 's/.*R=\([0-9.]*\).*/\1/p')
    f1=$(echo "$line"  | sed -n 's/.*F1=\([0-9.]*\).*/\1/p')
    miou=$(grep '^mIoU' "$log" | sed -n 's/.*: \([0-9.]*\).*/\1/p')
    miou="${miou:-0.0000}"

    # step number if name is like step_170000
    step=$(echo "$name" | sed -n 's/.*step_\([0-9]*\).*/\1/p')
    step="${step:-0}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$step" "$name" "$iou" "$p" "$r" "$f1" "$miou" >> "$SUMMARY"
    echo "  IoU=$iou  mIoU=$miou"
done

echo
echo "=== top 10 by IoU ==="
(head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -k3 -gr | head -10) | column -t -s $'\t'

echo
echo "=== top 10 by mIoU ==="
(head -1 "$SUMMARY"; tail -n +2 "$SUMMARY" | sort -k7 -gr | head -10) | column -t -s $'\t'

echo
echo "full summary: $SUMMARY"
