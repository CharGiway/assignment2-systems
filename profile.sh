#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="artifacts"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
VOCAB_SIZE="${VOCAB_SIZE:-10000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SIZES_STR="${SIZES:-small medium large xl 2.7B}"
SEQS_STR="${SEQS:-128 256 512 1024}"
MODES_STR="${MODES:-fwd fwd_bwd train}"

mkdir -p "$OUT_DIR"
INDEX="$OUT_DIR/nsys_runs.csv"
echo "size,seq,mode,report,kernels_csv" > "$INDEX"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found on PATH"
  exit 1
fi

read -r -a SIZES <<< "$SIZES_STR"
read -r -a SEQS <<< "$SEQS_STR"
read -r -a MODES <<< "$MODES_STR"

for s in "${SIZES[@]}"; do
  for seq in "${SEQS[@]}"; do
    for m in "${MODES[@]}"; do
      base="${OUT_DIR}/nsys_${s}_seq${seq}_${m}"
      uv run nsys profile -o "${base}" --trace=cuda,osrt,nvtx --sample=none --capture-range=nvtx --capture-range-end=nvtx python -m cs336_basics.profile_runner --size "$s" --context_length "$seq" --device "$DEVICE" --dtype "$DTYPE" --vocab_size "$VOCAB_SIZE" --batch_size "$BATCH_SIZE" --mode "$m"
      rep="${base}.nsys-rep"
      kernels_csv="${base}_kernels.csv"
      uv run nsys stats --report cuda_gpu_kernel_summary --format csv "$rep" > "$kernels_csv"
      echo "${s},${seq},${m},${rep},${kernels_csv}" >> "$INDEX"
    done
  done
done

echo "Done. Index: $INDEX"

