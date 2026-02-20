#!/usr/bin/env bash # 使用 env 查找 bash 解释器，提升可移植性
set -euo pipefail # 严格模式：出错退出、未定义变量报错、管道错误传播

SIZE="small"       # 预设模型尺寸：small/medium/large/xl/2.7B；留空则走自定义超参
D_MODEL="768"      # 自定义 d_model（当 SIZE 为空时生效）
D_FF="3072"        # 自定义前馈层宽度 d_ff（当 SIZE 为空时生效）
NUM_LAYERS="12"    # 自定义 Transformer 层数（当 SIZE 为空时生效）
NUM_HEADS="12"     # 自定义注意力头数（当 SIZE 为空时生效）
VOCAB_SIZE="10000" # 词表大小（作业建议使用 10,000）
CONTEXT_LENGTH="1024" # 上下文长度（序列长度）
BATCH_SIZE="4"     # 批大小
DEVICE="mps"       # 设备：cpu/cuda/mps
DTYPE="float32"    # 浮点精度：float32/float16/bfloat16
WARMUP="5"         # 预热步数（不计时）
STEPS="10"         # 计时步数
MODE="fwd_bwd"     # 计时模式：fwd（仅前向）或 fwd_bwd（前向+反向）
OUT_JSON=""        # 可选：结果写入的 JSON 路径（留空则仅打印）

# 单次运行或实验批量运行的开关：设置 RUN_SWEEP=1 进行批量实验
: "${RUN_SWEEP:=1}"

if [[ "$RUN_SWEEP" == "0" ]]; then
  ARGS=( --vocab_size "$VOCAB_SIZE" --context_length "$CONTEXT_LENGTH" --batch_size "$BATCH_SIZE" --device "$DEVICE" --dtype "$DTYPE" --warmup "$WARMUP" --steps "$STEPS" --mode "$MODE" )
  if [[ -n "$SIZE" ]]; then
    ARGS+=( --size "$SIZE" )
  else
    ARGS+=( --d_model "$D_MODEL" --d_ff "$D_FF" --num_layers "$NUM_LAYERS" --num_heads "$NUM_HEADS" )
  fi
  if [[ -n "$OUT_JSON" ]]; then
    ARGS+=( --out_json "$OUT_JSON" )
  fi
  uv run python -m cs336_basics.benchmark "${ARGS[@]}"
else
  ART_DIR="artifacts"
  mkdir -p "$ART_DIR"
  OUT_CSV="$ART_DIR/bench_results.csv"
  : > "$OUT_CSV"
  echo "seq,size,mode,warmup,mean_s,std_s" >> "$OUT_CSV"

  SIZES=("small" "medium" "large" "xl" "2.7B")
  WARMUPS=("0" "1" "2" "5")
  MODES=("fwd" "fwd_bwd")
  SWEEP_SEQS_STR="${SWEEP_SEQS:-"64 128 256 512 1024"}"
  IFS=' ' read -r -a SEQS <<< "$SWEEP_SEQS_STR"

  # 遍历顺序与 CSV 列顺序对齐：seq -> size -> mode -> warmup
  for seq in "${SEQS[@]}"; do
    for s in "${SIZES[@]}"; do
      for m in "${MODES[@]}"; do
        for wu in "${WARMUPS[@]}"; do
          args=( --size "$s" --vocab_size "$VOCAB_SIZE" --context_length "$seq" --batch_size "$BATCH_SIZE" --device "$DEVICE" --dtype "$DTYPE" --warmup "$wu" --steps "$STEPS" --mode "$m" )
          out_json="$ART_DIR/bench_seq${seq}_${s}_${m}_w${wu}.json"
          args+=( --out_json "$out_json" )
          echo "Running: seq=$seq size=$s mode=$m warmup=$wu"
          if uv run python -m cs336_basics.benchmark "${args[@]}"; then
            UV_OUT_JSON="$out_json" UV_OUT_CSV="$OUT_CSV" uv run python - <<'PY'
import os, json, sys
path = os.environ["UV_OUT_JSON"]
csvp = os.environ["UV_OUT_CSV"]
with open(path, "r", encoding="utf-8") as f:
    o = json.load(f)
row = f"{o['context_length']},{o['size']},{'fwd' if 'forward' in o['mode'] and 'backward' not in o['mode'] else 'fwd_bwd'},{o['warmup']},{o['mean_s']:.6f},{o['std_s']:.6f}\n"
with open(csvp, "a", encoding="utf-8") as f:
    f.write(row)
PY
          else
            echo "SKIP,seq=$seq,size=$s,mode=$m,warmup=$wu" >&2
          fi
        done
      done
    done
  done

  echo "Sweep finished. CSV: $OUT_CSV"
fi
