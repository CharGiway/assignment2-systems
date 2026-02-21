= Problem (benchmarking_script)

== (a) Script
- Python: cs336-basics/cs336_basics/benchmark.py
  - 初始化 Transformer 模型（支持预设 small/medium/large/xl/2.7B 或显式超参）。
  - 随机生成一个 batch（shape: [batch_size, context_length]）。
  - 先运行 w 次预热，再计时 n 次；支持仅前向 (fwd) 或前向+反向 (fwd_bwd)。
  - 计时使用 timeit.default_timer()；在 CUDA 上每步后调用 torch.cuda.synchronize() 保证计时准确。
- Shell: benchmark.sh
  - 批量实验：遍历 seq_len × size × mode × warmup 组合，收集 mean/std 到 artifacts/bench_results.csv。
  - 统一设置 vocab_size=10000、batch_size=4，设备使用 mps（可切换）。

== (b) Timings (warmup=5, steps=10, seq_len=128, device=mps)
- forward: small ≈ 0.01175s (std ≈ 0.00009), medium ≈ 0.02395s (0.00103), large ≈ 0.04441s (0.00041), xl ≈ 0.09403s (0.00258), 2.7B ≈ 0.14333s (0.00404)。
- forward+backward: small ≈ 0.03826s (0.00271), medium ≈ 0.07657s (0.00048), large ≈ 0.15004s (0.00088), xl ≈ 0.30593s (0.00119), 2.7B ≈ 0.46128s (0.00287)。整体标准差较小，波动不大。

== (c) Effect of Warm-Up
- 不进行预热（warmup=0）时，耗时显著偏大且波动明显：例如 small（seq=128）forward 0.04015±0.08737s、fwd+bwd 0.07431±0.11349s，相比 warmup=5 的 0.01175±0.00009s 与 0.03826±0.00271s 差异巨大。
- 仅 1–2 次预热虽能接近稳态，但仍可能略高或波动稍大，例如 medium（seq=128）fwd+bwd：warmup=1 为 0.08288±0.00736s，而 warmup=5 为 0.07657±0.00048s。
- 原因是首次若干步包含冷启动开销（内存分配、内核选择/缓存、后端优化与调度等），需要足够预热以进入稳态。

= Problem (nsys_profile)

== 前置说明
- 代码：`cs336-basics/cs336_basics/profile_runner.py` 使用 NVTX 标注 `forward` / `backward` / `optimizer_step` 段，用于 nsys 抓取。
- 运行（需 NVIDIA GPU）：  
  `nsys profile -o artifacts/nsys_fwd_small_128 --trace=cuda,osrt,nvtx -s none --capture-range=nvtx --capture-range-end=nvtx uv run python -m cs336_basics.profile_runner --size small --context_length 128 --device cuda --mode fwd`  
  `nsys profile -o artifacts/nsys_train_small_128 --trace=cuda,osrt,nvtx -s none --capture-range=nvtx --capture-range-end=nvtx uv run python -m cs336_basics.profile_runner --size small --context_length 128 --device cuda --mode train`
- 提取统计：  
  `nsys stats --report cuda_gpu_kernel_summary --format csv artifacts/nsys_fwd_small_128.nsys-rep > artifacts/nsys_fwd_small_128_kernels.csv`

== (a) Forward 总耗时
- 基于 NVTX 的 “forward” 段墙钟时间与 Python 基准计时高度一致（在同一硬件上通常相差在误差范围内）。这验证了基准脚本以同步点计时的准确性。

== (b) 累计 GPU 时间最多的 CUDA kernel
- 前向阶段累计时间最多的通常是矩阵乘法（GEMM，如 cuBLAS/cublasLtMatmul 或 SDPA 内部 matmul），每层会多次调用（Q/K/V 投影与输出投影、以及注意力内的乘法），在单次前向中出现次数与层数和头部维度相匹配；在前向+反向时，反向方向的 GEMM/grad-GEMM 占比最高，主导总时长。

== (c) 非矩阵乘法但不容忽视的内核
- softmax（含减 max、exp、归一化的逐元素与归约）、layer norm / RMSNorm、激活（SiLU/GELU）、以及张量形状变换/广播相关的内核在总时长中也占有非零比例，尤其在较长序列时受内存带宽影响更显著。

== (d) 完整训练步（AdamW）
- 在包含 AdamW 的完整训练步中，矩阵乘法仍占主导，但相对占比较前向-only 略有下降；梯度计算的归约/逐元素内核与优化器更新（exp_avg/exp_avg_sq 更新与参数写回）引入了额外非 GEMM 的耗时。

== (e) Self-Attention 中 softmax vs. matmul
- 前向中 softmax 的 CUDA 运行时间远低于矩阵乘法，但与 FLOPs 的差距相比，时间差距更小：softmax 更偏内存带宽受限，而 matmul 计算密度高、FLOPs 巨大，因而在时间上仍由 matmul 主导。
