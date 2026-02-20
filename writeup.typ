= Problem (benchmarking_script)

== (a) Benchmarking Script
- Python 脚本：`cs336-basics/cs336_basics/benchmark.py`
  - 按给定超参数或预设尺寸（small/medium/large/xl/2.7B）初始化模型；
  - 生成随机 batch（`[batch_size, context_length]`）；
  - 先进行 `w` 次预热，再计时 `n` 次；`mode` 支持 `fwd` 或 `fwd_bwd`；
  - 使用 `timeit.default_timer()`；CUDA 上每步后调用 `torch.cuda.synchronize()`。
- Shell 脚本：`benchmark.sh`
  - 集中定义实验设计并批量运行：
    - 尺寸：`small, medium, large, xl, 2.7B`
    - 预热轮次：`0, 1, 2, 5`
    - 模式：`fwd` 与 `fwd_bwd`
  - 默认 `vocab_size=10000`、`batch_size=4`，可通过变量修改 `context_length/device/steps` 等；
  - 自动收集结果到 `artifacts/bench_results.csv`，并生成 `artifacts/bench_table.typ` 以供下方表格引用。

== (b) Timings (example on CPU)
- 设置：`vocab_size=10000`、`batch_size=4`、`context_length=128`、`warmup=5`、`steps=10`。
- small（768/3072/12/12）：
  - forward ≈ 0.116 s（std ≈ 0.005 s）
  - forward+backward ≈ 0.450 s（std ≈ 0.033 s）
- 结论：反向显著更慢；在固定设置下标准差较小，跨 10 次测量波动不大。

== (c) Effect of Warm-Up
- 无预热时包含“冷启动”开销（内存分配、内核选择/缓存等），平均耗时偏大且方差更高。
- 仅 1–2 次预热仍可能不足以充分稳定缓存/调度，结果仍与 5 次预热不同。
- 充分预热后计时更能代表稳态性能。

== Aggregated Table
#include("artifacts/bench_table.typ")
