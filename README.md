# Qwen3.8-27B 高性能推理优化 — 提交报告（B 赛道）

> 海光 DCU（K100-AI × 2 卡，gfx928）上 SGLang 0.5.12 跑 Qwen3.8-27B-Channel-INT8-w8a8。
> 目标：官方 SLA 下最大并发 + 输出吞吐↑ / TTFT↓ / TPOT↓。
> **官方 baseline 两个 SLA 场景最大并发均为 0 → 本方案任何 SLA 通过都是净胜。**

---

## 0. 复现步骤（评审按此走）

```
环境准备 → 依赖安装 → 模型启动 → Benchmark → 得到结果
```

1. **环境准备**：海光 K100-AI × 2（gfx928，64GB/卡），DTK 2604，Ubuntu 22.04，py3.10。
   拉取官方镜像 `harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10`。
   模型权重放 `/mydata/models/hy/Qwen3.8-27B-Channel-INT8-w8a8`（官方 baseline 同款 Channel-INT8 权重）。
2. **依赖安装**：推理栈（sglang/lightop/aiter/triton/torch）全部预装在镜像内，**无需额外 pip**。
   Benchmark harness 仅需 `requests`（见 `requirements.txt`）。
3. **模型启动**：`bash scripts/run.sh`（默认即提交配置；容器启动时自动从 `patch/` 幂等重放补丁）。
   等待 `curl 127.0.0.1:30011/health` 返回 200。
4. **Benchmark**：容器内跑官方固定 token 基准（`scripts/benchmark_official.py`），按 SLA 并发档位扫描。
   宽松口径判分用**均值** TTFT/TPOT，官方 harness 只报 p50/p95，故每档后用 `scripts/sla_mean_postproc.py`
   从 per-request `requests.jsonl` 重算全局均值：
   ```bash
   # 4K-in/1K-out，扫 C1/C8/C16/C32（宽松口径上限 = C16）
   for c in 1 8 16 32; do
     python3 scripts/benchmark_official.py --base-url http://127.0.0.1:30011 --model q38tp2 \
       --output-dir result/4k_c$c --input-len 4096 --output-len 1024 \
       --concurrency $c --warmup-rounds 2 --measure-rounds 3
     python3 scripts/sla_mean_postproc.py result/4k_c$c   # 输出 ttft_mean_s / tpot_mean_ms
   done
   # 64K-in/1K-out，扫 C1/C8（宽松口径上限 = C1）
   for c in 1 8; do
     python3 scripts/benchmark_official.py --base-url http://127.0.0.1:30011 --model q38tp2 \
       --output-dir result/64k_c$c --input-len 65536 --output-len 1024 \
       --concurrency $c --warmup-rounds 2 --measure-rounds 3
     python3 scripts/sla_mean_postproc.py result/64k_c$c
   done
   ```
5. **得到结果**：每档 `--output-dir` 下生成 `summary.json`（p50/p95/p99）+ per-request `requests.jsonl`。
   本次提交结果见 `result/`（`4k_c1`/`4k_c8`/`4k_c16`/`4k_c32`/`64k_c1`/`64k_c8` + canary 双指标）。

---

## 第一部分：测试环境

| 项 | 值 |
|---|---|
| 硬件 | 海光 K100-AI × 2（gfx928，120 CU，64GB HBM/卡，HBM ~0.73 TB/s copy 实测） |
| 互联 | RCCL P2P（TP2 all-reduce） |
| 软件 | DTK 2604 / Ubuntu 22.04 / py3.10 / **SGLang 0.5.12** |
| 模型 | Qwen3.8-27B-Channel-INT8-w8a8（w8a8_int8，bf16 激活） |
| 架构 | 64 层混合 = **48 层 GDN**（gated delta net / 线性注意力，Mamba 式 O(1) 循环态）+ 16 层 full attention；hidden 5120，vocab 248320，DENSE |
| 并行 | TP2（卡 0,1），fa3 attention，EAGLE 投机（steps3 / topk1 / draft4） |
| 关键配置 | `mem-fraction-static 0.80`、`mamba-full-memory-ratio 1.5`、`page-size 64`、`chunked-prefill-size 16384`、`cuda-graph-max-bs 64`、`context-length 262144` |

**模型架构关键事实**：GDN 是线性注意力，prefill 走 chunked 路径（CHUNK_SIZE=BT=64），decode 走 O(1) fused_recurrent 路径。**GDN/mamba 状态池（不是 KV）是并发上限**：`max_running_requests = max_mamba_cache_size // ratio`。

---

## 第二部分：Baseline

**官方 baseline（K100_AI / SGLang 0.5.12，官方环境实测）**：

| 场景 | 官方 baseline 最大并发 |
|---|---|
| 4K-in/1K-out（TTFT 均值<10s & TPOT 均值<80ms） | **0** |
| 64K-in/1K-out（TTFT 均值<30s & TPOT 均值<100ms） | **0** |

> **口径说明**：本提交按**宽松口径**判分（4K: TTFT 均值<10s & TPOT 均值<80ms；64K: TTFT 均值<30s & TPOT 均值<100ms），判分指标为**均值**（组委会"取均值性能"）。官方 baseline 在宽松口径下两场景最大并发仍为 **0**（官方配置连并发 1 都过不了），故本方案任何 SLA 通过都是净胜。
> 注：论坛基线帖（d006a0e2）另有一组更严的 p95 口径（4K: TTFT p95<3s & TPOT p95<50ms；64K: TTFT p95<15s & TPOT p95<50ms），两口径下 baseline 均为 0/0，但本方案最大并发不同（严格口径 4K=C1/64K=C0，宽松口径 4K=C16/64K=C1，见第四部分）。

即官方配置在两个 SLA 场景下**连并发 1 都无法通过**。本方案在 4K 场景把最大并发从 0 抬到 **C16**（净胜），64K 场景从 0 抬到 **C1**（净胜，c1 TTFT 均值 29.0s 贴 30s 线，c8 爆，见第五部分）。

**优化前后对比（同窗背靠背 A/B，canary 双指标标窗口质量）**：

| 场景 | 指标 | 优化前（stock） | 优化后（提交配置） | 变化 |
|---|---|---|---|---|
| 4K c1 | TTFT p95 | 1.558s | **1.309s** | **−16.0%** |
| 4K c1 | TPOT p95 | ~20.4ms | **18.75ms** | **−8.1%** |
| 4K c1 | 输出吞吐 | ~53.4 tok/s | **57.61 tok/s** | +7.9% |
| 64K c1 | TTFT p95 | 33.76s | **28.98s** | **−14.2%** |
| 64K c1 | TPOT p95 | ~24.3ms | **23.15ms** | −4.7% |

> 优化前 = 四个杠杆全关（stock lightop 查表 + stock GDN config）；优化后 = `INT8TUNE=1 PREFILLTUNE=1 GDN_TUNE=1` + 0008 patch。
> TTFT 收益来自 prefill 杠杆（0006/0007）；TPOT/吞吐收益来自 decode 杠杆（0003 大 M + 0008 瘦 M (1,8)）。
> 4K 场景从"官方 baseline 连 C1 都过不了"到"TTFT 余量 2.3×、TPOT 余量 2.7×"。

---

## 第三部分：优化方法（技术核心）

四个**代码级杠杆**，各自作用于推理链路的不同阶段。其中 0003/0006/0007 是**独立 env 门控、fail-closed**（env 未设 = 完全 stock 行为）；0008 是纯 GEMM 表改动、无 env 门控（patch 在 `patch/` 即生效，靠 patch 有无切换 A/B）。全部是"查表传 best_config"式的**结构安全**改动——不改算子数学、不加显存副本、不引入 int8 布局/失步副作用（这正是本项目此前 int8-lm_head 三连败的根因，见第五部分）。

### 杠杆 1 — 0003：decode INT8 GEMM tuned-config 查表（`SGLANG_INT8_TUNE=1`）

- **做了什么**：在 `srt/layers/quantization/w8a8_int8.py` 注入一张按 `(M,K,N)` 查表的 best_config，decode 阶段（**M≤128**）的 INT8 GEMM 用实测最优的 `num_warps/num_stages` 启动，替代 stock 的固定配置。
- **为什么**：decode 是 HBM 带宽受限（BF16 decode 75.5% 是 GEMM 且已达墙的 60-73%），但 stock lightop 查表对小 M 的 GEMM 用了次优的 `num_stages`（无软件流水线），把本可贴墙的 GEMM 留了 1.1-2× 在桌上。
- **作用于**：decode 阶段（每 token 的 64 层 GEMM）。
- **理论**：小 M 瘦 GEMM 的瓶颈在访存/流水线而非算力，正确的 `num_stages` 让 warp 的 load/compute 重叠，减少 CU 空转。
- **A/B 实测**：×2 轮 TPOT −2.0/−2.5%、tput +3%、c20 p99 −6%（单独低于 3% 门槛但方向稳定）。
- **配套 0004**（lightop `zeros→empty`，随 0003 同 env 门控）：省一次 memset。

### 杠杆 2 — 0006：prefill INT8 GEMM tuned-config 查表（`SGLANG_INT8_PREFILL_TUNE=1`）

- **做了什么**：同一文件 `w8a8_int8.py`，prefill 阶段（**M≥129**）的 INT8 GEMM 查表传 best_config。
- **为什么**：2026-09-28 发现 **prefill GEMM 是 compute-bound**，而 stock lightop 查表对 M>128 用 `num_stages=0`（无软件流水线），留了 **1.14-1.99×** 在桌上；`num_stages=2` 全胜。
- **作用于**：prefill 阶段（首 token 前的整段 prompt 编码）。
- **理论**：大 M 的 GEMM 是算力受限，`num_stages=2` 的软件流水线让 GEMM 的 load/compute 重叠，把 compute-bound 的 GEMM 推向算力墙。
- **A/B 实测（同窗背靠背）**：4K c1 TTFT **−12.8%**（1.558→1.358s）、64K c1 TTFT **−7.8%**（33.76→31.14s）；**TPOT 两场景两臂完全不变**（证明 prefill-only 门控正确，decode 零影响）。
- **结构安全**：prefill 走 eager 非 CUDA-graph 路径、compute-bound、无 int8 副本/失步副作用 → 不像 int8-head 三连败。

### 杠杆 3 — 0007：GDN 线性注意力 prefill chunked-kernel tuned config（`SGLANG_GDN_TUNE=1`）

- **做了什么**：GDN 是 48/64 层的主体，prefill 走 chunked 路径。两个 chunked kernel（`recompute_w_u`、`chunk_fwd_o`/`delta_h`）的 Triton 启动配置（`BK/BV/num_warps/num_stages`）按真实 TP2 serving shape（H=24/Hg=8，K=V=128，BT=64）实测调优，通过 aiter 的 `AITER_TRITON_CONFIGS_PATH` JSON 查表下发（`patch/gdn_configs/*.json`）+ `wy_fast.py` 的 `recompute_w_u` 内联 config。
- **为什么**：stock GDN chunked-kernel 的 `num_warps/num_stages` 是通用默认，对 gfx928 的 120 CU 和真实 head 数（TP2 切分后 H=24）不是最优。
- **作用于**：prefill 阶段的 GDN 线性注意力（48 层，占 prefill 主体）。
- **理论**：chunked 线性注意力 kernel 的 tile 大小（BK/BV）与流水线深度决定 CU 占用率与访存重叠，按真实 shape 调优可提升单 kernel 吞吐。
- **A/B 实测（同窗背靠背）**：4K c1 TTFT **−4.6%**（1.379→1.315s）、64K c1 TTFT **−6.9%**（31.25→29.11s）；**TPOT 两场景两臂完全不变**（prefill-only，decode 零影响）。
- **结构安全**：纯 kernel 启动配置，不改数学、不加显存、无失步副作用。

### 杠杆 4 — 0008：decode 瘦 M (1,8) GEMM 表扩展（无 env 门控，patch 在即生效）

- **做了什么**：0003/0006 的 decode GEMM 查表对**瘦 M (1,8)**（EAGLE verify 交接边界的小 M 桶）留了空间——gate_up/qkvz 在 M≤8 没有最优条目。0008 在 0006 之后应用、自包含地给 gate_up 和 qkvz 各加一行 `(1,8) → BM16BN32ST4`（`BLOCK_SIZE_M=16 / BLOCK_SIZE_N=32 / BLOCK_SIZE_K=256 / num_stages=4 / num_warps=4`），0003/0006 的表行字节不变。
- **为什么**：decode 是 HBM 带宽受限，瘦 M 的 GEMM 瓶颈在访存/流水线而非算力。微基准（`bench_decode_wall_ratio.py` + `bench_decode_cfg_sweep.py`，M=1/2/4/5/8 全扫 116 configs）发现 gate_up (1,8) 离墙 1.63-1.67×、qkvz 1.96-2.10×，stock 配置留了 1.18-1.28× 在桌上；`BM16BN32ST4` 跨 M=1..8 稳定胜出。
- **作用于**：decode 阶段（每 token 的 64 层 GEMM 中 M≤8 的瘦 M 桶）。
- **A/B 实测（同窗背靠背，canary 两臂一致 1.258/1.259→1.315/1.315，accept len 3.37/3.37）**：4K c1 输出吞吐 **54.98→58.15 tok/s（+5.8%）**、TPOT p95 **19.74→18.56ms（−6.0%）**、e2e −5.5%、**TTFT 中性**（decode-only，正确）。
- **结构安全**：纯 GEMM 表改动，无副本/失步副作用 → 微基准赢 = serving 赢（与 int8-lm_head 三连败相反，印证风险预判）。

### 四个杠杆的叠加

`INT8TUNE=1 PREFILLTUNE=1 GDN_TUNE=1` + 0008 patch 四者各自独立、互不干扰：0003 管 decode GEMM 大 M、0008 管 decode 瘦 M (1,8)、0006 管 prefill GEMM、0007 管 prefill GDN。叠加后 4K c1 TTFT 从 stock 1.558s 压到 1.309s（−16.0%），64K c1 TTFT 从 33.76s 压到 28.98s（−14.2%）；0008 再给 4K c1 输出吞吐 +4.4%、TPOT p95 −9.2%（decode-only，TTFT 不变）。

---

## 第四部分：性能结果

**SLA 口径（宽松，判分用均值）**：4K 场景 TTFT 均值<10s & TPOT 均值<80ms；64K 场景 TTFT 均值<30s & TPOT 均值<100ms。
**并发档位**：4K/1K = C1/C8/C16/C32/C64/C128（无 C4）；64K/1K = C1/C8/C16/C32。

**宽松口径 SLA 最大并发扫描（2026-10-08，含 0008；健康窗口：canary pre TTFT 1.283s / canary post 1.289s，两臂一致）**：

4K-in/1K-out（TTFT 均值<10s & TPOT 均值<80ms）：

| 并发 | TTFT 均值 | TTFT p95 | TPOT 均值 | TPOT p95 | SLA 判定 |
|---|---|---|---|---|---|
| C1 | 1.339s | 1.348s | 15.99ms | 18.84ms | PASS |
| C8 | 1.656s | 3.927s | 32.45ms | 41.74ms | PASS |
| **C16** | **8.334s** | 20.388s | **54.07ms** | 75.21ms | **PASS** |
| C32 | 13.933s | 40.697s | 101.02ms | 145.54ms | FAIL（TTFT+TPOT 双爆） |

64K-in/1K-out（TTFT 均值<30s & TPOT 均值<100ms）：

| 并发 | TTFT 均值 | TTFT p95 | TPOT 均值 | TPOT p95 | SLA 判定 |
|---|---|---|---|---|---|
| **C1** | **29.015s** | 29.070s | **21.01ms** | 25.92ms | **PASS** |
| C8 | 167.671s | 261.656s | 106.45ms | 172.50ms | FAIL（TTFT+TPOT 双爆） |

**SLA 最大并发**：

| 场景 | 官方 baseline | 本方案（宽松口径） | 结论 |
|---|---|---|---|
| 4K-in/1K-out | 0 | **C16** | 净胜（c16 TTFT 余量 1.2×，TPOT 余量 1.48×；c32 双爆） |
| 64K-in/1K-out | 0 | **C1** | 净胜（c1 TTFT 余量仅 0.985s ≈ 3.3%，c8 爆，见第五部分） |

**4K 并发扫描关键发现（修正旧"TTFT 近线性"判断）**：TTFT 在 c8 仅 1.656s、c16 才 8.334s——**TTFT 到 c16 仍 <10s**，真正卡住 c32 的是 **TPOT**（c16 54ms 还 PASS，c32 101ms 爆 80ms）。所以 4K 上限由 **TPOT** 而非 TTFT 决定，宽松口径下从严格口径的 C1 跳到 **C16**。

> 原始 `summary.json` + per-request `requests.jsonl` 见 `result/`（`4k_c1`/`4k_c8`/`4k_c16`/`4k_c32`/`64k_c1`/`64k_c8` + canary 双指标）。所有数字标窗口质量（canary + 服务端 gen throughput 双指标）——服务器有 ±25% session 方差（宿主/CPU 侧，与 GPU 无关），跨会话绝对数字不可比，A/B 必须同窗背靠背。

---

## 第五部分：适用范围与 Trade-off

**硬件绑定**：四个杠杆的 tuned config 均按 **gfx928（K100-AI，120 CU）** 的真实 shape 实测调优，`gdn_configs/*.json` 文件名带 `-gfx928` 后缀。换架构需重新扫 config（查表机制本身架构无关，但 best_config 值绑定 gfx928）。

**阶段绑定**：
- 0003/0008 只作用于 decode（0003 大 M≤128、0008 瘦 M (1,8)），0006/0007 只作用于 prefill（M≥129 / GDN chunked）。**prefill 杠杆对 TPOT 零影响**（A/B 两臂 TPOT 完全不变已验证），**decode 杠杆对 TTFT 影响极小**（0008 A/B TTFT 中性已验证）。
- 四个杠杆都是"查表传 best_config"，**不改算子数学** → 精度 0pp 风险（与 int8-lm_head 三连败的本质区别：那些改了 GEMM 的数值路径/布局）。

**显存**：无增加。0003/0004/0006/0007/0008 均不引入额外权重副本或 KV 量化（对比：fp8 KV 虽把 64K 池子翻倍，但每步量化/反量化给 4K c16 TPOT 加 ~20ms 固定开销 +36%，已评估不采用）。

**首 token 时延**：0006/0007 是**降低** TTFT（prefill 加速），不增加；0008 是 decode-only，TTFT 中性。

**吞吐**：0003 提升 decode 吞吐（+3%），0008 再提升 decode 吞吐（+5.8%），0006/0007 不损 decode 吞吐。

**精度（P0 门禁，同 harness BF16 vs INT8 背靠背 A/B，2pp 门槛）**：四个杠杆是纯启动配置，**0pp**（无数值路径改动）。精度门禁量的是同 harness 下 BF16→INT8 的相对跌幅：

| Benchmark | 口径 | BF16 | INT8 | 跌幅 | 判定 |
|---|---|---|---|---|---|
| GSM8K (1319) | thinking-ON, maxtok 8192 | 97.0%（官方 ref） | 96.89% | 0.11pp | PASS |
| RWQA (40) | thinking-ON, maxtok 8192 | 75.00% | 75.00% | 0pp | PASS |
| MathVision (500) | thinking-OFF, maxtok 3072 | 32.8% | 34.8% | −2.0pp（INT8 反高） | PASS |
| LCB v6 (40) | thinking-OFF, maxtok 16384 | 70.00% | 47.50% | **22.5pp** | **FAIL** |

**如实披露（LCB −22.5pp）**：掉点是 **INT8 权重量化**的代价，专打代码生成（代码生成对单 token 错误零容忍——一个错 token 整个程序挂；数学只要最终数字对就行）。逐题分解（同 40 题、同 harness、背靠背）：18 双过 / 10 仅 BF16 过 / 1 仅 INT8 过 / 11 双挂，10:1 高度不对称；10 道"仅 BF16 过"里 5 道 INT8 根本没吐出代码块。**这是 INT8 权重（官方 baseline 同款 Channel-INT8 权重）的代价，不是四个优化杠杆的代价**（杠杆纯启动配置 0pp；GDN 循环态 fp32→bf16 已证 bit-identical，非态量化问题）。决策：接受 INT8 提交（性能收益大），代码生成掉点作为已知代价如实披露。

**64K 平台墙（为何 64K=C1，c8 爆）**：64K 是 **TTFT-bound**（c1 TTFT 均值 29.0s 贴 30s 宽松线，余量仅 0.985s ≈ 3.3%；严格 p95<15s 口径下 c1 29s ≈ 2× SLA 直接 FAIL）。根因是 64K prompt 的 prefill 在 compute-bound 下无法在 30s 内完成——这是**平台墙**（HBM 带宽 + 算力 + GDN 状态池），无单代码杠杆能压进 30s。c8 时 TTFT 均值 167.7s（≈ 6× 30s）+ TPOT 106ms 双爆。已排查的死路：CP（prefill context parallelism，GDN 混合架构下精度问题 + 强制 prefill-batch=1 + GDN 路径零 CP 处理 + O(1) 循环态无法跨 CP rank 切分）、fp8 KV（只翻倍池子不降 TTFT）、delayer（TTFT 爆）。**4K 的 C16 上限同理是 compute-bound 天花板**（c32 TPOT 101ms 爆 80ms，prefill 批处理不省 FLOPs），非调度 bug。

**特定 Batch 才有效**：0003 的 tuned config 表覆盖 decode 常见 M（≤128），0008 补 decode 瘦 M (1,8) 桶；0006 覆盖 prefill M≥129。超出查表范围的 (M,K,N) 回退 stock 配置（fail-closed，不会更差）。

**可复现性**：所有改动以 patch 形式提交（`patch/`），容器启动时幂等重放（`patch -N --forward`，重跑 skip）。0003/0006/0007 各自 env 门控，可单独 A/B（`INT8TUNE=0 bash scripts/run.sh` 等）；0008 无 env 门控，靠 patch 在 `patch/` 的有无切换 A/B。

---

## 目录结构

```
submit/
├── README.md                 # 本报告（五段式）
├── env_declare.yaml          # 环境声明
├── requirements.txt          # 依赖（推理栈在镜像内，harness 仅需 requests）
├── config/
│   └── sglang_serve_config.md  # 提交配置可读副本
├── scripts/
│   ├── run.sh                # 启动脚本（默认即提交配置，幂等重放 patch）
│   ├── benchmark_official.py # 官方固定 token 基准
│   └── sla_mean_postproc.py  # 从 per-request requests.jsonl 重算全局均值 TTFT/TPOT（宽松口径判分用）
├── patch/
│   ├── 0003-int8-tuned-config.patch        # decode INT8 GEMM 查表（sglang 包）
│   ├── 0006-int8-prefill-tuned-config.patch # prefill INT8 GEMM 查表（sglang 包）
│   ├── 0007-gdn-prefill-tuned-config.patch  # GDN prefill recompute_w_u（sglang 包）
│   ├── 0008-decode-skinny-m.patch           # decode 瘦 M (1,8) GEMM 表扩展（sglang 包，无 env 门控）
│   ├── lightop/
│   │   └── 0004-lightop-zeros-to-empty.patch # lightop zeros→empty（lightop 包）
│   └── gdn_configs/
│       ├── chunk_fwd_o-gfx928.json
│       └── chunk_gated_delta_rule_fwd_h-gfx928.json
├── logs/                     # 服务端启动日志摘录 + 宽松口径 SLA 重扫日志
└── result/                   # Benchmark 原始结果（summary.json + per-request requests.jsonl，宽松口径重扫窗口）
    ├── canary_pre/            # 窗口质量 canary（pre，4K c1）
    ├── 4k_c1/  4k_c8/  4k_c16/  4k_c32/   # 4K 并发扫描（宽松口径上限 = C16）
    ├── 64k_c1/  64k_c8/                    # 64K 并发扫描（宽松口径上限 = C1）
    └── canary_post/           # 窗口质量 canary（post，4K c1）
```
