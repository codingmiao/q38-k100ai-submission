# 提交配置（2026-09-29 最终复扫锁定）

> 本文件是 `scripts/run.sh` 实际下发参数的可读副本，供评审核对。
> 启动方式：`bash scripts/run.sh`（默认即提交配置）。

## 一、环境变量（容器内 export）

| 变量 | 值 | 说明 |
|---|---|---|
| `HIP_VISIBLE_DEVICES` | `0,1` | TP2，卡 0,1 |
| `SGLANG_USE_LIGHTOP` | `1` | lightop INT8 GEMM 栈 |
| `SGLANG_ENABLE_SPEC_V2` | `1` | EAGLE spec v2 |
| `SGLANG_USE_AITER_LINEAR_ATTN` | `1` | aiter 线性注意力（GDN） |
| `SGLANG_USE_CAUSAL_CONV1D` | `1` | causal conv1d |
| `SGLANG_USE_FUSED_TOPK_SOFTMAX` | `1` | 融合 topk+softmax |
| `SGLANG_USE_MARLIN_W16A16_MOE` | `0` | 关 marlin（DENSE 无 MoE） |
| `SGLANG_ROCM_USE_AITER_MOE` | `False` | 关 aiter MoE |
| `SGLANG_USE_CUDA_IPC_TRANSPORT` | `1` | CUDA IPC 传输 |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN` | `1` | 允许覆盖长上下文 |
| `SGLANG_INT8_TUNE` | `1` | **0003** decode INT8 GEMM tuned-config（M≤128） |
| `SGLANG_INT8_PREFILL_TUNE` | `1` | **0006** prefill INT8 GEMM tuned-config（M≥129） |
| `SGLANG_GDN_TUNE` | `1` | **0007** GDN prefill chunked-kernel tuned config |

## 二、sglang serve 参数

```
sglang serve \
  --model-path /mydata/models/hy/Qwen3.8-27B-Channel-INT8-w8a8 \
  --dtype bfloat16 \
  --host 0.0.0.0 --port 30011 \
  --kv-cache-dtype auto \
  --served-model-name q38tp2 \
  --quantization w8a8_int8 \
  --mm-attention-backend fa3 \
  --attention-backend fa3 \
  --enable-piecewise-cuda-graph \
  --tp-size 2 --pp-size 1 \
  --context-length 262144 \
  --page-size 64 \
  --chunked-prefill-size 16384 \
  --mamba-scheduler-strategy extra_buffer \
  --cuda-graph-max-bs 64 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.80 \
  --mamba-full-memory-ratio 1.5 \
  --keep-mm-feature-on-device \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --disable-custom-all-reduce
```

## 三、patch（容器启动时幂等重放，见 `patch/`）

| patch | 目标包 | 门控 env | 作用 |
|---|---|---|---|
| `0003-int8-tuned-config.patch` | sglang `srt/layers/quantization/w8a8_int8.py` | `SGLANG_INT8_TUNE` | decode INT8 GEMM 按 (M,K,N) 查表传 best_config（M≤128） |
| `0004-lightop-zeros-to-empty.patch` | lightop | 随 0003 | zeros→empty（省一次 memset） |
| `0006-int8-prefill-tuned-config.patch` | sglang `w8a8_int8.py` | `SGLANG_INT8_PREFILL_TUNE` | prefill INT8 GEMM 查表传 best_config（M≥129，num_stages=2） |
| `0007-gdn-prefill-tuned-config.patch` | sglang `srt/layers/attention/fla/wy_fast.py` | `SGLANG_GDN_TUNE` | GDN prefill recompute_w_u tuned config |
| `0008-decode-skinny-m.patch` | sglang `w8a8_int8.py` | **无**（patch 在 `patch/` 即生效） | decode 瘦 M (1,8) GEMM 表扩展：gate_up/qkvz 加 (1,8)→BM16BN32ST4（0003/0006 表行不变，自包含追加） |
| `gdn_configs/*.json` | aiter `ops/triton/configs/` | `SGLANG_GDN_TUNE` | GDN chunked-kernel（chunk_fwd_o / delta_h）tuned config |

> 0003/0006/0007 各自独立 env 门控、fail-closed（env 未设 = 完全 stock 行为），可单独 A/B；0008 是纯 GEMM 表改动、无 env 门控（patch 在 `patch/` 即生效，靠 patch 有无切换 A/B）。
