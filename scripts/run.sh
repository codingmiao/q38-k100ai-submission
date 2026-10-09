#!/bin/bash
# ============================================================================
# Qwen3.8-27B Channel-INT8 高性能推理 — 提交启动脚本（B 赛道，K100-AI × 2 卡 TP2）
#
# 提交配置（2026-09-29 最终复扫锁定，见 ../result/ 与 ../README.md）：
#   INT8TUNE=1   (0003) decode INT8 GEMM tuned-config 查表（M<=128）
#   PREFILLTUNE=1(0006) prefill INT8 GEMM tuned-config 查表（M>=129）
#   GDN_TUNE=1   (0007) GDN 线性注意力 prefill chunked-kernel tuned config
#   (0008)       decode 瘦 M (1,8) GEMM 表扩展（无 env 门控，patch 在 patch/ 即生效）
#   MEM=0.80 MRATIO=1.5  GDN/mamba 状态池抬升（max_running_requests 11->35）
#   DELAYER=0    prefill delayer 关（新 SLA TTFT<3s 下无用，见 README 第五部分）
#
# 用法：
#   bash run.sh                 # 用提交配置启动（默认即提交配置）
#   INT8TUNE=0 bash run.sh      # 单独 A/B 某个杠杆（各 env 独立门控）
#
# 依赖：sglang 0.5.12 镜像（DTK 2604, py3.10）+ /work/patches 下的幂等 patch。
# 本脚本每次运行删除并重建容器；patch 在容器启动时从 ../patch/ 幂等重放。
# ============================================================================
set -e

CONTAINER_NAME="q38_tp2"
IMAGE_NAME="harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10"
MODEL_PATH="/mydata/models/hy/Qwen3.8-27B-Channel-INT8-w8a8"
LOG_DIR="/mydata/models/logs"
WORK_DIR="/mydata/q38tp2"

# patch 目录 = 本脚本所在目录的上一级（submit/patch/）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}/../patch"

HIP_DEVICES="0,1"
PORT=30011
TP_SIZE=2
AR=${AR:-disable}          # disable(官方基线) | custom(实验: 开 custom all-reduce)
NOGRAPH=${NOGRAPH:-0}      # 1 = --disable-cuda-graph（全量kernel→代码定位用）
MEM=${MEM:-0.80}           # mem-fraction-static（提交值 0.80）
MRATIO=${MRATIO:-1.5}      # mamba-full-memory-ratio（提交值 1.5，抬 GDN 状态池）
DELAYER=${DELAYER:-0}      # 0 = 默认（提交值）| 1 = --enable-prefill-delayer
DELAYER_PASSES=${DELAYER_PASSES:-30}
DELAYER_MS=${DELAYER_MS:-5000}
KVDTYPE=${KVDTYPE:-auto}   # kv-cache-dtype: auto(bf16, 提交值) | fp8_e5m2(可选杠杆)
INT8TUNE=${INT8TUNE:-1}    # 1 = SGLANG_INT8_TUNE=1（0003 decode 查表 + 0004 zeros→empty）
PREFILLTUNE=${PREFILLTUNE:-1}  # 1 = SGLANG_INT8_PREFILL_TUNE=1（0006 prefill 查表）
GDN_TUNE=${GDN_TUNE:-1}    # 1 = SGLANG_GDN_TUNE=1（0007 GDN prefill tuned config）
RECVINT=${RECVINT:-1}      # --scheduler-recv-interval（默认 1，全场景无效，保留开关）

mkdir -p ${WORK_DIR}

if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
  echo "🧹 删除旧容器 ${CONTAINER_NAME}"
  docker stop ${CONTAINER_NAME} >/dev/null 2>&1 || true
  docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
fi

AR_FLAG="--disable-custom-all-reduce"
if [ "${AR}" = "custom" ]; then AR_FLAG=""; fi
GRAPH_FLAG=""
if [ "${NOGRAPH}" = "1" ]; then GRAPH_FLAG="--disable-cuda-graph"; fi
DELAYER_FLAG=""
if [ "${DELAYER}" = "1" ]; then
  DELAYER_FLAG="--enable-prefill-delayer --prefill-delayer-max-delay-passes ${DELAYER_PASSES} --prefill-delayer-max-delay-ms ${DELAYER_MS}"
fi
RECVINT_ARG=""
if [ "${RECVINT}" != "1" ]; then
  RECVINT_ARG="--scheduler-recv-interval ${RECVINT}"
fi

echo "🚀 ${CONTAINER_NAME}: GPU=${HIP_DEVICES} TP=${TP_SIZE} PORT=${PORT} AR=${AR} NOGRAPH=${NOGRAPH} MEM=${MEM} MRATIO=${MRATIO} DELAYER=${DELAYER} KVDTYPE=${KVDTYPE} INT8TUNE=${INT8TUNE} PREFILLTUNE=${PREFILLTUNE} GDN_TUNE=${GDN_TUNE}"

docker run -d \
  --ipc=host \
  --shm-size 128g \
  --network=host \
  --name=${CONTAINER_NAME} \
  --privileged \
  --restart=no \
  -e INT8TUNE=${INT8TUNE} \
  -e PREFILLTUNE=${PREFILLTUNE} \
  -e GDN_TUNE=${GDN_TUNE} \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v ${MODEL_PATH}:${MODEL_PATH}:ro \
  -v ${LOG_DIR}:/modellogs \
  -v ${WORK_DIR}:/work \
  ${IMAGE_NAME} \
  bash -c "
    set -e
    export HIP_VISIBLE_DEVICES=${HIP_DEVICES}
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    export SGLANG_USE_LIGHTOP=1
    export SGLANG_USE_MARLIN_W16A16_MOE=0
    export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
    export SGLANG_ROCM_USE_AITER_MOE=False
    export SGLANG_USE_CAUSAL_CONV1D=1
    export SGLANG_USE_CUDA_IPC_TRANSPORT=1
    export SGLANG_USE_AITER_LINEAR_ATTN=1
    export SGLANG_ENABLE_SPEC_V2=1
    export SGLANG_TORCH_PROFILER_DIR=/work/traces
    export PYTHONWARNINGS='ignore::UserWarning:numpy.core.getlimits'
    if [ \"\${INT8TUNE}\" = \"1\" ]; then
      export SGLANG_INT8_TUNE=1
      echo '[q38-opt] SGLANG_INT8_TUNE=1 (0003 decode tuned-config)'
    fi
    if [ \"\${PREFILLTUNE}\" = \"1\" ]; then
      export SGLANG_INT8_PREFILL_TUNE=1
      echo '[q38-opt] SGLANG_INT8_PREFILL_TUNE=1 (0006 prefill tuned-config)'
    fi
    if [ \"\${GDN_TUNE}\" = \"1\" ]; then
      export SGLANG_GDN_TUNE=1
      echo '[q38-opt] SGLANG_GDN_TUNE=1 (0007 GDN prefill tuned-config)'
    fi
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

    # ============================================================
    # [q38-opt] 应用 patch（幂等，patch -N --forward 重跑 skip）
    #   *.patch            -> sglang 包
    #   lightop/*.patch    -> lightop 包
    # ============================================================
    if ls ${PATCH_DIR}/*.patch >/dev/null 2>&1; then
      cd /usr/local/lib/python3.10/dist-packages/sglang
      for p in ${PATCH_DIR}/*.patch; do
        echo applying q38 patch \$(basename \$p)
        patch -p1 -N --forward < \$p || true
      done
      cd /
    fi
    if ls ${PATCH_DIR}/lightop/*.patch >/dev/null 2>&1; then
      cd /usr/local/lib/python3.10/dist-packages/lightop
      for p in ${PATCH_DIR}/lightop/*.patch; do
        echo applying q38 lightop patch \$(basename \$p)
        patch -p1 -N --forward < \$p || true
      done
      cd /
    fi

    # ============================================================
    # [q38-opt 0007] GDN prefill chunked-kernel tuned configs
    #   aiter 从 AITER_TRITON_CONFIGS_PATH 按 arch 读 JSON；容器每次重建
    #   → 启动时从 ${PATCH_DIR}/gdn_configs/ 拷入。仅 GDN_TUNE=1 时拷。
    # ============================================================
    if [ \"\${GDN_TUNE}\" = \"1\" ]; then
      AITER_CFG=/usr/local/lib/python3.10/dist-packages/aiter/ops/triton/configs
      cp ${PATCH_DIR}/gdn_configs/chunk_fwd_o-gfx928.json \
         \$AITER_CFG/chunk_fwd_o/chunk_fwd_o-gfx928.json
      cp ${PATCH_DIR}/gdn_configs/chunk_gated_delta_rule_fwd_h-gfx928.json \
         \$AITER_CFG/chunk_gated_delta_rule_fwd_h/chunk_gated_delta_rule_fwd_h-gfx928.json
      echo '[q38-opt] 0007 GDN tuned configs copied into aiter'
    fi

    option=' --chunked-prefill-size 16384'
    option+=' --mamba-scheduler-strategy extra_buffer'
    option+=' --cuda-graph-max-bs 64'

    time=\$(date '+%m%d-%H%M')
    LOG_PATH=\"/modellogs/q38tp2-sglang-\${time}.log\"
    echo '🚀 启动 SGLang (q38_tp2 提交配置)'
    echo \"📌 AR=${AR} NOGRAPH=${NOGRAPH} MEM=${MEM} MRATIO=${MRATIO} DELAYER=${DELAYER} KVDTYPE=${KVDTYPE} INT8TUNE=${INT8TUNE} PREFILLTUNE=${PREFILLTUNE} GDN_TUNE=${GDN_TUNE} Log: \${LOG_PATH}\"

    sglang serve \${option} \
      --model-path '${MODEL_PATH}' \
      --dtype bfloat16 \
      --host 0.0.0.0 \
      --port ${PORT} \
      --kv-cache-dtype ${KVDTYPE} \
      --served-model-name q38tp2 \
      --quantization w8a8_int8 \
      --mm-attention-backend fa3 \
      --enable-piecewise-cuda-graph \
      --tp-size ${TP_SIZE} \
      --pp-size 1 \
      --attention-backend fa3 \
      --context-length 262144 \
      --page-size 64 \
      --speculative-algorithm EAGLE \
      --speculative-num-steps 3 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 4 \
      --mem-fraction-static ${MEM} \
      --mamba-full-memory-ratio ${MRATIO} \
      --keep-mm-feature-on-device \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      ${AR_FLAG} ${GRAPH_FLAG} ${DELAYER_FLAG} ${RECVINT_ARG} \
      2>&1 | tee \${LOG_PATH}
  "
