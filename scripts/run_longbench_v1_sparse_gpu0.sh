#!/bin/bash
# LongBench Sparse Token Selection 测试流程
# 包括: 启动Sparse vLLM服务 -> 运行测试 -> 分析结果
#
# 使用方式:
#   1. 直接运行（使用默认参数）:
#      ./run_longbench_v1_sparse_gpu0.sh
#
#   2. 通过环境变量指定参数:
#      GPU_DEVICES=0,1 SPARSE_K_HEAD=4032 ./run_longbench_v1_sparse_gpu0.sh
#
#   3. 常用配置示例:
#      # 单卡测试
#      GPU_DEVICES=0 ./run_longbench_v1_sparse_gpu0.sh
#
#      # 双卡 TP=2，调整 k_head
#      GPU_DEVICES=0,1 SPARSE_K_HEAD=4032 SPARSE_SINK=8 ./run_longbench_v1_sparse_gpu0.sh
#
#      # 禁用 Sparse（baseline 对比）
#      ENABLE_SPARSE=0 ./run_longbench_v1_sparse_gpu0.sh
#
# 可配置的环境变量（括号内为默认值）:
#   GPU_DEVICES       - GPU 设备 ID，逗号分隔 (0)
#   TP_SIZE           - Tensor Parallel 大小，默认根据 GPU 数量自动计算
#   ENABLE_SPARSE     - 是否启用 Sparse (1)
#   BS                - Batch Size (1)
#   PORT              - vLLM 服务端口 (8000)
#   GPU_MEMORY_UTIL   - GPU 显存利用率 (0.8)
#   MAX_MODEL_LEN     - 最大模型长度 (262144)
#   MAX_BATCHED_TOKENS- 最大批处理 token 数 (16384)
#   SPARSE_K_HEAD     - 每个 head 保留的 token 数 (2016, aligned to kBlockN=112)
#   SPARSE_SINK       - Sink token 数量 (4)
#   SPARSE_RECENT     - Recent token 数量 (256)
#   SPARSE_REFRESH_INTERVAL - 刷新间隔 (64)
#   MODEL_NAME        - 模型名称 (Qwen3-4B-Instruct-2507)
#   REPO_ROOT         - 项目根目录，默认自动检测

set -e

# ============== 可配置参数 ==============
GPU_DEVICES="${GPU_DEVICES:-0}"
ENABLE_SPARSE="${ENABLE_SPARSE:-1}"
BS="${BS:-1}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"

# Sparse 核心参数
SPARSE_K_HEAD="${SPARSE_K_HEAD:-2016}"  # aligned to kBlockN=112 (2016=18*112)
SPARSE_SINK="${SPARSE_SINK:-4}"
SPARSE_RECENT="${SPARSE_RECENT:-256}"
SPARSE_REFRESH_INTERVAL="${SPARSE_REFRESH_INTERVAL:-64}"

# ============== 自动推断参数 ==============
# 自动检测 REPO_ROOT（脚本所在目录的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(dirname "${SCRIPT_DIR}")}"
LONGBENCH_ROOT="${LONGBENCH_ROOT:-/mnt/bn/ecom-ai-platform-1/yzc/fast-slow/attention-xx-point/LongBench}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/bn/ecom-ai-platform-1/yzc/models/Qwen}"
VLLM_ROOT="${VLLM_ROOT:-/mnt/bn/ecom-ai-platform-1/yzc/fast-slow/attention-xx-point/vllm}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/bn/ecom-ai-platform-1/yzc/miniconda}"
CONDA_ENV="${CONDA_ENV:-fst}"
CONDA_PREFIX="${CONDA_ROOT}/envs/${CONDA_ENV}"
CONDA_DEFAULT_ENV="${CONDA_ENV}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
export CONDA_PREFIX CONDA_DEFAULT_ENV
export PATH="${CONDA_PREFIX}/bin:${PATH}"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "❌ Python 不存在或不可执行: ${PYTHON_BIN}"
    exit 1
fi

# 自动计算 TP_SIZE（根据 GPU 数量）
IFS=',' read -ra GPU_ARRAY <<< "${GPU_DEVICES}"
TP_SIZE="${TP_SIZE:-${#GPU_ARRAY[@]}}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"
RESULT_DIR="${RESULT_DIR:-results_sparse_v1_bs${BS}_tp${TP_SIZE}}"
REQUEST_LOG="${REPO_ROOT}/logs/pred_v1_request_ids.log"
PRED_LOG="${REPO_ROOT}/logs/pred_v1.out"
LOG_SUFFIX="${GPU_DEVICES//,/_}"  # 0,1 -> 0_1

echo "=========================================="
echo "🚀 LongBench Sparse Token Selection 测试流程"
echo "=========================================="
echo "GPU: ${GPU_DEVICES}"
echo "模型: ${MODEL_NAME}"
echo "结果目录: LongBench/${RESULT_DIR}"
echo "Sparse配置: k_head=${SPARSE_K_HEAD}, sink=${SPARSE_SINK}"
echo "=========================================="

# 检查端口是否占用
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  端口 ${PORT} 已被占用，尝试清理..."
    lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true
    sleep 2
fi

cd "${REPO_ROOT}"

echo ""
echo "=========================================="
echo "步骤 0/3: 预编译 Sparse 扩展 (避免 TP=2 并发编译卡住)"
echo "=========================================="

# 预编译所需环境变量（REPO_ROOT 必须在 PYTHONPATH 中，否则 sitecustomize.py 不会被自动加载）
export PYTHONPATH="${REPO_ROOT}:${VLLM_ROOT}:${PYTHONPATH}"

# 自动检测 GPU 架构并设置 TORCH_CUDA_ARCH_LIST
TORCH_CUDA_ARCH_LIST=$("${PYTHON_BIN}" -c "
import torch
if torch.cuda.is_available():
    caps = set()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        caps.add(f'{major}.{minor}')
    print(';'.join(sorted(caps)))
else:
    print('8.0')  # fallback
")
export TORCH_CUDA_ARCH_LIST
echo "检测到 GPU 架构: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

"${PYTHON_BIN}" - <<'PY'
from utils.bounds_kernel_ext import _require_ext as _require_bounds
from utils.selector_pipeline_ext import _require_ext as _require_pipeline
print("[sparse precompile] start")
_require_bounds()
_require_pipeline()
print("[sparse precompile] done")
PY

echo ""
echo "=========================================="
echo "步骤 1/3: 启动 Sparse vLLM 服务 (GPU ${GPU_DEVICES})"
echo "=========================================="

# 停止现有的sparse服务器
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 3

# [SERVE-LIVENESS 2026-07-09] residency 三字段是双代默认 ON(758185e)后的硬性
# 前提:缺失=安装期合同直接 fail-fast(此前是运行中 rebuild 深处 raise)。
#   slots 必须 >= --max-num-seqs(本脚本两者同源 =BS,启动预检
#   W_SPARSE_SLOTS_LT_MAX_NUM_SEQS 亦会核对);
#   blocks_per_slot 公式 = ceil((sink + eff_k_head)/16),eff_k_head =
#   align(sink+k_head, kBlockN=112) - sink;k_head=2016,sink=4 → 133,
#   默认 140 留余量(显存账 = slots x blocks x 16 x KV-B/token x gen_count)。
SPARSE_BLOCKS_PER_SLOT="${SPARSE_BLOCKS_PER_SLOT:-140}"

if [ "${ENABLE_SPARSE}" = "1" ]; then
    # Sparse配置（使用顶部定义的参数）
    export VLLM_SPARSE_CONTROLLER_JSON='{
        "enabled": true,
        "tau": 1.0,
        "k_min": 32,
        "k_max": null,
        "sink": '"${SPARSE_SINK}"',
        "recent": '"${SPARSE_RECENT}"',
        "refresh_interval": '"${SPARSE_REFRESH_INTERVAL}"',
        "prefill_last_n_query": 16,
        "one_shot_bootstrap_only": true,
        "continuous_producer_enabled": true,
        "compact_page_residency_enabled": true,
        "max_live_sparse_slots": '"${BS}"',
        "compact_blocks_per_slot": '"${SPARSE_BLOCKS_PER_SLOT}"',
        "alpha_fair": {
            "k_head": '"${SPARSE_K_HEAD}"',
            "soft_alpha": 1.0,
            "cross_head_alpha": 0.85,
            "cross_head_temperature": 2.0,
            "cross_head_power": 0.0,
            "prior_pos_power": 1.8,
            "prior_pos_eta": 0.4,
            "lambda_tail_kappa": 0.0,
            "lambda_tail_pivot": 0.7,
            "lambda_clip_single": 0.35,
            "lambda_clip_multi": 0.25
        },
        "adaptive": {
            "enabled": false
        },
        "trigger": {
            "refresh_interval": '"${SPARSE_REFRESH_INTERVAL}"',
            "enable_sentence_triggers": true
        },
        "log_prefix": "[vllm-sparse-gpu'"${LOG_SUFFIX}"']",
        "log_interval": 10,
        "alpha_debug": false
    }'
else
    unset VLLM_SPARSE_CONTROLLER_JSON
fi

# 启用Sparse调试模式
export VLLM_SPARSE_DEBUG=0
export VLLM_SPARSE_ALPHA_DEBUG=0
# # 完全关闭异步刷新与 prefill 异步，验证同步路径
# # export VLLM_SPARSE_ASYNC_REFRESH=1
# # Decode bounds CUDA kernel 开关（1=启用，0=禁用）
# export VLLM_SPARSE_DECODE_BOUNDS_KERNEL="${VLLM_SPARSE_DECODE_BOUNDS_KERNEL:-1}"
# # 异步 trace（用于定位 wait_event/flush 行为）
# export VLLM_SPARSE_TRACE_ASYNC=1
# export VLLM_SPARSE_TRACE_ASYNC_LOG="logs/vllm_sparse_async_trace.log"
# # bootstrap/slot 调试日志（仅在报错时写一行）
# export VLLM_SPARSE_BOOTSTRAP_DEBUG=0
# export VLLM_SPARSE_BOOTSTRAP_DEBUG_LOG="logs/vllm_sparse_bootstrap_errors.log"
# # bootstrap 状态机追踪日志
# export VLLM_SPARSE_BOOTSTRAP_TRACE=0
# export VLLM_SPARSE_BOOTSTRAP_TRACE_LOG="logs/vllm_sparse_bootstrap_trace.log"
# # slot 生命周期调试（每个 step 写一行）
# export VLLM_SPARSE_SLOT_DEBUG=0
# export VLLM_SPARSE_SLOT_DEBUG_LOG="logs/vllm_sparse_slot_debug.log"
# # 关闭 mem trace（如需排查内存再手动开启）
# export VLLM_SPARSE_MEM_TRACE=0
# export VLLM_SPARSE_MEM_TRACE_EVERY=50
# export VLLM_SPARSE_MEM_TRACE_LOG="logs/vllm_sparse_mem_trace_v2.log"

# 其他环境变量 - 与baseline对齐
# export VLLM_USE_V1=1
# [TRITON-LINE-RETIRED 39e0e33] TRITON backend 已整链下线,挡板在 _install_patch
# 咽喉会 fail-fast;生产形态=FA3 native(vendored 树)。
export VLLM_ATTENTION_BACKEND="FLASH_ATTN_VLLM_V1"
export VLLM_FLASH_ATTN_VERSION="${VLLM_FLASH_ATTN_VERSION:-3}"
export VLLM_SPARSE_FA3_UPSTREAM_ROOT="${VLLM_SPARSE_FA3_UPSTREAM_ROOT:-${REPO_ROOT}/third_party_upstreams/vllm-project-flash-attention}"

# [SPARSE-LIVENESS-JUDGE] 活性判官产物(路径带 run 时间戳,防残留 append 假
# PASS);跑完 LongBench 后由本脚本尾部自动开箱判定,FAIL 则精度数不可采信。
LIVENESS_RUN_TS="$(date +%s)"
export VLLM_SPARSE_REFRESH_PROFILE=1
export VLLM_SPARSE_REFRESH_PROFILE_LOG="${REPO_ROOT}/logs/lb_refresh_profile.${LIVENESS_RUN_TS}.log"
export VLLM_SPARSE_FA3_ROUTE_TRACE_LOG="${REPO_ROOT}/logs/lb_route.${LIVENESS_RUN_TS}.jsonl"
export VLLM_SPARSE_FA3_ROUTE_COUNTER_MMAP="${REPO_ROOT}/logs/lb_route_counter.${LIVENESS_RUN_TS}.bin"
export VLLM_NO_USAGE_REPORT="1"
export CPUINFO_NO_DMI="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# # export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
# export NCCL_DEBUG="INFO"
# export NCCL_ASYNC_ERROR_HANDLING="1"

# PYTHONPATH 已在步骤 0 设置（包含 REPO_ROOT 和 REPO_ROOT/vllm）

echo "✅ GPU 环境变量设置完成"
echo "   GPU: ${GPU_DEVICES}, TP: ${TP_SIZE}"
echo "   k_head: ${SPARSE_K_HEAD}, sink: ${SPARSE_SINK}, recent: ${SPARSE_RECENT}"
echo "   max_model_len: ${MAX_MODEL_LEN}"
echo "   端口: ${PORT}"

# 创建日志目录
mkdir -p logs

# 后台启动修改后的vLLM服务
VLLM_LOG="logs/vllm_sparse_gpu${LOG_SUFFIX}.log"
nohup "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --port ${PORT} \
    --api-key token-abc123 \
    --trust-remote-code \
    --tensor-parallel-size ${TP_SIZE} \
    --dtype bfloat16 \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --max-num-seqs ${BS} \
    --max-num-partial-prefills 1 \
    --max-num-batched-tokens ${MAX_BATCHED_TOKENS} \
    --enable-chunked-prefill \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "GPU ${GPU_DEVICES} vLLM服务已启动，PID: ${VLLM_PID}"
echo "日志文件: ${VLLM_LOG}"

# 等待服务启动
echo "等待服务启动..."
MAX_WAIT=300  # 最多等待5分钟
WAIT_TIME=0
while ! curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; do
    if [ $WAIT_TIME -ge $MAX_WAIT ]; then
        echo "❌ 服务启动超时!"
        echo "请查看日志: tail -f ${VLLM_LOG}"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo -n "."
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
done
echo ""
echo "✅ GPU ${GPU_DEVICES} vLLM服务启动成功!"

# 验证sparse功能是否启用
echo "验证sparse功能..."
sleep 5
if grep -q "Sparse controller enabled" "${VLLM_LOG}" 2>/dev/null; then
    echo "✅ Sparse功能已启用"
else
    echo "⚠️  未检测到Sparse功能启用消息，但服务已启动"
fi

# 测试服务连通性
echo "测试服务连通性..."
curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 服务连通性测试通过"
else
    echo "❌ 服务连通性测试失败"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# GPU 显存监控（用于判断是否存在缓慢增长/泄漏）
GPU_MEM_LOG="logs/gpu_mem_trace.log"
echo "timestamp,gpu_mem_used_mb,gpu_mem_total_mb,proc_mem_mb" > "${GPU_MEM_LOG}"
(
    while true; do
        ts=$(date +"%F %T")
        gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n 1)
        proc_mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr '\n' ';' | sed 's/;$//')
        echo "${ts},${gpu_mem},\"${proc_mem}\"" >> "${GPU_MEM_LOG}"
        sleep 30
    done
) &
GPU_MEM_PID=$!
cleanup() {
    if [ -n "${GPU_MEM_PID:-}" ]; then
        kill ${GPU_MEM_PID} 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo ""
echo "=========================================="
echo "步骤 2/3: 运行 LongBench 测试 (GPU ${GPU_DEVICES})"
echo "=========================================="

cd "${LONGBENCH_ROOT}"

# 避免 httpx 因 SSL_CERT_FILE=/dev/null 初始化失败
unset SSL_CERT_FILE
# 禁用代理，避免 httpx 走 SOCKS 代理导致 ImportError
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export NO_PROXY="127.0.0.1,localhost"
# 强制离线模式（避免 datasets / HF hub 触网）
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# 创建结果目录
mkdir -p ${RESULT_DIR}

# 运行测试
echo "开始运行LongBench GPU ${GPU_DEVICES} Sparse测试..."
echo "配置: k_head=${SPARSE_K_HEAD}, sink=${SPARSE_SINK}"
echo "样本量取决于 data/ 内容（本机全量 4750 样本远超 4-5 小时；子集另备 data 目录）"
echo "可以使用 Ctrl+C 中断，支持断点续传"
echo ""

# [RECIPE-PORT-FIX 2026-07-09] pred_v1.py 没有 --port 参数（只有 --url/--api_key，
# 见 pred_v1.py argparse），旧写法 `--port ${PORT}` 会直接 argparse 崩；
# 且崩溃退出码被 `| tee` 吞掉（$? 是 tee 的 0）——改用 PIPESTATUS 取 pred 真实退出码。
"${PYTHON_BIN}" pred_v1.py \
  --model ${MODEL_NAME} \
  --n_proc ${BS} \
  --url "http://127.0.0.1:${PORT}/v1" \
  --api_key "token-abc123" \
  --request_log "${REQUEST_LOG}" \
  --save_dir ${RESULT_DIR} 2>&1 | tee -a "${PRED_LOG}"

PRED_EXIT_CODE=${PIPESTATUS[0]}

if [ $PRED_EXIT_CODE -ne 0 ]; then
    echo "❌ 测试运行失败，退出码: ${PRED_EXIT_CODE}"
    echo "vLLM服务仍在运行，PID: ${VLLM_PID}"
    echo "手动停止: kill ${VLLM_PID}"
    exit 1
fi

echo ""
echo "✅ GPU ${GPU_DEVICES} 测试运行完成!"

# [SPARSE-LIVENESS-JUDGE] 活性开箱判定:sparse 若静默退化成 dense,精度数字
# 看着正常但测的是 dense。FAIL 时结果目录改名隔离,防误用。
if [ "${ENABLE_SPARSE}" = "1" ]; then
    echo ""
    echo "运行 sparse 活性判官..."
    if "${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_sparse_liveness.py" \
        --refresh-profile-log "${VLLM_SPARSE_REFRESH_PROFILE_LOG}" \
        --route-trace "${VLLM_SPARSE_FA3_ROUTE_TRACE_LOG}" \
        --route-counter-mmap "${VLLM_SPARSE_FA3_ROUTE_COUNTER_MMAP}" \
        --run-since "${LIVENESS_RUN_TS}" \
        --min-world-publish 1; then
        echo "✅ sparse 活性判官 PASS"
    else
        echo "❌ sparse 活性判官 FAIL —— 本轮精度数不可采信(疑跑成 dense)"
        if [ -d "${RESULT_DIR}/${MODEL_NAME}" ]; then
            mv "${RESULT_DIR}/${MODEL_NAME}" \
               "${RESULT_DIR}/${MODEL_NAME}.SPARSE_LIVENESS_FAIL.${LIVENESS_RUN_TS}"
            echo "结果目录已隔离: ${RESULT_DIR}/${MODEL_NAME}.SPARSE_LIVENESS_FAIL.${LIVENESS_RUN_TS}"
        fi
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "步骤 3/3: 分析结果"
echo "=========================================="

# 检查结果目录
RESULT_DIR_PATH="${RESULT_DIR}/${MODEL_NAME}"
if [ ! -d "${RESULT_DIR_PATH}" ]; then
    echo "❌ 结果目录不存在: ${RESULT_DIR_PATH}"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# 运行 v1 评估
echo "运行结果评估..."
"${PYTHON_BIN}" LongBench/eval.py --model ${MODEL_NAME} --f ${RESULT_DIR}

echo ""
echo "=========================================="
echo "✅ LongBench GPU ${GPU_DEVICES} Sparse 测试完成!"
echo "=========================================="
echo "GPU: ${GPU_DEVICES}, 配置: k_head=${SPARSE_K_HEAD}, sink=${SPARSE_SINK}"
echo "结果目录: ${RESULT_DIR_PATH}"
echo "vLLM服务PID: ${VLLM_PID}"
echo "端口: ${PORT}"
echo ""
echo "停止vLLM服务: kill ${VLLM_PID}"
echo "=========================================="

# 询问是否停止服务（非交互模式下 read 会 EOF 失败并在 set -e 下把整个成功
# 流程判成失败——无人值守时自动停服务释放显存）
if [ -t 0 ]; then
    read -p "是否停止vLLM服务? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "停止vLLM服务..."
        kill $VLLM_PID 2>/dev/null || true
        echo "✅ 服务已停止"
    else
        echo "vLLM服务继续运行，PID: ${VLLM_PID}"
    fi
else
    echo "非交互模式：自动停止vLLM服务 (PID ${VLLM_PID})"
    kill $VLLM_PID 2>/dev/null || true
fi
