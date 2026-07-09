#!/bin/bash
# LongBench Sparse Token Selection 测试流程 - GPU 1 (k_head=4032, sink=8)
# 包括: 启动Sparse vLLM服务 -> 运行测试 -> 分析结果

set -e

GPU_DEVICES="${GPU_DEVICES:-4,5}"
TP_SIZE="${TP_SIZE:-2}"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
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
MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/${MODEL_NAME}}"
BS="${BS:-4}"
ENABLE_SPARSE=1
MAX_BATCHED_TOKENS=16384
RESULT_DIR="results_v1_bs${BS}_tp${TP_SIZE}_refresh32_4_256_3836_cha_085_pr_18_04_lam001001_nms10"
REQUEST_LOG="${REPO_ROOT}/logs/pred_v1_request_ids.log"
PRED_LOG="${REPO_ROOT}/logs/pred_v1.out"
PORT="${PORT:-8001}"
GPU_MEMORY_UTIL=0.8
LOG_FILE="logs/vllm_sparse_gpu1.log"
SKIP_KILL="${SKIP_KILL:-0}"

echo "=========================================="
echo "🚀 LongBench Sparse Token Selection 测试流程 - GPU 1"
echo "=========================================="
echo "GPU: ${GPU_DEVICES}"
echo "模型: ${MODEL_NAME}"
echo "结果目录: LongBench/${RESULT_DIR}"
echo "Sparse配置: k_head=4032, sink=8"
echo "=========================================="

# 检查端口是否占用
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    if [ "${SKIP_KILL}" = "1" ]; then
        echo "❌ 端口 ${PORT} 已被占用，且 SKIP_KILL=1，拒绝清理其他进程。"
        exit 1
    fi
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
TORCH_CUDA_ARCH_LIST=$("${PYTHON_BIN}" -c "
import torch
if torch.cuda.is_available():
    caps = set()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        caps.add(f'{major}.{minor}')
    print(';'.join(sorted(caps)))
else:
    print('8.0')
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
if [ "${SKIP_KILL}" != "1" ]; then
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 3
else
    echo "SKIP_KILL=1，跳过 pkill。"
fi

if [ "${ENABLE_SPARSE}" = "1" ]; then
    # Sparse配置 (GPU 1: k_head=4032, sink=8)
    export VLLM_SPARSE_CONTROLLER_JSON='{
        "enabled": true,
        "tau": 1.0,
        "k_min": 32,
        "k_max": null,
        "sink": 4,
        "recent": 256,
        "refresh_interval": 32,
        "prefill_last_n_query": 16,
        "alpha_fair": {
            "k_head": 4032,
            "soft_alpha": 0.5,
            "cross_head_alpha": 0.35,
            "cross_head_temperature": 2.0,
            "cross_head_power": 0.0,
            "prior_pos_power": 1.8,
            "prior_pos_eta": 0.4,
            "lambda_tail_kappa": 0.0,
            "lambda_tail_pivot": 0.7,
            "lambda_clip_single": 0.01,
            "lambda_clip_multi": 0.01
        },
        "adaptive": {
            "enabled": false
        },
        "trigger": {
            "refresh_interval": 32,
        "enable_sentence_triggers": true,
        "min_refresh_gap": 16
    },
    "log_prefix": "[vllm-sparse-gpu0]",
    "log_interval": 10
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
export VLLM_ATTENTION_BACKEND="TRITON_ATTN_VLLM_V1"
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
echo "   recent: 256"
echo "   decode_bounds_kernel: ${VLLM_SPARSE_DECODE_BOUNDS_KERNEL}"
echo "   端口: ${PORT}"

# 创建日志目录
mkdir -p logs

# 验证vLLM与patches加载路径（用于确认auto-patch是否可用）
echo "🔍 验证vLLM/patches路径:"
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import vllm

print(f"vllm.__file__: {vllm.__file__}")
print(f"vllm.__path__: {list(getattr(vllm, '__path__', []))}")
print(f"patches spec: {importlib.util.find_spec('patches')}")
PY

# 后台启动修改后的vLLM服务
nohup "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --port ${PORT} \
    --api-key token-abc123 \
    --trust-remote-code \
    --tensor-parallel-size ${TP_SIZE} \
    --dtype bfloat16 \
    --max-model-len 262144 \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --max-num-seqs ${BS} \
    --max-num-partial-prefills 1 \
    --max-num-batched-tokens ${MAX_BATCHED_TOKENS} \
    --enable-chunked-prefill \
    > "${LOG_FILE}" 2>&1 &

VLLM_PID=$!
echo "GPU 1 Sparse vLLM服务已启动，PID: ${VLLM_PID}"
echo "日志文件: ${LOG_FILE}"

# 等待服务启动
echo "等待服务启动..."
MAX_WAIT=300  # 最多等待5分钟
WAIT_TIME=0
while ! curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; do
    if [ $WAIT_TIME -ge $MAX_WAIT ]; then
        echo "❌ 服务启动超时!"
        echo "请查看日志: tail -f ${LOG_FILE}"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo -n "."
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
done
echo ""
echo "✅ GPU 1 Sparse vLLM服务启动成功!"

# 验证sparse功能是否启用
echo "验证sparse功能..."
sleep 5
if grep -q "Sparse controller enabled" "${LOG_FILE}" 2>/dev/null; then
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

# 关键：/health 与 /v1/models 可用并不代表推理引擎已完成初始化（仍可能在加载/编译/图捕获）。
# 在启动 LongBench 前，先用最小 chat/completions 请求做就绪探针，避免首批样本触发 connection error。
echo "等待推理引擎就绪（chat/completions 探针）..."
READY_WAIT=0
READY_MAX_WAIT=900
while true; do
    READY_CODE=$(curl -s -o /tmp/vllm_ready_probe.json -w "%{http_code}" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer token-abc123" \
        -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -d '{"model":"'"${MODEL_PATH}"'","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0.0}')
    if [ "${READY_CODE}" = "200" ] && grep -q "\"choices\"" /tmp/vllm_ready_probe.json; then
        echo "✅ 推理引擎就绪"
        break
    fi
    if [ ${READY_WAIT} -ge ${READY_MAX_WAIT} ]; then
        echo "❌ 推理引擎就绪超时（${READY_MAX_WAIT}s）"
        echo "最近探针响应:"
        tail -n 5 /tmp/vllm_ready_probe.json 2>/dev/null || true
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    echo -n "."
    sleep 5
    READY_WAIT=$((READY_WAIT + 5))
done

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
echo "步骤 2/3: 运行 LongBench 测试 (GPU 1)"
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
export LONGBENCH_URL="http://127.0.0.1:${PORT}/v1"

# 创建结果目录
mkdir -p ${RESULT_DIR}

# 运行测试
echo "开始运行LongBench GPU 1 Sparse测试..."
echo "样本量取决于 data/ 内容（本机全量 4750 样本远超 4-5 小时；子集另备 data 目录）"
echo "可以使用 Ctrl+C 中断，支持断点续传"
echo ""

# [RECIPE-PORT-FIX 2026-07-09] pred_v1.py 没有 --port 参数（只有 --url/--api_key），
# 旧写法直接 argparse 崩且退出码被 tee 吞——同 gpu0 脚本的修法。
"${PYTHON_BIN}" pred_v1.py \
  --model ${MODEL_NAME} \
  --n_proc ${BS} \
  --url "http://127.0.0.1:${PORT}/v1" \
  --api_key "token-abc123" \
  --save_dir ${RESULT_DIR} 2>&1 | tee -a "${PRED_LOG}"

PRED_EXIT_CODE=${PIPESTATUS[0]}

if [ $PRED_EXIT_CODE -ne 0 ]; then
    echo "❌ 测试运行失败，退出码: ${PRED_EXIT_CODE}"
    echo "vLLM服务仍在运行，PID: ${VLLM_PID}"
    echo "手动停止: kill ${VLLM_PID}"
    exit 1
fi

echo ""
echo "✅ GPU 1 测试运行完成!"

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
echo "✅ LongBench GPU 1 Sparse 测试完成!"
echo "=========================================="
echo "GPU: 1, 配置: k_head=4032, sink=8"
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
