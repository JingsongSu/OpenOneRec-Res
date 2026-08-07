#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# GPU 机器：
#   1. 检测 DATA_SUCCESS 是否更新；
#   2. 每个新版本只训练成功一次；
#   3. 每次 SFT 都固定从原始 stg2 的 21000 step 模型开始；
#   4. 当天生成的 SFT checkpoint/converted 保留给推理使用；
#   5. 下一轮新 SFT 真正开始前，再删除上一轮 SFT 输出目录。
#
# 建议由 cron 每 2 分钟调用一次。
# =============================================================================

readonly REPO_ROOT="/home/jovyan/ceph-1/sujinsong/online/openonerec-res"
readonly PRETRAIN_ROOT="${REPO_ROOT}/pretrain"
readonly PYTHON_ENV="/home/jovyan/ceph-1/sujinsong/env/onerec"

readonly LOG_DIR="${REPO_ROOT}/logs"
readonly STATE_DIR="${REPO_ROOT}/state"
readonly LOG_FILE="${LOG_DIR}/sft_$(date +%Y%m%d).log"

readonly DATA_SUCCESS="/home/jovyan/zhouyuhang-cloud1/DSP/rqvae_decoder_base_pipeline/data/train_ad_daily_sample/DATA_SUCCESS"

# 只有本轮数据处理、训练和 converted 转换全部成功后才更新。
readonly LAST_TRAINED_FINGERPRINT_FILE="${STATE_DIR}/last_trained_data_success.fingerprint"
readonly LAST_TRAINED_INFO_FILE="${STATE_DIR}/last_trained_data_success.info"

readonly RUN_DAILY_LOCK_FILE="/tmp/openonerec_run_daily.lock"
readonly GPU_EXCLUSIVE_LOCK_FILE="/tmp/openonerec_gpu_exclusive.lock"

readonly SFT_RUNNING_FILE="${REPO_ROOT}/_RUNNING"
readonly SFT_SUCCESS_FILE="${REPO_ROOT}/_SFT_SUCCESS"
readonly SFT_FAILED_FILE="${REPO_ROOT}/_SFT_FAILED"

# 永远不再覆盖或移动这个原始 stg2 模型。
readonly BASE_STG2_MODEL_DIR="${PRETRAIN_ROOT}/model_output/stg2_residual_add_feature/step21000/global_step21000/converted"

# 每一轮 SFT 都写入这个固定输出目录。
# 新一轮训练开始前删除上一轮目录；本轮成功后完整保留，供 infer_daily.sh 使用。
readonly SFT_OUTPUT_DIR="${PRETRAIN_ROOT}/model_output/sft_full_residual_add_feature_daily"
readonly SFT_CONVERTED_MODEL_DIR="${SFT_OUTPUT_DIR}/step15000/global_step15000/converted"

mkdir -p "$LOG_DIR" "$STATE_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

data_success_fingerprint() {
    # mtime + ctime + size + inode。
    # DATA_SUCCESS 即使一直存在，只要被 touch 或替换，指纹就会变化。
    stat -Lc '%y|%z|%s|%i' "$DATA_SUCCESS"
}

read_last_fingerprint() {
    [[ -f "$LAST_TRAINED_FINGERPRINT_FILE" ]] || return 1
    cat "$LAST_TRAINED_FINGERPRINT_FILE"
}

if [[ ! -f "$DATA_SUCCESS" ]]; then
    echo "[$(date '+%F %T')] DATA_SUCCESS 不存在，本次不训练。"
    exit 0
fi

current_fingerprint="$(data_success_fingerprint)"
last_fingerprint="$(read_last_fingerprint || true)"

if [[ -n "$last_fingerprint" && "$current_fingerprint" == "$last_fingerprint" ]]; then
    echo "[$(date '+%F %T')] DATA_SUCCESS 未变化，本次不重复训练。"
    exit 0
fi

# 防止 cron 重复启动同一轮训练。
exec 9>"$RUN_DAILY_LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] 另一个模型更新脚本正在运行，本次退出。"
    exit 0
fi

# 获得脚本锁后再读取一次，避免两个 cron 进程重复触发。
if [[ ! -f "$DATA_SUCCESS" ]]; then
    echo "[$(date '+%F %T')] DATA_SUCCESS 已消失，本次退出。"
    exit 0
fi

trigger_fingerprint="$(data_success_fingerprint)"
last_fingerprint="$(read_last_fingerprint || true)"

if [[ -n "$last_fingerprint" && "$trigger_fingerprint" == "$last_fingerprint" ]]; then
    echo "[$(date '+%F %T')] 该 DATA_SUCCESS 已经训练成功，本次退出。"
    exit 0
fi

# 只检查原始 stg2；此目录以后保持只读，不再被日更脚本替换。
for required in \
    "${BASE_STG2_MODEL_DIR}/config.json" \
    "${BASE_STG2_MODEL_DIR}/tokenizer_config.json" \
    "${BASE_STG2_MODEL_DIR}/residual_sid_config.json"
do
    if [[ ! -f "$required" ]]; then
        echo "必要原始 stg2 模型文件不存在：${required}"
        exit 1
    fi
done

completed=false

cleanup() {
    local rc=$?
    trap - EXIT

    rm -f "$SFT_RUNNING_FILE"

    if [[ "$completed" != true && $rc -ne 0 ]]; then
        {
            echo "trigger_fingerprint=${trigger_fingerprint}"
            echo "base_model=${BASE_STG2_MODEL_DIR}"
            echo "sft_output=${SFT_OUTPUT_DIR}"
            echo "exit_code=${rc}"
            echo "host=$(hostname)"
            echo "failed_at=$(date '+%F %T')"
        } > "${SFT_FAILED_FILE}.tmp"
        mv -f "${SFT_FAILED_FILE}.tmp" "$SFT_FAILED_FILE"
    fi

    exit "$rc"
}
trap cleanup EXIT

# 发布本轮训练意图。
{
    echo "trigger_fingerprint=${trigger_fingerprint}"
    echo "data_success_mtime=$(stat -Lc '%y' "$DATA_SUCCESS")"
    echo "base_model=${BASE_STG2_MODEL_DIR}"
    echo "sft_output=${SFT_OUTPUT_DIR}"
    echo "pid=$$"
    echo "host=$(hostname)"
    echo "started_at=$(date '+%F %T')"
} > "${SFT_RUNNING_FILE}.tmp"
mv -f "${SFT_RUNNING_FILE}.tmp" "$SFT_RUNNING_FILE"

# 新一轮开始后，旧的成功标记不能再代表当前 SFT 输出目录。
# 如果本轮失败，infer_daily.sh 不会误用半成品目录。
rm -f "$SFT_SUCCESS_FILE" "$SFT_FAILED_FILE"

# 与推理共用同一把 GPU 锁。如果推理已经开始，等待其结束。
exec 8>"$GPU_EXCLUSIVE_LOCK_FILE"
echo "[$(date '+%F %T')] 等待 GPU 独占锁……"
flock 8
echo "[$(date '+%F %T')] 已获得 GPU 独占锁。"

set +u
source "${PYTHON_ENV}/bin/activate"
set -u

process_data() {
    cd "$REPO_ROOT"
    python3 process_train_data_daily.py

    cd "${REPO_ROOT}/data/onerec_data"
    bash run_daily.sh

    cd "${REPO_ROOT}/data"
    bash prepare_sft_daily.sh
}

train_model() {
    # 仅在新一轮 SFT 真正开始时删除上一轮的全部 checkpoint/converted。
    # 原始 BASE_STG2_MODEL_DIR 永远不会被删除或覆盖。
    echo "[$(date '+%F %T')] 删除上一轮 SFT 输出：${SFT_OUTPUT_DIR}"
    rm -rf "$SFT_OUTPUT_DIR"

    cd "$PRETRAIN_ROOT"

    # 该训练脚本应固定以原始 stg2 21000 step 为初始模型，
    # 输出仍为 SFT_OUTPUT_DIR，并在 15000 step 产出 checkpoint。
    bash examples/posttrain_sft_residual_sid_4layer.sh
}

convert_model() {
    cd "$PRETRAIN_ROOT"

    # 固定使用原始 stg2 converted 作为基础配置/模型来源，
    # 将本轮 SFT 的 15000 step 转为 HF converted。
    bash scripts/convert_residual_checkpoint_to_hf.sh \
        "$BASE_STG2_MODEL_DIR" \
        "$SFT_OUTPUT_DIR" \
        15000

    for required in \
        "${SFT_CONVERTED_MODEL_DIR}/config.json" \
        "${SFT_CONVERTED_MODEL_DIR}/tokenizer_config.json" \
        "${SFT_CONVERTED_MODEL_DIR}/residual_sid_config.json" \
        "${SFT_CONVERTED_MODEL_DIR}/model.safetensors.index.json"
    do
        if [[ ! -f "$required" ]]; then
            echo "本轮 SFT converted 缺少文件：${required}"
            return 1
        fi
    done

    # 不再复制到 stg2，不再替换 stg2，也不删除 SFT_OUTPUT_DIR。
    echo "[$(date '+%F %T')] 本轮 SFT converted 已保留：${SFT_CONVERTED_MODEL_DIR}"
}

echo "============================================================"
echo "OpenOneRec DATA_SUCCESS 触发式 SFT"
echo "DATA_SUCCESS        : ${DATA_SUCCESS}"
echo "trigger fingerprint : ${trigger_fingerprint}"
echo "base stg2 model     : ${BASE_STG2_MODEL_DIR}"
echo "SFT output          : ${SFT_OUTPUT_DIR}"
echo "SFT converted       : ${SFT_CONVERTED_MODEL_DIR}"
echo "log                 : ${LOG_FILE}"
echo "============================================================"

process_data
train_model
convert_model

# 只记录本轮开始时捕获的指纹。
# 若训练期间 DATA_SUCCESS 又更新，下次 cron 会再次检测到差异。
printf '%s\n' "$trigger_fingerprint" > "${LAST_TRAINED_FINGERPRINT_FILE}.tmp"
mv -f \
    "${LAST_TRAINED_FINGERPRINT_FILE}.tmp" \
    "$LAST_TRAINED_FINGERPRINT_FILE"

{
    echo "fingerprint=${trigger_fingerprint}"
    echo "data_success_mtime=$(stat -Lc '%y' "$DATA_SUCCESS" 2>/dev/null || true)"
    echo "base_model=${BASE_STG2_MODEL_DIR}"
    echo "model=${SFT_CONVERTED_MODEL_DIR}"
    echo "sft_output=${SFT_OUTPUT_DIR}"
    echo "host=$(hostname)"
    echo "completed_at=$(date '+%F %T')"
} > "${LAST_TRAINED_INFO_FILE}.tmp"
mv -f "${LAST_TRAINED_INFO_FILE}.tmp" "$LAST_TRAINED_INFO_FILE"

cp -f "$LAST_TRAINED_INFO_FILE" "${SFT_SUCCESS_FILE}.tmp"
mv -f "${SFT_SUCCESS_FILE}.tmp" "$SFT_SUCCESS_FILE"

rm -f "$SFT_FAILED_FILE"
completed=true

echo "============================================================"
echo "模型更新完成"
echo "trained fingerprint : ${trigger_fingerprint}"
echo "base stg2 model     : ${BASE_STG2_MODEL_DIR}"
echo "SFT converted       : ${SFT_CONVERTED_MODEL_DIR}"
echo "SFT checkpoints     : 已保留，下一轮 SFT 开始前删除"
echo "log                 : ${LOG_FILE}"
echo "============================================================"
