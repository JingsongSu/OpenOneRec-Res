#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# GPU 机器：
#   day-2 输入完整且当天尚未推理时，尝试抢占 GPU 锁执行推理。
#   推理固定读取最近一次成功保留下来的 SFT 21000 step converted，
#   不再读取或依赖被日更覆盖的 stg2 目录。
#
# 与 run_daily.sh 没有优先级，谁先拿到锁谁运行。
# 建议由 cron 每分钟调用一次。
# =============================================================================

# 本次处理的数据日期：D-2
readonly DAY="$(TZ=Asia/Shanghai date -d "2 days ago" +%Y%m%d)"

# 日志文件日期：当天 D
readonly LOG_DAY="$(TZ=Asia/Shanghai date +%Y%m%d)"

readonly REPO_ROOT="/home/jovyan/ceph-1/sujinsong/online/openonerec-res"
readonly PYTHON_ENV="/home/jovyan/ceph-1/sujinsong/env/onerec"

readonly LOG_DIR="${REPO_ROOT}/logs"
readonly LOG_FILE="${LOG_DIR}/infer_daily_${LOG_DAY}.log"

readonly CONVERT_SCRIPT="${REPO_ROOT}/convert_all_parts_to_one_parquet_fast.py"
readonly INFER_SCRIPT="${REPO_ROOT}/online_residual_sid_infer_by_history_ctype_tuned.py"

# GPU 机器看到的同一共享盘。
readonly SHARED_ROOT="/home/jovyan/zhouyuhang-cloud1/sujingsong"
readonly SHARED_DATA_DIR="${SHARED_ROOT}/data"

# 必须与 pull_data.sh 在同一共享盘位置对应。
readonly INPUT_ACCESS_LOCK_DIR="${SHARED_ROOT}/.openonerec_input_access.lock"

readonly ONLINE_INFER_DIR="${SHARED_ROOT}/online_infer"
readonly OUTPUT_PARQUET="${ONLINE_INFER_DIR}/all_parts_infer.parquet"
readonly CONVERT_TEMP_DIR="${ONLINE_INFER_DIR}/.all_parts_infer_tmp"
readonly INFER_OUTPUT_DIR="${ONLINE_INFER_DIR}/infer_adid_parts"

# 直接读取 run_daily.sh 本轮成功后保留的 SFT 21000 step converted。
readonly CONVERTED_MODEL_PATH="${REPO_ROOT}/pretrain/model_output/sft_full_residual_add_feature_daily/step15000/global_step15000/converted"

# run_daily.sh 只有在训练和 converted 转换全部成功后才写这个标记。
readonly SFT_SUCCESS_FILE="${REPO_ROOT}/_SFT_SUCCESS"

readonly INPUT_READY_FILE="${SHARED_DATA_DIR}/_INPUT_READY"
readonly INFER_RUNNING_FILE="${SHARED_DATA_DIR}/_INFER_RUNNING"
readonly INFER_SUCCESS_FILE="${SHARED_DATA_DIR}/_INFER_SUCCESS"
readonly INFER_FAILED_FILE="${SHARED_DATA_DIR}/_INFER_FAILED"

readonly SCRIPT_LOCK_FILE="/tmp/openonerec_infer_daily.lock"

# 必须与 run_daily.sh 完全相同。
readonly GPU_EXCLUSIVE_LOCK_FILE="/tmp/openonerec_gpu_exclusive.lock"

mkdir -p "$LOG_DIR" "$SHARED_DATA_DIR" "$ONLINE_INFER_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

read_marker_day() {
    local marker="$1"
    [[ -f "$marker" ]] || return 1
    tr -d '[:space:]' < "$marker"
}

input_is_ready() {
    local ready_day
    ready_day="$(read_marker_day "$INPUT_READY_FILE" || true)"

    [[ "$ready_day" == "$DAY" ]] || return 1
    [[ -f "${SHARED_DATA_DIR}/_SUCCESS" ]] || return 1
    compgen -G "${SHARED_DATA_DIR}/part-*" >/dev/null || return 1
}

inference_is_complete() {
    local success_day
    success_day="$(read_marker_day "$INFER_SUCCESS_FILE" || true)"

    [[ "$success_day" == "$DAY" ]] || return 1
    compgen -G "${INFER_OUTPUT_DIR}/part-*" >/dev/null || return 1
}

sft_model_is_ready() {
    local required

    # 必须是 run_daily.sh 最近一次完整成功后发布的模型。
    [[ -f "$SFT_SUCCESS_FILE" ]] || return 1
    grep -Fqx "model=${CONVERTED_MODEL_PATH}" "$SFT_SUCCESS_FILE" || return 1

    for required in \
        "${CONVERTED_MODEL_PATH}/config.json" \
        "${CONVERTED_MODEL_PATH}/tokenizer_config.json" \
        "${CONVERTED_MODEL_PATH}/residual_sid_config.json" \
        "${CONVERTED_MODEL_PATH}/model.safetensors.index.json"
    do
        [[ -f "$required" ]] || return 1
    done
}

if inference_is_complete; then
    echo "[$(date '+%F %T')] ${DAY} 已经推理完成，无需重复执行。"
    exit 0
fi

if ! input_is_ready; then
    echo "[$(date '+%F %T')] ${DAY} 的 Hadoop 输入尚未准备完成。"
    exit 0
fi

# 这里只提前检查不会被 SFT 删除或重建的脚本。
# SFT 模型必须等拿到 GPU 锁后再检查，避免日更正在重建目录时误报失败。
for required in \
    "$CONVERT_SCRIPT" \
    "$INFER_SCRIPT"
do
    if [[ ! -f "$required" ]]; then
        echo "必要脚本不存在：${required}"
        exit 1
    fi
done

# 防止 cron 重复启动本脚本。
exec 9>"$SCRIPT_LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] 另一个 infer_daily.sh 正在执行，本次退出。"
    exit 0
fi

# 获得自身锁后再次检查。
if inference_is_complete || ! input_is_ready; then
    exit 0
fi

# SFT 与推理没有优先级：随机等待少量时间，再抢同一把锁。
sleep "$((RANDOM % 15))"

exec 8>"$GPU_EXCLUSIVE_LOCK_FILE"
if ! flock -n 8; then
    echo "[$(date '+%F %T')] GPU 正被 SFT 或其他推理占用，本次退出，等待下次 cron。"
    exit 0
fi

# 获得 GPU 后再次确认输入和完成状态。
if inference_is_complete; then
    echo "[$(date '+%F %T')] 抢到 GPU 后发现当天推理已完成。"
    exit 0
fi

if ! input_is_ready; then
    echo "[$(date '+%F %T')] 抢到 GPU 后发现输入状态变化。"
    exit 0
fi

# 必须在拿到 GPU 锁后检查：
# run_daily.sh 删除旧 SFT 目录、训练、转换期间一直持有同一把锁。
if ! sft_model_is_ready; then
    echo "[$(date '+%F %T')] 尚无完整可用的 SFT 15000 step 模型，本次退出，等待下次 cron。"
    echo "expected model : ${CONVERTED_MODEL_PATH}"
    echo "success marker : ${SFT_SUCCESS_FILE}"
    exit 0
fi

input_lock_acquired=false
completed=false

release_input_lock() {
    if [[ "$input_lock_acquired" == true ]]; then
        rm -rf "$INPUT_ACCESS_LOCK_DIR"
        input_lock_acquired=false
    fi
}

cleanup() {
    local rc=$?
    trap - EXIT

    release_input_lock
    rm -f "$INFER_RUNNING_FILE"

    if [[ "$completed" != true && $rc -ne 0 ]]; then
        {
            echo "day=${DAY}"
            echo "model=${CONVERTED_MODEL_PATH}"
            echo "exit_code=${rc}"
            echo "host=$(hostname)"
            echo "failed_at=$(date '+%F %T')"
        } > "${INFER_FAILED_FILE}.tmp"
        mv -f "${INFER_FAILED_FILE}.tmp" "$INFER_FAILED_FILE"
    fi

    exit "$rc"
}
trap cleanup EXIT

# 与 pull_data.sh 互斥，防止转换原始 part-* 时目录被删除。
if ! mkdir "$INPUT_ACCESS_LOCK_DIR" 2>/dev/null; then
    echo "[$(date '+%F %T')] Hadoop 正在删除或下载输入，本次退出。"
    exit 0
fi
input_lock_acquired=true

{
    echo "owner=infer_daily"
    echo "day=${DAY}"
    echo "pid=$$"
    echo "host=$(hostname)"
    echo "created_at=$(date '+%F %T')"
} > "${INPUT_ACCESS_LOCK_DIR}/owner"

{
    echo "day=${DAY}"
    echo "pid=$$"
    echo "host=$(hostname)"
    echo "model=${CONVERTED_MODEL_PATH}"
    echo "started_at=$(date '+%F %T')"
} > "${INFER_RUNNING_FILE}.tmp"
mv -f "${INFER_RUNNING_FILE}.tmp" "$INFER_RUNNING_FILE"

rm -f "$INFER_SUCCESS_FILE" "$INFER_FAILED_FILE"

set +u
source "${PYTHON_ENV}/bin/activate"
set -u

cd "$REPO_ROOT"

input_part_count="$(
    find "$SHARED_DATA_DIR" -maxdepth 1 -type f -name 'part-*' | wc -l
)"

echo "============================================================"
echo "开始转换推理输入"
echo "day         : ${DAY}"
echo "input parts : ${input_part_count}"
echo "parquet     : ${OUTPUT_PARQUET}"
echo "============================================================"

rm -f "$OUTPUT_PARQUET"
rm -rf "$CONVERT_TEMP_DIR"

python3 "$CONVERT_SCRIPT"

if [[ ! -s "$OUTPUT_PARQUET" ]]; then
    echo "转换程序没有生成有效 Parquet：${OUTPUT_PARQUET}"
    exit 1
fi

# 转换完成后不再读取共享盘原始 part-*，可释放输入锁。
release_input_lock

# 正式推理前删除过去的全部推理结果。
echo "[$(date '+%F %T')] 删除过去推理结果：${INFER_OUTPUT_DIR}"
rm -rf "$INFER_OUTPUT_DIR"
mkdir -p "$INFER_OUTPUT_DIR"

echo "============================================================"
echo "开始 GPU 推理"
echo "model  : ${CONVERTED_MODEL_PATH}"
echo "input  : ${OUTPUT_PARQUET}"
echo "output : ${INFER_OUTPUT_DIR}"
echo "============================================================"

# 重要：
# 推理 Python 内部的 CONVERTED_MODEL_PATH 和 RESIDUAL_CONFIG_PATH
# 必须与上面的 CONVERTED_MODEL_PATH 一致。
python3 "$INFER_SCRIPT"

output_part_count="$(
    find "$INFER_OUTPUT_DIR" -maxdepth 1 -type f -name 'part-*' | wc -l
)"
nonempty_part_count="$(
    find "$INFER_OUTPUT_DIR" -maxdepth 1 -type f -name 'part-*' -size +0c | wc -l
)"

if [[ "$output_part_count" -le 0 ]]; then
    echo "推理没有生成 part-*。"
    exit 1
fi

if [[ "$nonempty_part_count" -le 0 ]]; then
    echo "推理结果 part 全部为空。"
    exit 1
fi

{
    echo "day=${DAY}"
    echo "model=${CONVERTED_MODEL_PATH}"
    echo "input_parts=${input_part_count}"
    echo "output_parts=${output_part_count}"
    echo "nonempty_output_parts=${nonempty_part_count}"
    echo "output_dir=${INFER_OUTPUT_DIR}"
    echo "host=$(hostname)"
    echo "completed_at=$(date '+%F %T')"
} > "${ONLINE_INFER_DIR}/infer_info_${DAY}.txt.tmp"
mv -f \
    "${ONLINE_INFER_DIR}/infer_info_${DAY}.txt.tmp" \
    "${ONLINE_INFER_DIR}/infer_info_${DAY}.txt"

printf '%s\n' "$DAY" > "${INFER_SUCCESS_FILE}.tmp"
mv -f "${INFER_SUCCESS_FILE}.tmp" "$INFER_SUCCESS_FILE"

rm -f "$INFER_FAILED_FILE"
completed=true

echo "============================================================"
echo "每日推理完成"
echo "day          : ${DAY}"
echo "model        : ${CONVERTED_MODEL_PATH}"
echo "output parts : ${output_part_count}"
echo "result       : ${INFER_OUTPUT_DIR}"
echo "log          : ${LOG_FILE}"
echo "============================================================"
