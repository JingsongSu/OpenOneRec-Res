#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# Hadoop 机器：
# 自动检查 day-2 HDFS 目录；发现 _SUCCESS 后，先删除共享盘旧输入，
# 再拉取新数据，校验完成后发布 _INPUT_READY。
# =============================================================================

# 实际拉取的数据日期：D-2
readonly DAY="$(TZ=Asia/Shanghai date -d "2 days ago" +%Y%m%d)"

# 日志文件日期：脚本运行当天 D
readonly LOG_DAY="$(TZ=Asia/Shanghai date +%Y%m%d)"

readonly REPO_ROOT="/home/jovyan/ceph-1/sujinsong/online/openonerec-res"
readonly LOG_DIR="${OPENONEREC_LOG_DIR:-${REPO_ROOT}/logs}"
readonly LOG_FILE="${LOG_DIR}/pull_data_${LOG_DAY}.log"

readonly HDFS_BASE="/home/hdp-ads-algo/project/user/zhouyuhang/dsp_recall/rqvae/common/merge_search_id_query_tag_ad_all"
readonly HDFS_DAY_DIR="${HDFS_BASE}/${DAY}"

# Hadoop 机器看到的共享盘路径。
readonly SHARED_ROOT="/sujingsong/sujingsong"
readonly SHARED_DATA_DIR="${SHARED_ROOT}/data"

# GPU 数据转换与 Hadoop 删除/下载共用的跨机器目录锁。
# 两台机器必须看到同一块共享盘：
# Hadoop: /sujingsong/sujingsong
# GPU   : /home/jovyan/zhouyuhang-cloud1/sujingsong
readonly INPUT_ACCESS_LOCK_DIR="${SHARED_ROOT}/.openonerec_input_access.lock"

# 每 2 分钟检查一次，最多约 20 小时。
readonly CHECK_INTERVAL_SECONDS=120
readonly MAX_CHECKS=600

# cron 环境若找不到 hadoop，请改成 Hadoop 可执行文件的绝对路径。
readonly HADOOP_BIN="${LOCAL_HADOOP:-hadoop}"

readonly INPUT_READY_FILE="${SHARED_DATA_DIR}/_INPUT_READY"
readonly INPUT_INFO_FILE="${SHARED_DATA_DIR}/input_info.txt"

readonly PULL_RUNNING_FILE="${SHARED_ROOT}/_PULL_RUNNING"
readonly PULL_FAILED_FILE="${SHARED_ROOT}/_PULL_FAILED"

readonly LOCAL_SCRIPT_LOCK="/tmp/openonerec_pull_data.lock"

mkdir -p "$LOG_DIR" "$SHARED_ROOT" "$SHARED_DATA_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

# 防止 Hadoop 机器上重复运行多个 pull_data.sh。
exec 9>"$LOCAL_SCRIPT_LOCK"
if ! flock -n 9; then
    echo "[$(date '+%F %T')] 另一个 pull_data.sh 正在运行，本次退出。"
    exit 0
fi

hdfs() {
    "$HADOOP_BIN" fs "$@"
}

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

input_lock_acquired=false
pull_completed=false

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
    rm -f "$PULL_RUNNING_FILE"

    if [[ "$pull_completed" != true && $rc -ne 0 ]]; then
        {
            echo "day=${DAY}"
            echo "hdfs_dir=${HDFS_DAY_DIR}"
            echo "exit_code=${rc}"
            echo "host=$(hostname)"
            echo "failed_at=$(date '+%F %T')"
        } > "${PULL_FAILED_FILE}.tmp"
        mv -f "${PULL_FAILED_FILE}.tmp" "$PULL_FAILED_FILE"
    fi

    exit "$rc"
}
trap cleanup EXIT

try_acquire_input_lock() {
    if mkdir "$INPUT_ACCESS_LOCK_DIR" 2>/dev/null; then
        input_lock_acquired=true

        {
            echo "owner=pull_data"
            echo "day=${DAY}"
            echo "pid=$$"
            echo "host=$(hostname)"
            echo "created_at=$(date '+%F %T')"
        } > "${INPUT_ACCESS_LOCK_DIR}/owner"

        return 0
    fi

    return 1
}

pull_new_data() {
    local remote_part_count
    local local_part_count

    remote_part_count="$(
        { hdfs -ls "${HDFS_DAY_DIR}/part-*" 2>/dev/null || true; } |
        awk '$1 ~ /^-/ {count++} END {print count + 0}'
    )"

    if [[ "$remote_part_count" -le 0 ]]; then
        echo "[$(date '+%F %T')] HDFS 中没有找到 part-*。"
        return 10
    fi

    if ! try_acquire_input_lock; then
        echo "[$(date '+%F %T')] GPU 正在转换输入数据，本次暂不删除共享盘数据。"
        return 20
    fi

    {
        echo "day=${DAY}"
        echo "hdfs_dir=${HDFS_DAY_DIR}"
        echo "expected_parts=${remote_part_count}"
        echo "pid=$$"
        echo "host=$(hostname)"
        echo "started_at=$(date '+%F %T')"
    } > "${PULL_RUNNING_FILE}.tmp"
    mv -f "${PULL_RUNNING_FILE}.tmp" "$PULL_RUNNING_FILE"

    echo "[$(date '+%F %T')] 检测到 ${DAY}/_SUCCESS。"
    echo "[$(date '+%F %T')] 先删除共享盘旧输入目录：${SHARED_DATA_DIR}"

    # 按要求：先完整删除过去的数据和状态，再重新拉取。
    rm -rf "$SHARED_DATA_DIR"
    mkdir -p "$SHARED_DATA_DIR"

    echo "[$(date '+%F %T')] 开始下载：${HDFS_DAY_DIR}/*"

    if ! hdfs -get "${HDFS_DAY_DIR}/*" "$SHARED_DATA_DIR/"; then
        echo "[$(date '+%F %T')] hadoop fs -get 失败，旧输入已经删除，后续会继续重试。"
        return 30
    fi

    if [[ ! -f "${SHARED_DATA_DIR}/_SUCCESS" ]]; then
        echo "[$(date '+%F %T')] 下载结果缺少 _SUCCESS。"
        return 31
    fi

    local_part_count="$(
        find "$SHARED_DATA_DIR" -maxdepth 1 -type f -name 'part-*' | wc -l
    )"

    if [[ "$local_part_count" -ne "$remote_part_count" ]]; then
        echo "[$(date '+%F %T')] part 数量不一致：remote=${remote_part_count}, local=${local_part_count}"
        return 32
    fi

    {
        echo "day=${DAY}"
        echo "hdfs_dir=${HDFS_DAY_DIR}"
        echo "part_count=${local_part_count}"
        echo "host=$(hostname)"
        echo "completed_at=$(date '+%F %T')"
    } > "${INPUT_INFO_FILE}.tmp"
    mv -f "${INPUT_INFO_FILE}.tmp" "$INPUT_INFO_FILE"

    # 必须最后发布，GPU 只以这个日期标记判断输入完整。
    printf '%s\n' "$DAY" > "${INPUT_READY_FILE}.tmp"
    mv -f "${INPUT_READY_FILE}.tmp" "$INPUT_READY_FILE"

    rm -f "$PULL_RUNNING_FILE" "$PULL_FAILED_FILE"
    pull_completed=true

    echo "============================================================"
    echo "Hadoop 数据拉取完成"
    echo "day        : ${DAY}"
    echo "parts      : ${local_part_count}"
    echo "shared dir : ${SHARED_DATA_DIR}"
    echo "log        : ${LOG_FILE}"
    echo "============================================================"

    release_input_lock
    return 0
}

echo "============================================================"
echo "OpenOneRec pull_data.sh"
echo "target day : ${DAY}"
echo "HDFS       : ${HDFS_DAY_DIR}"
echo "shared     : ${SHARED_DATA_DIR}"
echo "interval   : ${CHECK_INTERVAL_SECONDS}s"
echo "max checks : ${MAX_CHECKS}"
echo "log        : ${LOG_FILE}"
echo "============================================================"

if input_is_ready; then
    echo "[$(date '+%F %T')] ${DAY} 已经完整拉取，无需重复执行。"
    exit 0
fi

for ((attempt = 1; attempt <= MAX_CHECKS; attempt++)); do
    echo "[$(date '+%F %T')] 第 ${attempt}/${MAX_CHECKS} 次检查。"

    if input_is_ready; then
        echo "[$(date '+%F %T')] ${DAY} 已经完整拉取，结束循环。"
        exit 0
    fi

    if hdfs -test -e "${HDFS_DAY_DIR}/_SUCCESS"; then
        if pull_new_data; then
            exit 0
        else
            rc=$?
            echo "[$(date '+%F %T')] 本次拉取未完成，code=${rc}，稍后继续重试。"
            release_input_lock
            rm -f "$PULL_RUNNING_FILE"
        fi
    else
        echo "[$(date '+%F %T')] HDFS 尚未就绪：${HDFS_DAY_DIR}/_SUCCESS"
    fi

    if [[ "$attempt" -lt "$MAX_CHECKS" ]]; then
        sleep "$CHECK_INTERVAL_SECONDS"
    fi
done

{
    echo "day=${DAY}"
    echo "hdfs_dir=${HDFS_DAY_DIR}"
    echo "max_checks=${MAX_CHECKS}"
    echo "check_interval_seconds=${CHECK_INTERVAL_SECONDS}"
    echo "host=$(hostname)"
    echo "failed_at=$(date '+%F %T')"
    echo "reason=input_not_ready_after_max_checks"
} > "${PULL_FAILED_FILE}.tmp"
mv -f "${PULL_FAILED_FILE}.tmp" "$PULL_FAILED_FILE"

echo "[$(date '+%F %T')] ${MAX_CHECKS} 次检查后仍未成功拉取。"
exit 1
