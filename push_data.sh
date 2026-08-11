#!/bin/bash

set -ex
set -o pipefail

# =============================================================================
# OpenOneRec 推理结果上线脚本
#
# 脚本位置：
#   /sujingsong/sujingsong/push_data.sh
#
# 输入：
#   data/_INFER_SUCCESS
#   online_infer/infer_adid_parts/ 下所有严格四位编号的 part 文件：
#     part-0000
#     part-0001
#     ...
#
# 流程：
#   1. 读取最近一次已经完成推理的数据版本；
#   2. 如果该推理版本尚未上线，则执行 push；
#   3. 自动检测所有四位编号 part，并逐个检查非空；
#   4. 上传全部检测到的 part 到 HDFS；
#   5. 调用 build_kv_3shard；
#   6. 调用 push_kv_3shard；
#   7. 写入 _PUSH_SUCCESS。
#
# 日期规则：
#   _INFER_SUCCESS 中记录推理数据日期。
#   upload_date 固定等于 infer_success_date + 2 天。
#
# 例如：
#   _INFER_SUCCESS=20260805
#   则 upload_date=20260807
#
#   即使脚本实际到 20260808 才执行，
#   HDFS/build/push 仍然使用 20260807。
#
# 日志：
#   logs/push_data_YYYYMMDD.log
#   每行带 Asia/Shanghai 时间戳。
# =============================================================================


# -----------------------------------------------------------------------------
# 基础目录
# -----------------------------------------------------------------------------

readonly DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${DATA_DIR}/config/conf.sh"
source "${DATA_DIR}/config/utility.sh"
source /data/hdp-ads-algo/zhangguozhu/rqvae_recall_daily/rq-pipeline/rq_env/bin/activate


# -----------------------------------------------------------------------------
# 日志：每天一个文件，每行添加上海时间戳
# -----------------------------------------------------------------------------

readonly LOG_DIR="${DATA_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 日志文件按照脚本实际运行当天命名。
readonly LOG_DAY="$(TZ=Asia/Shanghai date +%Y%m%d)"
readonly LOG_FILE="${LOG_DIR}/push_data_${LOG_DAY}.log"


exec > >(
    TZ=Asia/Shanghai awk '
        {
            print strftime("[%Y-%m-%d %H:%M:%S]"), $0
            fflush()
        }
    ' | tee -a "${LOG_FILE}"
) 2>&1


# -----------------------------------------------------------------------------
# step0. 防止多个 push 流程同时运行
#
# 这是 push 任务自身的运行标记，不涉及 GPU。
# -----------------------------------------------------------------------------

readonly RUNNING_FILE="${DATA_DIR}/_PUSH_RUNNING"

if [ -f "${RUNNING_FILE}" ]; then
    echo "last online push is running.."
    exit 0
fi

touch "${RUNNING_FILE}"


cleanup() {
    echo "exit delete _PUSH_RUNNING.."
    rm -f "${RUNNING_FILE}"
}

trap cleanup EXIT


# -----------------------------------------------------------------------------
# step1. 相关路径
# -----------------------------------------------------------------------------

readonly INFER_SUCCESS_FILE="${DATA_DIR}/data/_INFER_SUCCESS"
readonly INFER_RESULT_DIR="${DATA_DIR}/online_infer/infer_adid_parts"

readonly PUSH_SUCCESS_FILE="${DATA_DIR}/_PUSH_SUCCESS"

readonly BUILD_DIR="${DATA_DIR}/build_kv_3shard"
readonly PUSH_DIR="${DATA_DIR}/push_kv_3shard"

readonly BUILD_CONF="${BUILD_DIR}/config/conf.sh"
readonly PUSH_CONF="${PUSH_DIR}/config/conf.sh"


# -----------------------------------------------------------------------------
# step2. 检查最近一次已经完成的推理版本
#
# 不再要求必须等于当前系统日期的 D-2。
#
# _INFER_SUCCESS 中记录哪个数据日期，
# 就认为哪个版本已经完整推理完成。
#
# upload_date 固定为：
#
#   infer_success_date + 2 days
#
# -----------------------------------------------------------------------------

if [ ! -f "${INFER_SUCCESS_FILE}" ]; then
    echo "${INFER_SUCCESS_FILE} not found, inference is not ready.."
    exit 0
fi

infer_success_date=$(tr -d '[:space:]' < "${INFER_SUCCESS_FILE}")

if [ -z "${infer_success_date}" ]; then
    echo "${INFER_SUCCESS_FILE} is empty, inference is not ready.."
    exit 0
fi


# 为了尽量保持后续原有逻辑不变，
# current_date 仍然表示当前准备上线的推理数据日期。
current_date="${infer_success_date}"


# HDFS 上传目录日期，以及 build/push 参数：
# 固定使用推理数据日期 + 2 天。
upload_date=$(
    TZ=Asia/Shanghai date \
        -d "${infer_success_date} +2 days" \
        +%Y%m%d
)


echo "============================================================"
echo "OpenOneRec online push check"
echo "infer_data_date=${current_date}"
echo "upload_date=${upload_date}"
echo "infer_success_file=${INFER_SUCCESS_FILE}"
echo "infer_result_dir=${INFER_RESULT_DIR}"
echo "log_file=${LOG_FILE}"
echo "============================================================"


# -----------------------------------------------------------------------------
# step3. 判断当前最新推理结果是否已经上线
#
# 只比较：
#   _INFER_SUCCESS
#   _PUSH_SUCCESS
#
# 两者相同：
#   当前最新推理结果已经上线，不重复执行。
#
# 两者不同：
#   说明 infer 数据版本发生变化，立即执行 push。
# -----------------------------------------------------------------------------

if [ -f "${PUSH_SUCCESS_FILE}" ]; then
    last_push_date=$(tr -d '[:space:]' < "${PUSH_SUCCESS_FILE}")

    if [ "${last_push_date}" = "${current_date}" ]; then
        echo "${current_date} inference result has already been pushed.."
        exit 0
    fi

    echo "new inference result detected:"
    echo "last pushed infer date=${last_push_date}"
    echo "current infer date=${current_date}"
else
    echo "no previous _PUSH_SUCCESS found."
    echo "current infer date=${current_date}"
fi


# -----------------------------------------------------------------------------
# step4. 自动检测所有严格四位编号 part
#
# 匹配：
#   part-0000
#   part-0001
#   part-0099
#   part-0100
#   part-0299
#
# 不匹配：
#   part-000
#   part-00000
#   part-old
#   part-0001.tmp
# -----------------------------------------------------------------------------

if [ ! -d "${INFER_RESULT_DIR}" ]; then
    echo "${INFER_RESULT_DIR} not found.."
    exit 0
fi


declare -a PART_FILES=()

while IFS= read -r part_file; do
    PART_FILES+=("${part_file}")
done < <(
    find "${INFER_RESULT_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'part-[0-9][0-9][0-9][0-9]' \
        -print |
    sort
)


NUM_PART="${#PART_FILES[@]}"


if [ "${NUM_PART}" -le 0 ]; then
    echo "no valid part files found in:"
    echo "${INFER_RESULT_DIR}"
    exit 0
fi


# 逐个检查所有检测到的 part 必须非空。
for part_file in "${PART_FILES[@]}"; do
    if [ ! -f "${part_file}" ]; then
        echo "${part_file} not found.."
        exit 0
    fi

    if [ ! -s "${part_file}" ]; then
        echo "${part_file} is empty.."
        exit 0
    fi
done


FIRST_PART="$(basename "${PART_FILES[0]}")"
LAST_PART="$(basename "${PART_FILES[$((NUM_PART - 1))]}")"


echo "inference result is ready:"
echo "current_date=${current_date}"
echo "upload_date=${upload_date}"
echo "result_dir=${INFER_RESULT_DIR}"
echo "part_count=${NUM_PART}"
echo "part_range=${FIRST_PART} ~ ${LAST_PART}"


# -----------------------------------------------------------------------------
# step5. 检查 build/push 配置和入口脚本
#
# HDFS ROOT_DIR 读取自：
#   build_kv_3shard/config/conf.sh
#   push_kv_3shard/config/conf.sh
#
# 两边必须完全相同。
# -----------------------------------------------------------------------------

if [ ! -f "${BUILD_CONF}" ]; then
    echo "${BUILD_CONF} not found.."
    exit 1
fi

if [ ! -f "${PUSH_CONF}" ]; then
    echo "${PUSH_CONF} not found.."
    exit 1
fi

if [ ! -f "${BUILD_DIR}/entrance.sh" ]; then
    echo "${BUILD_DIR}/entrance.sh not found.."
    exit 1
fi

if [ ! -f "${PUSH_DIR}/entrance.sh" ]; then
    echo "${PUSH_DIR}/entrance.sh not found.."
    exit 1
fi


BUILD_UPLOAD_DIR=$(
    bash -c 'source "$1"; printf "%s" "${ROOT_DIR}"' _ "${BUILD_CONF}"
)

PUSH_UPLOAD_DIR=$(
    bash -c 'source "$1"; printf "%s" "${ROOT_DIR}"' _ "${PUSH_CONF}"
)


if [ -z "${BUILD_UPLOAD_DIR}" ]; then
    echo "ROOT_DIR is empty in ${BUILD_CONF}"
    exit 1
fi

if [ -z "${PUSH_UPLOAD_DIR}" ]; then
    echo "ROOT_DIR is empty in ${PUSH_CONF}"
    exit 1
fi


if [ "${BUILD_UPLOAD_DIR}" != "${PUSH_UPLOAD_DIR}" ]; then
    echo "build and push ROOT_DIR are different.."
    echo "build ROOT_DIR=${BUILD_UPLOAD_DIR}"
    echo "push  ROOT_DIR=${PUSH_UPLOAD_DIR}"
    exit 1
fi


UPLOAD_DIR="${BUILD_UPLOAD_DIR}"
HDFS_UPLOAD_PATH="${UPLOAD_DIR}/${upload_date}"

echo "UPLOAD_DIR=${UPLOAD_DIR}"
echo "HDFS_UPLOAD_PATH=${HDFS_UPLOAD_PATH}"


# -----------------------------------------------------------------------------
# step6. 上传自动检测到的全部 part 到 HDFS
# -----------------------------------------------------------------------------

if "${LOCAL_HADOOP}" fs -test -e "${HDFS_UPLOAD_PATH}"; then
    echo "HDFS path already exists, delete before upload:"
    echo "${HDFS_UPLOAD_PATH}"

    "${LOCAL_HADOOP}" fs -rm -r -f "${HDFS_UPLOAD_PATH}"
fi


"${LOCAL_HADOOP}" fs -mkdir -p "${HDFS_UPLOAD_PATH}"


# 只上传前面自动检测并逐个校验通过的 part 文件。
"${LOCAL_HADOOP}" fs -put \
    "${PART_FILES[@]}" \
    "${HDFS_UPLOAD_PATH}/"


# 检查 HDFS 中严格四位编号的 part 数量。
hdfs_part_count=$(
    "${LOCAL_HADOOP}" fs -ls "${HDFS_UPLOAD_PATH}" |
    awk '
        NF >= 8 &&
        $NF ~ /\/part-[0-9][0-9][0-9][0-9]$/ {
            count++
        }
        END {
            print count + 0
        }
    '
)


# HDFS part 数必须和本地实际检测到的 part 数完全一致。
if [ "${hdfs_part_count}" -ne "${NUM_PART}" ]; then
    echo "HDFS part count error:"
    echo "expected=${NUM_PART}"
    echo "actual=${hdfs_part_count}"
    exit 1
fi


echo "upload inference result success:"
echo "HDFS path=${HDFS_UPLOAD_PATH}"
echo "local part count=${NUM_PART}"
echo "HDFS part count=${hdfs_part_count}"


# -----------------------------------------------------------------------------
# step7. build/push 失败重试函数
# -----------------------------------------------------------------------------

run_cmd_with_retries() {
    local max_retries=$1
    local delay=$2

    shift 2

    local attempt=1

    while [ "${attempt}" -le "${max_retries}" ]; do
        echo "Attempt ${attempt} of ${max_retries}: $*"

        if "$@"; then
            echo "Command succeeded."
            return 0
        fi

        if [ "${attempt}" -lt "${max_retries}" ]; then
            echo "Retrying after ${delay} seconds..."
            sleep "${delay}"
        else
            echo "Command failed after ${max_retries} attempts."
            return 1
        fi

        attempt=$((attempt + 1))
    done
}


# -----------------------------------------------------------------------------
# step8. 调用 build_kv_3shard
#
# 输入：
#   ${ROOT_DIR}/${upload_date}/
#
# 输出：
#   ${ROOT_DIR}/${upload_date}/build3shard/
# -----------------------------------------------------------------------------

cd "${BUILD_DIR}"

if run_cmd_with_retries 10 600 bash entrance.sh "${upload_date}"; then
    echo "build success..."
else
    echo "build failed"
    exit 1
fi


# -----------------------------------------------------------------------------
# step9. 调用 push_kv_3shard
# -----------------------------------------------------------------------------

cd "${PUSH_DIR}"

if run_cmd_with_retries 10 600 bash entrance.sh "${upload_date}"; then
    echo "push success..."
else
    echo "push failed"
    exit 1
fi


cd "${DATA_DIR}"


# -----------------------------------------------------------------------------
# step10. 原子写入 push 成功标记
#
# 标记记录实际已经成功上线的 infer 数据日期。
# -----------------------------------------------------------------------------

PUSH_SUCCESS_TMP="${PUSH_SUCCESS_FILE}.tmp.$$"

printf "%s\n" "${current_date}" > "${PUSH_SUCCESS_TMP}"

mv -f \
    "${PUSH_SUCCESS_TMP}" \
    "${PUSH_SUCCESS_FILE}"


echo "============================================================"
echo "online push complete"
echo "infer_data_date=${current_date}"
echo "upload_date=${upload_date}"
echo "part_count=${NUM_PART}"
echo "part_range=${FIRST_PART} ~ ${LAST_PART}"
echo "HDFS path=${HDFS_UPLOAD_PATH}"
echo "success marker=${PUSH_SUCCESS_FILE}"
echo "log=${LOG_FILE}"
echo "============================================================"

exit 0
