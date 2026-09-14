#!/bin/bash

set -ex
set -o pipefail

# =============================================================================
# OpenOneRec 推理结果上线脚本（增加线上正排 adid 白名单过滤）
#
# 输入：
#   data/_INFER_SUCCESS
#   online_infer/infer_adid_parts/part-XXXX
#
# 新增过滤流程：
#   1. 发现新的已完成推理版本；
#   2. 在 HDFS 正排根目录中寻找“日期最新且存在 _SUCCESS”的目录；
#   3. 只抽取该目录所有 part-* 的第一列，缓存成 data_zhengpai/legal_adids.txt；
#   4. 使用 legal_adids.txt 过滤推理结果中第 2 列及以后所有 adid；
#   5. 保留第一列 mid_ctype，候选 adid 顺序不变；
#   6. 一行若所有 adid 都被过滤掉，则整行删除；
#   7. 过滤结果写到 online_infer/infer_adid_parts_filtered；
#   8. 真正上传/build/push 只使用过滤后的目录；
#   9. 原始 infer_adid_parts 永远不改，便于失败重试或问题回溯。
#
# 注意：
#   推理输出格式实际是：
#       mid_ctype<TAB>adid1<TAB>adid2<...>
#   所以 HDFS 正排数据“第一列”是白名单 adid，
#   但推理结果中需要过滤的是第 2 列到最后一列。
# =============================================================================

# -----------------------------------------------------------------------------
# 基础目录
# -----------------------------------------------------------------------------

readonly DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${DATA_DIR}/config/conf.sh"
source "${DATA_DIR}/config/utility.sh"
source /data/hdp-ads-algo/zhangguozhu/rqvae_recall_daily/rq-pipeline/rq_env/bin/activate

# -----------------------------------------------------------------------------
# 日志
# -----------------------------------------------------------------------------

readonly LOG_DIR="${DATA_DIR}/logs"
mkdir -p "${LOG_DIR}"

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

# 原始推理结果：永远不在 push 阶段修改。
readonly INFER_RESULT_DIR="${DATA_DIR}/online_infer/infer_adid_parts"

# 真正用于上线的过滤后结果。
readonly FILTERED_INFER_RESULT_DIR="${DATA_DIR}/online_infer/infer_adid_parts_filtered"
readonly FILTER_SCRIPT="${DATA_DIR}/filter_infer_adids.py"

readonly PUSH_SUCCESS_FILE="${DATA_DIR}/_PUSH_SUCCESS"

readonly BUILD_DIR="${DATA_DIR}/build_kv_3shard"
readonly PUSH_DIR="${DATA_DIR}/push_kv_3shard"
readonly BUILD_CONF="${BUILD_DIR}/config/conf.sh"
readonly PUSH_CONF="${PUSH_DIR}/config/conf.sh"

# -----------------------------------------------------------------------------
# 正排白名单 HDFS / 本地缓存
# -----------------------------------------------------------------------------

readonly ZHENGPAI_HDFS_BASE="/home/hdp-ads-algo/project/user/wuji/strategy/ann_constrastive_learning_tn/lm_ad_ann/whole_ad_zhengpai_add_ctype_imgsize_all"

# push_data.sh 若部署在 /sujingsong/sujingsong，
# 这里正好就是 /sujingsong/sujingsong/data_zhengpai。
readonly ZHENGPAI_LOCAL_DIR="${DATA_DIR}/data_zhengpai"
readonly LEGAL_ADID_FILE="${ZHENGPAI_LOCAL_DIR}/legal_adids.txt"
readonly ZHENGPAI_READY_DATE_FILE="${ZHENGPAI_LOCAL_DIR}/_READY_DATE"
readonly ZHENGPAI_LOCAL_SUCCESS_FILE="${ZHENGPAI_LOCAL_DIR}/_SUCCESS"
readonly ZHENGPAI_SOURCE_INFO="${ZHENGPAI_LOCAL_DIR}/source_info.txt"

mkdir -p "${ZHENGPAI_LOCAL_DIR}"

# -----------------------------------------------------------------------------
# step2. 检查最近一次已经完成的推理版本
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

current_date="${infer_success_date}"

upload_date=$(
    TZ=Asia/Shanghai date \
        -d "${infer_success_date} +2 days" \
        +%Y%m%d
)

echo "============================================================"
echo "OpenOneRec online push check"
echo "infer_data_date=${current_date}"
echo "upload_date=${upload_date}"
echo "raw_infer_result_dir=${INFER_RESULT_DIR}"
echo "filtered_infer_result_dir=${FILTERED_INFER_RESULT_DIR}"
echo "zhengpai_hdfs_base=${ZHENGPAI_HDFS_BASE}"
echo "zhengpai_local_dir=${ZHENGPAI_LOCAL_DIR}"
echo "log_file=${LOG_FILE}"
echo "============================================================"

# -----------------------------------------------------------------------------
# step3. 判断当前最新推理结果是否已经上线
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
# step4. 先检查原始推理结果
# -----------------------------------------------------------------------------

if [ ! -d "${INFER_RESULT_DIR}" ]; then
    echo "${INFER_RESULT_DIR} not found.."
    exit 1
fi

declare -a RAW_PART_FILES=()
while IFS= read -r part_file; do
    RAW_PART_FILES+=("${part_file}")
done < <(
    find "${INFER_RESULT_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'part-[0-9][0-9][0-9][0-9]' \
        -print |
    sort
)

RAW_NUM_PART="${#RAW_PART_FILES[@]}"

if [ "${RAW_NUM_PART}" -le 0 ]; then
    echo "no valid raw inference part files found in:"
    echo "${INFER_RESULT_DIR}"
    exit 1
fi

for part_file in "${RAW_PART_FILES[@]}"; do
    if [ ! -s "${part_file}" ]; then
        echo "raw inference part is missing or empty: ${part_file}"
        exit 1
    fi
done

echo "raw inference result is ready:"
echo "raw_part_count=${RAW_NUM_PART}"
echo "raw_first_part=$(basename "${RAW_PART_FILES[0]}")"
echo "raw_last_part=$(basename "${RAW_PART_FILES[$((RAW_NUM_PART - 1))]}")"

if [ ! -f "${FILTER_SCRIPT}" ]; then
    echo "filter script not found: ${FILTER_SCRIPT}"
    exit 1
fi

# -----------------------------------------------------------------------------
# step5. 找 HDFS 上“最新且成功”的正排日期
#
# 只识别 YYYY-MM-DD 目录，例如：
#   2026-09-07
#
# 不要求它与 infer_success_date 相同；严格按用户要求取 HDFS 当前最新成功日。
# -----------------------------------------------------------------------------

find_latest_success_zhengpai_date() {
    local listing
    local candidate
    local -a candidate_dates=()

    if ! listing=$("${LOCAL_HADOOP}" fs -ls "${ZHENGPAI_HDFS_BASE}" 2>&1); then
        echo "failed to list zhengpai HDFS base:" >&2
        echo "${ZHENGPAI_HDFS_BASE}" >&2
        echo "${listing}" >&2
        return 1
    fi

    mapfile -t candidate_dates < <(
        printf '%s\n' "${listing}" |
        awk 'NF >= 8 {print $NF}' |
        sed 's#/*$##' |
        awk -F/ '{print $NF}' |
        grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' |
        LC_ALL=C sort -r
    )

    if [ "${#candidate_dates[@]}" -le 0 ]; then
        echo "no YYYY-MM-DD directories found under ${ZHENGPAI_HDFS_BASE}" >&2
        return 1
    fi

    for candidate in "${candidate_dates[@]}"; do
        if "${LOCAL_HADOOP}" fs -test -e \
            "${ZHENGPAI_HDFS_BASE}/${candidate}/_SUCCESS"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    echo "no successful zhengpai date found under ${ZHENGPAI_HDFS_BASE}" >&2
    return 1
}

if ! latest_zhengpai_date="$(find_latest_success_zhengpai_date)"; then
    echo "cannot find latest successful zhengpai data, stop push."
    exit 1
fi

readonly LATEST_ZHENGPAI_DATE="${latest_zhengpai_date}"
readonly LATEST_ZHENGPAI_HDFS_DIR="${ZHENGPAI_HDFS_BASE}/${LATEST_ZHENGPAI_DATE}"

echo "latest successful zhengpai:"
echo "date=${LATEST_ZHENGPAI_DATE}"
echo "hdfs_dir=${LATEST_ZHENGPAI_HDFS_DIR}"

# -----------------------------------------------------------------------------
# step6. 将最新成功日的“第一列 adid”缓存到本地
#
# 这里只落第一列，不把所有无关列长期保存到本地。
# awk 的 FS 同时兼容：
#   - tab
#   - 空格
#   - Ctrl-A (\001)
#
# 本地最终文件：
#   data_zhengpai/legal_adids.txt
#   data_zhengpai/_READY_DATE
#   data_zhengpai/_SUCCESS
#   data_zhengpai/source_info.txt
# -----------------------------------------------------------------------------

prepare_legal_adid_whitelist() {
    local local_ready_date=""
    local remote_part_count
    local legal_row_count
    local tmp_legal_file
    local tmp_info_file
    local tmp_ready_file
    local tmp_success_file

    if [ -f "${ZHENGPAI_READY_DATE_FILE}" ]; then
        local_ready_date=$(tr -d '[:space:]' < "${ZHENGPAI_READY_DATE_FILE}")
    fi

    if [ "${local_ready_date}" = "${LATEST_ZHENGPAI_DATE}" ] \
        && [ -s "${LEGAL_ADID_FILE}" ] \
        && [ -f "${ZHENGPAI_LOCAL_SUCCESS_FILE}" ]; then
        echo "reuse local legal adid whitelist:"
        echo "date=${local_ready_date}"
        echo "file=${LEGAL_ADID_FILE}"
        return 0
    fi

    remote_part_count=$(
        {
            "${LOCAL_HADOOP}" fs -ls \
                "${LATEST_ZHENGPAI_HDFS_DIR}/part-*" \
                2>/dev/null || true
        } |
        awk '$1 ~ /^-/ {count++} END {print count + 0}'
    )

    if [ "${remote_part_count}" -le 0 ]; then
        echo "no part-* found in latest successful zhengpai dir:"
        echo "${LATEST_ZHENGPAI_HDFS_DIR}"
        return 1
    fi

    tmp_legal_file="${ZHENGPAI_LOCAL_DIR}/.legal_adids.${LATEST_ZHENGPAI_DATE}.tmp.$$"
    tmp_info_file="${ZHENGPAI_LOCAL_DIR}/.source_info.tmp.$$"
    tmp_ready_file="${ZHENGPAI_LOCAL_DIR}/._READY_DATE.tmp.$$"
    tmp_success_file="${ZHENGPAI_LOCAL_DIR}/._SUCCESS.tmp.$$"

    rm -f \
        "${tmp_legal_file}" \
        "${tmp_info_file}" \
        "${tmp_ready_file}" \
        "${tmp_success_file}"

    echo "============================================================"
    echo "pull legal adids from latest successful zhengpai data"
    echo "date=${LATEST_ZHENGPAI_DATE}"
    echo "remote_parts=${remote_part_count}"
    echo "source=${LATEST_ZHENGPAI_HDFS_DIR}/part-*"
    echo "target=${LEGAL_ADID_FILE}"
    echo "============================================================"

    # 只保留第一列。
    # 使用 hadoop fs -cat 是为了不在本地保留正排数据的其余大字段。
    if ! "${LOCAL_HADOOP}" fs -cat \
        "${LATEST_ZHENGPAI_HDFS_DIR}/part-*" |
        awk '
            BEGIN {
                FS="[ \t\001]+"
            }
            NF > 0 {
                adid=$1
                sub(/\r$/, "", adid)
                if (adid != "") {
                    print adid
                }
            }
        ' > "${tmp_legal_file}"
    then
        echo "failed to extract first-column adids from HDFS."
        rm -f "${tmp_legal_file}"
        return 1
    fi

    if [ ! -s "${tmp_legal_file}" ]; then
        echo "legal adid whitelist is empty, stop push."
        rm -f "${tmp_legal_file}"
        return 1
    fi

    legal_row_count=$(wc -l < "${tmp_legal_file}")
    legal_row_count=$(echo "${legal_row_count}" | tr -d '[:space:]')

    if [ "${legal_row_count}" -le 0 ]; then
        echo "legal adid row count is zero, stop push."
        rm -f "${tmp_legal_file}"
        return 1
    fi

    {
        echo "date=${LATEST_ZHENGPAI_DATE}"
        echo "hdfs_dir=${LATEST_ZHENGPAI_HDFS_DIR}"
        echo "remote_part_count=${remote_part_count}"
        echo "legal_adid_rows=${legal_row_count}"
        echo "host=$(hostname)"
        echo "completed_at=$(TZ=Asia/Shanghai date '+%F %T')"
    } > "${tmp_info_file}"

    printf '%s\n' "${LATEST_ZHENGPAI_DATE}" > "${tmp_ready_file}"
    : > "${tmp_success_file}"

    # _READY_DATE 最后发布；如果前面任意一步失败，下次不会误复用半成品。
    mv -f "${tmp_legal_file}" "${LEGAL_ADID_FILE}"
    mv -f "${tmp_info_file}" "${ZHENGPAI_SOURCE_INFO}"
    mv -f "${tmp_success_file}" "${ZHENGPAI_LOCAL_SUCCESS_FILE}"
    mv -f "${tmp_ready_file}" "${ZHENGPAI_READY_DATE_FILE}"

    echo "legal adid whitelist ready:"
    echo "date=${LATEST_ZHENGPAI_DATE}"
    echo "rows=${legal_row_count}"
    echo "file=${LEGAL_ADID_FILE}"

    return 0
}

if ! prepare_legal_adid_whitelist; then
    echo "prepare legal adid whitelist failed, stop push."
    exit 1
fi

# -----------------------------------------------------------------------------
# step7. 过滤推理结果
#
# 缓存条件：
#   infer_date / legal_date / raw_part_count 三者都一致，且过滤目录中有非空 part，
#   则直接复用，避免 build/push 失败重跑脚本时重复扫描海量结果。
# -----------------------------------------------------------------------------

filter_cache_is_ready() {
    local marker="${FILTERED_INFER_RESULT_DIR}/_FILTER_SUCCESS"

    [ -d "${FILTERED_INFER_RESULT_DIR}" ] || return 1
    [ -f "${marker}" ] || return 1

    grep -Fqx "infer_date=${current_date}" "${marker}" || return 1
    grep -Fqx "legal_date=${LATEST_ZHENGPAI_DATE}" "${marker}" || return 1
    grep -Fqx "raw_part_count=${RAW_NUM_PART}" "${marker}" || return 1

    find "${FILTERED_INFER_RESULT_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'part-[0-9][0-9][0-9][0-9]' \
        -size +0c \
        -print -quit |
    grep -q .
}

run_filter() {
    local tmp_filter_dir
    local old_filter_dir
    local filtered_part_count

    tmp_filter_dir="${FILTERED_INFER_RESULT_DIR}.tmp.$$"
    old_filter_dir="${FILTERED_INFER_RESULT_DIR}.old.$$"

    rm -rf "${tmp_filter_dir}" "${old_filter_dir}"
    mkdir -p "${tmp_filter_dir}"

    echo "============================================================"
    echo "filter inference candidates by legal adid whitelist"
    echo "infer_date=${current_date}"
    echo "legal_date=${LATEST_ZHENGPAI_DATE}"
    echo "raw_dir=${INFER_RESULT_DIR}"
    echo "legal_file=${LEGAL_ADID_FILE}"
    echo "tmp_output=${tmp_filter_dir}"
    echo "============================================================"

    if ! python3 "${FILTER_SCRIPT}" \
        --input-dir "${INFER_RESULT_DIR}" \
        --legal-adids "${LEGAL_ADID_FILE}" \
        --output-dir "${tmp_filter_dir}"
    then
        echo "filter inference result failed."
        rm -rf "${tmp_filter_dir}"
        return 1
    fi

    filtered_part_count=$(
        find "${tmp_filter_dir}" \
            -maxdepth 1 \
            -type f \
            -name 'part-[0-9][0-9][0-9][0-9]' \
            -size +0c |
        wc -l
    )
    filtered_part_count=$(echo "${filtered_part_count}" | tr -d '[:space:]')

    if [ "${filtered_part_count}" -le 0 ]; then
        echo "all inference rows were filtered out; stop push."
        rm -rf "${tmp_filter_dir}"
        return 1
    fi

    {
        echo "infer_date=${current_date}"
        echo "legal_date=${LATEST_ZHENGPAI_DATE}"
        echo "raw_part_count=${RAW_NUM_PART}"
        echo "filtered_part_count=${filtered_part_count}"
        echo "legal_file=${LEGAL_ADID_FILE}"
        echo "completed_at=$(TZ=Asia/Shanghai date '+%F %T')"
    } > "${tmp_filter_dir}/_FILTER_SUCCESS"

    # 尽量原子替换过滤目录；原始 infer_adid_parts 不受影响。
    if [ -d "${FILTERED_INFER_RESULT_DIR}" ]; then
        mv "${FILTERED_INFER_RESULT_DIR}" "${old_filter_dir}"
    fi

    if ! mv "${tmp_filter_dir}" "${FILTERED_INFER_RESULT_DIR}"; then
        echo "failed to publish filtered result dir."
        if [ -d "${old_filter_dir}" ]; then
            mv "${old_filter_dir}" "${FILTERED_INFER_RESULT_DIR}" || true
        fi
        return 1
    fi

    rm -rf "${old_filter_dir}"

    echo "filtered inference result ready:"
    echo "dir=${FILTERED_INFER_RESULT_DIR}"
    echo "part_count=${filtered_part_count}"

    if [ -f "${FILTERED_INFER_RESULT_DIR}/filter_stats.json" ]; then
        cat "${FILTERED_INFER_RESULT_DIR}/filter_stats.json"
    fi

    return 0
}

if filter_cache_is_ready; then
    echo "reuse filtered inference result:"
    echo "infer_date=${current_date}"
    echo "legal_date=${LATEST_ZHENGPAI_DATE}"
    echo "dir=${FILTERED_INFER_RESULT_DIR}"
else
    if ! run_filter; then
        echo "filter step failed, stop push."
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# step8. 自动检测过滤后的所有严格四位编号 part，并逐个检查非空
# -----------------------------------------------------------------------------

declare -a PART_FILES=()
while IFS= read -r part_file; do
    PART_FILES+=("${part_file}")
done < <(
    find "${FILTERED_INFER_RESULT_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'part-[0-9][0-9][0-9][0-9]' \
        -print |
    sort
)

NUM_PART="${#PART_FILES[@]}"

if [ "${NUM_PART}" -le 0 ]; then
    echo "no valid filtered part files found in:"
    echo "${FILTERED_INFER_RESULT_DIR}"
    exit 1
fi

for part_file in "${PART_FILES[@]}"; do
    if [ ! -s "${part_file}" ]; then
        echo "${part_file} is empty.."
        exit 1
    fi
done

FIRST_PART="$(basename "${PART_FILES[0]}")"
LAST_PART="$(basename "${PART_FILES[$((NUM_PART - 1))]}")"

echo "filtered inference result is ready for push:"
echo "infer_date=${current_date}"
echo "legal_date=${LATEST_ZHENGPAI_DATE}"
echo "upload_date=${upload_date}"
echo "result_dir=${FILTERED_INFER_RESULT_DIR}"
echo "part_count=${NUM_PART}"
echo "part_range=${FIRST_PART} ~ ${LAST_PART}"

# -----------------------------------------------------------------------------
# step9. 检查 build/push 配置和入口脚本
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
# step10. 上传过滤后的全部 part 到 HDFS
# -----------------------------------------------------------------------------

if "${LOCAL_HADOOP}" fs -test -e "${HDFS_UPLOAD_PATH}"; then
    echo "HDFS path already exists, delete before upload:"
    echo "${HDFS_UPLOAD_PATH}"

    "${LOCAL_HADOOP}" fs -rm -r -f "${HDFS_UPLOAD_PATH}"
fi

"${LOCAL_HADOOP}" fs -mkdir -p "${HDFS_UPLOAD_PATH}"

"${LOCAL_HADOOP}" fs -put \
    "${PART_FILES[@]}" \
    "${HDFS_UPLOAD_PATH}/"

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

if [ "${hdfs_part_count}" -ne "${NUM_PART}" ]; then
    echo "HDFS part count error:"
    echo "expected=${NUM_PART}"
    echo "actual=${hdfs_part_count}"
    exit 1
fi

echo "upload filtered inference result success:"
echo "HDFS path=${HDFS_UPLOAD_PATH}"
echo "local filtered part count=${NUM_PART}"
echo "HDFS part count=${hdfs_part_count}"

# -----------------------------------------------------------------------------
# step11. build/push 失败重试函数
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
# step12. 调用 build_kv_3shard
# -----------------------------------------------------------------------------

cd "${BUILD_DIR}"

if run_cmd_with_retries 10 600 bash entrance.sh "${upload_date}"; then
    echo "build success..."
else
    echo "build failed"
    exit 1
fi

# -----------------------------------------------------------------------------
# step13. 调用 push_kv_3shard
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
# step14. 原子写入 push 成功标记
# -----------------------------------------------------------------------------

PUSH_SUCCESS_TMP="${PUSH_SUCCESS_FILE}.tmp.$$"

printf "%s\n" "${current_date}" > "${PUSH_SUCCESS_TMP}"

mv -f \
    "${PUSH_SUCCESS_TMP}" \
    "${PUSH_SUCCESS_FILE}"

echo "============================================================"
echo "online push complete"
echo "infer_data_date=${current_date}"
echo "legal_zhengpai_date=${LATEST_ZHENGPAI_DATE}"
echo "upload_date=${upload_date}"
echo "filtered_part_count=${NUM_PART}"
echo "part_range=${FIRST_PART} ~ ${LAST_PART}"
echo "HDFS path=${HDFS_UPLOAD_PATH}"
echo "filtered_result_dir=${FILTERED_INFER_RESULT_DIR}"
echo "legal_adid_file=${LEGAL_ADID_FILE}"
echo "success marker=${PUSH_SUCCESS_FILE}"
echo "log=${LOG_FILE}"
echo "============================================================"

exit 0
