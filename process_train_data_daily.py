import glob
import os
import pandas as pd


############################################
# 产出 adid2sid.parquet 文件
# 加载 adid 码本字典
semid_path = (
    "/home/jovyan/ceph-1/zhouyuhang/data/onerec_data/"
    "search_join_dsp_tag_ad_hash_semid.txt.v3"
)

adid_sid_output_path = (
    "/home/jovyan/ceph-1/zhouyuhang/data/onerec_data/"
    "adid2sid.parquet"
)

# 将 ad 索引转成 adid
adidx2adid = {}

adid_sid_results = []

with open(semid_path, "r") as f:
    for line in f:
        cols = line.strip().split("\t")
        adidx2adid[cols[0]] = cols[1]


############################################
# MID SID 文件已经生成，直接使用现有 Parquet
mid_output_path = (
    "/home/jovyan/ceph-1/sujinsong/online/"
    "openonerec-res-v2/raw_data/onerec_data/"
    "mid2sid.parquet"
)

if not os.path.exists(mid_output_path):
    raise FileNotFoundError(
        f"mid2sid parquet 文件不存在: {mid_output_path}"
    )

print(f"Use existing MID SID file: {mid_output_path}")


#######################################
# 产出 user_behavior_sequence.parquet 用户序列文件
# 获取指定文件夹下所有 part-* 文件
trian_data_path = (
    "/home/jovyan/zhouyuhang-cloud1/DSP/"
    "rqvae_decoder_base_pipeline/data/"
    "train_ad_daily_sample"
)

output_path = (
    "/home/jovyan/ceph-1/sujinsong/online/"
    "openonerec-res-v2/raw_data/onerec_data/"
    "user_behavior_sequence_daily.parquet"
)

part_files = glob.glob(
    os.path.join(trian_data_path, "part-*")
)

print(f"找到 {len(part_files)} 个 part 文件:")

results = []

for file in part_files:
    print(file)

    with open(f"{file}", "r") as f:
        for line in f:
            d = {}
            row = line.strip().split("\t")

            d["mid"] = row[0]

            adid_list = [
                adidx2adid[i]
                for i in row[2].split(",")
            ]

            d["hist_adid"] = adid_list[:-1]
            d["target_adid"] = adid_list[-1:]

            # 用户序列由搜索 query 和 ad 组成，最后一个一定是 ad
            mask_list = [
                int(i)
                for i in row[3].split(",")
            ]

            d["hist_adid_mask"] = mask_list[:-1]

            # 时间信息
            time_list = [
                int(i)
                for i in row[4].split(",")
            ]

            d["hist_adid_time"] = time_list[:-1]
            d["target_adid_time"] = time_list[-1:]

            ctypeid_list = [
                int(i)
                for i in row[5].split(",")
            ]

            d["hist_ctypeid"] = ctypeid_list[:-1]
            d["target_ctypeid"] = ctypeid_list[-1:]

            results.append(d)

df_train = pd.DataFrame(results)

df_train.to_parquet(
    output_path,
    index=False,
)

print(
    f"Saved: {output_path} "
    f"({len(df_train):,} rows)"
)

print("Done!")
