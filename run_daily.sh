#!/bin/bash

set -ex

#激活虚拟环境  这是模型更新脚本
source /home/jovyan/ceph-1/sujinsong/env/onerec/bin/activate


if [ -f "./_RUNNING" ]; then
    echo "_RUNNING 存在"
    exit 1
else
    touch ./_RUNNING
fi


function process_data()
{
    
    max_attempts=30
    attempt=1
    check_interval=600  # 10分钟，单位秒
    success=false
    
    while [ $attempt -le $max_attempts ]; do
        echo "第 ${attempt}/${max_attempts} 次检测..."
        if [ -f "/home/jovyan/zhouyuhang-cloud1/DSP/rqvae_decoder_base_pipeline/data/train_ad_daily_sample/DATA_SUCCESS" ]; then
            echo "今日数据ready"
            success=true
            break
        fi
        attempt=$((attempt + 1))
        sleep ${check_interval}
    done
    
    if [ "$success" = false ]; then
        echo "今日数据未ready"
        exit 1
    fi
    
    #准备数据
    cd /home/jovyan/ceph-1/sujinsong/online/openonerec-res

    python process_train_data_daily.py

    #构造训练样本
    cd /home/jovyan/ceph-1/sujinsong/online/openonerec-res/data/onerec_data

    bash run_daily.sh

    #split数据
    cd ..

    bash prepare_sft_daily.sh

}


function train_model()
{
    
    #训练模型
    cd /home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain

    bash examples/posttrain_sft_residual_sid_4layer.sh

}


function update_model()
{
    #转换模型参数
    cd /home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain
    
    bash scripts/convert_residual_checkpoint_to_hf.sh    /home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/model_output/stg2_residual_add_feature/step22000/global_step22000/converted  /home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/model_output/sft_full_residual_add_feature_daily 22000


    #模型文件更新
    rm -rf /home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/model_output/stg2_residual_add_feature/step22000/global_step22000/converted
    
    cp -r /home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/model_output/sft_full_residual_add_feature_daily/step22000/global_step22000/converted  /home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/model_output/stg2_residual_add_feature/step22000/global_step22000/converted
    
    rm -rf /home/jovyan/ceph-1/sujinsong/online/openonerec-res/pretrain/model_output/sft_full_residual_add_feature_daily

}


function main()
{

    process_data
    
    train_model
    
    update_model

}


main

cd /home/jovyan/ceph-1/sujinsong/online/openonerec-res
rm ./_RUNNING
