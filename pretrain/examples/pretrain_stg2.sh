sed 's/=1/=8/g' /etc/mpi/hostfile > /etc/mpi/hostfile_seq

# MODEL_DIR=/code/hf_models/Qwen3-1.7B_itemic
STAGE1_OUTPUT_DIR=/code/onerec_pretrain/model_output/stg1_opt_utils_big
MODEL_DIR=${STAGE1_OUTPUT_DIR}/step2000/global_step2000/converted
OUTPUT_DIR=/code/onerec_pretrain/model_output/stg2_opt_utils_big


mkdir -p $OUTPUT_DIR
mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

set -x

SCRIPT_FILE=$(readlink -f $0)
echo `date '+%Y-%m-%d %H:%M:%S'` >> $OUTPUT_DIR/task_info.log
echo "script: ${SCRIPT_FILE}" >> $OUTPUT_DIR/task_info.log
echo "=========================" >> $OUTPUT_DIR/task_info.log

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=$PWD:$PYTHONPATH

source set_env.sh

hostfile=/etc/mpi/hostfile_seq
TCP_NIC=$(ifconfig | grep -B1 " "$(hostname -i)" " | grep -o "^\w*")

MASTER_ADDR=$MY_NODE_IP
MASTER_PORT=8499

mpirun --allow-run-as-root \
    -hostfile $hostfile \
    -mca btl self,tcp -mca pml ob1 \
    -mca plm_rsh_num_concurrent 600 \
    -mca routed_radix 600 \
    -mca btl_tcp_if_include $TCP_NIC \
    -mca oob_tcp_if_include $TCP_NIC \
    -mca btl_openib_allow_ib false \
    -mca opal_set_max_sys_limits 1 \
    -x OMPI_MCA_btl=self,tcp \
    -x OMPI_MCA_pml=ob1 \
    -x OMPI_MCA_btl_tcp_if_include=$TCP_NIC \
    -x OMPI_MCA_oob_tcp_if_include=$TCP_NIC \
    -x OMPI_MCA_btl_openib_allow_ib=false \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_SOCKET_IFNAME=$TCP_NIC \
    -x NCCL_IB_HCA=mlx5 \
    -x NCCL_DEBUG=WARN \
    -x NCCL_IB_QPS_PER_CONNECTION=4 \
    -x NCCL_NET_OVERHEAD=1000 \
    -x NCCL_IB_TIMEOUT=20 \
    -x LD_PRELOAD=$LD_PRELOAD \
    -x http_proxy="" \
    -x https_proxy="" \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
    -x MPI_THREAD_SINGLE=1 \
    -x NO_COLOR=1 \
    -x TERM=dumb \
    -x COLORTERM=0 \
    -x PYTHONIOENCODING=utf-8 \
    -x LD_LIBRARY_PATH=$LIBRARY_PATH \
    -x PATH \
    -x PYTHONPATH=$PYTHONPATH \
    -x JAVA_HOME=$JAVA_HOME \
    -x HIVE_HOME=$HIVE_HOME \
    -x CLASSPATH=$CLASSPATH \
    -x HADOOP_USER_NAME=$HADOOP_USER_NAME \
    -x HADOOP_HOME=$HADOOP_HOME \
    -x SPARK_HOME=$SPARK_HOME \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x TOKENIZERS_PARALLELISM=false \
    with_nccl_local_env \
    bash -c "bash scripts/numa_runner.sh python3 recipes/train_qwen3.py \
        --model_dir $MODEL_DIR \
        --output_dir $OUTPUT_DIR \
        --dataset_config examples/dataset_config/pretrain.json \
        --use_tie_weights \
        --model_class Qwen3ForCausalLM \
        --monitor_datasource_loss \
        --monitor_datasource_cnt \
        --max_length 32768 \
        --learning_rate 2e-4 \
        --min_lr 1e-4 \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 500 \
        --num_training_steps 5000 \
        --save_checkpoint_per_step 50 \
        --minibatch_size 16384 \
        --logging_per_step 5 \
        --use_fp32_weight \
        --seed 19260817 \
        --enable_profiler \
        --enable_gradient_checkpointing \
        --use_chunked_loss_computer \
    " > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &