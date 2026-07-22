import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault(
    "VLLM_PLUGINS",
    "openonerec_residual_sid_v085",
)

import json
import time
from multiprocessing import Process, Queue, set_start_method

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig


MODEL_PATH = (
    "/home/jovyan/ceph-1/sujinsong/sujinsong/OpenOneRec-res/pretrain/model_output/sft_full_res_vllm085_b100-20500step"
)

DATA_PATH = (
    "/home/jovyan/ceph-1/zhangguozhu/generative_recommendation/OpenOneRec_data/output/eval/sft/sft_video_rec.parquet"
)

NUM_GPUS = 8
BATCH_SIZE = 128

GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 32768



def content_to_text(content):
    """
    兼容 OpenOneRec parquet 中不同的 content 格式。
    """
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        return ""

    if isinstance(content, list):
        result = []

        for item in content:
            if isinstance(item, str):
                result.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text":
                    result.append(item.get("text", ""))

        return "".join(result)

    raise ValueError(
        f"Unsupported content type: {type(content)}"
    )


def convert_messages(messages):
    """
    保持和原评测代码一致。
    """
    msg_list = []

    for msg in messages:
        msg_list.append(
            {
                "role": msg["role"],
                "content": content_to_text(
                    msg["content"]
                ),
            }
        )

    return msg_list


def load_residual_config(model_path):
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    layer_starts = list(
        config.residual_sid_layer_starts
    )

    layer_sizes = list(
        config.residual_sid_layer_sizes
    )

    sid_begin_token_id = int(
        config.residual_sid_begin_token_id
    )

    beam_size = int(
        getattr(
            config,
            "residual_sid_beam_size",
            0,
        )
    )

    if len(layer_starts) != 4:
        raise ValueError(
            "Expected exactly 4 SID layers, "
            f"got {len(layer_starts)}"
        )

    if beam_size <= 0:
        raise ValueError(
            "residual_sid_beam_size missing "
            "from exported config.json"
        )

    return (
        layer_starts,
        layer_sizes,
        sid_begin_token_id,
        beam_size,
    )


def extract_target_sid(
    answer_text,
    tokenizer,
    layer_starts,
    layer_sizes,
):
    """
    从 parquet 最后一条 assistant answer 中，
    直接提取 A/B/C/D 四层 SID global token id。

    不依赖答案字符串中有没有 sid_begin/sid_end。
    """
    answer_ids = tokenizer.encode(
        answer_text,
        add_special_tokens=False,
    )

    target = []

    for start, size in zip(
        layer_starts,
        layer_sizes,
    ):
        end = start + size

        matched = [
            token_id
            for token_id in answer_ids
            if start <= token_id < end
        ]

        if len(matched) != 1:
            raise ValueError(
                "Cannot uniquely extract SID layer "
                f"[{start}, {end}) from answer.\n"
                f"answer={answer_text}\n"
                f"matched={matched}"
            )

        target.append(matched[0])

    return target


def decode_pooling_output(
    output,
    beam_size,
    num_sid_layers,
):
    """
    Pooler 输出：

    beam × [
        sid_a,
        sid_b,
        sid_c,
        sid_d,
        score,
    ]
    """
    data = output.outputs.data

    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    else:
        data = np.asarray(data)

    expected_width = num_sid_layers + 1

    if data.ndim == 1:
        expected_numel = (
            beam_size * expected_width
        )

        if data.size != expected_numel:
            raise ValueError(
                "Unexpected residual pooling output "
                f"shape={data.shape}, "
                f"numel={data.size}, "
                f"expected={expected_numel}"
            )

        data = data.reshape(
            beam_size,
            expected_width,
        )

    elif data.ndim == 2:
        if data.shape != (
            beam_size,
            expected_width,
        ):
            raise ValueError(
                "Unexpected residual pooling output "
                f"shape={data.shape}"
            )

    else:
        raise ValueError(
            "Unexpected residual pooling output "
            f"ndim={data.ndim}, "
            f"shape={data.shape}"
        )

    # 前四列是 global SID token IDs
    candidate_ids = np.rint(
        data[:, :num_sid_layers]
    ).astype(np.int64)

    # 最后一列是累计 score
    scores = data[:, num_sid_layers]

    return candidate_ids, scores


def run_batch(
    llm,
    prompt_list,
    answer_list,
    beam_size,
):
    """
    residual 模型的核心推理。

    注意这里不再使用：
        llm.beam_search()

    而是：
        llm.encode()

    Transformer 只执行 prompt prefill。
    """
    outputs = llm.encode(
        prompt_list,
        use_tqdm=False,
    )

    hit = 0
    cnt = len(outputs)

    for output, target in zip(
        outputs,
        answer_list,
    ):
        candidate_ids, _ = (
            decode_pooling_output(
                output=output,
                beam_size=beam_size,
                num_sid_layers=4,
            )
        )

        target_array = np.asarray(
            target,
            dtype=np.int64,
        )

        matched = np.all(
            candidate_ids == target_array[None, :],
            axis=1,
        )

        if matched.any():
            hit += 1

    return hit, cnt


def evaluate_worker(
    rank,
    world_size,
    result_queue,
):
    try:
        # 必须在 import vllm / 创建 LLM 前设置
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            rank
        )

        # 每个进程独占一张 GPU，
        # 与你原来的评测代码完全一致。
        from vllm import LLM

        print(
            f"[Rank {rank}] "
            f"CUDA_VISIBLE_DEVICES={rank}"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )

        (
            layer_starts,
            layer_sizes,
            sid_begin_token_id,
            beam_size,
        ) = load_residual_config(
            MODEL_PATH
        )

        print(
            f"[Rank {rank}] "
            f"beam_size={beam_size}"
        )

        print(
            f"[Rank {rank}] "
            f"SID starts={layer_starts}"
        )

        print(
            f"[Rank {rank}] "
            "Loading residual vLLM model..."
        )

        llm = LLM(
            model=MODEL_PATH,
            task="embed",
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
        )

        df = pd.read_parquet(
            DATA_PATH
        )

        # 与你的原始代码完全一致的数据切分
        df = df.iloc[
            rank::world_size
        ].reset_index(drop=True)

        print(
            f"[Rank {rank}] "
            f"Assigned {len(df)} samples"
        )

        hit = 0
        cnt = 0

        prompt_list = []
        answer_list = []

        start_time = time.perf_counter()

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"GPU-{rank}",
        ):
            messages_all = json.loads(
                row["messages"]
            )

            # 最后一条是 ground truth answer
            messages = messages_all[:-1]

            answer_text = content_to_text(
                messages_all[-1]["content"]
            )

            messages = convert_messages(
                messages
            )

            # 直接生成 token IDs，
            # 不再先生成字符串再计算 len。
            prompt_token_ids = (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

            prompt_token_ids = list(
                prompt_token_ids
            )

            # residual 模型的 anchor hidden
            # 是 <|sid_begin|> 位置的 hidden state。
            #
            # 所以输入必须在 sid_begin 结束。
            if (
                len(prompt_token_ids) == 0
                or
                prompt_token_ids[-1]
                != sid_begin_token_id
            ):
                prompt_token_ids.append(
                    sid_begin_token_id
                )

            # vLLM 0.8.5 TokensPrompt
            prompt_list.append(
                {
                    "prompt_token_ids":
                        prompt_token_ids
                }
            )

            target_sid = extract_target_sid(
                answer_text=answer_text,
                tokenizer=tokenizer,
                layer_starts=layer_starts,
                layer_sizes=layer_sizes,
            )

            answer_list.append(
                target_sid
            )

            if (
                len(prompt_list)
                >= BATCH_SIZE
            ):
                batch_hit, batch_cnt = (
                    run_batch(
                        llm=llm,
                        prompt_list=prompt_list,
                        answer_list=answer_list,
                        beam_size=beam_size,
                    )
                )

                hit += batch_hit
                cnt += batch_cnt

                prompt_list = []
                answer_list = []

        # flush 最后一批
        if prompt_list:
            batch_hit, batch_cnt = (
                run_batch(
                    llm=llm,
                    prompt_list=prompt_list,
                    answer_list=answer_list,
                    beam_size=beam_size,
                )
            )

            hit += batch_hit
            cnt += batch_cnt

        elapsed = (
            time.perf_counter()
            - start_time
        )

        result_queue.put(
            {
                "rank": rank,
                "hit": hit,
                "cnt": cnt,
                "elapsed": elapsed,
            }
        )

        print(
            f"[Rank {rank}] "
            f"Hit={hit}, "
            f"Cnt={cnt}, "
            f"Ratio={hit / cnt:.6f}, "
            f"Time={elapsed:.2f}s"
        )

    except Exception as exc:
        import traceback

        traceback.print_exc()

        result_queue.put(
            {
                "rank": rank,
                "error": repr(exc),
            }
        )


def main():
    set_start_method(
        "spawn",
        force=True,
    )

    result_queue = Queue()
    processes = []

    for rank in range(NUM_GPUS):
        process = Process(
            target=evaluate_worker,
            args=(
                rank,
                NUM_GPUS,
                result_queue,
            ),
        )

        process.start()
        processes.append(
            process
        )

    total_hit = 0
    total_cnt = 0
    worker_times = []

    for _ in range(NUM_GPUS):
        result = result_queue.get()

        if "error" in result:
            raise RuntimeError(
                f"Rank {result['rank']} failed: "
                f"{result['error']}"
            )

        print(
            f"Receive rank "
            f"{result['rank']} result: "
            f"{result['hit']}/"
            f"{result['cnt']}"
        )

        total_hit += result["hit"]
        total_cnt += result["cnt"]

        worker_times.append(
            result["elapsed"]
        )

    for process in processes:
        process.join()

    wall_time = max(
        worker_times
    )

    print(
        "=" * 60
    )

    print(
        "FINAL RESULT"
    )

    print(
        f"Hit Ratio = "
        f"{total_hit}/{total_cnt} "
        f"= "
        f"{total_hit / total_cnt:.6f}"
    )

    print(
        f"Wall Time = "
        f"{wall_time:.2f} s"
    )

    print(
        f"Throughput = "
        f"{total_cnt / wall_time:.2f} "
        f"samples/s"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
