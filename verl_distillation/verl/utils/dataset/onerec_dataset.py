import ast
import copy
import logging
import os
import random
import re
from typing import Any, Optional

import datasets
import numpy as np
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

class OneRecDataset(Dataset):
    """Onerec数据集读取与预处理。

    - 缓存Parquet文件到本地；
    - 利用HF Dataset读取并转换chat结构；
    - 根据配置过滤超长prompt；
    - 支持多模态预处理与位置编码。
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ) -> None:
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.enable_think = config.get("enable_think", True)
        self.think_mode = config.get("think_mode", "force_think")
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)

        auto_workers = max(1, (os.cpu_count() or 4) // 4)
        self.num_workers = min(
            config.get("filter_overlong_prompts_workers", auto_workers),
            os.cpu_count() or auto_workers,
        )
        self.use_shm = config.get("use_shm", False)
        self.serialize_dataset = False

        #self._download()
        self._read_files_and_tokenize()

    # ---------------------------------------------------------------------
    # 数据准备
    # ---------------------------------------------------------------------
    def _download(self, use_origin_parquet: bool = False) -> None:
        from verl.utils.fs import copy_to_local

        target_files = self.original_data_files if use_origin_parquet else self.data_files
        for idx, parquet_file in enumerate(target_files):
            local_path = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
            target_files[idx] = local_path

        if use_origin_parquet:
            self.data_files = target_files

    def _read_files_and_tokenize(self) -> None:
        #dataframes: list[datasets.Dataset] = []
        self.dataframe = datasets.load_dataset("parquet", data_files=self.data_files)["train"]
        #for parquet_file in self.data_files:
        #    dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
        #    dataframes.append(dataframe)

        #self.dataframe = datasets.concatenate_datasets(dataframes)  # type: ignore[attr-defined]
        logger.info("dataset len: %s", len(self.dataframe))

        if self.max_samples > 0 and self.max_samples < len(self.dataframe):
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(len(self.dataframe), size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {len(self.dataframe)}")

        self.dataframe = self.dataframe.map(
            self._extract_prompt_fields,
            num_proc=self.num_workers,
            desc="Extract prompts and reward annotations",
        )

        # 过滤掉处理失败的样本
        original_len = len(self.dataframe)
        self.dataframe = self.dataframe.filter(
            self._is_valid_sample,
            num_proc=self.num_workers,
            desc="Filtering out failed samples",
        )
        filtered_len = len(self.dataframe)
        logger.info("Filtered out %s failed samples, remaining: %s", original_len - filtered_len, filtered_len)

        logger.info("processed dataset len: %s", len(self.dataframe))
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _extract_prompt_fields(self, row: dict[str, Any]) -> dict[str, Any]:
        try:
            raw_messages = row.get("messages")
            if isinstance(raw_messages, str):
                messages = ast.literal_eval(raw_messages)
            else:
                messages = raw_messages or []
            
            # 多轮对话清洗成单轮对话
            user_cnt = 0
            assistant_cnt = 0
            clean_chats = []

            for msg in messages:
                if user_cnt > 0 and assistant_cnt > 0:
                    break
                role = msg.get("role", None)
                content = msg.get("content", None)
                if role is None or content is None:
                    raise ValueError("role or content is None!")
                content_text = ""
                if isinstance(content, str):
                    content_text = content
                elif isinstance(content, dict) and content.get("type") == "text":
                    content_text = content["text"]
                elif isinstance(content, list):
                    for seg in content:
                        if isinstance(seg, str):
                            content_text += seg
                        elif isinstance(seg, dict) and seg.get("type") == "text":
                            content_text += seg.get("text", "")
                        
                if role == "user" and content_text.strip() == "":
                    raise ValueError("content is empty!")
                
                # # drop system prompt
                # if role == "system":
                #     if "<think></think>" in content_text or "<answer></answer>" in content_text:
                #         continue
                    
                clean_chats.append({
                    "role": role,
                    "content": content_text
                })
                if role == "user":
                    user_cnt += 1
                
                if role == "assistant":
                    assistant_cnt += 1

            if not clean_chats or len(clean_chats) < 2:
                raise ValueError("Sample has empty messages; please check data integrity.")

            prompt_messages = clean_chats[:-1]

            # 根据配置决定是否给 user 消息添加 /think /no_think 指令
            if self.enable_think:
                think_suffix = ""
                if self.think_mode == "force_think":
                    think_suffix = " /think"
                elif self.think_mode == "force_nothink":
                    think_suffix = " /no_think"
                elif self.think_mode == "auto":
                    tm_idx = random.randint(0, 2)
                    think_suffix = " /think" if tm_idx == 1 else " /no_think" if tm_idx == 2 else ""
                else:
                    raise ValueError("think_mode is unexcept")

                for message in prompt_messages:
                    if message["role"] == "user":
                        message["content"] = message["content"] + think_suffix

            ground_truth_message = clean_chats[-1]["content"]

            reward_payload = {
                "ground_truth": ground_truth_message,
                "style": "rule",
            }

            row[self.prompt_key] = prompt_messages
            row["reward_model"] = reward_payload
            return row
        except Exception as e:
            # 标记处理失败的样本
            row["_processing_failed"] = True
            row["_processing_error"] = str(e)
            return row

    def _is_valid_sample(self, row: dict[str, Any]) -> bool:
        """检查样本是否处理成功"""
        return not row.get("_processing_failed", False)

    # ---------------------------------------------------------------------
    # 过滤与恢复
    # ---------------------------------------------------------------------
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key

        if processor is not None:
            from verl.utils.dataset.vision_utils import (process_image,
                                                         process_video)

            def doc_length(doc: dict[str, Any]) -> int:
                messages = self._build_messages(dict(doc))
                raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                images = [process_image(image) for image in doc.get(image_key, [])]
                videos = [process_video(video) for video in doc.get(video_key, [])]
                encoded = processor(text=[raw_prompt], images=images or None, videos=videos or None, return_tensors="pt")
                return int(encoded["input_ids"].shape[-1])

        else:

            def doc_length(doc: dict[str, Any]) -> int:
                messages = doc[prompt_key]
                return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))

        filtered = dataframe.filter(
            lambda doc: doc_length(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        # 获取data_source字段值为"distill"和"sft"的indices
        if "data_source" in filtered.features:
            self.distill_indices = [i for i, doc in enumerate(filtered) if doc.get("data_source") == "distill"]
            self.sft_indices = [i for i, doc in enumerate(filtered) if doc.get("data_source") == "sft"]
            logger.info(f"distill samples: {len(self.distill_indices)}, sft samples: {len(self.sft_indices)}")
        else:
            logger.warning("data_source field not found in filtered dataset")

        logger.info("filtered dataset len: %s", len(filtered))
        return filtered

    def resume_dataset_state(self) -> None:
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            logger.warning("resume with serialized dataloader, consider restarting from scratch for better perf")

    # ---------------------------------------------------------------------
    # Dataset 接口
    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataframe)

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                segments = [segment for segment in re.split(r"(<image>|<video>)", content) if segment]
                parsed_segments = []
                for segment in segments:
                    if segment == "<image>":
                        parsed_segments.append({"type": "image"})
                    elif segment == "<video>":
                        parsed_segments.append({"type": "video"})
                    else:
                        parsed_segments.append({"type": "text", "text": segment})
                message["content"] = parsed_segments

        return messages

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        row: dict[str, Any] = dict(self.dataframe[index])
        messages = self._build_messages(dict(row))
        model_inputs: dict[str, Any] = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import (process_image,
                                                         process_video)

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data: dict[str, Any] = {}

            images = None
            if self.image_key in row and row.get(self.image_key):
                images = [process_image(image) for image in row.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row and row.get(self.video_key):
                videos = [process_video(video) for video in row.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                return_tensors="pt",
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            row["multi_modal_data"] = multi_modal_data
            if self.return_multi_modal_inputs:
                mm_inputs = dict(model_inputs)
                mm_inputs.pop("second_per_grid_ts", None)
                row["multi_modal_inputs"] = mm_inputs
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row["input_ids"] = input_ids[0]
        row["attention_mask"] = attention_mask[0]
        row["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = self._truncate_ids(raw_prompt_ids)

        row["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row["raw_prompt"] = messages
        if self.return_full_prompt:
            row["full_prompts"] = raw_prompt

        extra_info = row.get("extra_info", {}) or {}
        row["index"] = extra_info.get("index", index)
        row["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})

        # 确保 data_source 或 source 字段被保留（用于按task统计）
        # 原始 parquet 数据中应该包含 source 或 data_source 字段
        # 如果都不存在，设置一个默认值
        if "source" in row or "data_source" in row:
            # 字段已存在，无需处理（会自动被 collate_fn 收集）
            pass
        else:
            # 如果两个字段都不存在，设置一个默认值
            row["data_source"] = "unknown"
            logger.warning("No source/data_source field found for index %s, set to 'unknown'", row["index"])

        if self.need_tools_kwargs and not row["tools_kwargs"]:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", row["index"], row.get("data_source", row.get("source", "unknown")))

        return row

    def _truncate_ids(self, token_ids: list[int]) -> list[int]:
        if self.truncation == "left":
            return token_ids[-self.max_prompt_length :]
        if self.truncation == "right":
            return token_ids[: self.max_prompt_length]
        if self.truncation == "middle":
            left = self.max_prompt_length // 2
            right = self.max_prompt_length - left
            return token_ids[:left] + token_ids[-right:]
        if self.truncation == "error":
            raise RuntimeError(
                f"Prompt length {len(token_ids)} exceeds max_prompt_length={self.max_prompt_length}. "
                "Consider increasingmax_prompt_length or enabling truncation."
            )
        raise ValueError(f"Unsupported truncation mode: {self.truncation}")

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()