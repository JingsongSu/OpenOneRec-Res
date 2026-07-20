"""
Base Loader for all task data loaders

Provides common functionality for data loading, sampling, and file path resolution.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional
from abc import ABC

from benchmark.console import *


class BaseLoader(ABC):
    """Base class for all task data loaders"""
    
    def __init__(
        self,
        task_config: Dict[str, Any],
        data_dir: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        enable_thinking: Optional[bool] = None,
    ):
        """Initialize base loader"""
        self.task_config = task_config
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.task_name = task_config.get("name", "unknown")

        # Validate tokenizer is provided for messages-based format
        if self.tokenizer is None:
            raise ValueError(
                f"{self.task_name} requires tokenizer for messages-based format. "
                f"Please provide model_path when initializing Benchmark.\n"
                f"Example: Benchmark(task_types=['{self.task_name}'], model_path='your-model-path')"
            )

    def load_data(self, split: str = "test", sample_size: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        Load data for the task in messages-based format

        Args:
            split: Dataset split (default "test")
            sample_size: Override sample size (can be int, "full", or None to use task config)

        Returns:
            Dictionary mapping sample_id to sample data:
            {
                sample_id: {
                    "prompt": "formatted prompt from apply_chat_template",
                    "ground_truth": "answer",
                    "metadata": {
                        "row_index": idx,
                        "messages": [...]
                    }
                }
            }
        """
        # Determine effective sample size
        if sample_size is not None:
            if sample_size == "full":
                effective_sample_size = self.task_config.get("size")
            else:
                effective_sample_size = int(sample_size)
        else:
            effective_sample_size = self.task_config.get("sample_size")
        
        full_size = self.task_config.get("size")

        # Try to load cached sample dataframe
        df = None
        if effective_sample_size is not None and full_size is not None and effective_sample_size < full_size:
            df = self._load_sample_dataframe(split, effective_sample_size)

        # If no cache, load and sample original data
        if df is None:
            df = self._load_dataframe(split)

            # Perform sampling if needed
            if effective_sample_size is not None and effective_sample_size < len(df):
                df = self._sample_data(df, effective_sample_size)

                # Save sampled data
                if full_size is not None and effective_sample_size < full_size:
                    self._save_sample_data(df, split, effective_sample_size)

        if 'messages' not in df.columns:
            raise ValueError(
                f"{self.task_name} requires 'messages' column in data file. "
                f"Found columns: {list(df.columns)}\n"
                f"Please ensure your data is in messages-based format."
            )

        if 'metadata' not in df.columns:
            raise ValueError(
                f"{self.task_name} requires 'metadata' column in data file. "
                f"Found columns: {list(df.columns)}\n"
                f"Please ensure your data is in messages-based format."
            )

        console.print(f"[green]Processing {self.task_name} data in messages-based format[/green]")

        result = self._process_dataframe(df)

        return result
    
    @staticmethod
    def _is_empty_value(value) -> bool:
        """Check if a value is None, NaN, or empty"""
        if value is None:
            return True
        
        if isinstance(value, float):
            try:
                return pd.isna(value)
            except (ValueError, TypeError):
                return False
        
        if isinstance(value, str):
            return len(value.strip()) == 0
        
        try:
            if hasattr(value, '__len__'):
                return len(value) == 0
        except (ValueError, TypeError):
            pass
        
        return False

    @staticmethod
    def _convert_messages_format(messages: list) -> list:
        """
        Convert message format.

        {"role": "user", "content": [{"type": "text", "text": "..."}]} 
        -> 
        {"role": "user", "content": "..."}
        """
        converted = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                # Extract text from content list
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                converted.append({
                    "role": msg.get("role"),
                    "content": "".join(text_parts)
                })
            else:
                # Already in old format
                converted.append(msg)
        return converted

    def _load_custom_chat_template(self):
        """Load custom chat template based on configuration"""
        if not self.tokenizer:
            return

        prompt_config = self.task_config.get("prompt_config", {})
        custom_template = prompt_config.get("custom_chat_template")

        template_path = os.path.join(
            os.path.dirname(__file__),
            custom_template
        )

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"✗ Custom chat template not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            self.tokenizer.chat_template = f.read()
        console.print(f"✓ Loaded custom chat template: {custom_template}", style=success_style)

    def _get_data_file_path(self, split: str) -> str:
        """Get data file path for the given split"""
        if self.data_dir:
            base_dir = self.data_dir
        else:
            base_dir = "./data"
        
        filename = f"{self.task_name}_{split}.parquet"
        
        possible_paths = [
            os.path.join(base_dir, self.task_name, filename),
        ]
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                return file_path
        
        return possible_paths[0]
    
    def _get_sample_data_file_path(self, split: str, sample_size: int) -> str:
        """Get sample data file path"""
        if self.data_dir:
            base_dir = self.data_dir
        else:
            base_dir = "./data"
        
        possible_paths = [
            os.path.join(base_dir, self.task_name, f"{self.task_name}_{split}_sample_{sample_size}.parquet"),
            os.path.join(base_dir, f"{self.task_name}_{split}_sample_{sample_size}.parquet"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return possible_paths[0]
    
    def _load_dataframe(self, split: str) -> pd.DataFrame:
        """Load DataFrame from data file"""
        data_file = self._get_data_file_path(split)
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        console.print(f"Loading data file: {data_file}")
        
        if data_file.endswith('.parquet'):
            df = pd.read_parquet(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
        
        return df
    
    def _sample_data(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Sample data from DataFrame"""
        if sample_size >= len(df):
            return df
        
        console.print(f"Sampling {sample_size} samples (total: {len(df)})")
        return df.head(sample_size)
    
    def _save_sample_data(
        self,
        df: pd.DataFrame,
        split: str,
        sample_size: int
    ):
        """Save sample data in parquet format"""
        sample_file = self._get_sample_data_file_path(split, sample_size)

        sample_dir = os.path.dirname(sample_file)
        if sample_dir:
            os.makedirs(sample_dir, exist_ok=True)

        df.to_parquet(sample_file, index=False)
        console.print(f"Sample data saved to: {sample_file}")
    
    def _load_sample_dataframe(self, split: str, sample_size: int) -> Optional[pd.DataFrame]:
        """Load sample dataframe from cache if exists"""
        sample_file = self._get_sample_data_file_path(split, sample_size)

        if not os.path.exists(sample_file):
            return None

        console.print(f"Loading sample data from cache: {sample_file}")

        df = pd.read_parquet(sample_file)
        return df

    def _process_dataframe(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Process DataFrame and convert to model input format"""
        self._load_custom_chat_template()

        result = {}

        prompt_config = self.task_config.get("prompt_config", {})
        # Command-line parameter has higher priority than config
        if self.enable_thinking is not None:
            enable_thinking = self.enable_thinking
        else:
            enable_thinking = prompt_config.get("enable_thinking", False)

        console.print(f"[cyan]Auto Thinking: {'✓ Enabled' if enable_thinking else '✗ Disabled'}[/cyan]")

        for idx, row in df.iterrows():
            sample_id = str(idx)

            messages = row.get('messages')
            if self._is_empty_value(messages):
                console.print(f"Sample {sample_id}: messages is empty, skipping")
                continue

            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except Exception:
                    console.print(f"Sample {sample_id}: failed to parse messages, skipping")
                    continue

            messages = self._convert_messages_format(messages)

            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except Exception as e:
                console.print(f"Sample {sample_id}: failed to apply chat template: {e}, skipping")
                continue

            metadata_raw = row.get('metadata')
            if self._is_empty_value(metadata_raw):
                console.print(f"Sample {sample_id}: metadata is empty, skipping")
                continue

            if isinstance(metadata_raw, str):
                try:
                    metadata_dict = json.loads(metadata_raw)
                except Exception:
                    console.print(f"Sample {sample_id}: failed to parse metadata, skipping")
                    continue
            elif isinstance(metadata_raw, dict):
                metadata_dict = metadata_raw
            else:
                console.print(f"Sample {sample_id}: invalid metadata format, skipping")
                continue

            answer = metadata_dict.get('answer')
            if self._is_empty_value(answer):
                console.print(f"Sample {sample_id}: answer is empty in metadata, skipping")
                continue

            ground_truth_str = str(answer).strip()

            result_item = {
                "prompt": formatted_prompt,
                "ground_truth": ground_truth_str,
                "metadata": self._make_metadata_serializable(idx, metadata_dict)
            }

            result[sample_id] = result_item

        console.print(f"[green]Loaded {len(result)} samples for {self.task_name}[/green]")

        return result

    def _make_metadata_serializable(
        self,
        idx: Any,
        metadata_dict: dict,
    ) -> dict:
        """Convert metadata to JSON-serializable format"""
        del metadata_dict["answer"]

        metadata = {
            "row_index": int(idx) if hasattr(idx, '__int__') else str(idx),
            **metadata_dict,
        }


        return metadata

