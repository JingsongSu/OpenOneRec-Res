import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from benchmark.console import console


WIP_EXTRACTION_PROMPT = """你是一位顶级的【信息抽取专家】，擅长从非结构化的文本中解析出结构化的信息。

### 你的核心任务
你的任务是分析我提供的描述性文字，并将其分解为结构化的【原子化且唯一】的"信息点"列表。

### 输出结构
对于列表中的每一个信息点，你必须提供：
1.  **info_point**: 一个简洁的、陈述事实的短语。
2.  **importance_score**: 一个 [1, 5] 之间的【整数】，代表该信息点的重要性。

---
### 关键原则 (必须遵守)

1.  **原子性 (Atomic):** 每个 `info_point` 应只包含一个独立的事实。
    * (好): `{{"info_point": "女孩在吃饭", "importance_score": 4}}`
    * (差): `{{"info_point": "女孩在吃饭，妈妈在旁边看", "importance_score": 4}}`
2.  **唯一性 (Unique):** 确保你提取的每个 `info_point` 都是**概念上唯一**的。
3.  **合并 (Consolidate):** 如果原始文本中的多个短语描述的是【同一个核心思想】，你【必须】将它们合并成一个单一的、最具代表性的 `info_point`。
    * (例如): 如果文本说 "活动环境是温馨的" 和 "视频色彩营造温馨氛围"，你应该只提取一个，如：`{{"info_point": "视频氛围温馨", "importance_score": 5}}`。
    * **不要创建重复或语义高度重叠的条目。**

---
### 打分指南 (1-5分制)

* **5分 (绝对核心):** 视频的"灵魂"。如果缺少这个点，整个摘要就毫无意义。（例如："如何制作煎蛋卷"、"XX游戏的评测"）
* **4分 (关键信息):** 视频的"骨架"。关键的事件、步骤或场景。（例如："打散三个鸡蛋"、"使用了不粘锅"、"游戏画面评测"）
* **3分 (重要细节):** 视频的"肉"。支撑骨架的具体、重要的细节。（例如："加入了盐和胡椒"、"用中火加热黄油"、"角色动作流畅"）
* **2分 (补充细节):** 补充性的上下文或次要信息。（例如："煎蛋卷折叠了三次"、"背景音乐很好听"）
* **1分 (琐碎信息):** 琐碎的、风格化的或背景性的描述。（例如："主持人穿着蓝色围裙"、"视频光线很好"）

---
### 格式与示例

你的输出必须是【纯粹的 JSON 格式】，可以被 `json.loads` 直接解析。JSON应包含一个 "wips" 键，其值为一个列表。如果文本中没有可提取的信息点，请返回 `{{"wips": []}}`。

**[示例输入]**
这是一段关于如何制作法式煎蛋卷的教程视频。主持人首先将三个鸡蛋打入碗中，并加入了盐和一小撮胡椒进行搅拌。视频强调了使用中火和不粘锅的重要性。接着，她在锅中融化了一块黄油，然后倒入蛋液。在烹饪过程中，她不断晃动平底锅，并将边缘的蛋液推向中心。最后，她将煎蛋卷折叠成三折，盛入盘中。整个过程非常快速。

**[示例输出]**
```json
{{
  "wips": [
    {{
      "info_point": "教程：如何制作法式煎蛋卷",
      "importance_score": 5
    }},
    {{
      "info_point": "使用三个鸡蛋，加盐和胡椒搅拌",
      "importance_score": 3
    }},
    {{
      "info_point": "强调使用中火",
      "importance_score": 4
    }},
    {{
      "info_point": "使用不粘锅和黄油",
      "importance_score": 4
    }},
    {{
      "info_point": "晃动锅并将蛋液边缘推向中心",
      "importance_score": 3
    }},
    {{
      "info_point": "煎蛋卷被折叠成三折",
      "importance_score": 2
    }},
    {{
      "info_point": "烹饪过程快速",
      "importance_score": 1
    }}
  ]
}}
```

现在，请开始分析我提供的描述性文字:
{}

你的输出结果 (请严格按照上述要求返回一个格式规整的 JSON，可以被 json.loads 直接解析。请不要在 JSON 数据前后添加任何额外的解释性文字或代码块标记): """


WIP_MATCHING_PROMPT = """你是一位极其严谨的**语义匹配专家**。你的任务是精确地对比两组关于同一个视频摘要的结构化信息点 (WIPs)，并找出它们之间的匹配关系。

**背景信息:**
- **Ground Truth WIPs (GT列表)**: 这是视频摘要的"事实标准"，代表视频中真实存在的所有核心信息。每个点都有一个 [1-5] 的重要性分数 (`importance_score`)。
- **Model-Generated WIPs (模型列表)**: 这是由一个AI模型生成的摘要信息点，代表它"声称"在视频中看到的内容。每个点也有一个 [1-5] 的重要性分数。

**你的核心任务:**
对比这两个列表，并输出一个包含三类结果的JSON对象：
1.  **`matches`**: 一个匹配对的列表。对于"模型列表"中的每一个项，如果在"GT列表"中找到了一个**语义上非常相似**的对应项，就将它们配对。
2.  **`unmatched_model_wips` (幻觉)**: "模型列表"中，那些在"GT列表"里找不到任何合理对应项的条目。这些代表了模型的**幻觉 (False Positives)**。
3.  **`unmatched_gt_wips` (漏报)**: "GT列表"中，那些没有被"模型列表"中任何条目匹配到的条目。这些代表了模型的**漏报 (False Negatives)**。

**至关重要的匹配规则:**
1.  **语义核心**: 匹配的核心是 `info_point` 的语义。
2.  **部分匹配**: 如果两个 `info_point` 语义上"部分重叠"但"不完全相同"，你【也应该】将它们匹配。
    * (例如): GT的 `"一场激烈精彩的篮球比赛"` 和 Gen的 `"球员在打篮球"` 应该被【匹配】(因为核心"篮球"匹配上了)。
    * (例如): GT的 `"评测《魔龙巢穴：暗影崛起》"` 和 Gen的 `"评测《魔龙巢穴：冰封王座》"` 应该被【匹配】(因为核心"《魔龙巢穴》评测"匹配上了)。
3.  **一对一匹配**: 找出最佳的匹配组合。

---
**[输出结构示例]**

**[输入]**
- GT列表: `[
    {{"info_point": "节气是秋分", "importance_score": 5}},
    {{"info_point": "农民在收割稻谷", "importance_score": 4}}
  ]`
- 模型列表: `[
    {{"info_point": "这是一个关于秋分的视频", "importance_score": 4}},
    {{"info_point": "狗在田里跑", "importance_score": 1}}
  ]`

**[你的输出]**
```json
{{
  "matches": [
    {{
      "model_wip": {{"info_point": "这是一个关于秋分的视频", "importance_score": 4}},
      "gt_wip": {{"info_point": "节气是秋分", "importance_score": 5}}
    }}
  ],
  "unmatched_model_wips": [
    {{
      "info_point": "狗在田里跑",
      "importance_score": 1
    }}
  ],
  "unmatched_gt_wips": [
    {{
      "info_point": "农民在收割稻谷",
      "importance_score": 4
    }}
  ]
}}
```

现在，请开始你的匹配工作:

[Ground Truth WIPs (GT列表)]

{}

[Model-Generated WIPs (模型列表)]

{}

你的匹配结果 (请严格按照上述要求返回一个格式规整的 JSON，可以被 json.loads 直接解析。请不要在 JSON 数据前后添加任何额外的解释性文字或代码块标记): """


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON from LLM response (simplified version for well-behaved LLMs).
    """
    if not response:
        return None
    
    try:
        response = response.rstrip('```').lstrip('```json')
        return json.loads(response.strip())
    except json.JSONDecodeError:
        print(response)
        return None


def extract_wips_single(
    text: str,
    llm_client
) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Extract WIPs from a single text using LLM.

    Args:
        text: Input text to extract WIPs from
        llm_client: LLM client instance (with built-in retry mechanism)

    Returns:
        Tuple of (wips_list, error_message)
        - wips_list: List of WIP dicts if successful, None if failed
        - error_message: Error message if failed, None if successful
    """
    prompt = WIP_EXTRACTION_PROMPT.format(text)

    try:
        response = llm_client.generate(prompt)
        result = extract_json_from_response(response)

        if result is not None and "wips" in result:
            return result["wips"], None

        return None, "Failed to parse JSON from response"

    except Exception as e:
        return None, f"API error: {str(e)}"


def extract_wips_batch(
    texts: Dict[str, str],
    llm_client,
    max_workers: int = 5,
    desc: str = "Extracting WIPs"
) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
    """
    Extract WIPs from multiple texts in parallel.

    Args:
        texts: Dict of {sample_id: text}
        llm_client: LLM client instance (with built-in retry mechanism)
        max_workers: Number of concurrent workers
        desc: Progress bar description

    Returns:
        Tuple of (results, errors):
        - results: Dict of {sample_id: wips_list}
        - errors: Dict of {sample_id: error_message}
    """
    results = {}
    errors = {}

    def process_single(sample_id: str, text: str):
        wips, error = extract_wips_single(text, llm_client)
        return sample_id, wips, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, sid, text): sid
            for sid, text in texts.items()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            sample_id, wips, error = future.result()
            if wips:
                results[sample_id] = wips
            if error:
                errors[sample_id] = error

    # Statistics: count valid (non-empty) extraction results
    total_attempted = len(texts)
    total_parsed = len(results)
    valid_results = sum(1 for wips in results.values() if wips)  # Count non-empty lists

    console.print(f"[cyan]{desc} statistics: {total_attempted} attempted, {total_parsed} parsed, {valid_results} valid (non-empty)[/cyan]")

    return results, errors


def match_wips_single(
    gt_wips: List[Dict],
    model_wips: List[Dict],
    llm_client
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Match WIPs between ground truth and model generation.

    Args:
        gt_wips: Ground truth WIPs list
        model_wips: Model-generated WIPs list
        llm_client: LLM client instance (with built-in retry mechanism)

    Returns:
        Tuple of (match_result, error_message)
    """
    gt_str = json.dumps(gt_wips, ensure_ascii=False, indent=2)
    model_str = json.dumps(model_wips, ensure_ascii=False, indent=2)
    prompt = WIP_MATCHING_PROMPT.format(gt_str, model_str)

    try:
        response = llm_client.generate(prompt)
        result = extract_json_from_response(response)

        if result is not None and all(k in result for k in ["matches", "unmatched_model_wips", "unmatched_gt_wips"]):
            return result, None

        return None, "Failed to parse match JSON from response"

    except Exception as e:
        return None, f"API error: {str(e)}"


def match_wips_batch(
    gt_wips_dict: Dict[str, List[Dict]],
    model_wips_dict: Dict[str, List[Dict]],
    llm_client,
    max_workers: int = 5
) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Match WIPs for multiple samples in parallel.

    Args:
        gt_wips_dict: Dict of {sample_id: gt_wips_list}
        model_wips_dict: Dict of {sample_id: model_wips_list}
        llm_client: LLM client instance (with built-in retry mechanism)
        max_workers: Number of concurrent workers

    Returns:
        Tuple of (results, errors)
    """
    results = {}
    errors = {}

    # Only match samples that have both GT and model WIPs (and both are non-empty)
    common_ids = {
        id for id in (set(gt_wips_dict.keys()) & set(model_wips_dict.keys()))
        if gt_wips_dict[id] and model_wips_dict[id]
    }

    def process_single(sample_id: str):
        gt_wips = gt_wips_dict[sample_id]
        model_wips = model_wips_dict[sample_id]
        match_result, error = match_wips_single(gt_wips, model_wips, llm_client)
        return sample_id, match_result, error

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single, sid): sid
            for sid in common_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Matching WIPs"):
            sample_id, match_result, error = future.result()
            if match_result is not None:
                results[sample_id] = match_result
            if error is not None:
                errors[sample_id] = error

    # Statistics: count valid (non-empty) match results
    total_attempted = len(common_ids)
    total_parsed = len(results)
    valid_results = 0

    for sample_id, match_result in results.items():
        # Check if result is not empty (has at least one non-empty field)
        if match_result:
            matches = match_result.get("matches", [])
            unmatched_model = match_result.get("unmatched_model_wips", [])
            unmatched_gt = match_result.get("unmatched_gt_wips", [])

            # Consider valid if result has any content
            if matches or unmatched_model or unmatched_gt:
                valid_results += 1

    console.print(f"[cyan]Matching statistics: {total_attempted} attempted, {total_parsed} parsed, {valid_results} valid (non-empty)[/cyan]")

    return results, errors


def get_wip_score_int(wip: Optional[Dict]) -> int:
    """Get importance score from WIP, defaulting to 1."""
    if not wip:
        return 1
    return wip.get("importance_score", 1)


def calculate_unweighted_metrics(match_results: Dict[str, Dict], core_threshold: int = 5) -> Dict[str, Any]:
    """
    Calculate unweighted metrics (count-based) with macro and per-sample versions.

    Args:
        match_results: Dict of {sample_id: match_result}
        core_threshold: Threshold for core WIPs (importance_score >= threshold)

    Returns:
        Dict with macro F1, core versions, and per-sample F1s (unweighted)
    """
    if not match_results:
        return {}

    # Per-sample metrics (for macro calculation)
    per_sample = {}

    for sample_id, result in match_results.items():
        if not result:
            per_sample[sample_id] = {"overall_f1": 0.0, "core_f1": 0.0}
            continue

        # Sample-level counts
        sample_tp = len(result.get("matches", []))
        sample_fp = len(result.get("unmatched_model_wips", []))
        sample_fn = len(result.get("unmatched_gt_wips", []))

        sample_core_tp = 0
        sample_core_fp = 0
        sample_core_fn = 0

        # Core: count only WIPs with importance_score >= threshold
        for match in result.get("matches", []):
            gt_wip = match.get("gt_wip", {})
            if get_wip_score_int(gt_wip) >= core_threshold:
                sample_core_tp += 1

        for fp_wip in result.get("unmatched_model_wips", []):
            if get_wip_score_int(fp_wip) >= core_threshold:
                sample_core_fp += 1

        for fn_wip in result.get("unmatched_gt_wips", []):
            if get_wip_score_int(fn_wip) >= core_threshold:
                sample_core_fn += 1

        # Calculate per-sample F1s
        sample_overall_f1 = 2 * sample_tp / (2 * sample_tp + sample_fp + sample_fn) if (2 * sample_tp + sample_fp + sample_fn) > 0 else 0.0
        sample_core_f1 = 2 * sample_core_tp / (2 * sample_core_tp + sample_core_fp + sample_core_fn) if (2 * sample_core_tp + sample_core_fp + sample_core_fn) > 0 else 0.0

        per_sample[sample_id] = {
            "overall_f1": sample_overall_f1,
            "core_f1": sample_core_f1,
        }

    # Calculate macro F1 (average of per-sample F1s)
    valid_samples = [v for v in per_sample.values() if v]
    macro_f1 = sum(s["overall_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0
    macro_core_f1 = sum(s["core_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0

    return {
        "macro_wip_unweighted_f1": macro_f1,
        "macro_wip_unweighted_core_f1": macro_core_f1,
        "per_sample": per_sample,
    }


def calculate_importance_weighted_metrics(
    match_results: Dict[str, Dict],
    core_threshold: int = 5
) -> Dict[str, Any]:
    """
    Calculate importance-weighted metrics (weighted by importance_score only) with macro and per-sample versions.

    Args:
        match_results: Dict of {sample_id: match_result}
        core_threshold: Threshold for core WIPs (importance_score >= threshold)

    Returns:
        Dict with macro F1, core versions, and per-sample F1s (importance-weighted)
    """
    if not match_results:
        return {}

    # Per-sample metrics (for macro calculation)
    per_sample = {}

    for sample_id, result in match_results.items():
        if not result:
            per_sample[sample_id] = {"overall_f1": 0.0, "core_f1": 0.0}
            continue

        # Sample-level metrics
        sample_tp, sample_fp, sample_fn = 0.0, 0.0, 0.0
        sample_core_tp, sample_core_fp, sample_core_fn = 0.0, 0.0, 0.0

        # TP from matches (use GT score)
        for match in result.get("matches", []):
            gt_wip = match.get("gt_wip")
            gt_score = get_wip_score_int(gt_wip)
            sample_tp += gt_score
            if gt_score >= core_threshold:
                sample_core_tp += gt_score

        # FP from unmatched model WIPs
        for fp_wip in result.get("unmatched_model_wips", []):
            fp_score = get_wip_score_int(fp_wip)
            sample_fp += fp_score
            if fp_score >= core_threshold:
                sample_core_fp += fp_score

        # FN from unmatched GT WIPs
        for fn_wip in result.get("unmatched_gt_wips", []):
            fn_score = get_wip_score_int(fn_wip)
            sample_fn += fn_score
            if fn_score >= core_threshold:
                sample_core_fn += fn_score

        # Calculate per-sample F1s
        sample_overall_f1 = 2 * sample_tp / (2 * sample_tp + sample_fp + sample_fn) if (2 * sample_tp + sample_fp + sample_fn) > 0 else 0.0
        sample_core_f1 = 2 * sample_core_tp / (2 * sample_core_tp + sample_core_fp + sample_core_fn) if (2 * sample_core_tp + sample_core_fp + sample_core_fn) > 0 else 0.0

        per_sample[sample_id] = {
            "overall_f1": sample_overall_f1,
            "core_f1": sample_core_f1,
        }

    # Calculate macro F1 (average of per-sample F1s)
    valid_samples = [v for v in per_sample.values() if v]
    macro_f1 = sum(s["overall_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0
    macro_core_f1 = sum(s["core_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0

    return {
        "macro_wip_importance_weighted_f1": macro_f1,
        "macro_wip_importance_weighted_core_f1": macro_core_f1,
        "per_sample": per_sample,
    }


def calculate_double_weighted_metrics(
    match_results: Dict[str, Dict],
    core_threshold: int = 5,
) -> Dict[str, Any]:
    """
    Calculate double-weighted metrics using V6.2 logic (importance_score × match_quality) with macro and per-sample versions.

    NOTE: This function now uses pre-computed match_quality from match results (no BERTScore computation here).

    V6.2 Logic:
    - For matched pairs:
        - TP = gt_score × match_quality
        - FN = gt_score × (1 - match_quality)
        - FP = model_score × (1 - match_quality)
    - For unmatched GT WIPs: FN += gt_score (complete miss)
    - For unmatched model WIPs: FP += model_score (complete hallucination)

    Args:
        match_results: Dict of {sample_id: match_result} (with pre-computed match_quality)
        core_threshold: Threshold for core WIPs (importance_score >= threshold)

    Returns:
        Dict with macro F1, core versions, and per-sample F1s (double-weighted)
    """
    if not match_results:
        return {}

    # Per-sample metrics (for macro calculation)
    per_sample = {}

    for sample_id, result in match_results.items():
        if not result:
            per_sample[sample_id] = {"overall_f1": 0.0, "core_f1": 0.0}
            continue

        # Sample-level metrics
        sample_tp, sample_fp, sample_fn = 0.0, 0.0, 0.0
        sample_core_tp, sample_core_fp, sample_core_fn = 0.0, 0.0, 0.0

        # Process matched pairs using pre-computed match_quality
        matches = result.get("matches", [])
        for match in matches:
            gt_wip = match.get("gt_wip", {})
            model_wip = match.get("model_wip", {})
            match_quality = match.get("match_quality")

            # Skip if match_quality not computed
            if match_quality is None:
                continue

            gt_score = get_wip_score_int(gt_wip)
            model_score = get_wip_score_int(model_wip)

            # V6.2 formulas for all WIPs
            tp_contrib = gt_score * match_quality
            fn_contrib = gt_score * (1 - match_quality)
            fp_contrib = model_score * (1 - match_quality)

            sample_tp += tp_contrib
            sample_fn += fn_contrib
            sample_fp += fp_contrib

            # Core: V6.2 formulas only for WIPs with importance_score >= threshold
            if gt_score >= core_threshold:
                sample_core_tp += tp_contrib
                sample_core_fn += fn_contrib
            if model_score >= core_threshold:
                sample_core_fp += fp_contrib

        # Complete misses (unmatched GT WIPs)
        for fn_wip in result.get("unmatched_gt_wips", []):
            fn_score = get_wip_score_int(fn_wip)
            sample_fn += fn_score
            if fn_score >= core_threshold:
                sample_core_fn += fn_score

        # Complete hallucinations (unmatched model WIPs)
        for fp_wip in result.get("unmatched_model_wips", []):
            fp_score = get_wip_score_int(fp_wip)
            sample_fp += fp_score
            if fp_score >= core_threshold:
                sample_core_fp += fp_score

        # Calculate per-sample F1s
        sample_overall_f1 = 2 * sample_tp / (2 * sample_tp + sample_fp + sample_fn) if (2 * sample_tp + sample_fp + sample_fn) > 0 else 0.0
        sample_core_f1 = 2 * sample_core_tp / (2 * sample_core_tp + sample_core_fp + sample_core_fn) if (2 * sample_core_tp + sample_core_fp + sample_core_fn) > 0 else 0.0

        per_sample[sample_id] = {
            "overall_f1": sample_overall_f1,
            "core_f1": sample_core_f1,
        }

    # Calculate macro F1 (average of per-sample F1s)
    valid_samples = [v for v in per_sample.values() if v]
    macro_f1 = sum(s["overall_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0
    macro_core_f1 = sum(s["core_f1"] for s in valid_samples) / len(valid_samples) if valid_samples else 0.0

    return {
        "macro_wip_double_weighted_f1": macro_f1,
        "macro_wip_double_weighted_core_f1": macro_core_f1,
        "per_sample": per_sample,
    }


def save_wip_detailed_results(
    save_dir: str,
    sample_ids: List[str],
    gt_wips_dict: Dict[str, List[Dict]],
    model_wips_dict: Dict[str, List[Dict]],
    match_results_dict: Dict[str, Dict],
    per_sample_f1s_dict: Dict[str, Dict],
    predictions_dict: Dict[str, str],
    references_dict: Dict[str, str],
    filename: str = "wip_results.json"
):
    """
    Save detailed WIP evaluation results to file.

    This saves:
    - prediction: Model prediction text for each sample
    - reference: Ground truth reference text for each sample
    - gt_wips: Ground truth information points for each sample
    - model_wips: Model-generated information points for each sample
    - match_result: Matching results (matches, unmatched_gt_wips, unmatched_model_wips)
    - per_sample_f1s: 6 F1 scores for each sample (3 types × 2 versions)

    Args:
        save_dir: Directory to save the file
        sample_ids: List of sample IDs
        gt_wips_dict: Dict of {sample_id: gt_wips_list}
        model_wips_dict: Dict of {sample_id: model_wips_list}
        match_results_dict: Dict of {sample_id: match_result}
        per_sample_f1s_dict: Dict of {sample_id: f1_scores_dict} with 6 F1 scores
        predictions_dict: Dict of {sample_id: prediction_text}
        references_dict: Dict of {sample_id: reference_text}
        filename: Output filename
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    detailed_results = {}

    for sample_id in sample_ids:
        detailed_results[sample_id] = {
            "prediction": predictions_dict.get(sample_id, ""),
            "reference": references_dict.get(sample_id, ""),
            "gt_wips": gt_wips_dict.get(sample_id, []),
            "model_wips": model_wips_dict.get(sample_id, []),
            "match_result": match_results_dict.get(sample_id, {}),
            "f1_scores": per_sample_f1s_dict.get(sample_id, {}),
        }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    console.print(f"[green]WIP detailed results saved to {save_path}[/green]")


def get_gt_cache_path(cache_dir: str, model_name: str) -> str:
    """Get the path for GT WIPs cache file."""
    return os.path.join(cache_dir, f"test_gt_wip_{model_name}.parquet")


def load_wip_results_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    """
    Load previously saved WIP results (model_wips and match_results).

    Args:
        cache_path: Path to wip_{model_name}.json file

    Returns:
        Dict with "model_wips" and "match_results" or None if not found
    """
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract model_wips and match_results from saved file
        model_wips = {}
        match_results = {}

        for sample_id, sample_data in data.items():
            # Load model_wips except it's an empty list
            if "model_wips" in sample_data and sample_data["model_wips"]:
                model_wips[sample_id] = sample_data["model_wips"]
            # Load match_result only if it's not empty (empty dict means no matching was done)
            if "match_result" in sample_data and sample_data["match_result"]:
                match_results[sample_id] = sample_data["match_result"]

        return {
            "model_wips": model_wips,
            "match_results": match_results,
        }

    except Exception as e:
        console.print(f"[yellow]Failed to load WIP results cache: {e}[/yellow]")
        return None


def load_gt_wips_cache(cache_path: str) -> Optional[Dict[str, List[Dict]]]:
    """
    Load GT WIPs from cache file.

    Args:
        cache_path: Path to parquet cache file

    Returns:
        Dict of {sample_id: wips_list} or None if not found
    """
    if not os.path.exists(cache_path):
        return None

    try:
        df = pd.read_parquet(cache_path)
        result = {}
        for _, row in df.iterrows():
            sample_id = str(row["sample_id"])
            wips = row["wips"]
            if isinstance(wips, str):
                wips = json.loads(wips)
            if wips:
                result[sample_id] = wips
        console.print(f"[green]Loaded GT WIPs cache from {cache_path} ({len(result)} samples)[/green]")
        return result
    except Exception as e:
        console.print(f"[yellow]Failed to load GT cache: {e}[/yellow]")
        return None


def save_gt_wips_cache(gt_wips: Dict[str, List[Dict]], cache_path: str):
    """
    Save GT WIPs to cache file.

    Args:
        gt_wips: Dict of {sample_id: wips_list}
        cache_path: Path to save parquet file
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    data = []
    for sample_id, wips in gt_wips.items():
        data.append({
            "sample_id": sample_id,
            "wips": json.dumps(wips, ensure_ascii=False)
        })

    df = pd.DataFrame(data)
    df.to_parquet(cache_path, index=False)
    console.print(f"[green]Saved GT WIPs cache to {cache_path} ({len(gt_wips)} samples)[/green]")


def _load_or_extract_gt_wips(
    sample_ids: List[str],
    references: Dict[str, str],
    llm_client,
    max_workers: int,
    gt_cache_dir: Optional[str],
    model_name: str,
) -> Dict[str, List[Dict]]:
    """Load GT WIPs from cache or extract if missing."""
    gt_wips = None
    if gt_cache_dir:
        cache_path = get_gt_cache_path(gt_cache_dir, model_name)
        full_gt_cache = load_gt_wips_cache(cache_path)

        if full_gt_cache is not None:
            gt_wips = {id: full_gt_cache[id] for id in sample_ids if id in full_gt_cache}

            missing_ids = set(sample_ids) - set(gt_wips.keys())
            if missing_ids:
                console.print(f"[yellow]Missing {len(missing_ids)} samples in GT cache, extracting...[/yellow]")
                missing_refs = {id: references[id] for id in missing_ids}
                new_gt_wips, gt_errors = extract_wips_batch(
                    missing_refs, llm_client, max_workers, "Extracting GT WIPs"
                )
                gt_wips.update(new_gt_wips)
                full_gt_cache.update(new_gt_wips)
                save_gt_wips_cache(full_gt_cache, cache_path)

                if gt_errors:
                    console.print(f"[red]GT extraction errors: {len(gt_errors)} samples[/red]")

    if gt_wips is None:
        console.print("[cyan]Extracting GT WIPs...[/cyan]")
        gt_wips, gt_errors = extract_wips_batch(
            references, llm_client, max_workers, "Extracting GT WIPs"
        )

        if gt_cache_dir:
            cache_path = get_gt_cache_path(gt_cache_dir, model_name)
            save_gt_wips_cache(gt_wips, cache_path)

        if gt_errors:
            console.print(f"[red]GT extraction errors: {len(gt_errors)} samples[/red]")

    return gt_wips

# Extract text after last </think> tag if present
def extract_after_think(text: str) -> str:
    """Extract text after the last </think> tag"""
    if '</think>' in text:
        return text.split('</think>')[-1].strip()
    return text

def _load_or_extract_model_wips(
    sample_ids: List[str],
    predictions: Dict[str, str],
    gt_wips: Dict[str, List[Dict]],
    llm_client,
    max_workers: int,
    save_dir: Optional[str],
    model_name: str,
) -> Dict[str, List[Dict]]:
    """Load Model WIPs from cache or extract if missing (incremental)."""
    model_wips = {}

    if save_dir:
        wip_cache_path = os.path.join(save_dir, f"wip_{model_name}.json")
        cached_data = load_wip_results_cache(wip_cache_path)
        if cached_data:
            model_wips = cached_data.get("model_wips", {})
            console.print(f"[green]Loaded {len(model_wips)} cached model_wips[/green]")

    missing_model_ids = set(sample_ids) - set(model_wips.keys())
    missing_model_ids = {id for id in missing_model_ids if id in gt_wips}

    if missing_model_ids:
        console.print(f"[cyan]Extracting Model WIPs for {len(missing_model_ids)} missing samples...[/cyan]")
        missing_predictions = {id: extract_after_think(predictions[id]) for id in missing_model_ids}
        new_model_wips, model_errors = extract_wips_batch(
            missing_predictions, llm_client, max_workers, "Extracting Model WIPs"
        )
        model_wips.update(new_model_wips)

        if model_errors:
            console.print(f"[red]Model extraction errors: {len(model_errors)} samples[/red]")
    else:
        console.print(f"[green]All {len(sample_ids)} samples already have Model WIPs (from cache)[/green]")

    return model_wips


def _load_or_match_wips(
    sample_ids: List[str],
    gt_wips: Dict[str, List[Dict]],
    model_wips: Dict[str, List[Dict]],
    llm_client,
    max_workers: int,
    save_dir: Optional[str],
    model_name: str,
) -> Dict[str, Dict]:
    """Load match results from cache or match if missing (incremental)."""
    match_results = {}

    if save_dir:
        wip_cache_path = os.path.join(save_dir, f"wip_{model_name}.json")
        cached_data = load_wip_results_cache(wip_cache_path)
        if cached_data:
            match_results = cached_data.get("match_results", {})
            console.print(f"[green]Loaded {len(match_results)} cached match_results[/green]")

    missing_match_ids = set(sample_ids) - set(match_results.keys())
    # Only match if both gt_wips and model_wips exist and are non-empty
    missing_match_ids = {
        id for id in missing_match_ids
        if id in gt_wips and id in model_wips and gt_wips[id] and model_wips[id]
    }

    if missing_match_ids:
        console.print(f"[cyan]Matching WIPs for {len(missing_match_ids)} missing samples...[/cyan]")
        missing_gt_wips = {id: gt_wips[id] for id in missing_match_ids}
        missing_model_wips = {id: model_wips[id] for id in missing_match_ids}
        new_match_results, match_errors = match_wips_batch(
            missing_gt_wips, missing_model_wips, llm_client, max_workers
        )
        match_results.update(new_match_results)

        if match_errors:
            console.print(f"[red]Matching errors: {len(match_errors)} samples[/red]")
    else:
        console.print(f"[green]All {len(sample_ids)} samples already have match results (from cache)[/green]")

    return match_results


def _compute_bertscore_incremental(
    sample_ids: List[str],
    match_results: Dict[str, Dict],
    bertscore_model: str,
    bertscore_num_layers: int,
) -> None:
    """Compute BERTScore for matches that don't have it yet (incremental, in-place update)."""
    import torch
    from bert_score import BERTScorer

    console.print("[cyan]Computing BERTScore for matched pairs...[/cyan]")

    all_gt_texts = []
    all_model_texts = []
    sample_match_indices = []

    for sample_id in sample_ids:
        if sample_id in match_results:
            matches = match_results[sample_id].get("matches", [])
            for match_idx, match in enumerate(matches):
                if match.get("match_quality") is not None:
                    continue

                gt_wip = match.get("gt_wip")
                model_wip = match.get("model_wip")

                # Skip if either wip is None or not a dict
                if not gt_wip or not isinstance(gt_wip, dict) or not model_wip or not isinstance(model_wip, dict):
                    continue

                gt_text = gt_wip.get("info_point", "")
                model_text = model_wip.get("info_point", "")

                if gt_text and model_text:
                    batch_idx = len(all_gt_texts)
                    all_gt_texts.append(gt_text)
                    all_model_texts.append(model_text)
                    sample_match_indices.append((sample_id, match_idx, batch_idx))

    if all_gt_texts and all_model_texts:
        console.print(f"[cyan]Computing BERTScore for {len(all_gt_texts)} new matched pairs...[/cyan]")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = BERTScorer(
            model_type=bertscore_model,
            num_layers=bertscore_num_layers,
            device=device,
            lang="zh",
            rescale_with_baseline=False,
        )

        try:
            P, R, F1 = scorer.score(all_model_texts, all_gt_texts)
            match_qualities = F1.tolist()

            for sample_id, match_idx, batch_idx in sample_match_indices:
                match_results[sample_id]["matches"][match_idx]["match_quality"] = match_qualities[batch_idx]

            console.print(f"[green]Computed BERTScore for {len(match_qualities)} matched pairs[/green]")
        except Exception as e:
            console.print(f"[red]BERTScore computation failed: {e}[/red]")
            for sample_id, match_idx, _ in sample_match_indices:
                match_results[sample_id]["matches"][match_idx]["match_quality"] = None
    else:
        console.print(f"[green]All matches already have BERTScore (from cache)[/green]")


def evaluate_wip(
    predictions: Dict[str, str],
    references: Dict[str, str],
    llm_client,
    max_workers: int = 5,
    max_samples: Optional[int] = None,
    gt_cache_dir: Optional[str] = None,
    model_name: str = "unknown",
    save_dir: Optional[str] = None,
    bertscore_model: str = "bert-base-chinese",
    bertscore_num_layers: int = 9,
    core_threshold: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Main WIP evaluation function with six types of F1 metrics (3 types × 2 versions).

    Computes:
    1. Unweighted F1 (count-based) - overall and core
    2. Importance-weighted F1 (weighted by importance_score only) - overall and core
    3. Double-weighted F1 (V6.2 logic: importance_score × match_quality) - overall and core

    Args:
        predictions: Dict of {sample_id: prediction_text}
        references: Dict of {sample_id: reference_text}
        llm_client: LLM client instance for judge (with built-in retry mechanism)
        max_workers: Number of concurrent workers
        max_samples: Maximum samples to evaluate (None for all)
        gt_cache_dir: Directory for GT WIPs cache
        model_name: Model name for cache file naming
        save_dir: Directory to save results
        bertscore_model: BERT model for computing match quality
        bertscore_num_layers: Number of layers for BERTScore
        core_threshold: Threshold for core WIPs (importance_score >= threshold)

    Returns:
        Tuple of (metrics, per_sample_metrics):
        - metrics: Dict with 6 F1 scores and sample count (flattened)
        - per_sample_metrics: Dict of {sample_id: {6 F1 scores}}
    """
    # Select samples (sorted by sample_id for consistency)
    all_sample_ids = sorted(set(predictions.keys()) & set(references.keys()))

    if max_samples is not None and max_samples < len(all_sample_ids):
        sample_ids = all_sample_ids[:max_samples]
        console.print(f"[cyan]Selected {len(sample_ids)} samples for WIP evaluation (sorted by sample_id)[/cyan]")
    else:
        sample_ids = all_sample_ids
        console.print(f"[cyan]Evaluating all {len(sample_ids)} samples for WIP[/cyan]")

    selected_predictions = {id: predictions[id] for id in sample_ids}
    selected_references = {id: references[id] for id in sample_ids}

    # Step 1: Load/Extract GT WIPs
    gt_wips = _load_or_extract_gt_wips(
        sample_ids, selected_references, llm_client, max_workers, gt_cache_dir, model_name
    )

    # Step 2: Load/Extract Model WIPs (incremental)
    model_wips = _load_or_extract_model_wips(
        sample_ids, selected_predictions, gt_wips, llm_client, max_workers, save_dir, model_name
    )

    # Step 3: Load/Match WIPs (incremental)
    match_results = _load_or_match_wips(
        sample_ids, gt_wips, model_wips, llm_client, max_workers, save_dir, model_name
    )

    # Step 3.5: Compute BERTScore (incremental)
    _compute_bertscore_incremental(sample_ids, match_results, bertscore_model, bertscore_num_layers)

    # Step 4: Calculate three types of metrics (each with overall, core, and per-sample)
    console.print("[cyan]Computing all metrics (unweighted, importance-weighted, double-weighted)...[/cyan]")

    # 4.1: Unweighted metrics
    unweighted_metrics = calculate_unweighted_metrics(
        match_results,
        core_threshold=core_threshold
    )

    # 4.2: Importance-weighted metrics
    importance_metrics = calculate_importance_weighted_metrics(
        match_results,
        core_threshold=core_threshold
    )

    # 4.3: Double-weighted metrics (using pre-computed BERTScore)
    double_metrics = calculate_double_weighted_metrics(
        match_results,
        core_threshold=core_threshold
    )

    # Flattened overall metrics (6 F1 scores: 3 types × 2 versions)
    metrics = {
        # Macro F1 (average of per-sample F1s)
        "macro_wip_unweighted_f1": unweighted_metrics.get("macro_wip_unweighted_f1", 0.0),
        "macro_wip_unweighted_core_f1": unweighted_metrics.get("macro_wip_unweighted_core_f1", 0.0),
        "macro_wip_importance_weighted_f1": importance_metrics.get("macro_wip_importance_weighted_f1", 0.0),
        "macro_wip_importance_weighted_core_f1": importance_metrics.get("macro_wip_importance_weighted_core_f1", 0.0),
        "macro_wip_double_weighted_f1": double_metrics.get("macro_wip_double_weighted_f1", 0.0),
        "macro_wip_double_weighted_core_f1": double_metrics.get("macro_wip_double_weighted_core_f1", 0.0),
        "wip_num_samples": len(match_results),
    }

    # Merge per-sample metrics from all three types (6 F1 scores per sample, same as before since per-sample is used for macro)
    per_sample_metrics = {}
    for sample_id in sample_ids:
        unweighted_per_sample = unweighted_metrics.get("per_sample", {}).get(sample_id, {"overall_f1": 0.0, "core_f1": 0.0})
        importance_per_sample = importance_metrics.get("per_sample", {}).get(sample_id, {"overall_f1": 0.0, "core_f1": 0.0})
        double_per_sample = double_metrics.get("per_sample", {}).get(sample_id, {"overall_f1": 0.0, "core_f1": 0.0})

        per_sample_metrics[sample_id] = {
            "wip_unweighted_f1": unweighted_per_sample["overall_f1"],
            "wip_unweighted_core_f1": unweighted_per_sample["core_f1"],
            "wip_importance_weighted_f1": importance_per_sample["overall_f1"],
            "wip_importance_weighted_core_f1": importance_per_sample["core_f1"],
            "wip_double_weighted_f1": double_per_sample["overall_f1"],
            "wip_double_weighted_core_f1": double_per_sample["core_f1"],
        }

    # Step 5: Save detailed results to file
    if save_dir:
        console.print("[cyan]Saving WIP detailed results...[/cyan]")
        save_wip_detailed_results(
            save_dir=save_dir,
            sample_ids=sample_ids,
            gt_wips_dict=gt_wips,
            model_wips_dict=model_wips,
            match_results_dict=match_results,
            per_sample_f1s_dict=per_sample_metrics,
            predictions_dict=selected_predictions,
            references_dict=selected_references,
            filename=f"wip_{model_name}.json"
        )

    console.print(f"[green]WIP evaluation completed: {metrics['wip_num_samples']} samples[/green]")
    console.print(f"[green]  Macro Unweighted F1: {metrics['macro_wip_unweighted_f1']:.4f} (Core: {metrics['macro_wip_unweighted_core_f1']:.4f})[/green]")
    console.print(f"[green]  Macro Importance-weighted F1: {metrics['macro_wip_importance_weighted_f1']:.4f} (Core: {metrics['macro_wip_importance_weighted_core_f1']:.4f})[/green]")
    console.print(f"[green]  Macro Double-weighted F1: {metrics['macro_wip_double_weighted_f1']:.4f} (Core: {metrics['macro_wip_double_weighted_core_f1']:.4f})[/green]")

    return metrics, per_sample_metrics
