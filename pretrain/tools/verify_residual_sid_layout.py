"""Print the automatically discovered residual SID layout."""

from __future__ import annotations

import argparse
import collections
import json
import re

from transformers import AutoTokenizer


def checked_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or int(token_id) < 0:
        raise ValueError(f"Tokenizer does not contain {token!r}.")
    if (
        tokenizer.unk_token_id is not None
        and int(token_id) == int(tokenizer.unk_token_id)
        and token != tokenizer.unk_token
    ):
        raise ValueError(f"Tokenizer maps {token!r} to unk_token_id.")
    return int(token_id)


def discover(tokenizer) -> list[dict]:
    pattern = re.compile(r"^<s_(.+)_([0-9]+)>$")
    groups = collections.defaultdict(dict)
    for token, global_id in tokenizer.get_vocab().items():
        match = pattern.match(token)
        if match:
            groups[match.group(1)][int(match.group(2))] = int(global_id)

    layers = []
    for name, mapping in groups.items():
        if 0 not in mapping:
            continue
        size = max(mapping) + 1
        if set(mapping) != set(range(size)):
            raise ValueError(f"Layer {name!r} has missing local IDs.")
        start = mapping[0]
        for local_id, global_id in mapping.items():
            if global_id != start + local_id:
                raise ValueError(
                    f"Layer {name!r} is not globally contiguous."
                )
        layers.append({"name": name, "start": start, "size": size})
    layers.sort(key=lambda item: item["start"])
    return layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected_layers", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    layers = discover(tokenizer)
    if len(layers) != args.expected_layers:
        raise ValueError(
            f"Expected {args.expected_layers} layers, discovered {layers}."
        )

    print(
        json.dumps(
            {
                "model": args.model,
                "layers": layers,
                "sid_begin_token_id": checked_id(
                    tokenizer, "<|sid_begin|>"
                ),
                "sid_end_token_id": checked_id(
                    tokenizer, "<|sid_end|>"
                ),
                "itemic_id_range": [
                    layers[0]["start"],
                    max(
                        layer["start"] + layer["size"] - 1
                        for layer in layers
                    ),
                ],
                "num_residual_blocks": len(layers) - 1,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
